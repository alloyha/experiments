"""
Text Assignment Manager for Agentic Clusterizer
Handles dry assignment of texts to workflows, batches, and slots.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import tiktoken  # OpenAI's tokenizer library


@dataclass
class Text:
    """Represents a text document to be clustered."""
    id: str
    content: str
    token_count: Optional[int] = None
    
    def __post_init__(self):
        if self.token_count is None:
            self.token_count = 0  # Will be set by tokenizer


@dataclass
class Chunk:
    """Represents a chunk of a big text."""
    text_id: str
    chunk_idx: int
    content: str
    token_count: int
    start_pos: int
    end_pos: int


@dataclass
class SlotAssignment:
    """Represents assignment of text/chunk to a specific slot."""
    workflow_id: int
    batch_id: int
    slot_id: int
    type: str  # 'text' or 'chunk'
    text_id: str
    chunk_idx: Optional[int] = None
    content: str = ""
    token_count: int = 0


@dataclass
class BigTextInfo:
    """Tracks information about a big text across chunks."""
    text_id: str
    total_chunks: int
    original_size: int
    chunks: List[Dict] = field(default_factory=list)


@dataclass
class DryAssignmentResult:
    """Result of dry assignment phase."""
    workflow_assignments: List[List[List[SlotAssignment]]] = field(default_factory=list)
    text_placement_map: Dict[str, List[Tuple[int, int, int]]] = field(default_factory=dict)
    big_text_registry: Dict[str, BigTextInfo] = field(default_factory=dict)
    total_workflows_needed: int = 0
    total_batches: int = 0
    total_slots_used: int = 0
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Dry Assignment Summary ===",
            f"Total Workflows: {self.total_workflows_needed}",
            f"Total Batches: {self.total_batches}",
            f"Total Slots Used: {self.total_slots_used}",
            f"Total Texts: {len(self.text_placement_map)}",
            f"Big Texts (chunked): {len(self.big_text_registry)}",
            "",
            f"=== Big Text Details ===",
        ]
        
        for text_id, info in self.big_text_registry.items():
            workflows = set(chunk['workflow_id'] for chunk in info.chunks)
            batches = set((chunk['workflow_id'], chunk['batch_id']) for chunk in info.chunks)
            lines.append(
                f"  {text_id}: {info.total_chunks} chunks, "
                f"{info.original_size} tokens, "
                f"spans {len(workflows)} workflow(s), {len(batches)} batch(es)"
            )
        
        return "\n".join(lines)


class Tokenizer:
    """Wrapper around tiktoken for counting tokens."""
    
    def __init__(self, model: str = "cl100k_base"):
        """
        Initialize tokenizer.
        
        Args:
            model: Tokenizer model to use. Options:
                - "cl100k_base" (GPT-4, GPT-3.5-turbo)
                - "p50k_base" (Codex, text-davinci-002)
                - "r50k_base" (GPT-3 models)
        """
        self.encoding = tiktoken.get_encoding(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.encoding.decode(tokens)
    
    def chunk_text(self, text: str, max_tokens: int) -> List[Tuple[str, int, int]]:
        """
        Chunk text into pieces of max_tokens.
        
        Returns:
            List of (chunk_text, start_token_idx, end_token_idx)
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append((chunk_text, start, end))
            start = end
        
        return chunks


class TextAssignmentManager:
    """Manages dry assignment of texts to workflows."""
    
    def __init__(self, batch_size: int, max_batches_per_workflow: int, 
                 token_threshold: int, tokenizer: Optional[Tokenizer] = None):
        """
        Initialize assignment manager.
        
        Args:
            batch_size: Number of slots per batch (b)
            max_batches_per_workflow: Maximum batches per workflow (n_b)
            token_threshold: Token threshold for big texts (M)
            tokenizer: Tokenizer instance (creates default if None)
        """
        self.b = batch_size
        self.n_b = max_batches_per_workflow
        self.M = token_threshold
        self.tokenizer = tokenizer or Tokenizer()
    
    def prepare_texts(self, texts: List[Text]) -> List[Text]:
        """Count tokens for all texts."""
        for text in texts:
            text.token_count = self.tokenizer.count_tokens(text.content)
        return texts
    
    def dry_assign(self, texts: List[Text]) -> DryAssignmentResult:
        """
        Perform dry assignment of texts to workflows.
        
        Args:
            texts: List of texts to assign
            
        Returns:
            DryAssignmentResult with complete assignment plan
        """
        # Prepare texts (count tokens)
        texts = self.prepare_texts(texts)
        
        result = DryAssignmentResult()
        
        # State tracking
        current_workflow_id = 0
        current_batch_id = 0
        current_slot = 0
        current_batch = []
        current_workflow_batches = []
        
        for text in texts:
            if self._needs_chunking(text):
                # Big text - chunk and place
                chunks = self._create_chunks(text)
                
                # Register big text
                result.big_text_registry[text.id] = BigTextInfo(
                    text_id=text.id,
                    total_chunks=len(chunks),
                    original_size=text.token_count,
                    chunks=[]
                )
                
                # Place each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    state = self._place_item(
                        item_type='chunk',
                        text_id=text.id,
                        chunk_idx=chunk_idx,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        current_workflow_id=current_workflow_id,
                        current_batch_id=current_batch_id,
                        current_slot=current_slot,
                        current_batch=current_batch,
                        current_workflow_batches=current_workflow_batches,
                        result=result
                    )
                    
                    # Update state
                    current_workflow_id = state['workflow_id']
                    current_batch_id = state['batch_id']
                    current_slot = state['slot']
                    current_batch = state['batch']
                    current_workflow_batches = state['workflow_batches']
            else:
                # Regular text - place as-is
                state = self._place_item(
                    item_type='text',
                    text_id=text.id,
                    chunk_idx=None,
                    content=text.content,
                    token_count=text.token_count,
                    current_workflow_id=current_workflow_id,
                    current_batch_id=current_batch_id,
                    current_slot=current_slot,
                    current_batch=current_batch,
                    current_workflow_batches=current_workflow_batches,
                    result=result
                )
                
                # Update state
                current_workflow_id = state['workflow_id']
                current_batch_id = state['batch_id']
                current_slot = state['slot']
                current_batch = state['batch']
                current_workflow_batches = state['workflow_batches']
        
        # Finalize any remaining batch/workflow
        if current_batch:
            current_workflow_batches.append(current_batch)
        if current_workflow_batches:
            result.workflow_assignments.append(current_workflow_batches)
        
        # Set summary statistics
        result.total_workflows_needed = len(result.workflow_assignments)
        result.total_batches = sum(len(w) for w in result.workflow_assignments)
        result.total_slots_used = sum(
            len(batch) for workflow in result.workflow_assignments 
            for batch in workflow
        )
        
        return result
    
    def _needs_chunking(self, text: Text) -> bool:
        """Check if text needs chunking."""
        return text.token_count > self.M
    
    def _create_chunks(self, text: Text) -> List[Chunk]:
        """Create chunks from big text."""
        token_chunks = self.tokenizer.chunk_text(text.content, self.M)
        
        chunks = []
        for idx, (chunk_text, start_pos, end_pos) in enumerate(token_chunks):
            chunk = Chunk(
                text_id=text.id,
                chunk_idx=idx,
                content=chunk_text,
                token_count=self.tokenizer.count_tokens(chunk_text),
                start_pos=start_pos,
                end_pos=end_pos
            )
            chunks.append(chunk)
        
        return chunks
    
    def _place_item(self, item_type: str, text_id: str, chunk_idx: Optional[int],
                    content: str, token_count: int,
                    current_workflow_id: int, current_batch_id: int, current_slot: int,
                    current_batch: List, current_workflow_batches: List,
                    result: DryAssignmentResult) -> Dict:
        """Place a text or chunk into a slot."""
        
        # Check if fits in current batch
        if current_slot < self.b:
            # Create assignment
            assignment = SlotAssignment(
                workflow_id=current_workflow_id,
                batch_id=current_batch_id,
                slot_id=current_slot,
                type=item_type,
                text_id=text_id,
                chunk_idx=chunk_idx,
                content=content,
                token_count=token_count
            )
            
            current_batch.append(assignment)
            
            # Record placement
            if text_id not in result.text_placement_map:
                result.text_placement_map[text_id] = []
            result.text_placement_map[text_id].append(
                (current_workflow_id, current_batch_id, current_slot)
            )
            
            # If chunk, record in big text registry
            if item_type == 'chunk':
                result.big_text_registry[text_id].chunks.append({
                    'chunk_idx': chunk_idx,
                    'workflow_id': current_workflow_id,
                    'batch_id': current_batch_id,
                    'slot_id': current_slot,
                    'token_count': token_count
                })
            
            return {
                'workflow_id': current_workflow_id,
                'batch_id': current_batch_id,
                'slot': current_slot + 1,
                'batch': current_batch,
                'workflow_batches': current_workflow_batches
            }
        
        # Current batch full - move to next batch
        current_workflow_batches.append(current_batch)
        current_batch = []
        current_batch_id += 1
        current_slot = 0
        
        # Check if exceeded workflow capacity
        if current_batch_id >= self.n_b:
            # Need new workflow
            result.workflow_assignments.append(current_workflow_batches)
            current_workflow_batches = []
            current_workflow_id += 1
            current_batch_id = 0
        
        # Recursive call with new state
        return self._place_item(
            item_type, text_id, chunk_idx, content, token_count,
            current_workflow_id, current_batch_id, current_slot,
            current_batch, current_workflow_batches, result
        )


# ============================================================================
# TEST CODE
# ============================================================================

def create_test_texts() -> List[Text]:
    """Create sample texts for testing."""
    texts = [
        Text(
            id="short_1",
            content="This is a short text that fits easily in one slot."
        ),
        Text(
            id="short_2",
            content="Another short text for testing purposes."
        ),
        Text(
            id="medium_1",
            content=" ".join(["This is a medium-length text."] * 50)
        ),
        Text(
            id="big_1",
            content=" ".join([
                "This is a very long text that will need to be chunked into multiple pieces. "
                "It contains a lot of information spread across many sentences and paragraphs. "
                "The content is repetitive to ensure we exceed the token threshold. "
            ] * 100)
        ),
        Text(
            id="short_3",
            content="Yet another short text."
        ),
        Text(
            id="big_2",
            content=" ".join([
                "Another big text that spans multiple chunks. "
                "This one has different content to make testing more realistic. "
                "We want to see how the assignment manager handles multiple big texts. "
            ] * 80)
        ),
    ]
    
    # Add more short texts to fill batches
    for i in range(10):
        texts.append(Text(
            id=f"filler_{i}",
            content=f"Filler text number {i} to test batch allocation."
        ))
    
    return texts


def test_assignment_manager():
    """Test the assignment manager with sample texts."""
    print("=" * 80)
    print("Testing Text Assignment Manager")
    print("=" * 80)
    
    # Configuration
    batch_size = 5
    max_batches_per_workflow = 3
    token_threshold = 200
    
    print(f"\nConfiguration:")
    print(f"  Batch size (b): {batch_size}")
    print(f"  Max batches per workflow (n_b): {max_batches_per_workflow}")
    print(f"  Token threshold (M): {token_threshold}")
    
    # Create manager
    manager = TextAssignmentManager(
        batch_size=batch_size,
        max_batches_per_workflow=max_batches_per_workflow,
        token_threshold=token_threshold
    )
    
    # Create test texts
    texts = create_test_texts()
    print(f"\nCreated {len(texts)} test texts")
    
    # Perform dry assignment
    print("\nPerforming dry assignment...")
    result = manager.dry_assign(texts)
    
    # Print summary
    print("\n" + result.get_summary())
    
    # Print detailed workflow structure
    print("\n" + "=" * 80)
    print("Detailed Workflow Structure")
    print("=" * 80)
    
    for workflow_id, workflow in enumerate(result.workflow_assignments):
        print(f"\nWorkflow {workflow_id}:")
        for batch_id, batch in enumerate(workflow):
            print(f"  Batch {batch_id}:")
            for slot in batch:
                if slot.type == 'chunk':
                    print(f"    Slot {slot.slot_id}: [CHUNK] {slot.text_id} "
                          f"(chunk {slot.chunk_idx}, {slot.token_count} tokens)")
                else:
                    print(f"    Slot {slot.slot_id}: [TEXT] {slot.text_id} "
                          f"({slot.token_count} tokens)")
    
    # Test specific placement lookups
    print("\n" + "=" * 80)
    print("Sample Placement Lookups")
    print("=" * 80)
    
    for text_id in ["short_1", "big_1", "big_2"]:
        if text_id in result.text_placement_map:
            placements = result.text_placement_map[text_id]
            print(f"\n{text_id}: {len(placements)} placement(s)")
            for w_id, b_id, s_id in placements:
                print(f"  -> Workflow {w_id}, Batch {b_id}, Slot {s_id}")


if __name__ == "__main__":
    test_assignment_manager()