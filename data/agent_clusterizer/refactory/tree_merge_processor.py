"""
Tree Merge Processor for Agentic Clusterizer
Handles hierarchical merging of clustering results from the TextAssignmentManager.

Key Design:
- Each slot contains ≤ M tokens (enforced by TextAssignmentManager)
- All texts/chunks are treated uniformly as "small texts"
- No special handling needed for big texts during clustering
- Big text consolidation happens only at final merge
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


# Import your actual models
from agentic_clusterizer import (
    Category, CategoryAssignment, ClusterizerConfig,
    CONFIG_BALANCED_HYBRID, clusterize_texts
)


@dataclass
class ClusterResult:
    """Result from a single clusterizer execution."""
    categories: List[Category]
    assignments: List[CategoryAssignment]
    metadata: Dict[str, Any]
    execution_id: int


@dataclass
class BigTextTracker:
    """
    Tracks chunks of a big text across workflow executions.
    No locking needed - passive tracking only.
    """
    original_text_id: str
    total_chunks: int
    chunks_processed: int = 0
    chunk_assignments: List[Tuple[str, str, float]] = field(default_factory=list)  # (chunk_id, category_id, confidence)
    execution_ids: Set[int] = field(default_factory=set)
    
    def is_complete(self) -> bool:
        """Check if all chunks have been processed."""
        return self.chunks_processed >= self.total_chunks
    
    def add_chunk_assignment(self, chunk_id: str, category_id: str, confidence: float, execution_id: int):
        """Record a chunk's category assignment."""
        self.chunk_assignments.append((chunk_id, category_id, confidence))
        self.chunks_processed += 1
        self.execution_ids.add(execution_id)
    
    def get_consolidated_category(self) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Aggregate chunk assignments into final category assignment.
        
        Returns:
            (primary_category_id, primary_confidence, [(secondary_cat, confidence), ...])
        """
        if not self.chunk_assignments:
            return ("unknown", 0.0, [])
        
        # Vote by confidence-weighted frequency
        category_votes = defaultdict(list)
        for _, cat_id, confidence in self.chunk_assignments:
            category_votes[cat_id].append(confidence)
        
        # Calculate average confidence per category
        category_scores = {
            cat_id: sum(confidences) / len(confidences)
            for cat_id, confidences in category_votes.items()
        }
        
        # Sort by score
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_cat, primary_conf = sorted_categories[0]
        secondary = sorted_categories[1:4]  # Keep top 3 secondary categories
        
        return (primary_cat, primary_conf, secondary)


@dataclass
class MergeNode:
    """Node in the binary merge tree."""
    node_id: int
    level: int  # 0 = leaf (workflow execution), 1+ = merge result
    result: Optional[ClusterResult] = None
    children: List['MergeNode'] = field(default_factory=list)
    is_ready: bool = False


class TreeMergeProcessor:
    """
    Hierarchical merge processor for clustering results.
    
    Integration with TextAssignmentManager:
    1. TextAssignmentManager produces workflow assignments (with big text tracking)
    2. Each workflow processes texts/chunks (all ≤ M tokens)
    3. TreeMergeProcessor merges results hierarchically
    4. Final step consolidates big text chunks
    
    Key Insight: Since all slots are ≤ M tokens, we treat everything uniformly
    during clustering. Big text tracking is only for final consolidation.
    """
    
    def __init__(self, max_parallel_merges: int = 4):
        """
        Initialize merge processor.
        
        Args:
            max_parallel_merges: Maximum parallel merge operations
        """
        self.max_parallel_merges = max_parallel_merges
        self.big_text_trackers: Dict[str, BigTextTracker] = {}
        self.next_node_id = 0
        self.execution_counter = 0
    
    async def process_with_dry_assignment(
        self,
        dry_assignment_result,  # From TextAssignmentManager
        clusterizer_config: Optional[ClusterizerConfig] = None,
        max_passes: int = 2,
        prefilter_k: int = 3,
        batch_size: int = 20,
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Process texts using pre-computed dry assignment.
        
        Args:
            dry_assignment_result: DryAssignmentResult from TextAssignmentManager
            clusterizer_config: ClusterizerConfig for clustering
            ... (other clusterizer params)
        
        Returns:
            Final merged clustering result with consolidated big texts
        """
        
        if clusterizer_config is None:
            clusterizer_config = CONFIG_BALANCED_HYBRID
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TREE MERGE PROCESSOR")
        logger.info(f"{'='*70}")
        logger.info(f"Total workflows: {dry_assignment_result.total_workflows_needed}")
        logger.info(f"Total batches: {dry_assignment_result.total_batches}")
        logger.info(f"Total slots: {dry_assignment_result.total_slots_used}")
        logger.info(f"Big texts tracked: {len(dry_assignment_result.big_text_registry)}")
        logger.info(f"Max parallel merges: {self.max_parallel_merges}")
        logger.info(f"{'='*70}\n")
        
        # Initialize big text trackers
        self._initialize_big_text_trackers(dry_assignment_result)
        
        # Phase 1: Execute all workflow leaves in parallel
        leaf_nodes = await self._execute_workflow_leaves(
            dry_assignment_result,
            clusterizer_config,
            max_passes,
            prefilter_k,
            batch_size,
            max_concurrent
        )
        
        logger.info(f"\n✅ Completed {len(leaf_nodes)} workflow executions")
        
        # Phase 2: Build and execute merge tree bottom-up
        if len(leaf_nodes) == 1:
            # Single workflow - no merging needed
            root_node = leaf_nodes[0]
        else:
            root_node = await self._build_and_execute_merge_tree(leaf_nodes)
        
        logger.info(f"\n✅ Merge tree complete")
        
        # Phase 3: Consolidate big text chunks
        final_result = self._consolidate_big_texts(root_node.result)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"FINAL RESULT")
        logger.info(f"{'='*70}")
        logger.info(f"Categories: {len(final_result['categories'])}")
        logger.info(f"Assignments: {len(final_result['assignments'])}")
        logger.info(f"Big texts consolidated: {sum(1 for t in self.big_text_trackers.values() if t.is_complete())}")
        logger.info(f"{'='*70}\n")
        
        return final_result
    
    def _initialize_big_text_trackers(self, dry_assignment_result):
        """Initialize trackers for big texts from dry assignment."""
        for text_id, big_text_info in dry_assignment_result.big_text_registry.items():
            tracker = BigTextTracker(
                original_text_id=text_id,
                total_chunks=big_text_info.total_chunks
            )
            self.big_text_trackers[text_id] = tracker
            logger.info(f"📋 Tracking big text '{text_id}': {big_text_info.total_chunks} chunks")
    
    async def _execute_workflow_leaves(
        self,
        dry_assignment_result,
        config: ClusterizerConfig,
        max_passes: int,
        prefilter_k: int,
        batch_size: int,
        max_concurrent: int
    ) -> List[MergeNode]:
        """
        Execute all workflow leaves (first level of clustering).
        
        Each workflow processes texts from its assigned batches.
        """
        
        workflow_assignments = dry_assignment_result.workflow_assignments
        logger.info(f"📦 Executing {len(workflow_assignments)} workflow leaves in parallel...")
        
        # Prepare tasks for parallel execution
        tasks = []
        for workflow_id, workflow_batches in enumerate(workflow_assignments):
            task = self._execute_single_workflow(
                workflow_id=workflow_id,
                workflow_batches=workflow_batches,
                config=config,
                max_passes=max_passes,
                prefilter_k=prefilter_k,
                batch_size=batch_size,
                max_concurrent=max_concurrent
            )
            tasks.append(task)
        
        # Execute all workflows in parallel
        results = await asyncio.gather(*tasks)
        
        # Create leaf nodes
        leaf_nodes = []
        for result in results:
            node = MergeNode(
                node_id=self._get_next_node_id(),
                level=0,
                result=result,
                is_ready=True
            )
            leaf_nodes.append(node)
        
        return leaf_nodes
    
    async def _execute_single_workflow(
        self,
        workflow_id: int,
        workflow_batches: List[List[Any]],  # List[List[SlotAssignment]]
        config: ClusterizerConfig,
        max_passes: int,
        prefilter_k: int,
        batch_size: int,
        max_concurrent: int
    ) -> ClusterResult:
        """
        Execute a single workflow (cluster texts from assigned batches).
        
        Key: All slots are ≤ M tokens, so we treat all uniformly.
        """
        
        # Extract texts from all batches in this workflow
        texts = []
        text_metadata = []  # Track which text is which for big text tracking
        
        for batch_id, batch in enumerate(workflow_batches):
            for slot_assignment in batch:
                texts.append(slot_assignment.content)
                text_metadata.append({
                    'workflow_id': workflow_id,
                    'batch_id': batch_id,
                    'slot_id': slot_assignment.slot_id,
                    'text_id': slot_assignment.text_id,
                    'type': slot_assignment.type,
                    'chunk_idx': slot_assignment.chunk_idx
                })
        
        logger.info(f"⚙️  Workflow {workflow_id}: Processing {len(texts)} texts...")
        
        # Execute clustering on all texts in this workflow
        raw_result = await clusterize_texts(
            texts=texts,
            max_passes=max_passes,
            prefilter_k=prefilter_k,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            config=config
        )
        
        # Track big text chunk assignments
        self._track_big_text_assignments(
            workflow_id=workflow_id,
            text_metadata=text_metadata,
            assignments=raw_result['assignments']
        )
        
        # Wrap in ClusterResult
        result = ClusterResult(
            categories=raw_result['categories'],
            assignments=raw_result['assignments'],
            metadata=raw_result['metadata'],
            execution_id=workflow_id
        )
        
        logger.info(f"   ✓ Workflow {workflow_id} complete: "
                   f"{len(result.categories)} categories, "
                   f"{len(result.assignments)} assignments")
        
        return result
    
    def _track_big_text_assignments(
        self,
        workflow_id: int,
        text_metadata: List[Dict],
        assignments: List[CategoryAssignment]
    ):
        """Track which categories were assigned to big text chunks."""
        
        # Build assignment lookup by text content
        assignment_by_text = {a.text: a for a in assignments}
        
        for i, metadata in enumerate(text_metadata):
            if metadata['type'] == 'chunk':
                # This is a chunk of a big text
                original_text_id = metadata['text_id']
                chunk_idx = metadata['chunk_idx']
                
                if original_text_id in self.big_text_trackers:
                    # Find the assignment for this chunk
                    # Get text from assignments (assignments have full text)
                    matching_assignment = None
                    for assignment in assignments:
                        # Match by index (i-th text -> i-th assignment)
                        if assignments.index(assignment) == i:
                            matching_assignment = assignment
                            break
                    
                    if matching_assignment:
                        chunk_id = f"{original_text_id}__chunk_{chunk_idx}"
                        self.big_text_trackers[original_text_id].add_chunk_assignment(
                            chunk_id=chunk_id,
                            category_id=matching_assignment.category_id,
                            confidence=matching_assignment.confidence,
                            execution_id=workflow_id
                        )
                        
                        logger.debug(f"   📌 Tracked chunk {chunk_idx} of '{original_text_id}' -> {matching_assignment.category_id}")
    
    async def _build_and_execute_merge_tree(self, leaf_nodes: List[MergeNode]) -> MergeNode:
        """
        Build and execute binary merge tree bottom-up.
        
        Strategy: Pair nodes and merge until single root remains.
        """
        
        current_level_nodes = leaf_nodes
        level = 1
        
        while len(current_level_nodes) > 1:
            logger.info(f"\n🔀 Merge level {level}: Merging {len(current_level_nodes)} nodes...")
            
            # Pair nodes for merging
            merge_pairs = []
            for i in range(0, len(current_level_nodes), 2):
                if i + 1 < len(current_level_nodes):
                    merge_pairs.append((current_level_nodes[i], current_level_nodes[i + 1]))
                else:
                    # Odd one out - promote to next level
                    merge_pairs.append((current_level_nodes[i], None))
            
            # Execute merges in parallel batches
            next_level_nodes = []
            
            for batch_start in range(0, len(merge_pairs), self.max_parallel_merges):
                batch = merge_pairs[batch_start:batch_start + self.max_parallel_merges]
                
                tasks = []
                for node_a, node_b in batch:
                    if node_b is None:
                        # Single node - promote directly
                        next_level_nodes.append(node_a)
                    else:
                        task = self._merge_two_nodes(node_a, node_b, level)
                        tasks.append(task)
                
                if tasks:
                    merged_nodes = await asyncio.gather(*tasks)
                    next_level_nodes.extend(merged_nodes)
            
            logger.info(f"   ✓ Level {level} complete: {len(next_level_nodes)} nodes at next level")
            
            current_level_nodes = next_level_nodes
            level += 1
        
        return current_level_nodes[0]
    
    async def _merge_two_nodes(self, node_a: MergeNode, node_b: MergeNode, level: int) -> MergeNode:
        """
        Merge two nodes by re-clustering their categories.
        
        Strategy (from your code):
        1. Create representative texts for each category
        2. Re-cluster these representatives
        3. Map original assignments to new categories
        """
        
        logger.info(f"   🔄 Merging nodes {node_a.node_id} + {node_b.node_id}")
        
        result_a = node_a.result
        result_b = node_b.result
        
        # Step 1: Create category representatives
        category_representatives = []
        category_provenance = []
        
        for result in [result_a, result_b]:
            for category in result.categories:
                rep_text = self._create_category_representative(category, result)
                category_representatives.append(rep_text)
                category_provenance.append({
                    'execution_id': result.execution_id,
                    'category': category,
                    'source_result': result
                })
        
        logger.info(f"      📝 Created {len(category_representatives)} category representatives")
        
        # Step 2: Re-cluster the category representatives (using same config as original)
        config = CONFIG_BALANCED_HYBRID
                
        merged_clustering = await clusterize_texts(
            texts=category_representatives,
            max_passes=1,  # Single pass for meta-clustering
            batch_size=min(20, len(category_representatives)),
            config=config
        )
        
        logger.info(f"      ✓ Re-clustering: {len(category_representatives)} reps → {len(merged_clustering['categories'])} meta-categories")
        
        # Step 3: Remap original text assignments to new categories
        final_assignments = self._remap_assignments(
            merged_clustering,
            category_provenance,
            [result_a, result_b]
        )
        
        logger.info(f"      ✓ Remapped {len(final_assignments)} assignments")
        
        # Create merged result
        merged_result = ClusterResult(
            categories=merged_clustering['categories'],
            assignments=final_assignments,
            metadata={
                **merged_clustering['metadata'],
                'merge_level': level,
                'source_executions': [result_a.execution_id, result_b.execution_id]
            },
            execution_id=self.execution_counter
        )
        
        self.execution_counter += 1
        
        # Create parent node
        parent_node = MergeNode(
            node_id=self._get_next_node_id(),
            level=level,
            result=merged_result,
            children=[node_a, node_b],
            is_ready=True
        )
        
        return parent_node
    
    def _create_category_representative(self, category: Category, result: ClusterResult) -> str:
        """
        Create representative text for a category.
        
        Combines category metadata with sample assigned texts.
        """
        
        # Get texts assigned to this category
        assigned_texts = [
            a.text for a in result.assignments 
            if a.category_id == category.id
        ]
        
        # Sample up to 3 representative texts
        sample_texts = assigned_texts[:3]
        
        # Build representative
        parts = [
            f"Category: {category.name}",
            f"Description: {category.description or 'No description'}",
            f"Keywords: {', '.join(category.keywords) if category.keywords else 'none'}",
            f"Assigned texts: {len(assigned_texts)}",
            "",
            "Sample texts:"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            # Truncate long texts
            truncated = text[:150] + "..." if len(text) > 150 else text
            parts.append(f"{i}. {truncated}")
        
        return "\n".join(parts)
    
    def _remap_assignments(
        self,
        merged_clustering: Dict,
        category_provenance: List[Dict],
        source_results: List[ClusterResult]
    ) -> List[CategoryAssignment]:
        """
        Remap original text assignments to new merged categories.
        
        Strategy:
        1. Find which new category each old category was assigned to
        2. Update all original text assignments accordingly
        """
        
        # Build mapping: old_category_id -> new_category_id
        old_to_new_category = {}
        
        # Match representatives to their source categories
        for assignment in merged_clustering['assignments']:
            rep_text = assignment.text
            
            # Find which original category this representative came from
            for prov in category_provenance:
                prov_rep = self._create_category_representative(
                    prov['category'],
                    prov['source_result']
                )
                
                if rep_text == prov_rep:
                    old_to_new_category[prov['category'].id] = assignment.category_id
                    break
        
        # Remap all original text assignments
        final_assignments = []
        
        for result in source_results:
            for assignment in result.assignments:
                new_category_id = old_to_new_category.get(
                    assignment.category_id,
                    assignment.category_id  # Keep original if not found
                )
                
                new_assignment = CategoryAssignment(
                    text=assignment.text,
                    category_id=new_category_id,
                    confidence=assignment.confidence,
                    reasoning=assignment.reasoning
                )
                final_assignments.append(new_assignment)
        
        return final_assignments
    
    def _consolidate_big_texts(self, result: ClusterResult) -> Dict[str, Any]:
        """
        Consolidate assignments for big texts based on their chunks.
        
        Strategy:
        1. Identify chunk assignments in the result
        2. For each big text, aggregate chunk assignments
        3. Replace chunk assignments with single consolidated assignment
        """
        
        logger.info(f"\n📊 Consolidating big text chunks...")
        
        # Identify which texts are chunks
        chunk_text_ids = set()
        for tracker in self.big_text_trackers.values():
            for chunk_id, _, _ in tracker.chunk_assignments:
                chunk_text_ids.add(chunk_id)
        
        # Separate regular and chunk assignments
        regular_assignments = []
        chunk_assignments_by_text = defaultdict(list)
        
        for assignment in result.assignments:
            # Check if this assignment is for a chunk
            # Chunks are identified by their text content matching tracked chunks
            is_chunk = False
            for tracker in self.big_text_trackers.values():
                for chunk_id, cat_id, conf in tracker.chunk_assignments:
                    # Match by category assignment (chunks processed in same workflow)
                    # This is a heuristic - in production, need better chunk identification
                    if assignment.category_id == cat_id:
                        chunk_assignments_by_text[tracker.original_text_id].append(assignment)
                        is_chunk = True
                        break
                if is_chunk:
                    break
            
            if not is_chunk:
                regular_assignments.append(assignment)
        
        # Create consolidated assignments for big texts
        consolidated_assignments = []
        
        for original_text_id, tracker in self.big_text_trackers.items():
            if tracker.is_complete():
                primary_cat, primary_conf, secondary_cats = tracker.get_consolidated_category()
                
                # Build reasoning
                secondary_info = ""
                if secondary_cats:
                    secondary_info = f" | Secondary: {', '.join(f'{cat}({conf:.2f})' for cat, conf in secondary_cats[:2])}"
                
                consolidated = CategoryAssignment(
                    text=f"[BIG_TEXT:{original_text_id}]",
                    category_id=primary_cat,
                    confidence=primary_conf,
                    reasoning=f"Consolidated from {tracker.total_chunks} chunks{secondary_info}"
                )
                consolidated_assignments.append(consolidated)
                
                logger.info(f"   ✓ '{original_text_id}': {tracker.total_chunks} chunks → {primary_cat} (conf: {primary_conf:.3f})")
        
        # Combine all assignments
        all_assignments = regular_assignments + consolidated_assignments
        
        return {
            'categories': result.categories,
            'assignments': all_assignments,
            'metadata': {
                **result.metadata,
                'big_texts_consolidated': len(consolidated_assignments),
                'regular_texts': len(regular_assignments),
                'total_big_texts': len(self.big_text_trackers)
            }
        }
    
    def _get_next_node_id(self) -> int:
        """Get next node ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

async def example_integration():
    """
    Example showing complete integration:
    TextAssignmentManager -> TreeMergeProcessor
    """
    
    from text_assignment_manager import TextAssignmentManager, Text
    from agentic_clusterizer import CONFIG_BALANCED_HYBRID
    
    # Step 1: Create texts
    texts = [
        Text(id=f"text_{i}", content=f"Sample text content {i} " * 50)
        for i in range(500)  # 500 texts
    ]
    
    # Step 2: Dry assignment
    print("Step 1: Performing dry assignment...")
    assignment_manager = TextAssignmentManager(
        batch_size=5,           # 5 texts per batch
        max_batches_per_workflow=10,  # 10 batches per workflow = 50 texts/workflow
        token_threshold=200     # Texts > 200 tokens get chunked
    )
    
    dry_assignment = assignment_manager.dry_assign(texts)
    print(dry_assignment.get_summary())
    
    # Step 3: Process with TreeMergeProcessor
    print("\nStep 2: Processing with TreeMergeProcessor...")
    processor = TreeMergeProcessor(max_parallel_merges=4)
    
    result = await processor.process_with_dry_assignment(
        dry_assignment_result=dry_assignment,
        clusterizer_config=CONFIG_BALANCED_HYBRID,
        max_passes=2,
        batch_size=20,
        max_concurrent=5
    )
    
    # Step 4: Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Categories: {len(result['categories'])}")
    print(f"Assignments: {len(result['assignments'])}")
    print(f"Big texts consolidated: {result['metadata'].get('big_texts_consolidated', 0)}")
    
    # Show sample categories
    print("\nTop Categories:")
    sorted_cats = sorted(result['categories'], key=lambda c: c.text_count, reverse=True)
    for cat in sorted_cats[:5]:
        print(f"  - {cat.name}: {cat.text_count} texts")
        print(f"    Keywords: {', '.join(cat.keywords[:5])}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_integration())