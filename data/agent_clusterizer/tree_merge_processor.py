"""
Priority Queue-Based Dynamic Execution and Merging

Key Design:
- Single priority queue manages BOTH workflow execution AND merging
- As workflows complete, their results become available for merging
- Merge operations compete for slots alongside workflow execution
- Priority key determines what executes next (workflows first, then small merges)
- Terminates when queue is empty and only one final result remains

This is a producer-consumer pattern:
- Producers: Workflow completions generate merge opportunities
- Consumers: Merge operations consume results and produce new results
- Queue: Coordinates everything with priorities
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple
from enum import Enum
from collections import defaultdict
import logging

from agentic_clusterizer import (
    clusterize_texts,
    ClusterConfig,
    Category,
    CategoryAssignment,
    CONFIG_BALANCED_HYBRID,
)

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Type of task in priority queue."""
    WORKFLOW = "workflow"
    MERGE = "merge"

@dataclass(order=True)
class ExecutionTask:
    """Unified task for priority queue."""
    priority: int
    task_type: TaskType = field(compare=False)
    task_id: str = field(compare=False)
    
    # For workflows
    workflow_id: Optional[int] = field(default=None, compare=False)
    workflow_batches: Optional[List] = field(default=None, compare=False)
    
    # For merges
    node_ids_to_merge: Optional[tuple] = field(default=None, compare=False)
    merge_level: Optional[int] = field(default=None, compare=False)

class TaskType(Enum):
    """Type of task in priority queue."""
    WORKFLOW = "workflow"
    MERGE = "merge"


@dataclass(order=True)
class ExecutionTask:
    """Unified task for priority queue."""
    priority: int
    task_type: TaskType = field(compare=False)
    task_id: str = field(compare=False)
    
    # For workflows
    workflow_id: Optional[int] = field(default=None, compare=False)
    workflow_batches: Optional[List] = field(default=None, compare=False)
    
    # For merges
    node_ids_to_merge: Optional[tuple] = field(default=None, compare=False)
    merge_level: Optional[int] = field(default=None, compare=False)


@dataclass
class ExecutionState:
    """Encapsulates all execution state for clarity."""
    task_queue: List[ExecutionTask] = field(default_factory=list)
    result_pool: Dict = field(default_factory=dict)
    active_tasks: Set[str] = field(default_factory=set)
    active_futures: Dict = field(default_factory=dict)
    next_node_id: int = 0
    task_counter: int = 0
    iteration: int = 0
    
    def has_work(self) -> bool:
        """Check if there's any work remaining."""
        return bool(self.task_queue or self.active_futures)
    
    def can_schedule_more(self, max_parallel: int) -> bool:
        """Check if we can schedule more tasks."""
        return self.task_queue and len(self.active_futures) < max_parallel
    
    def get_pool_size(self) -> int:
        """Get current result pool size."""
        return len(self.result_pool)


class TaskScheduler:
    """Handles task scheduling logic."""
    
    def __init__(self, processor, config_params: Dict):
        self.processor = processor
        self.config_params = config_params
    
    async def create_workflow_future(self, task: ExecutionTask) -> asyncio.Task:
        """Create async task for workflow execution."""
        logger.info(f"🚀 Starting workflow {task.workflow_id}")
        return asyncio.create_task(
            self._run_workflow(task)
        )
    
    async def create_merge_future(
        self, 
        task: ExecutionTask, 
        result_pool: Dict,
        next_node_id: int
    ) -> asyncio.Task:
        """Create async task for merge execution."""
        logger.info(f"🔄 Merging {list(task.node_ids_to_merge)}")
        return asyncio.create_task(
            self._run_merge(task, result_pool, next_node_id)
        )
    
    async def _run_workflow(self, task: ExecutionTask):
        """Execute workflow using processor's method."""
        result = await self.processor._execute_single_workflow(
            workflow_id=task.workflow_id,
            workflow_batches=task.workflow_batches,
            **self.config_params
        )
        
        from tree_merge_processor import MergeNode
        
        return MergeNode(
            node_id=task.workflow_id,
            level=0,
            result=result,
            is_ready=True
        )
    
    async def _run_merge(
        self, 
        task: ExecutionTask, 
        result_pool: Dict,
        new_node_id: int
    ):
        """Execute merge using processor's method."""
        nodes = self._get_nodes_for_merge(task, result_pool)
        if not nodes:
            return None
        
        # Remove consumed nodes from pool
        self._consume_nodes(task.node_ids_to_merge, result_pool)
        
        # Perform merge
        merged = await self._merge_nodes(nodes, task.merge_level or 1)
        merged.node_id = new_node_id
        return merged
    
    def _get_nodes_for_merge(self, task: ExecutionTask, result_pool: Dict) -> Optional[List]:
        """Retrieve nodes for merge from pool."""
        nodes = []
        for nid in task.node_ids_to_merge:
            if nid not in result_pool:
                logger.error(f"Node {nid} missing from pool!")
                return None
            nodes.append(result_pool[nid])
        return nodes
    
    def _consume_nodes(self, node_ids: tuple, result_pool: Dict):
        """Remove consumed nodes from pool."""
        for nid in node_ids:
            result_pool.pop(nid, None)
    
    async def _merge_nodes(self, nodes: List, level: int):
        """Merge nodes using appropriate processor method."""
        if len(nodes) == 2:
            return await self.processor._merge_two_nodes(nodes[0], nodes[1], level)
        else:
            return await self.processor._merge_nodes(nodes, level)


class MergeTaskGenerator:
    """Generates merge tasks from completed results."""
    
    def create_merge_tasks(
        self, 
        result_pool: Dict, 
        task_counter: int
    ) -> List[ExecutionTask]:
        """Generate all pairwise merge tasks from current pool."""
        nodes = list(result_pool.values())
        
        if len(nodes) < 2:
            return []
        
        tasks = []
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                task = self._create_single_merge_task(
                    node_a, node_b, task_counter
                )
                tasks.append(task)
                task_counter += 1
        
        return tasks
    
    def _create_single_merge_task(
        self, 
        node_a, 
        node_b, 
        task_counter: int
    ) -> ExecutionTask:
        """Create merge task for a pair of nodes."""
        total_cats = (
            len(node_a.result.categories) + 
            len(node_b.result.categories)
        )
        
        return ExecutionTask(
            priority=1000 + total_cats,
            task_type=TaskType.MERGE,
            task_id=f"M{task_counter}_{node_a.node_id}_{node_b.node_id}",
            node_ids_to_merge=(node_a.node_id, node_b.node_id),
            merge_level=max(node_a.level, node_b.level) + 1
        )


class CompletionHandler:
    """Handles task completion and result processing."""
    
    def __init__(self, merge_generator: MergeTaskGenerator):
        self.merge_generator = merge_generator
    
    async def process_completed_task(
        self,
        task_id: str,
        future: asyncio.Task,
        state: ExecutionState
    ) -> None:
        """Process a completed task and update state."""
        try:
            result_node = await future
            
            if result_node:
                self._handle_successful_result(task_id, result_node, state)
            
        except Exception as e:
            logger.error(f"❌ {task_id} failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup_task(task_id, state)
    
    def _handle_successful_result(
        self, 
        task_id: str, 
        result_node, 
        state: ExecutionState
    ):
        """Handle successful task completion."""
        state.result_pool[result_node.node_id] = result_node
        
        logger.info(
            f"✅ {task_id} → node {result_node.node_id} "
            f"({len(result_node.result.categories)} cats)"
        )
        
        # Generate new merge tasks
        new_merges = self.merge_generator.create_merge_tasks(
            state.result_pool, 
            state.task_counter
        )
        
        self._enqueue_merge_tasks(new_merges, state)
    
    def _enqueue_merge_tasks(
        self, 
        merge_tasks: List[ExecutionTask], 
        state: ExecutionState
    ):
        """Add new merge tasks to queue."""
        for merge_task in merge_tasks:
            heapq.heappush(state.task_queue, merge_task)
            state.task_counter += 1
        
        if merge_tasks:
            logger.info(f"   📊 +{len(merge_tasks)} merge tasks")
    
    def _cleanup_task(self, task_id: str, state: ExecutionState):
        """Clean up task tracking."""
        state.active_tasks.discard(task_id)
        state.active_futures.pop(task_id, None)


class PriorityQueueExecutor:
    """
    Priority queue-based executor for TreeMergeProcessor.
    
    Refactored for lower cyclomatic complexity and better testability.
    """
    
    def __init__(self, processor):
        """
        Args:
            processor: Your actual TreeMergeProcessor instance
        """
        self.processor = processor
        self.merge_generator = MergeTaskGenerator()
        self.completion_handler = CompletionHandler(self.merge_generator)
    
    async def run(
        self,
        dry_assignment_result,
        config,
        max_passes: int,
        prefilter_k: int,
        batch_size: int,
        max_concurrent: int
    ):
        """Execute workflows and merges through priority queue."""
        
        # Initialize state
        state = self._initialize_state(dry_assignment_result)
        
        # Setup scheduler
        config_params = {
            'config': config,
            'max_passes': max_passes,
            'prefilter_k': prefilter_k,
            'batch_size': batch_size,
            'max_concurrent': max_concurrent
        }
        scheduler = TaskScheduler(self.processor, config_params)
        
        # Log startup
        self._log_startup(dry_assignment_result)
        
        # Main execution loop
        await self._execute_loop(state, scheduler)
        
        # Finalize and return result
        # ✅ CHANGE: Now we await since _finalize_result is async
        return await self._finalize_result(state)
    
    def _initialize_state(self, dry_assignment_result) -> ExecutionState:
        """Initialize execution state with workflow tasks."""
        state = ExecutionState(
            next_node_id=len(dry_assignment_result.workflow_assignments)
        )
        
        # Queue all workflows
        for workflow_id, batches in enumerate(dry_assignment_result.workflow_assignments):
            task = ExecutionTask(
                priority=0,
                task_type=TaskType.WORKFLOW,
                task_id=f"W{workflow_id}",
                workflow_id=workflow_id,
                workflow_batches=batches
            )
            heapq.heappush(state.task_queue, task)
        
        logger.info(f"📋 Queued {len(state.task_queue)} workflows")
        return state
    
    def _log_startup(self, dry_assignment_result):
        """Log execution startup information."""
        total_workflows = len(dry_assignment_result.workflow_assignments)
        max_parallel = self.processor.max_parallel_merges
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PRIORITY QUEUE EXECUTION")
        logger.info(f"{'='*70}")
        logger.info(f"Workflows: {total_workflows}")
        logger.info(f"Max parallel: {max_parallel}")
        logger.info(f"{'='*70}\n")
    
    async def _execute_loop(self, state: ExecutionState, scheduler: TaskScheduler):
        """Main execution loop - schedules and processes tasks."""
        max_parallel = self.processor.max_parallel_merges
        
        while state.has_work():
            state.iteration += 1
            
            # Schedule new tasks
            await self._schedule_tasks(state, scheduler, max_parallel)
            
            # Process completions
            await self._process_completions(state)
            
            # Periodic logging
            self._log_progress(state)
        
        self._log_completion(state)
    
    async def _schedule_tasks(
        self, 
        state: ExecutionState, 
        scheduler: TaskScheduler,
        max_parallel: int
    ):
        """Schedule tasks from queue up to parallelism limit."""
        while state.can_schedule_more(max_parallel):
            task = heapq.heappop(state.task_queue)
            
            if task.task_id in state.active_tasks:
                continue
            
            state.active_tasks.add(task.task_id)
            future = await self._create_task_future(task, state, scheduler)
            state.active_futures[task.task_id] = future
    
    async def _create_task_future(
        self, 
        task: ExecutionTask, 
        state: ExecutionState,
        scheduler: TaskScheduler
    ) -> asyncio.Task:
        """Create appropriate future based on task type."""
        if task.task_type == TaskType.WORKFLOW:
            return await scheduler.create_workflow_future(task)
        else:
            future = await scheduler.create_merge_future(
                task, 
                state.result_pool, 
                state.next_node_id
            )
            state.next_node_id += 1
            return future
    
    async def _process_completions(self, state: ExecutionState):
        """Wait for and process completed tasks."""
        if not state.active_futures:
            return
        
        done, _ = await asyncio.wait(
            state.active_futures.values(),
            timeout=0.5,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for future in done:
            task_id = self._find_task_id_for_future(future, state)
            if task_id:
                await self.completion_handler.process_completed_task(
                    task_id, future, state
                )
    
    def _find_task_id_for_future(
        self, 
        future: asyncio.Task, 
        state: ExecutionState
    ) -> Optional[str]:
        """Find task ID corresponding to a completed future."""
        for tid, fut in list(state.active_futures.items()):
            if fut == future:
                return tid
        return None
    
    def _log_progress(self, state: ExecutionState):
        """Log periodic progress updates."""
        if state.iteration % 5 == 0:
            logger.info(
                f"📈 Queue={len(state.task_queue)}, "
                f"Active={len(state.active_futures)}, "
                f"Pool={state.get_pool_size()}"
            )
    
    def _log_completion(self, state: ExecutionState):
        """Log execution completion."""
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPLETE: {state.iteration} iterations, pool size={state.get_pool_size()}")
        logger.info(f"{'='*70}\n")
    
    async def _emergency_merge(self, result_pool: Dict):
        """Perform emergency merge when multiple results remain."""
        nodes = list(result_pool.values())
        
        # Track initial count
        initial_count = sum(len(node.result.assignments) for node in nodes)
        
        logger.warning(
            f"⚠️ Emergency merge: {len(nodes)} nodes, "
            f"{initial_count} total assignments"
        )
        
        final = nodes[0]
        for i, node in enumerate(nodes[1:], 1):
            logger.info(f"   Emergency merge step {i}/{len(nodes)-1}")
            final = await self.processor._merge_two_nodes(final, node, 999)
        
        # Verify final count
        final_count = len(final.result.assignments)
        if final_count != initial_count:
            logger.error(
                f"❌ Emergency merge lost {initial_count - final_count} assignments!"
            )
        
        return final
    
    async def _finalize_result(self, state: ExecutionState):
        """Finalize and return the execution result."""
        pool_size = state.get_pool_size()
        
        if pool_size == 1:
            result_node = next(iter(state.result_pool.values()))
            
            assignment_count = len(result_node.result.assignments)
            logger.info(
                f"✅ Single result in pool: {assignment_count} assignments, "
                f"{len(result_node.result.categories)} categories"
            )
            
            return result_node
            
        elif pool_size == 0:
            raise RuntimeError("No results in pool!")
            
        else:
            # Track total before emergency merge
            total_assignments = sum(
                len(node.result.assignments) 
                for node in state.result_pool.values()
            )
            logger.warning(
                f"⚠️ {pool_size} results remain, performing emergency merge. "
                f"Total assignments: {total_assignments}"
            )
            
            # Perform emergency merge
            final = await self._emergency_merge(state.result_pool)
            
            # Verify after emergency merge
            final_count = len(final.result.assignments)
            if final_count != total_assignments:
                logger.error(
                    f"❌ Emergency merge lost assignments! "
                    f"Before={total_assignments}, After={final_count}"
                )
            
            return final

# ============================================================================
# INTEGRATION WITH TreeMergeProcessor
# ============================================================================

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

class ChunkIdentifier:
    """Identifies which assignments correspond to text chunks."""
    
    def __init__(self, big_text_trackers: Dict):
        self.big_text_trackers = big_text_trackers
        self._chunk_id_set = self._build_chunk_id_set()
    
    def _build_chunk_id_set(self) -> Set[str]:
        """Build set of all chunk IDs for fast lookup."""
        chunk_ids = set()
        for tracker in self.big_text_trackers.values():
            for chunk_id, _, _ in tracker.chunk_assignments:
                chunk_ids.add(chunk_id)
        return chunk_ids
    
    def is_chunk_assignment(self, assignment) -> Tuple[bool, str]:
        """
        Check if assignment is for a chunk.
        
        Returns:
            (is_chunk, original_text_id)
        """
        for tracker in self.big_text_trackers.values():
            if self._assignment_matches_tracker(assignment, tracker):
                return True, tracker.original_text_id
        
        return False, ""
    
    def _assignment_matches_tracker(self, assignment, tracker) -> bool:
        """Check if assignment matches any chunk in tracker."""
        for chunk_id, cat_id, conf in tracker.chunk_assignments:
            if assignment.category_id == cat_id:
                return True
        return False


class AssignmentClassifier:
    """Classifies assignments into regular vs chunk assignments."""
    
    def __init__(self, chunk_identifier: ChunkIdentifier):
        self.chunk_identifier = chunk_identifier
    
    def classify_assignments(
        self, 
        assignments: List
    ) -> Tuple[List, Dict[str, List]]:
        """
        Classify assignments into regular and chunk groups.
        
        Returns:
            (regular_assignments, chunk_assignments_by_text)
        """
        regular_assignments = []
        chunk_assignments_by_text = defaultdict(list)
        
        for assignment in assignments:
            is_chunk, text_id = self.chunk_identifier.is_chunk_assignment(assignment)
            
            if is_chunk:
                chunk_assignments_by_text[text_id].append(assignment)
            else:
                regular_assignments.append(assignment)
        
        return regular_assignments, chunk_assignments_by_text


class ConsolidatedAssignmentBuilder:
    """Builds consolidated assignments from chunk groups."""
    
    def build_consolidated_assignment(
        self,
        original_text_id: str,
        tracker
    ) -> 'CategoryAssignment':
        """
        Build single consolidated assignment from tracker data.
        
        Args:
            original_text_id: ID of the original big text
            tracker: BigTextTracker containing chunk assignments
            
        Returns:
            CategoryAssignment representing consolidated result
        """
        primary_cat, primary_conf, secondary_cats = tracker.get_consolidated_category()
        
        reasoning = self._build_reasoning(tracker, secondary_cats)
        
        from agentic_clusterizer import CategoryAssignment
        
        return CategoryAssignment(
            text=f"[BIG_TEXT:{original_text_id}]",
            category_id=primary_cat,
            confidence=primary_conf,
            reasoning=reasoning
        )
    
    def _build_reasoning(
        self,
        tracker,
        secondary_cats: List[Tuple[str, float]]
    ) -> str:
        """Build reasoning string for consolidated assignment."""
        base_reasoning = f"Consolidated from {tracker.total_chunks} chunks"
        
        if secondary_cats:
            secondary_info = ", ".join(
                f"{cat}({conf:.2f})" 
                for cat, conf in secondary_cats[:2]
            )
            return f"{base_reasoning} | Secondary: {secondary_info}"
        
        return base_reasoning


class ConsolidationLogger:
    """Handles logging for consolidation process."""
    
    @staticmethod
    def log_start():
        """Log consolidation start."""
        logger.info(f"\n📊 Consolidating big text chunks...")
    
    @staticmethod
    def log_consolidated_text(text_id: str, tracker, primary_cat: str, primary_conf: float):
        """Log individual text consolidation."""
        logger.info(
            f"   ✓ '{text_id}': {tracker.total_chunks} chunks → "
            f"{primary_cat} (conf: {primary_conf:.3f})"
        )
    
    @staticmethod
    def log_summary(num_consolidated: int, num_regular: int):
        """Log consolidation summary."""
        logger.info(
            f"\n📈 Consolidation complete: "
            f"{num_consolidated} big texts, {num_regular} regular texts"
        )


class BigTextConsolidator:
    """
    Main consolidator that orchestrates the consolidation process.
    
    Refactored for clarity and testability.
    """
    
    def __init__(self, big_text_trackers: Dict):
        self.big_text_trackers = big_text_trackers
        
        # Initialize components
        self.chunk_identifier = ChunkIdentifier(big_text_trackers)
        self.classifier = AssignmentClassifier(self.chunk_identifier)
        self.builder = ConsolidatedAssignmentBuilder()
        self.logger = ConsolidationLogger()
    
    def consolidate(self, result) -> Dict[str, Any]:
        """
        Consolidate assignments for big texts based on their chunks.
        
        Strategy:
        1. Classify assignments (regular vs chunks)
        2. Build consolidated assignments for complete big texts
        3. Combine and return results
        """
        self.logger.log_start()
        
        # Step 1: Classify assignments
        regular_assignments, chunk_groups = self._classify_all_assignments(
            result.assignments
        )
        
        # Step 2: Build consolidated assignments
        consolidated_assignments = self._build_all_consolidated_assignments()
        
        # Step 3: Combine and return
        return self._build_final_result(
            result,
            regular_assignments,
            consolidated_assignments
        )
    
    def _classify_all_assignments(
        self, 
        assignments: List
    ) -> Tuple[List, Dict[str, List]]:
        """Classify all assignments into regular and chunk groups."""
        return self.classifier.classify_assignments(assignments)
    
    def _build_all_consolidated_assignments(self) -> List:
        """Build consolidated assignments for all complete big texts."""
        consolidated = []
        
        for text_id, tracker in self.big_text_trackers.items():
            if not tracker.is_complete():
                continue
            
            assignment = self._build_single_consolidated_assignment(
                text_id, 
                tracker
            )
            consolidated.append(assignment)
        
        return consolidated
    
    def _build_single_consolidated_assignment(
        self,
        text_id: str,
        tracker
    ):
        """Build and log consolidated assignment for one big text."""
        assignment = self.builder.build_consolidated_assignment(text_id, tracker)
        
        self.logger.log_consolidated_text(
            text_id,
            tracker,
            assignment.category_id,
            assignment.confidence
        )
        
        return assignment
    
    def _build_final_result(
        self,
        result,
        regular_assignments: List,
        consolidated_assignments: List
    ) -> Dict[str, Any]:
        """Build final result dictionary."""
        all_assignments = regular_assignments + consolidated_assignments
        
        self.logger.log_summary(
            len(consolidated_assignments),
            len(regular_assignments)
        )
        
        return {
            'categories': result.categories,
            'assignments': all_assignments,
            'metadata': self._build_metadata(
                result.metadata,
                regular_assignments,
                consolidated_assignments
            )
        }
    
    def _build_metadata(
        self,
        original_metadata: Dict,
        regular_assignments: List,
        consolidated_assignments: List
    ) -> Dict[str, Any]:
        """Build enhanced metadata with consolidation info."""
        return {
            **original_metadata,
            'big_texts_consolidated': len(consolidated_assignments),
            'regular_texts': len(regular_assignments),
            'total_big_texts': len(self.big_text_trackers)
        }



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
        clusterizer_config: Optional[ClusterConfig] = None,
        max_passes: int = 2,
        prefilter_k: int = 3,
        batch_size: int = 20,
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Process texts using pre-computed dry assignment.
        
        Args:
            dry_assignment_result: DryAssignmentResult from TextAssignmentManager
            clusterizer_config: ClusterConfig for clustering
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
        config: ClusterConfig,
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
        workflow_batches: List[List[Any]],
        config: ClusterConfig,
        max_passes: int,
        prefilter_k: int,
        batch_size: int,
        max_concurrent: int
    ) -> ClusterResult:
        """Execute a single workflow (cluster texts from assigned batches)."""
        
        # Extract texts from all batches in this workflow
        texts = []
        text_metadata = []
        
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
        
        # ✅ ADD: Store expected count
        expected_count = len(texts)
        
        logger.info(f"⚙️  Workflow {workflow_id}: Processing {expected_count} texts...")
        
        # Execute clustering on all texts in this workflow
        raw_result = await clusterize_texts(texts=texts, config=config)
        
        # ✅ ADD: Verify assignment count immediately
        actual_count = len(raw_result['assignments'])
        if actual_count != expected_count:
            logger.error(
                f"❌ Workflow {workflow_id}: Assignment count mismatch! "
                f"Expected {expected_count}, got {actual_count}"
            )
            # List missing texts for debugging
            assigned_texts = {a.text for a in raw_result['assignments']}
            for i, text in enumerate(texts):
                if text not in assigned_texts:
                    logger.error(f"   Missing text {i}: {text[:50]}...")
        
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
        
        logger.info(
            f"   ✓ Workflow {workflow_id} complete: "
            f"{len(result.categories)} categories, "
            f"{len(result.assignments)} assignments (expected {expected_count})"
        )
        
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
        """Merge two nodes by re-clustering their categories."""
        
        logger.info(f"   🔄 Merging nodes {node_a.node_id} + {node_b.node_id}")
        
        result_a = node_a.result
        result_b = node_b.result
        
        # Track input assignment count
        input_assignment_count = (
            len(result_a.assignments) + len(result_b.assignments)
        )
        
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
        
        # Step 2: Re-cluster the category representatives
        config = CONFIG_BALANCED_HYBRID
                
        merged_clustering = await clusterize_texts(
            texts=category_representatives,
            max_passes=1,
            batch_size=min(20, len(category_representatives)),
            config=config
        )
        
        logger.info(
            f"      ✓ Re-clustering: {len(category_representatives)} reps → "
            f"{len(merged_clustering['categories'])} meta-categories"
        )
        
        # Step 3: Remap original text assignments to new categories
        final_assignments = self._remap_assignments(
            merged_clustering,
            category_provenance,
            [result_a, result_b]
        )
        
        # Verify no assignments lost during merge
        output_assignment_count = len(final_assignments)
        if output_assignment_count != input_assignment_count:
            logger.error(
                f"❌ Merge {node_a.node_id}+{node_b.node_id}: "
                f"Assignment loss! Input={input_assignment_count}, "
                f"Output={output_assignment_count}, "
                f"Lost={input_assignment_count - output_assignment_count}"
            )
        else:
            logger.info(
                f"      ✓ Verified: {output_assignment_count} assignments preserved"
            )
        
        # Create merged result
        merged_result = ClusterResult(
            categories=merged_clustering['categories'],
            assignments=final_assignments,
            metadata={
                **merged_clustering['metadata'],
                'merge_level': level,
                'source_executions': [result_a.execution_id, result_b.execution_id],
                'input_assignments': input_assignment_count,  # ✅ Track for debugging
                'output_assignments': output_assignment_count,
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
        
        FIXED: Use indexed mapping instead of string matching to prevent assignment loss.
        """
        
        # Build mapping: old_category_id -> new_category_id
        old_to_new_category = {}
        
        # Build mapping based on index
        for i, assignment in enumerate(merged_clustering['assignments']):
            if i < len(category_provenance):
                prov = category_provenance[i]
                old_category_id = prov['category'].id
                new_category_id = assignment.category_id
                
                old_to_new_category[old_category_id] = new_category_id
                
                logger.debug(
                    f"Category mapping: {old_category_id} -> {new_category_id}"
                )
        
        # ✅ ADD: Verification logging
        logger.info(
            f"Built category mapping: {len(old_to_new_category)} old categories -> "
            f"{len(set(old_to_new_category.values()))} new categories"
        )
        
        # Remap all original text assignments
        final_assignments = []
        unmapped_count = 0
        
        for result in source_results:
            for assignment in result.assignments:
                old_category_id = assignment.category_id
                
                # ✅ FIX: Handle unmapped categories gracefully
                if old_category_id in old_to_new_category:
                    new_category_id = old_to_new_category[old_category_id]
                else:
                    # Keep original if not in mapping (shouldn't happen, but defensive)
                    new_category_id = old_category_id
                    unmapped_count += 1
                    logger.warning(
                        f"Assignment for text '{assignment.text[:50]}...' "
                        f"references unmapped category {old_category_id}"
                    )
                
                new_assignment = CategoryAssignment(
                    text=assignment.text,
                    category_id=new_category_id,
                    confidence=assignment.confidence,
                    reasoning=assignment.reasoning
                )
                final_assignments.append(new_assignment)
        
        # Summary logging of remapping
        logger.info(
            f"Remapped {len(final_assignments)} assignments "
            f"({unmapped_count} unmapped categories)"
        )
        
        return final_assignments
    
    def _consolidate_big_texts(self, result) -> Dict[str, Any]:
        """
        Consolidate assignments for big texts based on their chunks.
        
        This is the new implementation with reduced complexity.
        """
        consolidator = BigTextConsolidator(self.big_text_trackers)
        return consolidator.consolidate(result)

    
    async def process_with_priority_queue(
        self,
        dry_assignment_result,
        clusterizer_config: Optional[ClusterConfig] = CONFIG_BALANCED_HYBRID,
        max_passes: int = 2,
        prefilter_k: int = 3,
        batch_size: int = 20,
        max_concurrent: int = 5
    ):
        '''Process using priority queue execution.'''
        
        if clusterizer_config is None:
            clusterizer_config = CONFIG_BALANCED_HYBRID
        
        # Initialize big text trackers
        self._initialize_big_text_trackers(dry_assignment_result)
        
        # Store original text count for verification
        original_text_count = dry_assignment_result.total_slots_used
        
        # Execute with priority queue
        executor = PriorityQueueExecutor(self)
        root_node = await executor.run(
            dry_assignment_result,
            clusterizer_config,
            max_passes,
            prefilter_k,
            batch_size,
            max_concurrent
        )
        
        # Consolidate big texts
        final_result = self._consolidate_big_texts(root_node.result)
        
        # Update metadata with original text count
        final_result['metadata']['total_texts'] = original_text_count
        
        # Verify assignment count matches original text count
        actual_count = len(final_result['assignments'])
        if actual_count != original_text_count:
            logger.error(
                f"❌ CRITICAL: Assignment count mismatch! "
                f"Expected {original_text_count}, got {actual_count} "
                f"({original_text_count - actual_count} assignments lost)"
            )
            # Continue but log error for visibility
        
        return final_result
    
    def _get_next_node_id(self) -> int:
        """Get next node ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_with_existing_processor():
    """
    Example using your ACTUAL TreeMergeProcessor from document 8.
    
    This will work with your existing code!
    """
    
    from text_assignment_manager import TextAssignmentManager, Text
    from agentic_clusterizer import CONFIG_BALANCED_HYBRID
    
    # Create test data
    texts = [Text(id=f"text_{i}", content=f"Sample {i} " * 20) for i in range(20)]
    
    # Dry assignment
    manager = TextAssignmentManager(
        batch_size=5,
        max_batches_per_workflow=2,
        token_threshold=200
    )
    dry_assignment = manager.dry_assign(texts)
    
    print(f"Setup: {len(texts)} texts → {dry_assignment.total_workflows_needed} workflows\n")
        
    # OPTION 1: Use priority queue (NEW!)
    print("=" * 70)
    print("USING PRIORITY QUEUE EXECUTION")
    print("=" * 70)
    
    # First, add the PriorityQueueExecutor integration to your processor
    processor = TreeMergeProcessor(max_parallel_merges=4)
    
    # Use priority queue execution
    result = await processor.process_with_priority_queue(
        dry_assignment_result=dry_assignment,
        clusterizer_config=CONFIG_BALANCED_HYBRID,
        max_passes=1,
        batch_size=20,
        max_concurrent=5
    )
    
    print(f"\n✅ Priority Queue Complete!")
    print(f"Categories: {len(result['categories'])}")
    print(f"Assignments: {len(result['assignments'])}")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_with_existing_processor())