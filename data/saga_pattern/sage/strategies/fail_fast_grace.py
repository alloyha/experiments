"""
FAIL_FAST_WITH_GRACE Strategy Implementation

Balanced approach: cancel remaining steps but allow grace period for in-flight ones.
Provides good balance between resource efficiency and graceful handling.
"""

import asyncio
from typing import List, Any
from sage.strategies.base import ParallelExecutionStrategy


class FailFastWithGraceStrategy(ParallelExecutionStrategy):
    """
    Implements FAIL_FAST_WITH_GRACE parallel failure strategy
    
    When one parallel step fails:
    1. Cancel remaining parallel steps that haven't started yet
    2. Wait gracefully for in-flight parallel steps to complete (with timeout)
    3. Begin compensation after graceful completion or timeout
    """
    
    def __init__(self, grace_period: float = 2.0):
        """
        Initialize strategy with grace period
        
        Args:
            grace_period: Maximum time to wait for in-flight steps to complete
        """
        self.grace_period = grace_period
    
    async def execute_parallel_steps(self, steps: List[Any]) -> List[Any]:
        """
        Execute steps in parallel with graceful failure handling
        
        Args:
            steps: List of steps to execute (each step should have an execute() method)
            
        Returns:
            List of results from all successful steps
            
        Raises:
            Exception: First exception encountered from any step
        """
        if not steps:
            return []
        
        # Create tasks for all steps
        tasks = [asyncio.create_task(step.execute()) for step in steps]
        
        try:
            # Wait for first failure or all completions
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )
            
            # Check if any completed task failed
            failed_task = None
            for task in done:
                if task.exception():
                    failed_task = task
                    break
            
            if failed_task:
                # Allow grace period for pending tasks to complete
                if pending:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=self.grace_period
                        )
                    except asyncio.TimeoutError:
                        # Cancel tasks that didn't complete within grace period
                        for task in pending:
                            if not task.done():
                                task.cancel()
                
                # Raise the original exception
                raise failed_task.exception()
            
            # All tasks completed successfully
            return [task.result() for task in tasks]
            
        except asyncio.CancelledError:
            # Handle case where this strategy itself is cancelled
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
                
            # Check if task has actually started executing
            if self._is_task_executing(task):
                in_flight_tasks.append((step_name, task))
            else:
                not_started_tasks.append((step_name, task))
        
        # Cancel tasks that haven't started
        for step_name, task in not_started_tasks:
            task.cancel()
            step_statuses[step_name] = SagaStepStatus.FAILED
        
        # Wait gracefully for in-flight tasks
        if in_flight_tasks:
            in_flight_task_objects = [task for _, task in in_flight_tasks]
            
            try:
                # Give in-flight tasks reasonable time to complete
                results = await asyncio.wait_for(
                    asyncio.gather(*in_flight_task_objects, return_exceptions=True),
                    timeout=30.0  # Grace period for completion
                )
                
                # Update statuses based on results
                for i, (step_name, _) in enumerate(in_flight_tasks):
                    result = results[i]
                    if isinstance(result, Exception):
                        step_statuses[step_name] = SagaStepStatus.FAILED
                    else:
                        step_statuses[step_name] = SagaStepStatus.COMPLETED
                        
            except asyncio.TimeoutError:
                # If grace period expires, cancel remaining tasks
                for step_name, task in in_flight_tasks:
                    if not task.done():
                        task.cancel()
                        step_statuses[step_name] = SagaStepStatus.FAILED
        
        # Quick cleanup of any cancelled tasks
        cancelled_tasks = [task for _, task in not_started_tasks if not task.done()]
        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
    
    def _is_task_executing(self, task: asyncio.Task) -> bool:
        """
        Heuristic to determine if a task has actually started executing
        This is a simplified implementation - in practice, you might want
        more sophisticated detection based on your step implementation
        """
        # Simple heuristic: if task is not done and not cancelled, assume executing
        return not task.done() and not task.cancelled()
    
    def should_wait_for_completion(self) -> bool:
        """FAIL_FAST_WITH_GRACE waits for in-flight tasks only"""
        return True
    
    def get_description(self) -> str:
        """Human-readable description of this strategy"""
        return "FAIL_FAST_WITH_GRACE: Cancel pending, wait for in-flight steps"