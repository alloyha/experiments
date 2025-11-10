"""
FAIL_FAST Strategy Implementation

Cancels all other parallel steps immediately when one fails.
Most aggressive failure handling - minimizes resource usage and execution time.
"""

import asyncio
from typing import List, Any
from sage.strategies.base import ParallelExecutionStrategy


class FailFastStrategy(ParallelExecutionStrategy):
    """
    Implements FAIL_FAST parallel failure strategy
    
    When one parallel step fails:
    1. Immediately cancel all other parallel steps that haven't started
    2. Cancel running parallel steps (best effort)  
    3. Raise the first exception encountered
    """
    
    async def execute_parallel_steps(self, steps: List[Any]) -> List[Any]:
        """
        Execute steps in parallel, failing fast on first error
        
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
            # Wait for any task to complete or fail
            done, pending = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_EXCEPTION
            )
            
            # Check if any completed task failed
            for task in done:
                if task.exception():
                    # Cancel all pending tasks immediately
                    for pending_task in pending:
                        pending_task.cancel()
                    
                    # Wait for cancellations to complete (with short timeout)
                    if pending:
                        await asyncio.wait(pending, timeout=1.0)
                    
                    # Re-raise the first exception
                    raise task.exception()
            
            # If we get here, all tasks completed successfully
            return [task.result() for task in tasks]
            
        except asyncio.CancelledError:
            # Handle case where this strategy itself is cancelled
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise