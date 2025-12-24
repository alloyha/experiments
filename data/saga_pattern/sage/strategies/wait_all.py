"""
WAIT_ALL Strategy Implementation

Waits for all parallel steps to complete before handling failures.
Allows maximum work to be done before starting compensation.
"""

import asyncio
from typing import Any

from sage.strategies.base import ParallelExecutionStrategy


class WaitAllStrategy(ParallelExecutionStrategy):
    """
    Implements WAIT_ALL parallel failure strategy
    
    When one or more parallel steps fail:
    1. Let all parallel steps run to completion
    2. Collect all results and exceptions
    3. Raise exception if any step failed, but only after all complete
    """

    async def execute_parallel_steps(self, steps: list[Any]) -> list[Any]:
        """
        Execute steps in parallel, waiting for all to complete
        
        Args:
            steps: List of steps to execute (each step should have an execute() method)
            
        Returns:
            List of results from all successful steps
            
        Raises:
            Exception: Any exception from failed steps (after all steps complete)
        """
        if not steps:
            return []

        # Create tasks for all steps
        tasks = [asyncio.create_task(step.execute()) for step in steps]

        # Wait for ALL tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful results from exceptions
        successful_results = []
        exceptions = []

        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                successful_results.append(result)

        # If any step failed, raise the first exception
        # (but only after all steps had a chance to complete)
        if exceptions:
            raise exceptions[0]

        return successful_results
