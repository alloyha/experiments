"""
Base classes for parallel execution strategies

Defines the interface for handling parallel step failures in DAG sagas.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from enum import Enum


class ParallelFailureStrategy(Enum):
    """Strategy for handling failures in parallel execution"""
    FAIL_FAST = "fail_fast"                    # Cancel others immediately
    WAIT_ALL = "wait_all"                      # Let all finish, then compensate
    FAIL_FAST_WITH_GRACE = "fail_fast_grace"   # Cancel new, wait for in-flight


class ParallelExecutionStrategy(ABC):
    """
    Base class for parallel execution strategies
    
    Defines how to handle parallel step execution and failures.
    """
    
    @abstractmethod
    async def execute_parallel_steps(self, steps: List[Any]) -> List[Any]:
        """
        Execute parallel steps according to the strategy
        
        Args:
            steps: List of steps to execute in parallel
            
        Returns:
            List of results from successful steps
            
        Raises:
            Exception: If strategy determines execution should fail
        """
        raise NotImplementedError("Subclasses must implement execute_parallel_steps")


