"""
Saga State Machine - Handles saga lifecycle state transitions

This module contains the state machine logic for saga execution,
providing robust state management with proper async support using python-statemachine.

State Diagram:
    PENDING → EXECUTING → COMPLETED
                     ↘ COMPENSATING → ROLLED_BACK  
                     ↘ FAILED (unrecoverable)

Step State Diagram:
    PENDING → EXECUTING → COMPLETED
                     ↘ COMPENSATING → COMPENSATED
                     ↘ FAILED (unrecoverable)
"""

from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.core import Saga


class SagaStateMachine(StateMachine):
    """
    State machine for managing saga lifecycle transitions
    
    Provides async state management with guards and hooks for saga execution.
    """
    
    # Define saga states
    pending = State("Pending", initial=True)
    executing = State("Executing") 
    completed = State("Completed", final=True)
    compensating = State("Compensating")
    rolled_back = State("RolledBack", final=True)
    failed = State("Failed", final=True)
    
    # Define state transitions with conditions
    start = pending.to(executing, cond="can_start")
    succeed = executing.to(completed)
    fail = executing.to(compensating, cond="can_compensate") 
    fail_unrecoverable = executing.to(failed)
    finish_compensation = compensating.to(rolled_back)
    compensation_failed = compensating.to(failed)
    
    def __init__(self, saga: "Saga"):
        self.saga = saga
        super().__init__()
    
    # Guard conditions
    def can_start(self) -> bool:
        """Guard: can only start if we have steps defined"""
        return len(self.saga.steps) > 0
    
    def can_compensate(self) -> bool:
        """Guard: can only compensate if we have completed steps to undo"""
        return len(self.saga.completed_steps) > 0
    
    # State entry hooks
    async def on_enter_executing(self) -> None:
        """Called when entering EXECUTING state - start saga execution"""
        await self.saga._on_enter_executing()
    
    async def on_enter_compensating(self) -> None:
        """Called when entering COMPENSATING state - start compensation"""
        await self.saga._on_enter_compensating()
    
    async def on_enter_completed(self) -> None:
        """Called when entering COMPLETED state - saga succeeded"""
        await self.saga._on_enter_completed()
    
    async def on_enter_rolled_back(self) -> None:
        """Called when entering ROLLED_BACK state - saga compensated successfully"""
        await self.saga._on_enter_rolled_back()
    
    async def on_enter_failed(self) -> None:
        """Called when entering FAILED state - unrecoverable failure"""
        await self.saga._on_enter_failed()


class SagaStepStateMachine(StateMachine):
    """
    State machine for individual saga step execution
    
    Manages the lifecycle of a single step within a saga.
    """
    
    # Define step states
    pending = State("Pending", initial=True)
    executing = State("Executing")
    completed = State("Completed")  # NOT final - can transition to compensating
    compensating = State("Compensating") 
    compensated = State("Compensated", final=True)
    failed = State("Failed", final=True)
    
    # Define transitions
    start = pending.to(executing)
    succeed = executing.to(completed)
    fail = executing.to(failed)
    compensate = completed.to(compensating)
    compensation_success = compensating.to(compensated)
    compensation_failure = compensating.to(failed)
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        super().__init__()
    
    async def on_enter_executing(self) -> None:
        """Called when step starts executing"""
        pass  # Override in saga implementation if needed
    
    async def on_enter_completed(self) -> None:
        """Called when step completes successfully"""
        pass  # Override in saga implementation if needed
    
    async def on_enter_compensating(self) -> None:
        """Called when step starts compensation"""
        pass  # Override in saga implementation if needed
    
    async def on_enter_compensated(self) -> None:
        """Called when step compensation completes"""
        pass  # Override in saga implementation if needed
    
    async def on_enter_failed(self) -> None:
        """Called when step fails"""
        pass  # Override in saga implementation if needed


def validate_state_transition(current_state: str, target_state: str) -> bool:
    """
    Validate if a state transition is allowed
    
    Args:
        current_state: Current state name
        target_state: Target state name
        
    Returns:
        True if transition is valid, False otherwise
    """
    valid_transitions = {
        "Pending": ["Executing"],
        "Executing": ["Completed", "Compensating", "Failed"],
        "Compensating": ["RolledBack", "Failed"],
        "Completed": [],  # Final state
        "RolledBack": [],  # Final state  
        "Failed": [],  # Final state
    }
    
    return target_state in valid_transitions.get(current_state, [])


def get_valid_next_states(current_state: str) -> list[str]:
    """
    Get list of valid next states from current state
    
    Args:
        current_state: Current state name
        
    Returns:
        List of valid next state names
    """
    valid_transitions = {
        "Pending": ["Executing"],
        "Executing": ["Completed", "Compensating", "Failed"], 
        "Compensating": ["RolledBack", "Failed"],
        "Completed": [],
        "RolledBack": [],
        "Failed": [],
    }
    
    return valid_transitions.get(current_state, [])