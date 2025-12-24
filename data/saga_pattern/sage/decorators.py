"""
Decorator API for declarative saga definitions.

Provides a user-friendly way to define sagas using decorators rather than
manual step registration. This makes saga definitions more readable and
easier to maintain.

Quick Start:
    >>> from sage import Saga, step, compensate
    >>> 
    >>> class OrderSaga(Saga):
    ...     @step(name="create_order")
    ...     async def create_order(self, ctx):
    ...         return await OrderService.create(ctx["order_data"])
    ...     
    ...     @compensate("create_order") 
    ...     async def cancel_order(self, ctx):
    ...         await OrderService.delete(ctx["order_id"])
    ...     
    ...     @step(name="charge_payment", depends_on=["create_order"])
    ...     async def charge(self, ctx):
    ...         return await PaymentService.charge(ctx["amount"])
    ...     
    ...     @compensate("charge_payment", depends_on=["create_order"])
    ...     async def refund(self, ctx):
    ...         await PaymentService.refund(ctx["charge_id"])
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from sage.compensation_graph import CompensationType, SagaCompensationGraph

# Type for saga step functions
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@dataclass
class StepMetadata:
    """Metadata attached to step functions via decorator."""
    name: str
    depends_on: list[str] = field(default_factory=list)
    aggregate_type: str | None = None
    event_type: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    description: str | None = None


@dataclass
class CompensationMetadata:
    """Metadata attached to compensation functions via decorator."""
    for_step: str
    depends_on: list[str] = field(default_factory=list)
    compensation_type: CompensationType = CompensationType.MECHANICAL
    timeout_seconds: float = 30.0
    max_retries: int = 3
    description: str | None = None


def step(
    name: str,
    depends_on: list[str] | None = None,
    aggregate_type: str | None = None,
    event_type: str | None = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 3,
    description: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to mark a method as a saga step.
    
    Args:
        name: Unique identifier for this step
        depends_on: List of step names that must complete before this step
        aggregate_type: Event aggregate type (for outbox pattern)
        event_type: Event type (for outbox pattern)
        timeout_seconds: Step execution timeout (default: 60s)
        max_retries: Maximum retry attempts (default: 3)
        description: Human-readable description
    
    Example:
        >>> class OrderSaga(Saga):
        ...     @step(name="create_order", aggregate_type="order")
        ...     async def create_order(self, ctx):
        ...         order = await OrderService.create(ctx["order_data"])
        ...         return {"order_id": order.id}
        ...     
        ...     @step(name="charge_payment", depends_on=["create_order"])
        ...     async def charge_payment(self, ctx):
        ...         charge = await PaymentService.charge(ctx["amount"])
        ...         return {"charge_id": charge.id}
    """
    def decorator(func: F) -> F:
        # Store metadata on the function
        func._saga_step_meta = StepMetadata(
            name=name,
            depends_on=depends_on or [],
            aggregate_type=aggregate_type,
            event_type=event_type,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            description=description or f"Execute {name}"
        )
        return func
    return decorator


def compensate(
    for_step: str,
    depends_on: list[str] | None = None,
    compensation_type: CompensationType = CompensationType.MECHANICAL,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    description: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to mark a method as compensation for a step.
    
    Compensation functions are called when a saga fails and needs to
    undo previously completed steps.
    
    Args:
        for_step: Name of the step this compensates
        depends_on: Steps whose compensations must complete BEFORE this one
        compensation_type: Type of compensation (MECHANICAL, SEMANTIC, MANUAL)
        timeout_seconds: Compensation timeout (default: 30s)
        max_retries: Maximum retry attempts (default: 3)
        description: Human-readable description
    
    Example:
        >>> class OrderSaga(Saga):
        ...     @step(name="charge_payment")
        ...     async def charge(self, ctx):
        ...         return await PaymentService.charge(ctx["amount"])
        ...     
        ...     @compensate("charge_payment", compensation_type=CompensationType.SEMANTIC)
        ...     async def refund(self, ctx):
        ...         # Semantic compensation: issue a refund (not exactly reversing charge)
        ...         await PaymentService.refund(ctx["charge_id"])
    """
    def decorator(func: F) -> F:
        func._saga_compensation_meta = CompensationMetadata(
            for_step=for_step,
            depends_on=depends_on or [],
            compensation_type=compensation_type,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            description=description or f"Compensate {for_step}"
        )
        return func
    return decorator


@dataclass
class SagaStepDefinition:
    """
    Complete definition of a saga step with its compensation.
    
    Used internally to track step and compensation pairs.
    """
    step_id: str
    forward_fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]
    compensation_fn: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    depends_on: list[str] = field(default_factory=list)
    compensation_depends_on: list[str] = field(default_factory=list)
    compensation_type: CompensationType = CompensationType.MECHANICAL
    aggregate_type: str | None = None
    event_type: str | None = None
    timeout_seconds: float = 60.0
    compensation_timeout_seconds: float = 30.0
    max_retries: int = 3
    description: str | None = None


class Saga:
    """
    Base class for declarative saga definitions.
    
    Subclass this and use @step and @compensate decorators to define
    your saga's steps declaratively. The saga will automatically:
    
    - Collect all decorated methods
    - Build the execution dependency graph
    - Build the compensation dependency graph
    - Execute steps in parallel where possible
    - Compensate in correct order on failure
    
    Example:
        >>> class OrderSaga(Saga):
        ...     '''Saga to place an order with payment processing.'''
        ...     
        ...     @step(name="validate_order")
        ...     async def validate(self, ctx):
        ...         # Validation has no compensation - pure check
        ...         if ctx["total"] <= 0:
        ...             raise ValueError("Invalid order total")
        ...         return {"validated": True}
        ...     
        ...     @step(name="create_order", depends_on=["validate_order"])
        ...     async def create_order(self, ctx):
        ...         order = await OrderRepository.create(ctx["order_data"])
        ...         return {"order_id": order.id}
        ...     
        ...     @compensate("create_order")
        ...     async def cancel_order(self, ctx):
        ...         await OrderRepository.delete(ctx["order_id"])
        ...     
        ...     @step(name="reserve_inventory", depends_on=["create_order"])
        ...     async def reserve(self, ctx):
        ...         reservation = await InventoryService.reserve(ctx["items"])
        ...         return {"reservation_id": reservation.id}
        ...     
        ...     @compensate("reserve_inventory")
        ...     async def release_inventory(self, ctx):
        ...         await InventoryService.release(ctx["reservation_id"])
        ...     
        ...     @step(name="charge_payment", depends_on=["create_order"])
        ...     async def charge(self, ctx):
        ...         # Runs in parallel with reserve_inventory
        ...         charge = await PaymentService.charge(ctx["amount"])
        ...         return {"charge_id": charge.id}
        ...     
        ...     @compensate("charge_payment", depends_on=["reserve_inventory"])
        ...     async def refund(self, ctx):
        ...         # Refund after inventory is released
        ...         await PaymentService.refund(ctx["charge_id"])
        ...
        >>> # Execute the saga
        >>> saga = OrderSaga()
        >>> result = await saga.run({"order_data": {...}, "amount": 99.99})
    """

    def __init__(self):
        self._steps: list[SagaStepDefinition] = []
        self._step_registry: dict[str, SagaStepDefinition] = {}
        self._compensation_graph = SagaCompensationGraph()
        self._context: dict[str, Any] = {}
        self._collect_steps()

    def _collect_steps(self) -> None:
        """Collect decorated methods into step definitions."""
        # First pass: collect all steps
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)

            if hasattr(attr, "_saga_step_meta"):
                meta: StepMetadata = attr._saga_step_meta
                step_def = SagaStepDefinition(
                    step_id=meta.name,
                    forward_fn=attr,
                    depends_on=meta.depends_on.copy(),
                    aggregate_type=meta.aggregate_type,
                    event_type=meta.event_type,
                    timeout_seconds=meta.timeout_seconds,
                    max_retries=meta.max_retries,
                    description=meta.description
                )
                self._steps.append(step_def)
                self._step_registry[meta.name] = step_def

        # Second pass: attach compensations
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)

            if hasattr(attr, "_saga_compensation_meta"):
                meta: CompensationMetadata = attr._saga_compensation_meta
                step_name = meta.for_step

                if step_name in self._step_registry:
                    step = self._step_registry[step_name]
                    step.compensation_fn = attr
                    step.compensation_depends_on = meta.depends_on.copy()
                    step.compensation_type = meta.compensation_type
                    step.compensation_timeout_seconds = meta.timeout_seconds

    def get_steps(self) -> list[SagaStepDefinition]:
        """Get all step definitions."""
        return self._steps.copy()

    def get_step(self, name: str) -> SagaStepDefinition | None:
        """Get a specific step by name."""
        return self._step_registry.get(name)

    def get_execution_order(self) -> list[list[SagaStepDefinition]]:
        """
        Compute step execution order respecting dependencies.
        
        Returns:
            List of levels, where steps in each level can run in parallel
        
        Raises:
            ValueError: If circular dependencies detected
        """
        if not self._steps:
            return []

        step_map = {s.step_id: s for s in self._steps}
        in_degree = {s.step_id: len(s.depends_on) for s in self._steps}
        remaining = set(step_map.keys())

        return self._topological_sort_steps(step_map, in_degree, remaining)

    def _topological_sort_steps(
        self,
        step_map: dict[str, SagaStepDefinition],
        in_degree: dict[str, int],
        remaining: set
    ) -> list[list[SagaStepDefinition]]:
        """Perform topological sort on steps."""
        levels: list[list[SagaStepDefinition]] = []

        while remaining:
            current_level = [step_map[sid] for sid in remaining if in_degree[sid] == 0]

            if not current_level:
                raise ValueError("Circular dependency detected in saga steps")

            levels.append(current_level)

            for step in current_level:
                remaining.remove(step.step_id)
                for other_id in remaining:
                    if step.step_id in step_map[other_id].depends_on:
                        in_degree[other_id] -= 1

        return levels

    async def run(
        self,
        initial_context: dict[str, Any],
        saga_id: str | None = None
    ) -> dict[str, Any]:
        """
        Execute this saga.
        
        Args:
            initial_context: Initial data for the saga
            saga_id: Optional saga identifier for tracing
        
        Returns:
            Final context with all step results merged
        
        Raises:
            Exception: If saga fails (compensations will be attempted first)
        """
        import uuid

        saga_id = saga_id or str(uuid.uuid4())
        self._context = initial_context.copy()
        self._compensation_graph.reset_execution()

        # Register all compensations in the graph
        for step in self._steps:
            if step.compensation_fn:
                self._compensation_graph.register_compensation(
                    step.step_id,
                    step.compensation_fn,
                    depends_on=step.compensation_depends_on,
                    compensation_type=step.compensation_type,
                    max_retries=step.max_retries,
                    timeout_seconds=step.compensation_timeout_seconds
                )

        try:
            # Execute steps level by level
            execution_levels = self.get_execution_order()

            for level in execution_levels:
                await self._execute_level(level)

            return self._context

        except Exception:
            # Compensate executed steps
            await self._compensate()
            raise

    async def _execute_level(self, level: list[SagaStepDefinition]) -> None:
        """Execute all steps in a level concurrently."""
        if not level:
            return

        tasks = [
            self._execute_step(step)
            for step in level
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        for result in results:
            if isinstance(result, Exception):
                raise result

    async def _execute_step(self, step: SagaStepDefinition) -> None:
        """Execute a single step."""
        try:
            # Apply timeout
            result = await asyncio.wait_for(
                step.forward_fn(self._context),
                timeout=step.timeout_seconds
            )

            # Merge result into context
            if result and isinstance(result, dict):
                self._context.update(result)

            # Mark step as executed for compensation tracking
            self._context[f"__{step.step_id}_completed"] = True
            self._compensation_graph.mark_step_executed(step.step_id)

        except TimeoutError:
            raise TimeoutError(f"Step '{step.step_id}' timed out after {step.timeout_seconds}s")

    async def _compensate(self) -> None:
        """Execute compensations in dependency order."""
        comp_levels = self._compensation_graph.get_compensation_order()

        for level in comp_levels:
            tasks = [
                self._execute_compensation(step_id)
                for step_id in level
            ]
            # Execute compensations in parallel, don't fail on individual errors
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_compensation(self, step_id: str) -> None:
        """Execute a single compensation."""
        node = self._compensation_graph.get_compensation_info(step_id)
        if not node:
            return

        try:
            await asyncio.wait_for(
                node.compensation_fn(self._context),
                timeout=node.timeout_seconds
            )
            self._context[f"__{step_id}_compensated"] = True
        except Exception as e:
            # Log but don't fail - we want all compensations to attempt
            self._context[f"__{step_id}_compensation_error"] = str(e)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(steps={len(self._steps)})"


# Backward compatibility alias
DeclarativeSaga = Saga

