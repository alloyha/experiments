"""
Constraint system — Open/Closed principle.

New constraints are added by implementing `Constraint` and registering
them in a `ConstraintRegistry`.  The pipeline never needs modification.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from elevator.domain.models import Route, ElevatorState


# ---------------------------------------------------------------------------
# Base Constraint  (Interface Segregation: only feasibility check)
# ---------------------------------------------------------------------------

class Constraint(ABC):
    """Abstract base for all hard and soft constraints.

    A constraint is *feasible* when `check` returns True.
    Hard constraints that return False cause the route to be discarded.
    Soft constraints can be converted to penalty terms via `penalty`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in reports."""

    @abstractmethod
    def check(self, route: Route, state: ElevatorState) -> bool:
        """Return True if the route satisfies this constraint."""

    def penalty(self, route: Route, state: ElevatorState) -> float:
        """Optional numeric penalty for soft-constraint usage.

        Default: 0 if feasible, +inf if infeasible.
        Override to provide graded penalties.
        """
        return 0.0 if self.check(route, state) else float("inf")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


# ---------------------------------------------------------------------------
# Constraint Registry
# ---------------------------------------------------------------------------

class ConstraintRegistry:
    """Container for a named set of constraints.

    Supports:
        registry.add(constraint)
        registry.remove("name")
        registry.check_all(route, state)   → bool (all hard constraints)
        registry.total_penalty(route, state) → float
    """

    def __init__(self) -> None:
        self._constraints: dict[str, Constraint] = {}

    def add(self, constraint: Constraint) -> "ConstraintRegistry":
        """Fluent add — returns self for chaining."""
        if constraint.name in self._constraints:
            raise ValueError(f"Constraint '{constraint.name}' already registered.")
        self._constraints[constraint.name] = constraint
        return self

    def remove(self, name: str) -> "ConstraintRegistry":
        self._constraints.pop(name, None)
        return self

    def get(self, name: str) -> Constraint | None:
        return self._constraints.get(name)

    def __iter__(self):
        return iter(self._constraints.values())

    def __len__(self) -> int:
        return len(self._constraints)

    def check_all(self, route: Route, state: ElevatorState) -> bool:
        """True only when every registered constraint is satisfied."""
        return all(c.check(route, state) for c in self._constraints.values())

    def violations(self, route: Route, state: ElevatorState) -> list[str]:
        """Return names of violated constraints (useful for debugging)."""
        return [
            name for name, c in self._constraints.items()
            if not c.check(route, state)
        ]

    def total_penalty(self, route: Route, state: ElevatorState) -> float:
        """Sum of all constraint penalties (used in soft-constraint mode)."""
        return sum(c.penalty(route, state) for c in self._constraints.values())
