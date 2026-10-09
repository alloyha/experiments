from __future__ import annotations

"""
Duwhal-guided regimen search with:

- greedy search;
- beam search;
- exhaustive exact oracle;
- regimen-evaluation memoization;
- Duwhal-query memoization;
- internal profiling;
- complete beam-frontier tracing;
- interleaved timing benchmark.

Architecture
============

Duwhal
    Candidate discovery / structural proximity.

solver.py
    Domain semantics:
        - residual burden;
        - therapeutic effects;
        - adverse effects;
        - feasibility;
        - interactions;
        - objective J_p(M).

guided_solver.py
    Search strategy and experimentation:
        - greedy;
        - beam;
        - exact oracle;
        - caching;
        - profiling;
        - benchmarking.

All Demo* data currently used by the project is synthetic and exists
only for mathematical/software experiments.
"""

import argparse
import statistics
import time

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import duckdb

from explore import (
    build_duwhal_interactions,
    candidate_coverage,
    generate_candidates,
    load_therapeutic_support,
)

from solver import (
    ClinicalModel,
    RegimenSolver,
    resolve_pathology,
)


# ============================================================
# CONFIGURATION
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "patologias.duckdb"

EPSILON = 1e-12


# ============================================================
# SEARCH DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class GuidedCandidate:
    presentation_id: int

    score: float
    hops: int
    reason: str

    direct: bool
    coverage: float

    covered_symptoms: frozenset[int]


@dataclass(frozen=True)
class GuidedMove:
    move_type: str

    from_regimen: tuple[int, ...]
    to_regimen: tuple[int, ...]

    added: tuple[int, ...]
    removed: tuple[int, ...]

    evaluation: object


@dataclass(frozen=True)
class SearchStep:
    iteration: int

    move: GuidedMove

    previous_objective: float
    new_objective: float
    delta_j: float

    candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class BeamNode:
    evaluation: object
    steps: tuple[SearchStep, ...]


@dataclass(frozen=True)
class ExactResult:
    evaluation: object

    evaluated_regimens: int
    feasible_regimens: int


# ============================================================
# PROFILING
# ============================================================


@dataclass
class SearchProfiler:
    """
    Internal wall-clock profiler.

    Components:

        solver
            Actual solver.evaluate() calls on cache misses.

        duwhal
            Candidate-cache lookup plus actual Duwhal work on misses.

        neighborhood
            ADD / REMOVE / SWITCH construction excluding solver time.

        frontier
            Beam deduplication, ranking, truncation and trace handling.

        other
            Difference between wall-clock search time and explicitly
            accounted components.
    """

    times: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    search_started_at: float | None = None
    search_finished_at: float | None = None

    def start_search(self) -> None:
        self.search_started_at = time.perf_counter()

    def stop_search(self) -> None:
        self.search_finished_at = time.perf_counter()

    def add_time(
        self,
        component: str,
        seconds: float,
    ) -> None:
        self.times[component] += seconds

    def increment(
        self,
        component: str,
        amount: int = 1,
    ) -> None:
        self.counts[component] += amount

    @property
    def total_seconds(self) -> float:
        if (
            self.search_started_at is None
            or self.search_finished_at is None
        ):
            return 0.0

        return (
            self.search_finished_at
            - self.search_started_at
        )

    @property
    def accounted_seconds(self) -> float:
        return sum(
            self.times[name]
            for name in (
                "solver",
                "duwhal",
                "neighborhood",
                "frontier",
            )
        )

    @property
    def other_seconds(self) -> float:
        return max(
            0.0,
            self.total_seconds
            - self.accounted_seconds,
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "total": self.total_seconds,
            "solver": self.times["solver"],
            "duwhal": self.times["duwhal"],
            "neighborhood": self.times["neighborhood"],
            "frontier": self.times["frontier"],
            "other": self.other_seconds,
        }


# ============================================================
# FRONTIER TRACE
# ============================================================


@dataclass(frozen=True)
class FrontierState:
    regimen: tuple[int, ...]
    objective: float

    parent_regimen: tuple[int, ...] | None

    move_type: str | None

    added: tuple[int, ...]
    removed: tuple[int, ...]

    delta_j: float | None

    selected: bool


@dataclass(frozen=True)
class FrontierExpansion:
    parent_regimen: tuple[int, ...]
    parent_objective: float

    candidate_ids: tuple[int, ...]

    generated: tuple[FrontierState, ...]


@dataclass(frozen=True)
class BeamDepthTrace:
    depth: int

    active_frontier: tuple[FrontierState, ...]

    expansions: tuple[FrontierExpansion, ...]

    generated_unique: tuple[FrontierState, ...]

    next_frontier: tuple[FrontierState, ...]


@dataclass
class BeamFrontierTrace:
    depths: list[BeamDepthTrace] = field(
        default_factory=list
    )

    def add(
        self,
        value: BeamDepthTrace,
    ) -> None:
        self.depths.append(value)


# ============================================================
# SEARCH RESULT
# ============================================================


@dataclass(frozen=True)
class SearchResult:
    strategy: str

    initial_evaluation: object
    final_evaluation: object

    steps: tuple[SearchStep, ...]

    converged: bool
    iterations: int

    discovered_regimens: int

    evaluate_calls: int
    unique_evaluations: int
    evaluation_cache_hits: int

    duwhal_calls: int
    unique_duwhal_queries: int
    duwhal_cache_hits: int

    profile: dict[str, float]

    frontier_trace: BeamFrontierTrace | None = None


# ============================================================
# BENCHMARK DATA
# ============================================================


@dataclass(frozen=True)
class BenchmarkSample:
    seconds: float

    objective: float
    regimen: tuple[int, ...]

    evaluate_calls: int
    unique_evaluations: int
    evaluation_cache_hits: int

    duwhal_calls: int
    unique_duwhal_queries: int
    duwhal_cache_hits: int

    profile: dict[str, float]


@dataclass(frozen=True)
class BenchmarkStats:
    strategy: str

    samples: tuple[BenchmarkSample, ...]

    median_seconds: float
    p25_seconds: float
    p75_seconds: float

    objective: float
    regimen: tuple[int, ...]

    evaluate_calls: int
    unique_evaluations: int
    evaluation_cache_hits: int

    duwhal_calls: int
    unique_duwhal_queries: int
    duwhal_cache_hits: int

    profile_median: dict[str, float]


# ============================================================
# BASIC UTILITIES
# ============================================================


def normalize_regimen(
    regimen: Iterable[int],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            set(regimen)
        )
    )


def parse_regimen(
    value: str | None,
) -> tuple[int, ...]:
    if value is None:
        return tuple()

    value = value.strip()

    if not value:
        return tuple()

    return normalize_regimen(
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    )


def parse_int_list(
    value: str,
) -> tuple[int, ...]:
    values = tuple(
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    )

    if not values:
        raise ValueError(
            "Integer list cannot be empty."
        )

    if any(
        value < 1
        for value in values
    ):
        raise ValueError(
            "All integer-list values must be >= 1."
        )

    return tuple(
        dict.fromkeys(
            values
        )
    )


def format_regimen(
    model: ClinicalModel,
    regimen: Iterable[int],
) -> str:
    regimen = normalize_regimen(
        regimen
    )

    if not regimen:
        return "∅"

    return " + ".join(
        model
        .presentation_by_id[
            presentation_id
        ]
        .label

        for presentation_id
        in regimen
    )


def format_symptoms(
    model: ClinicalModel,
    symptoms: Iterable[int],
) -> str:
    symptoms = tuple(
        sorted(
            symptoms
        )
    )

    if not symptoms:
        return "∅"

    return (
        "{ "
        + ", ".join(
            model.symptom_names[
                symptom_id
            ]
            for symptom_id
            in symptoms
        )
        + " }"
    )


def evaluation_key(
    evaluation,
):
    regimen = normalize_regimen(
        evaluation.regimen
    )

    return (
        evaluation.objective,
        len(regimen),
        regimen,
    )


def strictly_better(
    candidate,
    reference,
) -> bool:
    return (
        candidate.objective
        <
        reference.objective
        - EPSILON
    )


# ============================================================
# EVALUATION CACHE
# ============================================================


class EvaluationCache:
    """
    Memoized wrapper around RegimenSolver.evaluate().
    """

    def __init__(
        self,
        solver: RegimenSolver,
        profiler: SearchProfiler | None = None,
    ):
        self.solver = solver
        self.profiler = profiler

        self._cache: dict[
            tuple[int, ...],
            object,
        ] = {}

        self.calls = 0
        self.hits = 0
        self.misses = 0

    def evaluate(
        self,
        regimen: Iterable[int],
    ):
        self.calls += 1

        key = normalize_regimen(
            regimen
        )

        if key in self._cache:
            self.hits += 1

            if self.profiler is not None:
                self.profiler.increment(
                    "evaluation_cache_hit"
                )

            return self._cache[key]

        self.misses += 1

        if self.profiler is not None:
            self.profiler.increment(
                "evaluation_cache_miss"
            )

        started = time.perf_counter()

        evaluation = self.solver.evaluate(
            key
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        if self.profiler is not None:
            self.profiler.add_time(
                "solver",
                elapsed,
            )

        self._cache[key] = evaluation

        return evaluation

    @property
    def unique_evaluations(
        self,
    ) -> int:
        return self.misses


# ============================================================
# DUWHAL CANDIDATE CACHE
# ============================================================


class DuwhalCandidateCache:
    """
    Cache Duwhal results by unresolved symptom set.

    Candidate generation depends on:

        U_p(M)
        depth
        top
        candidate_mode

    Current-regimen filtering is done after cache retrieval.
    """

    def __init__(
        self,
        *,
        interactions,
        therapeutic_support,
        profiler: SearchProfiler | None = None,
    ):
        self.interactions = interactions
        self.therapeutic_support = therapeutic_support
        self.profiler = profiler

        self._cache: dict[
            tuple[
                frozenset[int],
                int,
                int,
                str,
            ],
            tuple[GuidedCandidate, ...],
        ] = {}

        self.calls = 0
        self.hits = 0
        self.misses = 0

    def get(
        self,
        *,
        evaluation,
        depth: int,
        top: int,
        candidate_mode: str,
    ) -> tuple[GuidedCandidate, ...]:
        started = time.perf_counter()

        self.calls += 1

        unresolved = frozenset(
            evaluation
            .symptom_space
            .unresolved_support
        )

        if not unresolved:
            if self.profiler is not None:
                self.profiler.add_time(
                    "duwhal",
                    time.perf_counter()
                    - started,
                )

            return tuple()

        key = (
            unresolved,
            depth,
            top,
            candidate_mode,
        )

        if key in self._cache:
            self.hits += 1

            if self.profiler is not None:
                self.profiler.increment(
                    "duwhal_cache_hit"
                )

            all_candidates = (
                self._cache[key]
            )

        else:
            self.misses += 1

            if self.profiler is not None:
                self.profiler.increment(
                    "duwhal_cache_miss"
                )

            all_candidates = (
                self._generate(
                    unresolved=unresolved,
                    depth=depth,
                    top=top,
                    candidate_mode=(
                        candidate_mode
                    ),
                )
            )

            self._cache[
                key
            ] = all_candidates

        current = frozenset(
            evaluation.regimen
        )

        result = tuple(
            candidate
            for candidate
            in all_candidates
            if (
                candidate.presentation_id
                not in current
            )
        )

        if self.profiler is not None:
            self.profiler.add_time(
                "duwhal",
                time.perf_counter()
                - started,
            )

        return result

    def _generate(
        self,
        *,
        unresolved: frozenset[int],
        depth: int,
        top: int,
        candidate_mode: str,
    ) -> tuple[GuidedCandidate, ...]:
        frame = generate_candidates(
            self.interactions,
            unresolved,
            top=top,
            depth=depth,
        )

        if frame.empty:
            return tuple()

        result: list[
            GuidedCandidate
        ] = []

        for row in frame.to_dict(
            orient="records"
        ):
            presentation_id = int(
                row[
                    "presentation_id"
                ]
            )

            covered, coverage = (
                candidate_coverage(
                    presentation_id,
                    unresolved,
                    self.therapeutic_support,
                )
            )

            direct = bool(
                covered
            )

            if (
                candidate_mode == "direct"
                and not direct
            ):
                continue

            result.append(
                GuidedCandidate(
                    presentation_id=(
                        presentation_id
                    ),

                    score=float(
                        row["score"]
                    ),

                    hops=int(
                        row["hops"]
                    ),

                    reason=str(
                        row["reason"]
                    ),

                    direct=direct,

                    coverage=float(
                        coverage
                    ),

                    covered_symptoms=(
                        covered
                    ),
                )
            )

        return tuple(
            result
        )

    @property
    def unique_queries(
        self,
    ) -> int:
        return self.misses


# ============================================================
# SEARCH CONTEXT
# ============================================================


@dataclass
class SearchContext:
    evaluator: EvaluationCache
    candidate_cache: DuwhalCandidateCache
    profiler: SearchProfiler


def make_search_context(
    *,
    solver,
    interactions,
    therapeutic_support,
) -> SearchContext:
    profiler = SearchProfiler()

    evaluator = EvaluationCache(
        solver,
        profiler=profiler,
    )

    candidate_cache = (
        DuwhalCandidateCache(
            interactions=interactions,
            therapeutic_support=(
                therapeutic_support
            ),
            profiler=profiler,
        )
    )

    return SearchContext(
        evaluator=evaluator,
        candidate_cache=candidate_cache,
        profiler=profiler,
    )


# ============================================================
# GUIDED NEIGHBORHOOD
# ============================================================


def build_guided_moves(
    *,
    evaluator: EvaluationCache,
    current_evaluation,
    candidates: tuple[
        GuidedCandidate,
        ...
    ],
    profiler: SearchProfiler | None = None,
) -> tuple[
    GuidedMove,
    ...
]:
    """
    Construct ADD / REMOVE / SWITCH neighborhood.

    neighborhood timing excludes actual solver.evaluate()
    cache-miss time.
    """

    started = time.perf_counter()

    solver_before = (
        profiler.times["solver"]
        if profiler is not None
        else 0.0
    )

    current = frozenset(
        current_evaluation.regimen
    )

    candidate_ids = frozenset(
        candidate.presentation_id
        for candidate
        in candidates
    )

    moves: dict[
        tuple[int, ...],
        GuidedMove,
    ] = {}

    # --------------------------------------------------------
    # ADD
    # --------------------------------------------------------

    for added_id in sorted(
        candidate_ids - current
    ):
        destination = (
            normalize_regimen(
                current
                | {
                    added_id
                }
            )
        )

        evaluation = (
            evaluator.evaluate(
                destination
            )
        )

        moves[
            destination
        ] = GuidedMove(
            move_type="ADD",

            from_regimen=(
                normalize_regimen(
                    current
                )
            ),

            to_regimen=destination,

            added=(
                added_id,
            ),

            removed=tuple(),

            evaluation=evaluation,
        )

    # --------------------------------------------------------
    # REMOVE
    # --------------------------------------------------------

    for removed_id in sorted(
        current
    ):
        destination = (
            normalize_regimen(
                current
                - {
                    removed_id
                }
            )
        )

        evaluation = (
            evaluator.evaluate(
                destination
            )
        )

        moves[
            destination
        ] = GuidedMove(
            move_type="REMOVE",

            from_regimen=(
                normalize_regimen(
                    current
                )
            ),

            to_regimen=destination,

            added=tuple(),

            removed=(
                removed_id,
            ),

            evaluation=evaluation,
        )

    # --------------------------------------------------------
    # SWITCH
    # --------------------------------------------------------

    for removed_id in sorted(
        current
    ):
        reduced = (
            current
            - {
                removed_id
            }
        )

        for added_id in sorted(
            candidate_ids - current
        ):
            destination = (
                normalize_regimen(
                    reduced
                    | {
                        added_id
                    }
                )
            )

            if destination in moves:
                continue

            evaluation = (
                evaluator.evaluate(
                    destination
                )
            )

            moves[
                destination
            ] = GuidedMove(
                move_type="SWITCH",

                from_regimen=(
                    normalize_regimen(
                        current
                    )
                ),

                to_regimen=(
                    destination
                ),

                added=(
                    added_id,
                ),

                removed=(
                    removed_id,
                ),

                evaluation=(
                    evaluation
                ),
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    if profiler is not None:
        solver_delta = (
            profiler.times["solver"]
            - solver_before
        )

        profiler.add_time(
            "neighborhood",
            max(
                0.0,
                elapsed
                - solver_delta,
            ),
        )

        profiler.increment(
            "neighborhood_builds"
        )

        profiler.increment(
            "neighbor_destinations",
            len(moves),
        )

    return tuple(
        moves.values()
    )


# ============================================================
# SEARCH RESULT BUILDER
# ============================================================


def make_search_result(
    *,
    strategy: str,
    initial,
    final,
    steps,
    converged: bool,
    iterations: int,
    discovered,
    context: SearchContext,
    frontier_trace: BeamFrontierTrace | None = None,
) -> SearchResult:
    return SearchResult(
        strategy=strategy,

        initial_evaluation=initial,
        final_evaluation=final,

        steps=tuple(
            steps
        ),

        converged=converged,
        iterations=iterations,

        discovered_regimens=len(
            discovered
        ),

        evaluate_calls=(
            context
            .evaluator
            .calls
        ),

        unique_evaluations=(
            context
            .evaluator
            .unique_evaluations
        ),

        evaluation_cache_hits=(
            context
            .evaluator
            .hits
        ),

        duwhal_calls=(
            context
            .candidate_cache
            .calls
        ),

        unique_duwhal_queries=(
            context
            .candidate_cache
            .unique_queries
        ),

        duwhal_cache_hits=(
            context
            .candidate_cache
            .hits
        ),

        profile=(
            context
            .profiler
            .snapshot()
        ),

        frontier_trace=(
            frontier_trace
        ),
    )


# ============================================================
# GREEDY SEARCH
# ============================================================


def greedy_search(
    *,
    solver,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth: int,
    top: int,
    candidate_mode: str,
    max_iterations: int,
    verbose: bool = False,
) -> SearchResult:
    context = make_search_context(
        solver=solver,
        interactions=interactions,
        therapeutic_support=(
            therapeutic_support
        ),
    )

    context.profiler.start_search()

    initial = (
        context
        .evaluator
        .evaluate(
            initial_regimen
        )
    )

    if not (
        initial
        .feasibility
        .feasible
    ):
        context.profiler.stop_search()

        raise ValueError(
            "Initial regimen is infeasible: "
            + "; ".join(
                initial
                .feasibility
                .violations
            )
        )

    current = initial

    steps: list[
        SearchStep
    ] = []

    discovered = {
        normalize_regimen(
            initial.regimen
        )
    }

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        candidates = (
            context
            .candidate_cache
            .get(
                evaluation=current,
                depth=depth,
                top=top,
                candidate_mode=(
                    candidate_mode
                ),
            )
        )

        moves = build_guided_moves(
            evaluator=(
                context.evaluator
            ),

            current_evaluation=(
                current
            ),

            candidates=candidates,

            profiler=(
                context.profiler
            ),
        )

        discovered.update(
            move.to_regimen
            for move
            in moves
        )

        improving = [
            move

            for move
            in moves

            if (
                move
                .evaluation
                .feasibility
                .feasible

                and strictly_better(
                    move.evaluation,
                    current,
                )
            )
        ]

        if verbose:
            print()

            print(
                f"[greedy iteration {iteration}]"
            )

            print(
                f"  M           : "
                f"{current.regimen}"
            )

            print(
                f"  J           : "
                f"{current.objective:.6f}"
            )

            print(
                "  unresolved  : "
                f"{current.symptom_space.unresolved_support}"
            )

            print(
                "  K           : "
                f"{[candidate.presentation_id for candidate in candidates]}"
            )

            print(
                f"  neighbors   : "
                f"{len(moves)}"
            )

        if not improving:
            context.profiler.stop_search()

            return make_search_result(
                strategy="greedy",

                initial=initial,
                final=current,

                steps=steps,

                converged=True,

                iterations=len(
                    steps
                ),

                discovered=discovered,

                context=context,
            )

        move = min(
            improving,

            key=lambda value: (
                evaluation_key(
                    value.evaluation
                )
            ),
        )

        previous = current
        current = move.evaluation

        step = SearchStep(
            iteration=iteration,

            move=move,

            previous_objective=(
                previous.objective
            ),

            new_objective=(
                current.objective
            ),

            delta_j=(
                current.objective
                - previous.objective
            ),

            candidate_ids=tuple(
                candidate.presentation_id
                for candidate
                in candidates
            ),
        )

        steps.append(
            step
        )

        if verbose:
            print(
                f"  accepted    : "
                f"{move.move_type} "
                f"{move.to_regimen}"
            )

            print(
                f"  ΔJ          : "
                f"{step.delta_j:+.6f}"
            )

    context.profiler.stop_search()

    return make_search_result(
        strategy="greedy",

        initial=initial,
        final=current,

        steps=steps,

        converged=False,

        iterations=max_iterations,

        discovered=discovered,

        context=context,
    )


# ============================================================
# BEAM TRACE HELPERS
# ============================================================


def active_frontier_trace(
    beam: list[BeamNode],
) -> tuple[
    FrontierState,
    ...
]:
    return tuple(
        FrontierState(
            regimen=normalize_regimen(
                node.evaluation.regimen
            ),

            objective=(
                node.evaluation.objective
            ),

            parent_regimen=None,
            move_type=None,

            added=tuple(),
            removed=tuple(),

            delta_j=None,

            selected=True,
        )

        for node
        in beam
    )


def node_to_frontier_state(
    node: BeamNode,
    selected: bool,
) -> FrontierState:
    regimen = normalize_regimen(
        node.evaluation.regimen
    )

    if node.steps:
        last_step = (
            node.steps[-1]
        )

        return FrontierState(
            regimen=regimen,

            objective=(
                node.evaluation.objective
            ),

            parent_regimen=(
                last_step
                .move
                .from_regimen
            ),

            move_type=(
                last_step
                .move
                .move_type
            ),

            added=(
                last_step
                .move
                .added
            ),

            removed=(
                last_step
                .move
                .removed
            ),

            delta_j=(
                last_step.delta_j
            ),

            selected=selected,
        )

    return FrontierState(
        regimen=regimen,

        objective=(
            node.evaluation.objective
        ),

        parent_regimen=None,
        move_type=None,

        added=tuple(),
        removed=tuple(),

        delta_j=None,

        selected=selected,
    )


# ============================================================
# BEAM SEARCH
# ============================================================


def beam_search(
    *,
    solver,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth: int,
    top: int,
    candidate_mode: str,
    max_iterations: int,
    beam_width: int,
    verbose: bool = False,
    trace_frontier: bool = False,
) -> SearchResult:
    if beam_width < 1:
        raise ValueError(
            "beam_width must be >= 1"
        )

    context = make_search_context(
        solver=solver,
        interactions=interactions,
        therapeutic_support=(
            therapeutic_support
        ),
    )

    context.profiler.start_search()

    trace = (
        BeamFrontierTrace()
        if trace_frontier
        else None
    )

    initial = (
        context
        .evaluator
        .evaluate(
            initial_regimen
        )
    )

    if not (
        initial
        .feasibility
        .feasible
    ):
        context.profiler.stop_search()

        raise ValueError(
            "Initial regimen is infeasible: "
            + "; ".join(
                initial
                .feasibility
                .violations
            )
        )

    beam = [
        BeamNode(
            evaluation=initial,
            steps=tuple(),
        )
    ]

    best = beam[0]

    discovered = {
        normalize_regimen(
            initial.regimen
        )
    }

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        active_trace = (
            active_frontier_trace(
                beam
            )
            if trace is not None
            else tuple()
        )

        generated: dict[
            tuple[int, ...],
            BeamNode,
        ] = {}

        expansion_traces: list[
            FrontierExpansion
        ] = []

        if verbose:
            print()

            print(
                f"[beam depth {iteration}]"
            )

            print(
                f"  active states: "
                f"{len(beam)}"
            )

        for rank, node in enumerate(
            beam,
            start=1,
        ):
            current = (
                node.evaluation
            )

            candidates = (
                context
                .candidate_cache
                .get(
                    evaluation=current,
                    depth=depth,
                    top=top,
                    candidate_mode=(
                        candidate_mode
                    ),
                )
            )

            moves = build_guided_moves(
                evaluator=(
                    context.evaluator
                ),

                current_evaluation=(
                    current
                ),

                candidates=candidates,

                profiler=(
                    context.profiler
                ),
            )

            discovered.update(
                move.to_regimen
                for move
                in moves
            )

            if verbose:
                print(
                    f"  {rank}. "
                    f"M={current.regimen} "
                    f"J={current.objective:.6f} "
                    f"K={[candidate.presentation_id for candidate in candidates]}"
                )

            frontier_started = (
                time.perf_counter()
            )

            child_trace: list[
                FrontierState
            ] = []

            for move in moves:
                candidate = (
                    move.evaluation
                )

                if not (
                    candidate
                    .feasibility
                    .feasible
                ):
                    continue

                if not strictly_better(
                    candidate,
                    current,
                ):
                    continue

                destination = (
                    normalize_regimen(
                        candidate.regimen
                    )
                )

                step = SearchStep(
                    iteration=iteration,

                    move=move,

                    previous_objective=(
                        current.objective
                    ),

                    new_objective=(
                        candidate.objective
                    ),

                    delta_j=(
                        candidate.objective
                        - current.objective
                    ),

                    candidate_ids=tuple(
                        item.presentation_id
                        for item
                        in candidates
                    ),
                )

                new_node = BeamNode(
                    evaluation=candidate,

                    steps=(
                        node.steps
                        + (
                            step,
                        )
                    ),
                )

                previous = (
                    generated.get(
                        destination
                    )
                )

                if (
                    previous is None
                    or evaluation_key(
                        new_node.evaluation
                    )
                    <
                    evaluation_key(
                        previous.evaluation
                    )
                ):
                    generated[
                        destination
                    ] = new_node

                if trace is not None:
                    child_trace.append(
                        FrontierState(
                            regimen=destination,

                            objective=(
                                candidate.objective
                            ),

                            parent_regimen=(
                                normalize_regimen(
                                    current.regimen
                                )
                            ),

                            move_type=(
                                move.move_type
                            ),

                            added=(
                                move.added
                            ),

                            removed=(
                                move.removed
                            ),

                            delta_j=(
                                candidate.objective
                                - current.objective
                            ),

                            selected=False,
                        )
                    )

            if trace is not None:
                expansion_traces.append(
                    FrontierExpansion(
                        parent_regimen=(
                            normalize_regimen(
                                current.regimen
                            )
                        ),

                        parent_objective=(
                            current.objective
                        ),

                        candidate_ids=tuple(
                            candidate.presentation_id
                            for candidate
                            in candidates
                        ),

                        generated=tuple(
                            sorted(
                                child_trace,

                                key=lambda state: (
                                    state.objective,
                                    len(
                                        state.regimen
                                    ),
                                    state.regimen,
                                ),
                            )
                        ),
                    )
                )

            context.profiler.add_time(
                "frontier",
                time.perf_counter()
                - frontier_started,
            )

        # ----------------------------------------------------
        # TERMINAL FRONTIER
        # ----------------------------------------------------

        if not generated:
            if trace is not None:
                trace.add(
                    BeamDepthTrace(
                        depth=iteration,

                        active_frontier=(
                            active_trace
                        ),

                        expansions=tuple(
                            expansion_traces
                        ),

                        generated_unique=tuple(),

                        next_frontier=tuple(),
                    )
                )

            context.profiler.stop_search()

            return make_search_result(
                strategy=(
                    f"beam[{beam_width}]"
                ),

                initial=initial,

                final=(
                    best.evaluation
                ),

                steps=(
                    best.steps
                ),

                converged=True,

                iterations=(
                    iteration - 1
                ),

                discovered=discovered,

                context=context,

                frontier_trace=trace,
            )

        # ----------------------------------------------------
        # RANK + PRUNE
        # ----------------------------------------------------

        frontier_started = (
            time.perf_counter()
        )

        ranked = sorted(
            generated.values(),

            key=lambda node: (
                evaluation_key(
                    node.evaluation
                )
            ),
        )

        beam = ranked[
            :beam_width
        ]

        selected_regimens = {
            normalize_regimen(
                node.evaluation.regimen
            )
            for node
            in beam
        }

        iteration_best = (
            beam[0]
        )

        if (
            evaluation_key(
                iteration_best.evaluation
            )
            <
            evaluation_key(
                best.evaluation
            )
        ):
            best = iteration_best

        if trace is not None:
            generated_trace = tuple(
                node_to_frontier_state(
                    node,
                    selected=(
                        normalize_regimen(
                            node.evaluation.regimen
                        )
                        in selected_regimens
                    ),
                )

                for node
                in ranked
            )

            next_trace = tuple(
                state
                for state
                in generated_trace
                if state.selected
            )

            trace.add(
                BeamDepthTrace(
                    depth=iteration,

                    active_frontier=(
                        active_trace
                    ),

                    expansions=tuple(
                        expansion_traces
                    ),

                    generated_unique=(
                        generated_trace
                    ),

                    next_frontier=(
                        next_trace
                    ),
                )
            )

        context.profiler.add_time(
            "frontier",
            time.perf_counter()
            - frontier_started,
        )

        if verbose:
            print(
                "  next beam:"
            )

            for rank, node in enumerate(
                beam,
                start=1,
            ):
                print(
                    f"    {rank}. "
                    f"M={node.evaluation.regimen} "
                    f"J={node.evaluation.objective:.6f}"
                )

    context.profiler.stop_search()

    return make_search_result(
        strategy=(
            f"beam[{beam_width}]"
        ),

        initial=initial,

        final=(
            best.evaluation
        ),

        steps=(
            best.steps
        ),

        converged=False,

        iterations=max_iterations,

        discovered=discovered,

        context=context,

        frontier_trace=trace,
    )


# ============================================================
# EXACT ORACLE
# ============================================================


def powerset(
    ids: Iterable[int],
):
    ids = tuple(
        sorted(
            ids
        )
    )

    for size in range(
        len(ids) + 1
    ):
        yield from combinations(
            ids,
            size,
        )


def exact_global_optimum(
    solver,
    model,
) -> ExactResult:
    best = None

    evaluated = 0
    feasible = 0

    for regimen in powerset(
        model
        .presentation_by_id
        .keys()
    ):
        evaluated += 1

        evaluation = solver.evaluate(
            regimen
        )

        if not (
            evaluation
            .feasibility
            .feasible
        ):
            continue

        feasible += 1

        if (
            best is None
            or evaluation_key(
                evaluation
            )
            <
            evaluation_key(
                best
            )
        ):
            best = evaluation

    if best is None:
        raise RuntimeError(
            "No feasible regimen found."
        )

    return ExactResult(
        evaluation=best,

        evaluated_regimens=(
            evaluated
        ),

        feasible_regimens=(
            feasible
        ),
    )


# ============================================================
# SEARCH DISPATCH
# ============================================================


def run_search(
    *,
    strategy: str,
    solver,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth: int,
    top: int,
    candidate_mode: str,
    max_iterations: int,
    beam_width: int,
    verbose: bool,
    trace_frontier: bool = False,
):
    common = dict(
        solver=solver,

        interactions=interactions,

        therapeutic_support=(
            therapeutic_support
        ),

        initial_regimen=(
            initial_regimen
        ),

        depth=depth,
        top=top,

        candidate_mode=(
            candidate_mode
        ),

        max_iterations=(
            max_iterations
        ),

        verbose=verbose,
    )

    if strategy == "greedy":
        return greedy_search(
            **common
        )

    if strategy == "beam":
        return beam_search(
            **common,

            beam_width=(
                beam_width
            ),

            trace_frontier=(
                trace_frontier
            ),
        )

    raise ValueError(
        f"Unknown strategy: "
        f"{strategy}"
    )


# ============================================================
# TIMING
# ============================================================


def time_call(
    function: Callable,
):
    started = time.perf_counter()

    result = function()

    elapsed = (
        time.perf_counter()
        - started
    )

    return (
        result,
        elapsed,
    )


def percentile_quartiles(
    values: list[float],
):
    if len(values) == 1:
        return (
            values[0],
            values[0],
        )

    quartiles = (
        statistics.quantiles(
            values,
            n=4,
            method="inclusive",
        )
    )

    return (
        quartiles[0],
        quartiles[2],
    )


# ============================================================
# BENCHMARK RUNNERS
# ============================================================


def make_guided_benchmark_runner(
    *,
    strategy,
    beam_width,
    solver,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth,
    top,
    candidate_mode,
    max_iterations,
):
    def run():
        return run_search(
            strategy=strategy,

            solver=solver,

            interactions=(
                interactions
            ),

            therapeutic_support=(
                therapeutic_support
            ),

            initial_regimen=(
                initial_regimen
            ),

            depth=depth,
            top=top,

            candidate_mode=(
                candidate_mode
            ),

            max_iterations=(
                max_iterations
            ),

            beam_width=(
                beam_width
            ),

            verbose=False,

            # Benchmark should not pay for trace construction.
            trace_frontier=False,
        )

    return run


def benchmark_sample_from_guided(
    result: SearchResult,
    seconds: float,
) -> BenchmarkSample:
    final = (
        result
        .final_evaluation
    )

    return BenchmarkSample(
        seconds=seconds,

        objective=(
            final.objective
        ),

        regimen=(
            normalize_regimen(
                final.regimen
            )
        ),

        evaluate_calls=(
            result.evaluate_calls
        ),

        unique_evaluations=(
            result.unique_evaluations
        ),

        evaluation_cache_hits=(
            result.evaluation_cache_hits
        ),

        duwhal_calls=(
            result.duwhal_calls
        ),

        unique_duwhal_queries=(
            result.unique_duwhal_queries
        ),

        duwhal_cache_hits=(
            result.duwhal_cache_hits
        ),

        profile=dict(
            result.profile
        ),
    )


def benchmark_sample_from_exact(
    result: ExactResult,
    seconds: float,
) -> BenchmarkSample:
    final = result.evaluation

    return BenchmarkSample(
        seconds=seconds,

        objective=(
            final.objective
        ),

        regimen=(
            normalize_regimen(
                final.regimen
            )
        ),

        evaluate_calls=(
            result
            .evaluated_regimens
        ),

        unique_evaluations=(
            result
            .evaluated_regimens
        ),

        evaluation_cache_hits=0,

        duwhal_calls=0,
        unique_duwhal_queries=0,
        duwhal_cache_hits=0,

        profile={
            "total": seconds,
            "solver": seconds,
            "duwhal": 0.0,
            "neighborhood": 0.0,
            "frontier": 0.0,
            "other": 0.0,
        },
    )


def aggregate_benchmark(
    *,
    strategy: str,
    samples: list[
        BenchmarkSample
    ],
) -> BenchmarkStats:
    times = [
        sample.seconds
        for sample
        in samples
    ]

    p25, p75 = (
        percentile_quartiles(
            times
        )
    )

    reference = samples[
        0
    ]

    def median_int(
        attribute: str,
    ) -> int:
        return int(
            statistics.median(
                getattr(
                    sample,
                    attribute
                )
                for sample
                in samples
            )
        )

    components = (
        "solver",
        "duwhal",
        "neighborhood",
        "frontier",
        "other",
    )

    profile_median = {
        component: statistics.median(
            sample.profile[
                component
            ]
            for sample
            in samples
        )

        for component
        in components
    }

    return BenchmarkStats(
        strategy=strategy,

        samples=tuple(
            samples
        ),

        median_seconds=(
            statistics.median(
                times
            )
        ),

        p25_seconds=p25,
        p75_seconds=p75,

        objective=(
            reference.objective
        ),

        regimen=(
            reference.regimen
        ),

        evaluate_calls=(
            median_int(
                "evaluate_calls"
            )
        ),

        unique_evaluations=(
            median_int(
                "unique_evaluations"
            )
        ),

        evaluation_cache_hits=(
            median_int(
                "evaluation_cache_hits"
            )
        ),

        duwhal_calls=(
            median_int(
                "duwhal_calls"
            )
        ),

        unique_duwhal_queries=(
            median_int(
                "unique_duwhal_queries"
            )
        ),

        duwhal_cache_hits=(
            median_int(
                "duwhal_cache_hits"
            )
        ),

        profile_median=(
            profile_median
        ),
    )


# ============================================================
# BENCHMARK
# ============================================================


def run_benchmark(
    *,
    solver,
    model,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth,
    top,
    candidate_mode,
    max_iterations,
    beam_widths,
    runs,
    warmup,
):
    runners: dict[
        str,
        Callable,
    ] = {}

    runners[
        "greedy"
    ] = make_guided_benchmark_runner(
        strategy="greedy",

        beam_width=1,

        solver=solver,

        interactions=(
            interactions
        ),

        therapeutic_support=(
            therapeutic_support
        ),

        initial_regimen=(
            initial_regimen
        ),

        depth=depth,
        top=top,

        candidate_mode=(
            candidate_mode
        ),

        max_iterations=(
            max_iterations
        ),
    )

    for width in beam_widths:
        name = (
            f"beam[{width}]"
        )

        runners[
            name
        ] = make_guided_benchmark_runner(
            strategy="beam",

            beam_width=width,

            solver=solver,

            interactions=(
                interactions
            ),

            therapeutic_support=(
                therapeutic_support
            ),

            initial_regimen=(
                initial_regimen
            ),

            depth=depth,
            top=top,

            candidate_mode=(
                candidate_mode
            ),

            max_iterations=(
                max_iterations
            ),
        )

    runners[
        "exact"
    ] = lambda: exact_global_optimum(
        solver,
        model,
    )

    names = list(
        runners
    )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    for warmup_round in range(
        warmup
    ):
        offset = (
            warmup_round
            % len(names)
        )

        ordered = (
            names[offset:]
            + names[:offset]
        )

        for name in ordered:
            runners[
                name
            ]()

    # --------------------------------------------------------
    # TIMED ROUNDS
    # --------------------------------------------------------

    samples = {
        name: []
        for name
        in names
    }

    for round_index in range(
        runs
    ):
        offset = (
            round_index
            % len(names)
        )

        ordered = (
            names[offset:]
            + names[:offset]
        )

        for name in ordered:
            result, seconds = (
                time_call(
                    runners[
                        name
                    ]
                )
            )

            if name == "exact":
                sample = (
                    benchmark_sample_from_exact(
                        result,
                        seconds,
                    )
                )

            else:
                sample = (
                    benchmark_sample_from_guided(
                        result,
                        seconds,
                    )
                )

            samples[
                name
            ].append(
                sample
            )

    stats = [
        aggregate_benchmark(
            strategy=name,
            samples=samples[
                name
            ],
        )

        for name
        in names
    ]

    print_benchmark(
        model=model,
        stats=stats,
    )

    print_benchmark_profile(
        stats
    )


# ============================================================
# SEARCH REPORT
# ============================================================


def print_search_result(
    model: ClinicalModel,
    result: SearchResult,
) -> None:
    print()

    print(
        "=" * 110
    )

    print(
        f"{result.strategy.upper()} SEARCH RESULT"
    )

    print(
        "=" * 110
    )

    for step in result.steps:
        print()

        print(
            f"Step {step.iteration}"
        )

        print(
            "-" * 110
        )

        print(
            f"move      : "
            f"{step.move.move_type}"
        )

        print(
            f"from      : "
            f"{step.move.from_regimen}"
        )

        print(
            f"to        : "
            f"{step.move.to_regimen}"
        )

        print(
            f"J         : "
            f"{step.previous_objective:.6f}"
            f" -> "
            f"{step.new_objective:.6f}"
        )

        print(
            f"ΔJ        : "
            f"{step.delta_j:+.6f}"
        )

        print(
            f"Duwhal K  : "
            f"{step.candidate_ids}"
        )

    final = (
        result.final_evaluation
    )

    print()

    print(
        "Regimen:"
    )

    print(
        " ",
        format_regimen(
            model,
            final.regimen,
        ),
    )

    print()

    print(
        f"J_p(M)                  : "
        f"{final.objective:.6f}"
    )

    print(
        f"Residual burden         : "
        f"{final.symptom_space.total_residual_burden:.6f}"
    )

    print(
        f"Feasible                : "
        f"{final.feasibility.feasible}"
    )

    print(
        f"Accepted moves          : "
        f"{len(result.steps)}"
    )

    print(
        f"Converged               : "
        f"{result.converged}"
    )

    print()

    print(
        "Controlled:"
    )

    print(
        " ",
        format_symptoms(
            model,
            final
            .symptom_space
            .controlled_support,
        ),
    )

    print()

    print(
        "Unresolved:"
    )

    print(
        " ",
        format_symptoms(
            model,
            final
            .symptom_space
            .unresolved_support,
        ),
    )

    print()

    print(
        "Search instrumentation"
    )

    print(
        "-" * 50
    )

    print(
        f"Discovered regimens     : "
        f"{result.discovered_regimens}"
    )

    print(
        f"evaluate() requests     : "
        f"{result.evaluate_calls}"
    )

    print(
        f"unique evaluations      : "
        f"{result.unique_evaluations}"
    )

    print(
        f"evaluation cache hits   : "
        f"{result.evaluation_cache_hits}"
    )

    print(
        f"Duwhal requests         : "
        f"{result.duwhal_calls}"
    )

    print(
        f"unique Duwhal queries   : "
        f"{result.unique_duwhal_queries}"
    )

    print(
        f"Duwhal cache hits       : "
        f"{result.duwhal_cache_hits}"
    )


# ============================================================
# INTERNAL PROFILE REPORT
# ============================================================


def print_profile(
    result: SearchResult,
) -> None:
    profile = (
        result.profile
    )

    total = profile[
        "total"
    ]

    print()

    print(
        "=" * 80
    )

    print(
        "INTERNAL SEARCH PROFILE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"{'component':<24}"
        f"{'time (ms)':>14}"
        f"{'% total':>14}"
    )

    print(
        "-" * 52
    )

    for component in (
        "duwhal",
        "solver",
        "neighborhood",
        "frontier",
        "other",
    ):
        seconds = (
            profile[
                component
            ]
        )

        percentage = (
            seconds
            / total
            * 100.0

            if total > EPSILON
            else 0.0
        )

        print(
            f"{component:<24}"
            f"{seconds * 1000:>14.3f}"
            f"{percentage:>13.2f}%"
        )

    print(
        "-" * 52
    )

    print(
        f"{'total':<24}"
        f"{total * 1000:>14.3f}"
        f"{100.0:>13.2f}%"
    )


# ============================================================
# FRONTIER TRACE REPORT
# ============================================================


def print_frontier_trace(
    model: ClinicalModel,
    trace: BeamFrontierTrace,
) -> None:
    print()

    print(
        "=" * 132
    )

    print(
        "BEAM FRONTIER TRACE"
    )

    print(
        "=" * 132
    )

    for depth in trace.depths:
        print()

        print(
            f"DEPTH {depth.depth}"
        )

        print(
            "-" * 132
        )

        print()

        print(
            "ACTIVE FRONTIER"
        )

        if not depth.active_frontier:
            print(
                "  ∅"
            )

        for rank, state in enumerate(
            depth.active_frontier,
            start=1,
        ):
            print(
                f"  {rank:>2}. "
                f"J={state.objective:.6f}  "
                f"M={state.regimen!s:<18}  "
                f"{format_regimen(model, state.regimen)}"
            )

        print()

        print(
            "EXPANSIONS"
        )

        if not depth.expansions:
            print(
                "  ∅"
            )

        for expansion in depth.expansions:
            print()

            print(
                f"  parent "
                f"M={expansion.parent_regimen} "
                f"J={expansion.parent_objective:.6f}"
            )

            print(
                f"    Duwhal K = "
                f"{expansion.candidate_ids}"
            )

            if not expansion.generated:
                print(
                    "    no feasible improving children"
                )

                continue

            for child in (
                expansion.generated
            ):
                print(
                    f"    "
                    f"{child.move_type:<7}"
                    f"M={str(child.regimen):<18}"
                    f" "
                    f"J={child.objective:.6f}"
                    f" "
                    f"ΔJ={child.delta_j:+.6f}"
                )

        print()

        print(
            "UNIQUE GENERATED STATES"
        )

        if not depth.generated_unique:
            print(
                "  ∅"
            )

        for rank, state in enumerate(
            depth.generated_unique,
            start=1,
        ):
            marker = (
                "★"
                if state.selected
                else " "
            )

            delta = (
                f"{state.delta_j:+.6f}"
                if state.delta_j is not None
                else "n/a"
            )

            print(
                f" {marker} "
                f"{rank:>2}. "
                f"J={state.objective:.6f} "
                f"ΔJ={delta:>10} "
                f"M={str(state.regimen):<18} "
                f"via {state.move_type} "
                f"from {state.parent_regimen}"
            )

        print()

        print(
            "NEXT FRONTIER"
        )

        if not depth.next_frontier:
            print(
                "  ∅"
            )

        for rank, state in enumerate(
            depth.next_frontier,
            start=1,
        ):
            print(
                f"  {rank:>2}. "
                f"J={state.objective:.6f} "
                f"M={state.regimen!s:<18} "
                f"{format_regimen(model, state.regimen)}"
            )


# ============================================================
# EXACT COMPARISON
# ============================================================


def print_exact_comparison(
    model,
    guided,
    exact_result: ExactResult,
) -> None:
    exact = (
        exact_result.evaluation
    )

    guided_ids = frozenset(
        guided.regimen
    )

    exact_ids = frozenset(
        exact.regimen
    )

    gap = (
        guided.objective
        - exact.objective
    )

    relative_gap = (
        gap
        / abs(
            exact.objective
        )

        if abs(
            exact.objective
        )
        > EPSILON

        else 0.0
    )

    print()

    print(
        "=" * 110
    )

    print(
        "EXACT ORACLE COMPARISON"
    )

    print(
        "=" * 110
    )

    print()

    print(
        "Guided:"
    )

    print(
        " ",
        format_regimen(
            model,
            guided.regimen,
        ),
    )

    print(
        f"  J = "
        f"{guided.objective:.6f}"
    )

    print()

    print(
        "Exact:"
    )

    print(
        " ",
        format_regimen(
            model,
            exact.regimen,
        ),
    )

    print(
        f"  J = "
        f"{exact.objective:.6f}"
    )

    print()

    print(
        f"Same regimen          : "
        f"{guided_ids == exact_ids}"
    )

    print(
        f"Objective gap         : "
        f"{gap:+.6f}"
    )

    print(
        f"Relative gap          : "
        f"{relative_gap:+.4%}"
    )

    print(
        f"Missing exact meds    : "
        f"{sorted(exact_ids - guided_ids)}"
    )

    print(
        f"Extra guided meds     : "
        f"{sorted(guided_ids - exact_ids)}"
    )

    print()

    print(
        f"Exact states evaluated: "
        f"{exact_result.evaluated_regimens}"
    )

    print(
        f"Exact feasible states : "
        f"{exact_result.feasible_regimens}"
    )


# ============================================================
# BENCHMARK REPORT
# ============================================================


def print_benchmark(
    *,
    model,
    stats: list[
        BenchmarkStats
    ],
) -> None:
    exact = next(
        row
        for row
        in stats
        if row.strategy == "exact"
    )

    print()

    print(
        "=" * 146
    )

    print(
        "TIME + SEARCH-SPACE BENCHMARK"
    )

    print(
        "=" * 146
    )

    print()

    print(
        f"{'strategy':<12}"
        f"{'median ms':>12}"
        f"{'p25 ms':>11}"
        f"{'p75 ms':>11}"
        f"{'speedup':>10}"
        f"{'J':>12}"
        f"{'gap':>11}"
        f"{'same':>8}"
        f"{'eval req':>11}"
        f"{'unique M':>11}"
        f"{'eval hit':>10}"
        f"{'D queries':>11}"
        f"{'D unique':>10}"
        f"{'D hit':>8}"
    )

    print(
        "-" * 146
    )

    for row in stats:
        gap = (
            row.objective
            - exact.objective
        )

        speedup = (
            exact.median_seconds
            / row.median_seconds

            if row.median_seconds
            > EPSILON

            else float(
                "inf"
            )
        )

        same = (
            row.regimen
            == exact.regimen
        )

        print(
            f"{row.strategy:<12}"
            f"{row.median_seconds * 1000:>12.2f}"
            f"{row.p25_seconds * 1000:>11.2f}"
            f"{row.p75_seconds * 1000:>11.2f}"
            f"{speedup:>9.2f}x"
            f"{row.objective:>12.6f}"
            f"{gap:>+11.6f}"
            f"{str(same):>8}"
            f"{row.evaluate_calls:>11}"
            f"{row.unique_evaluations:>11}"
            f"{row.evaluation_cache_hits:>10}"
            f"{row.duwhal_calls:>11}"
            f"{row.unique_duwhal_queries:>10}"
            f"{row.duwhal_cache_hits:>8}"
        )

    print()

    print(
        "Exact optimum:"
    )

    print(
        " ",
        format_regimen(
            model,
            exact.regimen,
        ),
    )

    print(
        f"  J = "
        f"{exact.objective:.6f}"
    )

    print()

    print(
        "Definitions:"
    )

    print(
        "  eval req : requests to evaluate a regimen"
    )

    print(
        "  unique M : actual solver evaluations after memoization"
    )

    print(
        "  eval hit : evaluation-cache hits"
    )

    print(
        "  D queries: Duwhal candidate-generation requests"
    )

    print(
        "  D unique : actual Duwhal queries after memoization"
    )

    print(
        "  D hit    : Duwhal-cache hits"
    )


def print_benchmark_profile(
    stats: list[
        BenchmarkStats
    ],
) -> None:
    print()

    print(
        "=" * 100
    )

    print(
        "MEDIAN INTERNAL PROFILE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"{'strategy':<12}"
        f"{'Duwhal ms':>14}"
        f"{'solver ms':>14}"
        f"{'neigh ms':>14}"
        f"{'frontier ms':>16}"
        f"{'other ms':>14}"
    )

    print(
        "-" * 84
    )

    for row in stats:
        profile = (
            row.profile_median
        )

        print(
            f"{row.strategy:<12}"
            f"{profile['duwhal'] * 1000:>14.2f}"
            f"{profile['solver'] * 1000:>14.2f}"
            f"{profile['neighborhood'] * 1000:>14.2f}"
            f"{profile['frontier'] * 1000:>16.2f}"
            f"{profile['other'] * 1000:>14.2f}"
        )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Duwhal-guided pharmacology regimen search."
        )
    )

    parser.add_argument(
        "--pathology",
        default="1",
    )

    parser.add_argument(
        "--regimen",
        default="",
        help=(
            "Initial comma-separated presentation IDs."
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "greedy",
            "beam",
        ],
        default="greedy",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--beam-widths",
        default="",
        help=(
            "Comma-separated beam widths for benchmark, "
            "e.g. 1,2,3,5,8."
        ),
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--candidate-mode",
        choices=[
            "direct",
            "all",
        ],
        default="direct",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lambda-complexity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-interaction",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-regimen-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-interaction-severity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--compare-exact",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
    )

    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print internal search timing decomposition."
        ),
    )

    parser.add_argument(
        "--trace-frontier",
        action="store_true",
        help=(
            "Record and print the complete beam frontier."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found.\n"
            "Run first:\n"
            "    python3 ingest.py"
        )

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    try:
        pathology_id = (
            resolve_pathology(
                con,
                args.pathology,
            )
        )

        model = ClinicalModel(
            con,
            pathology_id,
        )

        solver = RegimenSolver(
            model,

            lambda_complexity=(
                args.lambda_complexity
            ),

            lambda_interaction=(
                args.lambda_interaction
            ),

            max_regimen_size=(
                args.max_regimen_size
            ),

            max_interaction_severity=(
                args
                .max_interaction_severity
            ),
        )

        initial_regimen = (
            parse_regimen(
                args.regimen
            )
        )

        interactions = (
            build_duwhal_interactions(
                con
            )
        )

        therapeutic_support = (
            load_therapeutic_support(
                con
            )
        )

        print()

        print(
            f"Pathology      : "
            f"{model.pathology_name}"
        )

        print(
            f"|M|            : "
            f"{len(model.presentation_by_id)}"
        )

        print(
            f"|2^M|          : "
            f"{2 ** len(model.presentation_by_id)}"
        )

        print(
            f"Duwhal depth   : "
            f"{args.depth}"
        )

        print(
            f"Candidate mode : "
            f"{args.candidate_mode}"
        )

        # ====================================================
        # BENCHMARK MODE
        # ====================================================

        if args.benchmark:
            if args.beam_widths.strip():
                beam_widths = (
                    parse_int_list(
                        args.beam_widths
                    )
                )

            else:
                beam_widths = (
                    args.beam_width,
                )

            print(
                f"Beam widths    : "
                f"{beam_widths}"
            )

            print(
                f"Benchmark runs : "
                f"{args.benchmark_runs}"
            )

            print(
                f"Warmup rounds  : "
                f"{args.benchmark_warmup}"
            )

            run_benchmark(
                solver=solver,

                model=model,

                interactions=(
                    interactions
                ),

                therapeutic_support=(
                    therapeutic_support
                ),

                initial_regimen=(
                    initial_regimen
                ),

                depth=(
                    args.depth
                ),

                top=(
                    args.top
                ),

                candidate_mode=(
                    args.candidate_mode
                ),

                max_iterations=(
                    args.max_iterations
                ),

                beam_widths=(
                    beam_widths
                ),

                runs=(
                    args.benchmark_runs
                ),

                warmup=(
                    args.benchmark_warmup
                ),
            )

            return

        # ====================================================
        # NORMAL SEARCH MODE
        # ====================================================

        result, search_seconds = (
            time_call(
                lambda: run_search(
                    strategy=(
                        args.strategy
                    ),

                    solver=solver,

                    interactions=(
                        interactions
                    ),

                    therapeutic_support=(
                        therapeutic_support
                    ),

                    initial_regimen=(
                        initial_regimen
                    ),

                    depth=(
                        args.depth
                    ),

                    top=(
                        args.top
                    ),

                    candidate_mode=(
                        args.candidate_mode
                    ),

                    max_iterations=(
                        args.max_iterations
                    ),

                    beam_width=(
                        args.beam_width
                    ),

                    verbose=(
                        args.verbose
                    ),

                    trace_frontier=(
                        args.trace_frontier
                    ),
                )
            )
        )

        print_search_result(
            model,
            result,
        )

        print()

        print(
            f"Search time             : "
            f"{search_seconds * 1000:.3f} ms"
        )

        if args.profile:
            print_profile(
                result
            )

        if (
            args.trace_frontier
            and result.frontier_trace
            is not None
        ):
            print_frontier_trace(
                model,
                result.frontier_trace,
            )

        if args.compare_exact:
            exact, exact_seconds = (
                time_call(
                    lambda: exact_global_optimum(
                        solver,
                        model,
                    )
                )
            )

            print_exact_comparison(
                model,
                result.final_evaluation,
                exact,
            )

            print()

            print(
                f"Exact oracle time       : "
                f"{exact_seconds * 1000:.3f} ms"
            )

    finally:
        con.close()


if __name__ == "__main__":
    main()