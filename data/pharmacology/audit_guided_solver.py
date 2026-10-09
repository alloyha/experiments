from __future__ import annotations

"""
Fine-grained audit of Duwhal-guided regimen search.

Focus:

    where does generate_candidates() spend time?

The search semantics are unchanged.

Timing hierarchy
================

total search
    solver
    duwhal_total
        cache lookup
        generate_candidates
            graph init
            load interactions
            build topology
            prepare seeds
            rank nodes
            native -> pandas
            filter presentation nodes
            parse IDs
            sort / limit
            cleanup
        frame -> records
        candidate coverage
        candidate adaptation
        current-regimen filtering
    neighborhood
    frontier
    other
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
    generate_candidates_audit,
    load_therapeutic_support,
)

from solver import (
    ClinicalModel,
    RegimenSolver,
    resolve_pathology,
)


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "patologias.duckdb"

EPSILON = 1e-12


# ============================================================
# DATA STRUCTURES
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
# PROFILER
# ============================================================


PROFILE_COMPONENTS = (
    "solver",

    "duwhal_total",
    "duwhal_cache_lookup",
    "duwhal_generate",
    "duwhal_frame_to_records",
    "duwhal_coverage",
    "duwhal_adaptation",
    "duwhal_current_filter",

    "generate_graph_init",
    "generate_load_interactions",
    "generate_build_topology",
    "generate_prepare_seeds",
    "generate_rank_nodes",
    "generate_native_to_pandas",
    "generate_filter_presentations",
    "generate_parse_presentations",
    "generate_sort_limit",
    "generate_cleanup",

    "neighborhood",
    "frontier",
)


@dataclass
class SearchProfiler:
    times: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    started_at: float | None = None
    finished_at: float | None = None

    def start_search(
        self,
    ) -> None:
        self.started_at = (
            time.perf_counter()
        )

    def stop_search(
        self,
    ) -> None:
        self.finished_at = (
            time.perf_counter()
        )

    def add_time(
        self,
        name: str,
        seconds: float,
    ) -> None:
        self.times[
            name
        ] += seconds

    def increment(
        self,
        name: str,
        amount: int = 1,
    ) -> None:
        self.counts[
            name
        ] += amount

    @property
    def total_seconds(
        self,
    ) -> float:
        if (
            self.started_at is None
            or self.finished_at is None
        ):
            return 0.0

        return (
            self.finished_at
            - self.started_at
        )

    @property
    def other_seconds(
        self,
    ) -> float:
        top_level = (
            self.times[
                "solver"
            ]
            + self.times[
                "duwhal_total"
            ]
            + self.times[
                "neighborhood"
            ]
            + self.times[
                "frontier"
            ]
        )

        return max(
            0.0,
            self.total_seconds
            - top_level,
        )

    def snapshot(
        self,
    ) -> dict[
        str,
        float,
    ]:
        result = {
            name: self.times[
                name
            ]
            for name
            in PROFILE_COMPONENTS
        }

        result[
            "other"
        ] = self.other_seconds

        result[
            "total"
        ] = self.total_seconds

        return result


# ============================================================
# SEARCH RESULTS
# ============================================================


@dataclass(frozen=True)
class SearchResult:
    strategy: str

    initial_evaluation: object
    final_evaluation: object

    steps: tuple[
        SearchStep,
        ...
    ]

    converged: bool
    iterations: int

    discovered_regimens: int

    evaluate_calls: int
    unique_evaluations: int
    evaluation_cache_hits: int

    duwhal_calls: int
    unique_duwhal_queries: int
    duwhal_cache_hits: int

    profile: dict[
        str,
        float,
    ]

    profile_counts: dict[
        str,
        int,
    ]


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

    profile: dict[
        str,
        float,
    ]

    profile_counts: dict[
        str,
        int,
    ]


@dataclass(frozen=True)
class BenchmarkStats:
    strategy: str

    samples: tuple[
        BenchmarkSample,
        ...
    ]

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

    profile_median: dict[
        str,
        float,
    ]

    count_median: dict[
        str,
        int,
    ]


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
        for part
        in value.split(",")
        if part.strip()
    )


def parse_int_list(
    value: str,
) -> tuple[int, ...]:
    values = tuple(
        int(part.strip())
        for part
        in value.split(",")
        if part.strip()
    )

    if not values:
        raise ValueError(
            "Integer list cannot be empty."
        )

    if any(
        value < 1
        for value
        in values
    ):
        raise ValueError(
            "All values must be >= 1."
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

    def __init__(
        self,
        solver: RegimenSolver,
        profiler: SearchProfiler,
    ):
        self.solver = solver
        self.profiler = profiler

        self.cache: dict[
            tuple[int, ...],
            object,
        ] = {}

        self.calls = 0
        self.hits = 0
        self.misses = 0

    def evaluate(
        self,
        regimen,
    ):
        self.calls += 1

        key = normalize_regimen(
            regimen
        )

        if key in self.cache:
            self.hits += 1

            self.profiler.increment(
                "evaluation_cache_hit"
            )

            return self.cache[
                key
            ]

        self.misses += 1

        self.profiler.increment(
            "evaluation_cache_miss"
        )

        started = (
            time.perf_counter()
        )

        result = (
            self.solver.evaluate(
                key
            )
        )

        self.profiler.add_time(
            "solver",
            time.perf_counter()
            - started,
        )

        self.cache[
            key
        ] = result

        return result

    @property
    def unique_evaluations(
        self,
    ):
        return self.misses


# ============================================================
# DUWHAL CACHE
# ============================================================


class DuwhalCandidateCache:

    def __init__(
        self,
        *,
        interactions,
        therapeutic_support,
        profiler: SearchProfiler,
    ):
        self.interactions = interactions

        self.therapeutic_support = (
            therapeutic_support
        )

        self.profiler = profiler

        self.cache = {}

        self.calls = 0
        self.hits = 0
        self.misses = 0

    def get(
        self,
        *,
        evaluation,
        depth,
        top,
        candidate_mode,
    ):
        total_started = (
            time.perf_counter()
        )

        self.calls += 1

        lookup_started = (
            time.perf_counter()
        )

        unresolved = frozenset(
            evaluation
            .symptom_space
            .unresolved_support
        )

        key = (
            unresolved,
            depth,
            top,
            candidate_mode,
        )

        cached = (
            self.cache.get(
                key
            )
        )

        self.profiler.add_time(
            "duwhal_cache_lookup",
            time.perf_counter()
            - lookup_started,
        )

        if not unresolved:
            self.profiler.add_time(
                "duwhal_total",
                time.perf_counter()
                - total_started,
            )

            return tuple()

        if cached is not None:
            self.hits += 1

            self.profiler.increment(
                "duwhal_cache_hit"
            )

            all_candidates = cached

        else:
            self.misses += 1

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

            self.cache[
                key
            ] = all_candidates

        filter_started = (
            time.perf_counter()
        )

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

        self.profiler.add_time(
            "duwhal_current_filter",
            time.perf_counter()
            - filter_started,
        )

        self.profiler.increment(
            "duwhal_returned_candidates",
            len(result),
        )

        self.profiler.add_time(
            "duwhal_total",
            time.perf_counter()
            - total_started,
        )

        return result

    def _generate(
        self,
        *,
        unresolved,
        depth,
        top,
        candidate_mode,
    ):
        frame, audit = (
            generate_candidates_audit(
                self.interactions,
                unresolved,
                top=top,
                depth=depth,
            )
        )

        self.profiler.add_time(
            "duwhal_generate",
            audit.total_seconds,
        )

        for (
            component,
            seconds,
        ) in audit.timings.items():
            if component == "total":
                continue

            self.profiler.add_time(
                f"generate_{component}",
                seconds,
            )

        for (
            name,
            value,
        ) in audit.counts.items():
            self.profiler.increment(
                f"generate_{name}",
                value,
            )

        self.profiler.increment(
            "duwhal_generate_calls"
        )

        self.profiler.increment(
            "duwhal_frame_rows",
            len(frame),
        )

        started = (
            time.perf_counter()
        )

        records = frame.to_dict(
            orient="records"
        )

        self.profiler.add_time(
            "duwhal_frame_to_records",
            time.perf_counter()
            - started,
        )

        self.profiler.increment(
            "duwhal_record_rows",
            len(records),
        )

        intermediate = []

        for row in records:
            presentation_id = int(
                row[
                    "presentation_id"
                ]
            )

            started = (
                time.perf_counter()
            )

            covered, coverage = (
                candidate_coverage(
                    presentation_id,
                    unresolved,
                    self.therapeutic_support,
                )
            )

            self.profiler.add_time(
                "duwhal_coverage",
                time.perf_counter()
                - started,
            )

            self.profiler.increment(
                "candidate_coverage_calls"
            )

            intermediate.append(
                (
                    row,
                    presentation_id,
                    covered,
                    coverage,
                )
            )

        started = (
            time.perf_counter()
        )

        result = []

        for (
            row,
            presentation_id,
            covered,
            coverage,
        ) in intermediate:
            direct = bool(
                covered
            )

            if (
                candidate_mode
                == "direct"
                and not direct
            ):
                self.profiler.increment(
                    "duwhal_mode_filtered"
                )

                continue

            result.append(
                GuidedCandidate(
                    presentation_id=(
                        presentation_id
                    ),

                    score=float(
                        row[
                            "score"
                        ]
                    ),

                    hops=int(
                        row[
                            "hops"
                        ]
                    ),

                    reason=str(
                        row[
                            "reason"
                        ]
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

        self.profiler.add_time(
            "duwhal_adaptation",
            time.perf_counter()
            - started,
        )

        self.profiler.increment(
            "duwhal_adapted_candidates",
            len(result),
        )

        return tuple(
            result
        )

    @property
    def unique_queries(
        self,
    ):
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
):
    profiler = (
        SearchProfiler()
    )

    return SearchContext(
        evaluator=EvaluationCache(
            solver,
            profiler,
        ),

        candidate_cache=(
            DuwhalCandidateCache(
                interactions=interactions,

                therapeutic_support=(
                    therapeutic_support
                ),

                profiler=profiler,
            )
        ),

        profiler=profiler,
    )


# ============================================================
# NEIGHBORHOOD
# ============================================================


def build_guided_moves(
    *,
    evaluator,
    current_evaluation,
    candidates,
    profiler,
):
    started = (
        time.perf_counter()
    )

    solver_before = (
        profiler.times[
            "solver"
        ]
    )

    current = frozenset(
        current_evaluation.regimen
    )

    candidate_ids = frozenset(
        candidate.presentation_id

        for candidate
        in candidates
    )

    moves = {}

    for added_id in sorted(
        candidate_ids
        - current
    ):
        destination = normalize_regimen(
            current
            | {
                added_id
            }
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

            to_regimen=(
                destination
            ),

            added=(
                added_id,
            ),

            removed=tuple(),

            evaluation=evaluation,
        )

    for removed_id in sorted(
        current
    ):
        destination = normalize_regimen(
            current
            - {
                removed_id
            }
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

            to_regimen=(
                destination
            ),

            added=tuple(),

            removed=(
                removed_id,
            ),

            evaluation=evaluation,
        )

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
            candidate_ids
            - current
        ):
            destination = normalize_regimen(
                reduced
                | {
                    added_id
                }
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

    solver_delta = (
        profiler.times[
            "solver"
        ]
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
        "neighbor_destinations",
        len(moves),
    )

    profiler.increment(
        "neighborhood_builds"
    )

    return tuple(
        moves.values()
    )


# ============================================================
# RESULT BUILDER
# ============================================================


def make_search_result(
    *,
    strategy,
    initial,
    final,
    steps,
    converged,
    iterations,
    discovered,
    context,
):
    return SearchResult(
        strategy=strategy,

        initial_evaluation=initial,
        final_evaluation=final,

        steps=tuple(
            steps
        ),

        converged=converged,

        iterations=iterations,

        discovered_regimens=(
            len(discovered)
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

        profile_counts=dict(
            context
            .profiler
            .counts
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
    depth,
    top,
    candidate_mode,
    max_iterations,
):
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

    current = initial

    steps = []

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

            candidates=(
                candidates
            ),

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

                discovered=(
                    discovered
                ),

                context=context,
            )

        move = min(
            improving,

            key=lambda item: (
                evaluation_key(
                    item.evaluation
                )
            ),
        )

        previous = current

        current = (
            move.evaluation
        )

        steps.append(
            SearchStep(
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
# BEAM SEARCH
# ============================================================


def beam_search(
    *,
    solver,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth,
    top,
    candidate_mode,
    max_iterations,
    beam_width,
):
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

    beam = [
        BeamNode(
            evaluation=initial,

            steps=tuple(),
        )
    ]

    best = beam[
        0
    ]

    discovered = {
        normalize_regimen(
            initial.regimen
        )
    }

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        generated = {}

        for node in beam:
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

                candidates=(
                    candidates
                ),

                profiler=(
                    context.profiler
                ),
            )

            discovered.update(
                move.to_regimen

                for move
                in moves
            )

            started = (
                time.perf_counter()
            )

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

            context.profiler.add_time(
                "frontier",

                time.perf_counter()
                - started,
            )

        if not generated:
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
                    iteration
                    - 1
                ),

                discovered=(
                    discovered
                ),

                context=context,
            )

        started = (
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

        if (
            evaluation_key(
                beam[0].evaluation
            )
            <
            evaluation_key(
                best.evaluation
            )
        ):
            best = beam[
                0
            ]

        context.profiler.add_time(
            "frontier",

            time.perf_counter()
            - started,
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

        discovered=(
            discovered
        ),

        context=context,
    )


# ============================================================
# EXACT
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
        len(ids)
        + 1
    ):
        yield from combinations(
            ids,
            size,
        )


def exact_global_optimum(
    solver,
    model,
):
    best = None

    evaluated = 0
    feasible = 0

    for regimen in powerset(
        model
        .presentation_by_id
        .keys()
    ):
        evaluated += 1

        evaluation = (
            solver.evaluate(
                regimen
            )
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
# DISPATCH
# ============================================================


def run_search(
    *,
    strategy,
    solver,
    interactions,
    therapeutic_support,
    initial_regimen,
    depth,
    top,
    candidate_mode,
    max_iterations,
    beam_width,
):
    common = dict(
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

    if strategy == "greedy":
        return greedy_search(
            **common
        )

    return beam_search(
        **common,

        beam_width=(
            beam_width
        ),
    )


# ============================================================
# TIMING / BENCHMARK
# ============================================================


def time_call(
    function: Callable,
):
    started = (
        time.perf_counter()
    )

    result = function()

    return (
        result,
        time.perf_counter()
        - started,
    )


def percentile_quartiles(
    values,
):
    if len(values) == 1:
        return (
            values[0],
            values[0],
        )

    q = statistics.quantiles(
        values,
        n=4,

        method="inclusive",
    )

    return (
        q[0],
        q[2],
    )


def benchmark_sample_guided(
    result,
    seconds,
):
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
            result
            .evaluation_cache_hits
        ),

        duwhal_calls=(
            result.duwhal_calls
        ),

        unique_duwhal_queries=(
            result
            .unique_duwhal_queries
        ),

        duwhal_cache_hits=(
            result.duwhal_cache_hits
        ),

        profile=(
            result.profile
        ),

        profile_counts=(
            result.profile_counts
        ),
    )


def benchmark_sample_exact(
    result,
    seconds,
):
    return BenchmarkSample(
        seconds=seconds,

        objective=(
            result
            .evaluation
            .objective
        ),

        regimen=(
            normalize_regimen(
                result
                .evaluation
                .regimen
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

            "duwhal_total": 0.0,

            "duwhal_cache_lookup": 0.0,

            "duwhal_generate": 0.0,

            "duwhal_frame_to_records": 0.0,

            "duwhal_coverage": 0.0,

            "duwhal_adaptation": 0.0,

            "duwhal_current_filter": 0.0,

            "generate_graph_init": 0.0,

            "generate_load_interactions": 0.0,

            "generate_build_topology": 0.0,

            "generate_prepare_seeds": 0.0,

            "generate_rank_nodes": 0.0,

            "generate_native_to_pandas": 0.0,

            "generate_filter_presentations": 0.0,

            "generate_parse_presentations": 0.0,

            "generate_sort_limit": 0.0,

            "generate_cleanup": 0.0,

            "neighborhood": 0.0,

            "frontier": 0.0,

            "other": 0.0,
        },

        profile_counts={},
    )


def aggregate_benchmark(
    strategy,
    samples,
):
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

    reference = (
        samples[
            0
        ]
    )

    def med_int(
        attr,
    ):
        return int(
            statistics.median(
                getattr(
                    sample,
                    attr
                )

                for sample
                in samples
            )
        )

    profile_keys = set()

    count_keys = set()

    for sample in samples:
        profile_keys.update(
            sample.profile
        )

        count_keys.update(
            sample.profile_counts
        )

    profile_median = {
        key: statistics.median(
            sample
            .profile
            .get(
                key,
                0.0,
            )

            for sample
            in samples
        )

        for key
        in profile_keys
    }

    count_median = {
        key: int(
            statistics.median(
                sample
                .profile_counts
                .get(
                    key,
                    0,
                )

                for sample
                in samples
            )
        )

        for key
        in count_keys
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
            med_int(
                "evaluate_calls"
            )
        ),

        unique_evaluations=(
            med_int(
                "unique_evaluations"
            )
        ),

        evaluation_cache_hits=(
            med_int(
                "evaluation_cache_hits"
            )
        ),

        duwhal_calls=(
            med_int(
                "duwhal_calls"
            )
        ),

        unique_duwhal_queries=(
            med_int(
                "unique_duwhal_queries"
            )
        ),

        duwhal_cache_hits=(
            med_int(
                "duwhal_cache_hits"
            )
        ),

        profile_median=(
            profile_median
        ),

        count_median=(
            count_median
        ),
    )


# ============================================================
# REPORTS
# ============================================================


def print_profile(
    result,
):
    p = result.profile

    total = p[
        "total"
    ]

    print()

    print(
        "=" * 100
    )

    print(
        "TOP-LEVEL PROFILE"
    )

    print(
        "=" * 100
    )

    for name in (
        "duwhal_total",

        "solver",

        "neighborhood",

        "frontier",

        "other",
    ):
        value = p.get(
            name,
            0.0,
        )

        percent = (
            value
            / total
            * 100.0

            if total > EPSILON

            else 0.0
        )

        print(
            f"{name:<32}"
            f"{value * 1000:>12.3f} ms"
            f"{percent:>10.2f}%"
        )

    print()

    print(
        "=" * 100
    )

    print(
        "DUWHAL INTERNAL PROFILE"
    )

    print(
        "=" * 100
    )

    duwhal_total = p.get(
        "duwhal_total",
        0.0,
    )

    names = (
        "duwhal_cache_lookup",

        "duwhal_generate",

        "duwhal_frame_to_records",

        "duwhal_coverage",

        "duwhal_adaptation",

        "duwhal_current_filter",
    )

    for name in names:
        value = p.get(
            name,
            0.0,
        )

        percent = (
            value
            / duwhal_total
            * 100.0

            if duwhal_total > EPSILON

            else 0.0
        )

        print(
            f"{name:<32}"
            f"{value * 1000:>12.3f} ms"
            f"{percent:>10.2f}%"
        )


def print_generate_candidates_profile(
    result,
):
    p = result.profile

    total = p.get(
        "duwhal_generate",
        0.0,
    )

    components = (
        "generate_graph_init",

        "generate_load_interactions",

        "generate_build_topology",

        "generate_prepare_seeds",

        "generate_rank_nodes",

        "generate_native_to_pandas",

        "generate_filter_presentations",

        "generate_parse_presentations",

        "generate_sort_limit",

        "generate_cleanup",
    )

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATE_CANDIDATES INTERNAL PROFILE"
    )

    print(
        "=" * 100
    )

    accounted = 0.0

    for component in components:
        seconds = p.get(
            component,
            0.0,
        )

        accounted += seconds

        percent = (
            seconds
            / total
            * 100.0

            if total > EPSILON

            else 0.0
        )

        name = component.removeprefix(
            "generate_"
        )

        print(
            f"{name:<32}"
            f"{seconds * 1000:>12.3f} ms"
            f"{percent:>10.2f}%"
        )

    unaccounted = max(
        0.0,

        total
        - accounted,
    )

    percent = (
        unaccounted
        / total
        * 100.0

        if total > EPSILON

        else 0.0
    )

    print(
        "-" * 56
    )

    print(
        f"{'unaccounted':<32}"
        f"{unaccounted * 1000:>12.3f} ms"
        f"{percent:>10.2f}%"
    )

    print(
        f"{'total':<32}"
        f"{total * 1000:>12.3f} ms"
        f"{100.0:>10.2f}%"
    )

    print()

    print(
        "COUNTERS"
    )

    print(
        "-" * 56
    )

    for name in sorted(
        result.profile_counts
    ):
        if not name.startswith(
            "generate_"
        ):
            continue

        print(
            f"{name.removeprefix('generate_'):<36}"
            f"{result.profile_counts[name]:>12}"
        )


def print_benchmark(
    stats,
):
    exact = next(
        row

        for row
        in stats

        if row.strategy
        == "exact"
    )

    print()

    print(
        "=" * 110
    )

    print(
        "TIME + SEARCH SPACE"
    )

    print(
        "=" * 110
    )

    print(
        f"{'strategy':<12}"
        f"{'median ms':>12}"
        f"{'p25':>10}"
        f"{'p75':>10}"
        f"{'speedup':>10}"
        f"{'J':>12}"
        f"{'gap':>12}"
        f"{'unique M':>11}"
        f"{'D unique':>11}"
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

        print(
            f"{row.strategy:<12}"
            f"{row.median_seconds * 1000:>12.2f}"
            f"{row.p25_seconds * 1000:>10.2f}"
            f"{row.p75_seconds * 1000:>10.2f}"
            f"{speedup:>9.2f}x"
            f"{row.objective:>12.6f}"
            f"{gap:>+12.6f}"
            f"{row.unique_evaluations:>11}"
            f"{row.unique_duwhal_queries:>11}"
        )


def print_generate_benchmark_profile(
    stats,
):
    print()

    print(
        "=" * 150
    )

    print(
        "MEDIAN GENERATE_CANDIDATES PROFILE"
    )

    print(
        "=" * 150
    )

    print(
        f"{'strategy':<12}"
        f"{'graph':>10}"
        f"{'load':>10}"
        f"{'topology':>12}"
        f"{'rank':>12}"
        f"{'pandas':>10}"
        f"{'filter':>10}"
        f"{'parse':>10}"
        f"{'sort':>10}"
    )

    for row in stats:
        p = (
            row.profile_median
        )

        print(
            f"{row.strategy:<12}"

            f"{p.get('generate_graph_init', 0) * 1000:>10.2f}"

            f"{p.get('generate_load_interactions', 0) * 1000:>10.2f}"

            f"{p.get('generate_build_topology', 0) * 1000:>12.2f}"

            f"{p.get('generate_rank_nodes', 0) * 1000:>12.2f}"

            f"{p.get('generate_native_to_pandas', 0) * 1000:>10.2f}"

            f"{p.get('generate_filter_presentations', 0) * 1000:>10.2f}"

            f"{p.get('generate_parse_presentations', 0) * 1000:>10.2f}"

            f"{p.get('generate_sort_limit', 0) * 1000:>10.2f}"
        )


# ============================================================
# BENCHMARK EXECUTION
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
    runners = {}

    runners[
        "greedy"
    ] = lambda: run_search(
        strategy="greedy",

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

        beam_width=1,
    )

    for width in beam_widths:
        runners[
            f"beam[{width}]"
        ] = (
            lambda width=width:
            run_search(
                strategy="beam",

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

                beam_width=width,
            )
        )

    runners[
        "exact"
    ] = lambda: (
        exact_global_optimum(
            solver,
            model,
        )
    )

    names = list(
        runners
    )

    for round_index in range(
        warmup
    ):
        offset = (
            round_index
            % len(names)
        )

        order = (
            names[offset:]
            + names[:offset]
        )

        for name in order:
            runners[
                name
            ]()

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

        order = (
            names[offset:]
            + names[:offset]
        )

        for name in order:
            result, seconds = (
                time_call(
                    runners[
                        name
                    ]
                )
            )

            if name == "exact":
                sample = (
                    benchmark_sample_exact(
                        result,
                        seconds,
                    )
                )

            else:
                sample = (
                    benchmark_sample_guided(
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
            name,
            samples[
                name
            ],
        )

        for name
        in names
    ]

    print_benchmark(
        stats
    )

    print_generate_benchmark_profile(
        stats
    )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pathology",
        default="1",
    )

    parser.add_argument(
        "--regimen",
        default="",
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "greedy",
            "beam",
        ],

        default="beam",
    )

    parser.add_argument(
        "--beam-width",
        type=int,

        default=2,
    )

    parser.add_argument(
        "--beam-widths",
        default="",
    )

    parser.add_argument(
        "--depth",
        type=int,

        default=2,
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

        default="all",
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
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    con = duckdb.connect(
        str(
            DB_PATH
        ),

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

        initial_regimen = (
            parse_regimen(
                args.regimen
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

        if args.benchmark:
            if (
                args
                .beam_widths
                .strip()
            ):
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

        result, seconds = time_call(
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
                    args
                    .candidate_mode
                ),

                max_iterations=(
                    args
                    .max_iterations
                ),

                beam_width=(
                    args.beam_width
                ),
            )
        )

        final = (
            result
            .final_evaluation
        )

        print()

        print(
            f"Strategy       : "
            f"{result.strategy}"
        )

        print(
            f"Regimen        : "
            f"{format_regimen(model, final.regimen)}"
        )

        print(
            f"J              : "
            f"{final.objective:.6f}"
        )

        print(
            f"Search time    : "
            f"{seconds * 1000:.3f} ms"
        )

        if args.profile:
            print_profile(
                result
            )

            print_generate_candidates_profile(
                result
            )

        if args.compare_exact:
            exact, exact_seconds = (
                time_call(
                    lambda:
                    exact_global_optimum(
                        solver,
                        model,
                    )
                )
            )

            print()

            print(
                "EXACT"
            )

            print(
                "-" * 60
            )

            print(
                f"Regimen        : "
                f"{format_regimen(model, exact.evaluation.regimen)}"
            )

            print(
                f"J              : "
                f"{exact.evaluation.objective:.6f}"
            )

            print(
                f"Same regimen   : "
                f"{normalize_regimen(final.regimen) == normalize_regimen(exact.evaluation.regimen)}"
            )

            print(
                f"Gap            : "
                f"{final.objective - exact.evaluation.objective:+.6f}"
            )

            print(
                f"Exact time     : "
                f"{exact_seconds * 1000:.3f} ms"
            )

    finally:
        con.close()


if __name__ == "__main__":
    main()
