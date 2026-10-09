from __future__ import annotations

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
    PersistentDuwhalCandidateEngine,
    build_duwhal_interactions,
    candidate_coverage,
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
# SEARCH DATA
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

    def start(self):
        self.started_at = (
            time.perf_counter()
        )

    def stop(self):
        self.finished_at = (
            time.perf_counter()
        )

    def add_time(
        self,
        name: str,
        seconds: float,
    ):
        self.times[name] += seconds

    def increment(
        self,
        name: str,
        value: int = 1,
    ):
        self.counts[name] += value

    @property
    def total_seconds(self):
        if (
            self.started_at is None
            or self.finished_at is None
        ):
            return 0.0

        return (
            self.finished_at
            - self.started_at
        )

    def snapshot(self):
        known = (
            "duwhal_setup_graph_init",
            "duwhal_setup_load_interactions",
            "duwhal_setup_build_topology",
            "duwhal_setup_total",

            "duwhal_query_total",
            "duwhal_prepare_seeds",
            "duwhal_rank_nodes",
            "duwhal_native_to_pandas",
            "duwhal_filter_presentations",
            "duwhal_parse_presentations",
            "duwhal_sort_limit",

            "duwhal_adapter",

            "solver",
            "neighborhood",
            "frontier",
        )

        result = {
            name: self.times[name]
            for name in known
        }

        accounted = (
            result[
                "duwhal_setup_total"
            ]
            + result[
                "duwhal_query_total"
            ]
            + result[
                "duwhal_adapter"
            ]
            + result[
                "solver"
            ]
            + result[
                "neighborhood"
            ]
            + result[
                "frontier"
            ]
        )

        result[
            "total"
        ] = self.total_seconds

        result[
            "other"
        ] = max(
            0.0,
            self.total_seconds
            - accounted,
        )

        return result


# ============================================================
# RESULTS
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

    profile: dict[str, float]
    profile_counts: dict[str, int]


@dataclass(frozen=True)
class BenchmarkSample:
    seconds: float

    objective: float
    regimen: tuple[int, ...]

    unique_evaluations: int
    unique_duwhal_queries: int

    profile: dict[str, float]


@dataclass(frozen=True)
class BenchmarkStats:
    strategy: str

    median_seconds: float
    p25_seconds: float
    p75_seconds: float

    objective: float
    regimen: tuple[int, ...]

    unique_evaluations: int
    unique_duwhal_queries: int

    profile_median: dict[str, float]


# ============================================================
# UTILITIES
# ============================================================


def normalize_regimen(
    regimen: Iterable[int],
):
    return tuple(
        sorted(
            set(regimen)
        )
    )


def parse_regimen(
    value: str | None,
):
    if not value:
        return tuple()

    return normalize_regimen(
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    )


def parse_int_list(
    value: str,
):
    return tuple(
        dict.fromkeys(
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    )


def format_regimen(
    model,
    regimen,
):
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
):
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
        solver,
        profiler,
    ):
        self.solver = solver
        self.profiler = profiler

        self.cache = {}

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
            return self.cache[key]

        self.misses += 1

        started = (
            time.perf_counter()
        )

        result = self.solver.evaluate(
            key
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


# ============================================================
# PERSISTENT DUWHAL ADAPTER
# ============================================================


class PersistentCandidateCache:

    def __init__(
        self,
        *,
        engine,
        therapeutic_support,
        profiler,
    ):
        self.engine = engine

        self.therapeutic_support = (
            therapeutic_support
        )

        self.profiler = profiler

        self.cache = {}

        self.calls = 0
        self.hits = 0
        self.misses = 0

        self._import_engine_setup()

    def _import_engine_setup(
        self,
    ):
        audit = (
            self.engine.audit
        )

        for name, seconds in (
            audit
            .setup_timings
            .items()
        ):
            self.profiler.add_time(
                f"duwhal_setup_{name}",
                seconds,
            )

        self.profiler.add_time(
            "duwhal_setup_total",
            audit.setup_total,
        )

    def _snapshot_engine_queries(
        self,
    ):
        return dict(
            self.engine
            .audit
            .query_timings
        )

    def _record_engine_query_delta(
        self,
        before,
    ):
        after = (
            self.engine
            .audit
            .query_timings
        )

        mapping = {
            "query_total":
                "duwhal_query_total",

            "prepare_seeds":
                "duwhal_prepare_seeds",

            "rank_nodes":
                "duwhal_rank_nodes",

            "native_to_pandas":
                "duwhal_native_to_pandas",

            "filter_presentations":
                "duwhal_filter_presentations",

            "parse_presentations":
                "duwhal_parse_presentations",

            "sort_limit":
                "duwhal_sort_limit",
        }

        for source, target in (
            mapping.items()
        ):
            delta = (
                after.get(
                    source,
                    0.0,
                )
                - before.get(
                    source,
                    0.0,
                )
            )

            self.profiler.add_time(
                target,
                delta,
            )

    def get(
        self,
        *,
        evaluation,
        depth,
        top,
        candidate_mode,
    ):
        self.calls += 1

        unresolved = frozenset(
            evaluation
            .symptom_space
            .unresolved_support
        )

        if not unresolved:
            return tuple()

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

        if cached is not None:
            self.hits += 1
            all_candidates = cached

        else:
            self.misses += 1

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

        current = frozenset(
            evaluation.regimen
        )

        return tuple(
            candidate
            for candidate in all_candidates
            if (
                candidate.presentation_id
                not in current
            )
        )

    def _generate(
        self,
        *,
        unresolved,
        depth,
        top,
        candidate_mode,
    ):
        before = (
            self._snapshot_engine_queries()
        )

        frame = (
            self.engine.generate(
                unresolved,
                top=top,
                depth=depth,
            )
        )

        self._record_engine_query_delta(
            before
        )

        adapter_started = (
            time.perf_counter()
        )

        result = []

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
            "duwhal_adapter",
            time.perf_counter()
            - adapter_started,
        )

        return tuple(
            result
        )


# ============================================================
# SEARCH CONTEXT
# ============================================================


@dataclass
class SearchContext:
    profiler: SearchProfiler

    engine: PersistentDuwhalCandidateEngine

    evaluator: EvaluationCache

    candidate_cache: PersistentCandidateCache

    def close(
        self,
    ):
        self.engine.close()


def make_search_context(
    *,
    solver,
    interactions,
    therapeutic_support,
):
    profiler = (
        SearchProfiler()
    )

    profiler.start()

    engine = (
        PersistentDuwhalCandidateEngine(
            interactions
        )
    )

    evaluator = (
        EvaluationCache(
            solver,
            profiler,
        )
    )

    candidate_cache = (
        PersistentCandidateCache(
            engine=engine,

            therapeutic_support=(
                therapeutic_support
            ),

            profiler=profiler,
        )
    )

    return SearchContext(
        profiler=profiler,
        engine=engine,
        evaluator=evaluator,
        candidate_cache=(
            candidate_cache
        ),
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
        for candidate in candidates
    )

    moves = {}

    # ADD
    for added_id in sorted(
        candidate_ids
        - current
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

            to_regimen=(
                destination
            ),

            added=(
                added_id,
            ),

            removed=tuple(),

            evaluation=evaluation,
        )

    # REMOVE
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

            to_regimen=(
                destination
            ),

            added=tuple(),

            removed=(
                removed_id,
            ),

            evaluation=evaluation,
        )

    # SWITCH
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

                evaluation=evaluation,
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

    return tuple(
        moves.values()
    )


# ============================================================
# RESULT BUILDER
# ============================================================


def finish_search(
    *,
    context,
    strategy,
    initial,
    final,
    steps,
    converged,
    iterations,
    discovered,
):
    context.profiler.stop()

    profile = (
        context
        .profiler
        .snapshot()
    )

    result = SearchResult(
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
            context.evaluator.calls
        ),

        unique_evaluations=(
            context.evaluator.misses
        ),

        evaluation_cache_hits=(
            context.evaluator.hits
        ),

        duwhal_calls=(
            context
            .candidate_cache
            .calls
        ),

        unique_duwhal_queries=(
            context
            .candidate_cache
            .misses
        ),

        duwhal_cache_hits=(
            context
            .candidate_cache
            .hits
        ),

        profile=profile,

        profile_counts=dict(
            context.profiler.counts
        ),
    )

    context.close()

    return result


# ============================================================
# GREEDY
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

            candidates=candidates,

            profiler=(
                context.profiler
            ),
        )

        discovered.update(
            move.to_regimen
            for move in moves
        )

        improving = [
            move

            for move in moves

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
            return finish_search(
                context=context,

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
                    for candidate in candidates
                ),
            )
        )

    return finish_search(
        context=context,

        strategy="greedy",

        initial=initial,
        final=current,

        steps=steps,

        converged=False,

        iterations=max_iterations,

        discovered=discovered,
    )


# ============================================================
# BEAM
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

                candidates=candidates,

                profiler=(
                    context.profiler
                ),
            )

            discovered.update(
                move.to_regimen
                for move in moves
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
                        for item in candidates
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
            return finish_search(
                context=context,

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

                discovered=(
                    discovered
                ),
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
            best = beam[0]

        context.profiler.add_time(
            "frontier",
            time.perf_counter()
            - started,
        )

    return finish_search(
        context=context,

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
    )


# ============================================================
# EXACT
# ============================================================


def powerset(
    ids,
):
    ids = tuple(
        sorted(ids)
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
        evaluated_regimens=evaluated,
        feasible_regimens=feasible,
    )


# ============================================================
# SEARCH DISPATCH
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
    )

    if strategy == "greedy":
        return greedy_search(
            **common
        )

    return beam_search(
        **common,
        beam_width=beam_width,
    )


# ============================================================
# BENCHMARK
# ============================================================


def time_call(
    function: Callable,
):
    started = time.perf_counter()

    result = function()

    return (
        result,
        time.perf_counter()
        - started,
    )


def quartiles(
    values,
):
    if len(values) == 1:
        return (
            values[0],
            values[0],
        )

    values = statistics.quantiles(
        values,
        n=4,
        method="inclusive",
    )

    return (
        values[0],
        values[2],
    )


def guided_sample(
    result,
    seconds,
):
    return BenchmarkSample(
        seconds=seconds,

        objective=(
            result
            .final_evaluation
            .objective
        ),

        regimen=(
            normalize_regimen(
                result
                .final_evaluation
                .regimen
            )
        ),

        unique_evaluations=(
            result
            .unique_evaluations
        ),

        unique_duwhal_queries=(
            result
            .unique_duwhal_queries
        ),

        profile=(
            result.profile
        ),
    )


def exact_sample(
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

        unique_evaluations=(
            result
            .evaluated_regimens
        ),

        unique_duwhal_queries=0,

        profile={
            "total": seconds,
            "solver": seconds,
        },
    )


def aggregate(
    strategy,
    samples,
):
    times = [
        sample.seconds
        for sample in samples
    ]

    p25, p75 = quartiles(
        times
    )

    profile_keys = set()

    for sample in samples:
        profile_keys.update(
            sample.profile.keys()
        )

    profile_median = {
        key: statistics.median(
            sample.profile.get(
                key,
                0.0,
            )
            for sample in samples
        )

        for key in profile_keys
    }

    reference = samples[0]

    return BenchmarkStats(
        strategy=strategy,

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

        unique_evaluations=int(
            statistics.median(
                sample.unique_evaluations
                for sample in samples
            )
        ),

        unique_duwhal_queries=int(
            statistics.median(
                sample.unique_duwhal_queries
                for sample in samples
            )
        ),

        profile_median=(
            profile_median
        ),
    )


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

    # Warmup
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
            runners[name]()

    samples = {
        name: []
        for name in names
    }

    # Timed
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
                    runners[name]
                )
            )

            sample = (
                exact_sample(
                    result,
                    seconds,
                )

                if name == "exact"

                else guided_sample(
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
        aggregate(
            name,
            samples[name],
        )

        for name in names
    ]

    print_benchmark(
        stats
    )

    print_profile_benchmark(
        stats
    )


# ============================================================
# REPORTING
# ============================================================


def print_profile(
    result,
):
    p = result.profile

    print()

    print(
        "=" * 96
    )

    print(
        "PERSISTENT DUWHAL PROFILE"
    )

    print(
        "=" * 96
    )

    print()

    print(
        "ONE-TIME ENGINE SETUP"
    )

    print(
        "-" * 60
    )

    for name in (
        "duwhal_setup_graph_init",
        "duwhal_setup_load_interactions",
        "duwhal_setup_build_topology",
        "duwhal_setup_total",
    ):
        print(
            f"{name:<36}"
            f"{p.get(name, 0) * 1000:>12.3f} ms"
        )

    print()

    print(
        "REPEATED QUERY WORK"
    )

    print(
        "-" * 60
    )

    for name in (
        "duwhal_prepare_seeds",
        "duwhal_rank_nodes",
        "duwhal_native_to_pandas",
        "duwhal_filter_presentations",
        "duwhal_parse_presentations",
        "duwhal_sort_limit",
        "duwhal_query_total",
        "duwhal_adapter",
    ):
        print(
            f"{name:<36}"
            f"{p.get(name, 0) * 1000:>12.3f} ms"
        )

    queries = (
        result.unique_duwhal_queries
    )

    if queries:
        mean_query = (
            p.get(
                "duwhal_query_total",
                0.0,
            )
            / queries
        )

        mean_rank = (
            p.get(
                "duwhal_rank_nodes",
                0.0,
            )
            / queries
        )

        print()

        print(
            f"Unique Duwhal queries : "
            f"{queries}"
        )

        print(
            f"Mean query            : "
            f"{mean_query * 1000:.3f} ms"
        )

        print(
            f"Mean rank_nodes       : "
            f"{mean_rank * 1000:.3f} ms"
        )

    print()

    print(
        "SEARCH"
    )

    print(
        "-" * 60
    )

    for name in (
        "solver",
        "neighborhood",
        "frontier",
        "other",
        "total",
    ):
        print(
            f"{name:<36}"
            f"{p.get(name, 0) * 1000:>12.3f} ms"
        )


def print_benchmark(
    stats,
):
    exact = next(
        row
        for row in stats
        if row.strategy == "exact"
    )

    print()

    print(
        "=" * 118
    )

    print(
        "PERSISTENT-ENGINE BENCHMARK"
    )

    print(
        "=" * 118
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

    print(
        "-" * 100
    )

    for row in stats:
        speedup = (
            exact.median_seconds
            / row.median_seconds
        )

        gap = (
            row.objective
            - exact.objective
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


def print_profile_benchmark(
    stats,
):
    print()

    print(
        "=" * 132
    )

    print(
        "MEDIAN PERSISTENT DUWHAL PROFILE"
    )

    print(
        "=" * 132
    )

    print(
        f"{'strategy':<12}"
        f"{'setup':>11}"
        f"{'graph':>10}"
        f"{'load':>10}"
        f"{'topology':>11}"
        f"{'queries':>11}"
        f"{'rank':>11}"
        f"{'adapter':>11}"
        f"{'solver':>11}"
    )

    print(
        "-" * 98
    )

    for row in stats:
        p = (
            row.profile_median
        )

        print(
            f"{row.strategy:<12}"

            f"{p.get('duwhal_setup_total', 0) * 1000:>11.2f}"

            f"{p.get('duwhal_setup_graph_init', 0) * 1000:>10.2f}"

            f"{p.get('duwhal_setup_load_interactions', 0) * 1000:>10.2f}"

            f"{p.get('duwhal_setup_build_topology', 0) * 1000:>11.2f}"

            f"{p.get('duwhal_query_total', 0) * 1000:>11.2f}"

            f"{p.get('duwhal_rank_nodes', 0) * 1000:>11.2f}"

            f"{p.get('duwhal_adapter', 0) * 1000:>11.2f}"

            f"{p.get('solver', 0) * 1000:>11.2f}"
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
        "--profile",
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

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

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
                args.max_interaction_severity
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
            beam_widths = (
                parse_int_list(
                    args.beam_widths
                )

                if args.beam_widths.strip()

                else (
                    args.beam_width,
                )
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

                interactions=interactions,

                therapeutic_support=(
                    therapeutic_support
                ),

                initial_regimen=(
                    initial_regimen
                ),

                depth=args.depth,
                top=args.top,

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

                interactions=interactions,

                therapeutic_support=(
                    therapeutic_support
                ),

                initial_regimen=(
                    initial_regimen
                ),

                depth=args.depth,
                top=args.top,

                candidate_mode=(
                    args.candidate_mode
                ),

                max_iterations=(
                    args.max_iterations
                ),

                beam_width=(
                    args.beam_width
                ),
            )
        )

        final = (
            result.final_evaluation
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
            f"Unique M       : "
            f"{result.unique_evaluations}"
        )

        print(
            f"Duwhal queries : "
            f"{result.unique_duwhal_queries}"
        )

        print(
            f"Search time    : "
            f"{seconds * 1000:.3f} ms"
        )

        if args.profile:
            print_profile(
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