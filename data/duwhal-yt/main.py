from __future__ import annotations

import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from time import perf_counter
from typing import Any

import pandas as pd

from src.dataset import load_interactions, temporal_split
from src.evaluate import (
    BatchEvaluationRun,
    EvaluationRun,
    evaluate_cf_batch,
    evaluate_strategy,
    evaluation_users,
    future_items,
    latency_summary,
    quality_summary,
)
from src.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.model import DuwhalVideoRecommender


# ============================================================================
# HELPERS / ENVIRONMENT
# ============================================================================


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ============================================================================
# EXECUTION MODE
# ============================================================================

# fast:
#   - one isolated process / DuckDB connection per base strategy
#   - CF, Graph and Popularity all use native batch inference
#   - Hybrid reuses cached candidates and performs no model inference
#   - batch query time is separated from application-side wall time
#
# benchmark:
#   - one model in one process
#   - scalar strategies run sequentially for serving-latency distributions
#   - batch probes are measured separately
#
MODE = os.getenv("DUWHAL_YT_MODE", "fast").lower()

if MODE not in {"fast", "benchmark"}:
    raise ValueError("DUWHAL_YT_MODE must be 'fast' or 'benchmark'")


# ============================================================================
# DATASET
# ============================================================================

MAX_USERS = 100
MIN_WATCH_RATIO = 0.70
TEST_FRACTION = 0.20


# ============================================================================
# SESSIONIZATION
# ============================================================================

SESSION_GAP_MINUTES = 30
MAX_SESSION_ITEMS = 15


# ============================================================================
# ITEMCF
# ============================================================================

CF_TOP_K_SIMILAR = 50
CF_MIN_COOCCURRENCE = 2
CF_SHRINKAGE = 10


# ============================================================================
# GRAPH
# ============================================================================

GRAPH_MIN_COOCCURRENCE = 2
GRAPH_TOP_K_EDGES = 100
GRAPH_ALPHA = 0.10
GRAPH_MAX_DEPTH = 2
GRAPH_BEAM_WIDTH = 200


# ============================================================================
# POPULARITY
# ============================================================================

POPULARITY_WINDOW_DAYS = 30
POPULARITY_DECAY_HALF_LIFE = 7


# ============================================================================
# EVALUATION
# ============================================================================

HISTORY_SIZE = 5
CANDIDATE_POOL = 200
POPULARITY_CANDIDATE_POOL = 100
TOP_K = 10
EVALUATION_USERS = 100
FUTURE_HORIZON = 20
WARMUP_RUNS = 2


# ============================================================================
# HYBRID
# ============================================================================

RRF_CONSTANT = 60
HYBRID_CF_WEIGHT = 1.00
HYBRID_GRAPH_WEIGHT = 0.80
HYBRID_POPULARITY_WEIGHT = 0.30


# ============================================================================
# FAST MODE
# ============================================================================

# With all three base strategies now batch-native, 3 x 1 is the current
# throughput baseline: one worker per strategy, one DuckDB thread per worker.
FAST_WORKERS = int(os.getenv("DUWHAL_YT_WORKERS", "3"))
FAST_THREADS_PER_WORKER = int(os.getenv("DUWHAL_YT_THREADS_PER_WORKER", "1"))

# Keep the safe default, but make process-start overhead easy to benchmark.
# On Linux/WSL, try DUWHAL_YT_MP_START_METHOD=fork as a controlled experiment.
FAST_MP_START_METHOD = os.getenv("DUWHAL_YT_MP_START_METHOD", "spawn").lower()

if FAST_MP_START_METHOD not in get_all_start_methods():
    raise ValueError(
        "DUWHAL_YT_MP_START_METHOD must be one of "
        f"{sorted(get_all_start_methods())}; got {FAST_MP_START_METHOD!r}"
    )

# Explanation diagnostics deliberately stay outside the fast critical path.
FAST_GRAPH_DIAGNOSTIC = env_flag("DUWHAL_YT_GRAPH_DIAGNOSTIC", False)


# ============================================================================
# HISTORICAL REFERENCES
# ============================================================================

HISTORICAL_DUWHAL_QUERY_SECONDS = 458.98
EXTERNAL_CANDIDATE_INDEX_MEAN_SECONDS = 0.003634


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class FastStrategyResult:
    strategy: str
    recommendations: dict[int, pd.DataFrame]
    fit_seconds: float
    inference_wall_seconds: float
    batch_seconds: float
    batch_users: int

    @property
    def worker_seconds(self) -> float:
        return self.fit_seconds + self.inference_wall_seconds


@dataclass
class BatchProbe:
    strategy: str
    recommendations: dict[int, pd.DataFrame]
    query_seconds: float
    wall_seconds: float
    users: int


# ============================================================================
# MODEL FACTORY
# ============================================================================


def build_model(*, threads: int | None = None) -> DuwhalVideoRecommender:
    return DuwhalVideoRecommender(
        session_gap_minutes=SESSION_GAP_MINUTES,
        max_session_items=MAX_SESSION_ITEMS,
        cf_top_k_similar=CF_TOP_K_SIMILAR,
        cf_min_cooccurrence=CF_MIN_COOCCURRENCE,
        cf_shrinkage=CF_SHRINKAGE,
        graph_min_cooccurrence=GRAPH_MIN_COOCCURRENCE,
        graph_top_k_edges=GRAPH_TOP_K_EDGES,
        graph_alpha=GRAPH_ALPHA,
        graph_max_depth=GRAPH_MAX_DEPTH,
        graph_beam_width=GRAPH_BEAM_WIDTH,
        popularity_window_days=POPULARITY_WINDOW_DAYS,
        popularity_decay_half_life=POPULARITY_DECAY_HALF_LIFE,
        threads=threads,
    )


# ============================================================================
# REPORT HELPERS
# ============================================================================


def print_dataset_summary(
    interactions: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    print("Positive interactions:", f"{len(interactions):,}")
    print("Users:", f"{interactions['user_id'].nunique():,}")
    print("Videos:", f"{interactions['video_id'].nunique():,}")
    print()
    print("Train:", f"{len(train):,}")
    print("Test: ", f"{len(test):,}")
    print()
    print("Train users:", f"{train['user_id'].nunique():,}")
    print("Train videos:", f"{train['video_id'].nunique():,}")
    print("Test users:", f"{test['user_id'].nunique():,}")


def print_graph_diagnostic(benchmark: dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("GRAPH EXPLANATION DIAGNOSTIC")
    print("=" * 70)

    pathless_ms = benchmark["pathless_seconds"] * 1000
    paths_ms = benchmark["paths_seconds"] * 1000

    print()
    print(f"return_paths=False: {pathless_ms:,.3f} ms")
    print(f"return_paths=True:  {paths_ms:,.3f} ms")

    if pathless_ms > 0:
        print(f"Path overhead:       {paths_ms / pathless_ms:,.3f}x")

    print()
    print(f"{'Same items:':<22}{benchmark['same_items']}")
    print(f"{'Same ranking:':<22}{benchmark['same_ranking']}")
    print(f"{'Same hops:':<22}{benchmark['same_hops']}")
    print(f"{'Scores close:':<22}{benchmark['scores_close']}")

    max_delta = benchmark["max_score_delta"]
    print(
        f"{'Max score delta:':<22}"
        + ("N/A" if max_delta is None else f"{max_delta:.18e}")
    )
    print(f"{'Same semantics:':<22}{benchmark['same_semantics']}")


# ============================================================================
# RELEVANCE / QUALITY
# ============================================================================


def build_relevance_cache(
    test: pd.DataFrame,
    users: list[int],
) -> dict[int, list[int]]:
    return {
        user_id: future_items(
            test,
            user_id,
            horizon=FUTURE_HORIZON,
        )
        for user_id in users
    }


def metrics_from_cache(
    recommendations: dict[int, pd.DataFrame],
    relevance: dict[int, list[int]],
    users: list[int],
    *,
    k: int = TOP_K,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for user_id in users:
        relevant = relevance.get(user_id, [])
        frame = recommendations.get(user_id)

        if frame is None or frame.empty:
            predicted: list[int] = []
        else:
            predicted = frame["video_id"].head(k).astype(int).tolist()

        rows.append(
            {
                "user_id": user_id,
                f"precision@{k}": precision_at_k(predicted, relevant, k),
                f"recall@{k}": recall_at_k(predicted, relevant, k),
                f"hit_rate@{k}": hit_rate_at_k(predicted, relevant, k),
                f"ndcg@{k}": ndcg_at_k(predicted, relevant, k),
                "returned": len(predicted),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# HYBRID RRF
# ============================================================================


def fuse_hybrid(
    cf: pd.DataFrame | None,
    graph: pd.DataFrame | None,
    popular: pd.DataFrame | None,
    *,
    k: int = TOP_K,
) -> pd.DataFrame:
    """Fuse one user's cached candidate lists with lightweight Python RRF.

    For the small candidate sets used here (200 + 200 + 100), Python dicts are
    materially cheaper than constructing/concatenating/grouping/pivoting pandas
    DataFrames. Component scores are retained for inspection.
    """

    scores: dict[int, float] = {}
    cf_rrf: dict[int, float] = {}
    graph_rrf: dict[int, float] = {}
    popular_rrf: dict[int, float] = {}
    source_count: dict[int, int] = {}

    sources = (
        (cf, HYBRID_CF_WEIGHT, cf_rrf),
        (graph, HYBRID_GRAPH_WEIGHT, graph_rrf),
        (popular, HYBRID_POPULARITY_WEIGHT, popular_rrf),
    )

    for frame, weight, component in sources:
        if frame is None or frame.empty:
            continue

        # A recommender should already emit unique items. Keeping this set makes
        # retrieval_sources semantically robust if a duplicate ever slips in.
        seen_in_source: set[int] = set()

        for rank, raw_video_id in enumerate(frame["video_id"].array, start=1):
            video_id = int(raw_video_id)
            value = weight / (RRF_CONSTANT + rank)

            scores[video_id] = scores.get(video_id, 0.0) + value
            component[video_id] = component.get(video_id, 0.0) + value

            if video_id not in seen_in_source:
                source_count[video_id] = source_count.get(video_id, 0) + 1
                seen_in_source.add(video_id)

    if not scores:
        return pd.DataFrame(
            columns=[
                "video_id",
                "score",
                "cf_rrf",
                "graph_rrf",
                "popular_rrf",
                "retrieval_sources",
            ]
        )

    top_items = sorted(
        scores,
        key=lambda video_id: (
            -scores[video_id],
            -source_count.get(video_id, 0),
            video_id,
        ),
    )[:k]

    return pd.DataFrame.from_records(
        [
            {
                "video_id": video_id,
                "score": scores[video_id],
                "cf_rrf": cf_rrf.get(video_id, 0.0),
                "graph_rrf": graph_rrf.get(video_id, 0.0),
                "popular_rrf": popular_rrf.get(video_id, 0.0),
                "retrieval_sources": source_count.get(video_id, 0),
            }
            for video_id in top_items
        ]
    )


def build_hybrid_cache(
    cf_cache: dict[int, pd.DataFrame],
    graph_cache: dict[int, pd.DataFrame],
    popular_cache: dict[int, pd.DataFrame],
    users: list[int],
    *,
    k: int = TOP_K,
) -> dict[int, pd.DataFrame]:
    return {
        user_id: fuse_hybrid(
            cf_cache.get(user_id),
            graph_cache.get(user_id),
            popular_cache.get(user_id),
            k=k,
        )
        for user_id in users
    }


# ============================================================================
# FAST WORKER
# ============================================================================


def fast_strategy_worker(
    strategy: str,
    train: pd.DataFrame,
    users: list[int],
) -> FastStrategyResult:
    """Fit and batch-evaluate one base strategy in an isolated process."""

    model = build_model(threads=FAST_THREADS_PER_WORKER)

    try:
        component = {
            "cf": "cf",
            "graph": "graph",
            "popular": "popular",
        }[strategy]

        sink = io.StringIO()
        fit_started = perf_counter()

        with redirect_stdout(sink):
            model.fit(train, components={component})

        fit_seconds = perf_counter() - fit_started
        inference_started = perf_counter()

        if strategy == "cf":
            recommendations, batch_seconds = model.recommend_cf_batch(
                users,
                k=CANDIDATE_POOL,
                history_size=HISTORY_SIZE,
                candidate_pool=CANDIDATE_POOL,
            )

        elif strategy == "graph":
            recommendations, batch_seconds = model.recommend_graph_batch(
                users,
                k=CANDIDATE_POOL,
                history_size=HISTORY_SIZE,
                candidate_pool=CANDIDATE_POOL,
                return_paths=False,
            )

        elif strategy == "popular":
            recommendations, batch_seconds = model.recommend_popular_batch(
                users,
                k=POPULARITY_CANDIDATE_POOL,
            )

        else:
            raise ValueError(f"Unsupported fast strategy: {strategy}")

        inference_wall_seconds = perf_counter() - inference_started

        return FastStrategyResult(
            strategy=strategy,
            recommendations=recommendations,
            fit_seconds=fit_seconds,
            inference_wall_seconds=inference_wall_seconds,
            batch_seconds=batch_seconds,
            batch_users=len(users),
        )

    finally:
        model.close()


# ============================================================================
# FAST REPORTING
# ============================================================================


def print_fast_latency_report(
    results: dict[str, FastStrategyResult],
) -> None:
    print()
    print("=" * 70)
    print("FAST MODE LATENCY")
    print("=" * 70)
    print()
    print("All base strategies use native batch inference.")
    print("Fast mode measures throughput under concurrent CPU pressure.")
    print("Use benchmark mode for isolated scalar serving latency.")
    print()
    print("BATCH STRATEGIES")
    print("-" * 78)
    print(
        f"{'Strategy':<12}"
        f"{'Users':>8}"
        f"{'Query ms':>12}"
        f"{'Wall ms':>12}"
        f"{'Post ms':>12}"
        f"{'Query/user':>14}"
        f"{'Wall/user':>13}"
    )
    print("-" * 83)

    for strategy in ("cf", "graph", "popular"):
        result = results[strategy]
        users = result.batch_users
        query_ms = result.batch_seconds * 1000
        wall_ms = result.inference_wall_seconds * 1000
        post_ms = max(0.0, wall_ms - query_ms)
        query_per_user = query_ms / users if users else float("nan")
        wall_per_user = wall_ms / users if users else float("nan")

        print(
            f"{strategy:<12}"
            f"{users:>8d}"
            f"{query_ms:>12.3f}"
            f"{wall_ms:>12.3f}"
            f"{post_ms:>12.3f}"
            f"{query_per_user:>14.3f}"
            f"{wall_per_user:>13.3f}"
        )


# ============================================================================
# OPTIONAL FAST GRAPH DIAGNOSTIC
# ============================================================================


def run_fast_graph_diagnostic(
    train: pd.DataFrame,
    diagnostic_user: int,
) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    model = build_model(threads=FAST_THREADS_PER_WORKER)

    try:
        sink = io.StringIO()
        with redirect_stdout(sink):
            model.fit(train, components={"graph"})

        diagnostic = model.benchmark_graph_paths(
            diagnostic_user,
            history_size=HISTORY_SIZE,
            n=TOP_K,
            repeats=5,
        )

        return diagnostic, perf_counter() - started

    finally:
        model.close()


# ============================================================================
# FAST MODE
# ============================================================================


def run_fast(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    users = evaluation_users(test, max_users=EVALUATION_USERS)
    diagnostic_user = int(test["user_id"].value_counts().index[0])
    relevance = build_relevance_cache(test, users)

    print()
    print("=" * 70)
    print("FAST PARALLEL EVALUATION")
    print("=" * 70)
    print()
    print("Users:", len(users))
    print("Workers:", FAST_WORKERS)
    print("DuckDB threads / worker:", FAST_THREADS_PER_WORKER)
    print("Process start method:", FAST_MP_START_METHOD)
    print()
    print("Execution:")
    print("  Graph      -> vectorized batch")
    print("  CF         -> vectorized batch")
    print("  Popularity -> vectorized batch")
    print("  Hybrid     -> cached dict-RRF, no inference")
    print(
        "  Diagnostic -> outside critical path"
        if FAST_GRAPH_DIAGNOSTIC
        else "  Diagnostic -> disabled"
    )

    strategies = ("graph", "cf", "popular")
    results: dict[str, FastStrategyResult] = {}

    parallel_started = perf_counter()
    context = get_context(FAST_MP_START_METHOD)

    with ProcessPoolExecutor(
        max_workers=min(FAST_WORKERS, len(strategies)),
        mp_context=context,
    ) as pool:
        futures = {
            pool.submit(
                fast_strategy_worker,
                strategy,
                train,
                users,
            ): strategy
            for strategy in strategies
        }

        for future in as_completed(futures):
            strategy = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Fast worker failed for strategy={strategy!r}"
                ) from exc

            results[strategy] = result

            print(
                f"{strategy:<10} completed | "
                f"fit={result.fit_seconds:.3f}s | "
                f"inference_wall={result.inference_wall_seconds:.3f}s | "
                f"batch_query={result.batch_seconds:.3f}s"
            )

    parallel_wall = perf_counter() - parallel_started

    hybrid_started = perf_counter()
    hybrid_cache = build_hybrid_cache(
        results["cf"].recommendations,
        results["graph"].recommendations,
        results["popular"].recommendations,
        users,
    )
    hybrid_seconds = perf_counter() - hybrid_started

    caches = {
        "cf": results["cf"].recommendations,
        "graph": results["graph"].recommendations,
        "popular": results["popular"].recommendations,
        "hybrid": hybrid_cache,
    }

    print()
    print("=" * 70)
    print("QUALITY")
    print("=" * 70)

    for strategy in ("cf", "graph", "popular", "hybrid"):
        metrics = metrics_from_cache(
            caches[strategy],
            relevance,
            users,
        )
        print()
        print(strategy.upper())
        print("-" * 30)
        print(quality_summary(metrics).to_string())

    print_fast_latency_report(results)

    print()
    print("Hybrid inference queries: 0")
    print("Hybrid RRF fusion:", f"{hybrid_seconds * 1000:,.3f} ms")

    print()
    print("=" * 70)
    print(f"CACHED SINGLE-USER INSPECTION: {diagnostic_user}")
    print("=" * 70)

    relevant = relevance.get(diagnostic_user, [])

    for strategy in ("cf", "graph", "popular", "hybrid"):
        print()
        print("-" * 70)
        print(strategy.upper())
        print("-" * 70)

        frame = (
            caches[strategy]
            .get(diagnostic_user, pd.DataFrame())
            .head(TOP_K)
            .copy()
        )

        if frame.empty:
            print("No recommendations.")
            continue

        frame["future_hit"] = frame["video_id"].isin(relevant)
        print(frame.to_string(index=False))

    sequential_work = sum(result.worker_seconds for result in results.values())
    critical_worker = max(result.worker_seconds for result in results.values())
    orchestration_overhead = max(0.0, parallel_wall - critical_worker)

    print()
    print("=" * 70)
    print("FAST MODE EXECUTION SUMMARY")
    print("=" * 70)
    print()
    print("Parallel base-strategy wall time:", f"{parallel_wall:,.3f}s")
    print("Longest measured worker work:   ", f"{critical_worker:,.3f}s")
    print("Parallel orchestration overhead:", f"{orchestration_overhead:,.3f}s")
    print("Sum of worker work:              ", f"{sequential_work:,.3f}s")

    if parallel_wall > 0:
        print(
            "Observed parallel factor:       ",
            f"{sequential_work / parallel_wall:,.2f}x",
        )

    print("Hybrid cached fusion:            ", f"{hybrid_seconds:,.4f}s")
    print("Core fast workload:              ", f"{parallel_wall + hybrid_seconds:,.3f}s")

    if FAST_GRAPH_DIAGNOSTIC:
        diagnostic, diagnostic_seconds = run_fast_graph_diagnostic(
            train,
            diagnostic_user,
        )
        print_graph_diagnostic(diagnostic)
        print()
        print("Diagnostic wall time:", f"{diagnostic_seconds:,.3f}s")
        print("Diagnostic time is excluded from the core fast workload above.")


# ============================================================================
# BENCHMARK REPORT
# ============================================================================


def print_strategy_report(run: EvaluationRun) -> None:
    print()
    print("=" * 70)
    print(f"{run.strategy.upper()} RESULTS")
    print("=" * 70)

    if run.metrics.empty:
        print("No metrics produced.")
        return

    print()
    print("Quality")
    print("-------")
    print(quality_summary(run.metrics).to_string())

    print()
    print("Cold / warm-up")
    print("--------------")

    if run.cold_query_seconds is not None:
        print(f"First query:   {run.cold_query_seconds * 1000:,.3f} ms")

    if run.warmup_seconds:
        print(
            "Warm-up:       "
            + ", ".join(
                f"{seconds * 1000:,.3f} ms" for seconds in run.warmup_seconds
            )
        )

    stats = latency_summary(run)

    print()
    print("Steady-state latency")
    print("--------------------")

    for key, value in stats.items():
        print(f"{key:<12}{value:>10.3f}")

    print()
    print("Users evaluated:", len(run.metrics))


def print_cf_batch_report(run: BatchEvaluationRun) -> None:
    print()
    print("=" * 70)
    print("VECTORIZED CF BATCH")
    print("=" * 70)
    print()

    if run.cold_batch_seconds is not None:
        print("Cold small batch:", f"{run.cold_batch_seconds * 1000:,.3f} ms")

    if run.warmup_batch_seconds is not None:
        print("Warm small batch:", f"{run.warmup_batch_seconds * 1000:,.3f} ms")

    print()
    print("Measured batch users:", run.batch_users)
    print("Measured batch query:", f"{run.batch_seconds * 1000:,.3f} ms")

    if run.batch_users:
        print(
            "Effective per-user:",
            f"{run.batch_seconds / run.batch_users * 1000:,.3f} ms",
        )

    print()
    print("Batch quality")
    print("-------------")
    print(quality_summary(run.metrics).to_string())


def run_batch_probe(
    model: DuwhalVideoRecommender,
    strategy: str,
    users: list[int],
) -> BatchProbe:
    wall_started = perf_counter()

    if strategy == "graph":
        recommendations, query_seconds = model.recommend_graph_batch(
            users,
            k=CANDIDATE_POOL,
            history_size=HISTORY_SIZE,
            candidate_pool=CANDIDATE_POOL,
            return_paths=False,
        )

    elif strategy == "popular":
        recommendations, query_seconds = model.recommend_popular_batch(
            users,
            k=POPULARITY_CANDIDATE_POOL,
        )

    else:
        raise ValueError(f"Unsupported batch probe strategy: {strategy}")

    return BatchProbe(
        strategy=strategy,
        recommendations=recommendations,
        query_seconds=query_seconds,
        wall_seconds=perf_counter() - wall_started,
        users=len(users),
    )


def print_batch_probe_report(
    probe: BatchProbe,
    metrics: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print(f"VECTORIZED {probe.strategy.upper()} BATCH")
    print("=" * 70)
    print()
    print("Measured batch users:", probe.users)
    print("Measured batch query:", f"{probe.query_seconds * 1000:,.3f} ms")
    print("Measured batch wall: ", f"{probe.wall_seconds * 1000:,.3f} ms")

    if probe.users:
        print(
            "Query effective/user:",
            f"{probe.query_seconds / probe.users * 1000:,.3f} ms",
        )
        print(
            "Wall effective/user: ",
            f"{probe.wall_seconds / probe.users * 1000:,.3f} ms",
        )

    print()
    print("Batch quality")
    print("-------------")
    print(quality_summary(metrics).to_string())


# ============================================================================
# BENCHMARK MODE
# ============================================================================


def run_benchmark(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    model = build_model()

    try:
        print()
        print("=" * 70)
        print("DUWHAL TRAINING")
        print("=" * 70)
        print()

        fit_started = perf_counter()
        model.fit(train)
        build_seconds = perf_counter() - fit_started

        model.print_training_times()
        model.print_model_stats()

        print()
        print("Total model build:", f"{build_seconds:,.4f}s")

        users = evaluation_users(test, max_users=EVALUATION_USERS)
        relevance = build_relevance_cache(test, users)

        print()
        print("=" * 70)
        print("OFFLINE EVALUATION")
        print("=" * 70)
        print()
        print("Users:", len(users))
        print("Future horizon:", FUTURE_HORIZON)
        print("Top-K:", TOP_K)
        print("History size:", HISTORY_SIZE)
        print("Candidate pool:", CANDIDATE_POOL)
        print("Warm-up runs:", WARMUP_RUNS)

        reports: dict[str, EvaluationRun] = {}

        for strategy in ("cf", "graph", "popular", "hybrid"):
            print()
            print(f"Evaluating {strategy}...")

            run = evaluate_strategy(
                model,
                test,
                strategy=strategy,
                users=users,
                k=TOP_K,
                future_horizon=FUTURE_HORIZON,
                history_size=HISTORY_SIZE,
                candidate_pool=CANDIDATE_POOL,
                warmup_runs=WARMUP_RUNS,
            )

            reports[strategy] = run
            print_strategy_report(run)

        cf_batch = evaluate_cf_batch(
            model,
            test,
            users=users,
            k=TOP_K,
            future_horizon=FUTURE_HORIZON,
            history_size=HISTORY_SIZE,
            candidate_pool=CANDIDATE_POOL,
        )
        print_cf_batch_report(cf_batch)

        batch_probes: dict[str, BatchProbe] = {}

        for strategy in ("graph", "popular"):
            probe = run_batch_probe(model, strategy, users)
            batch_probes[strategy] = probe

            probe_metrics = metrics_from_cache(
                probe.recommendations,
                relevance,
                users,
            )
            print_batch_probe_report(probe, probe_metrics)

        diagnostic_user = int(test["user_id"].value_counts().index[0])
        diagnostic = model.benchmark_graph_paths(
            diagnostic_user,
            history_size=HISTORY_SIZE,
            n=TOP_K,
            repeats=5,
        )
        print_graph_diagnostic(diagnostic)

        print()
        print("=" * 70)
        print("STEADY-STATE SCALAR SUMMARY")
        print("=" * 70)
        print()
        print(
            f"{'Strategy':<12}"
            f"{'Cold ms':>12}"
            f"{'p50 ms':>12}"
            f"{'p95 ms':>12}"
            f"{'p99 ms':>12}"
            f"{'Mean ms':>12}"
        )
        print("-" * 72)

        for strategy, run in reports.items():
            stats = latency_summary(run)
            cold_ms = (
                run.cold_query_seconds * 1000
                if run.cold_query_seconds is not None
                else float("nan")
            )

            print(
                f"{strategy:<12}"
                f"{cold_ms:>12.3f}"
                f"{stats.get('p50_ms', float('nan')):>12.3f}"
                f"{stats.get('p95_ms', float('nan')):>12.3f}"
                f"{stats.get('p99_ms', float('nan')):>12.3f}"
                f"{stats.get('mean_ms', float('nan')):>12.3f}"
            )

        print()
        print("=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print()

        if cf_batch.batch_users:
            print(
                "CF batch effective/user:        ",
                f"{cf_batch.batch_seconds / cf_batch.batch_users * 1000:,.3f} ms",
            )

        for strategy in ("graph", "popular"):
            probe = batch_probes[strategy]
            if probe.users:
                print(
                    f"{strategy.capitalize()} batch effective/user:".ljust(32),
                    f"{probe.query_seconds / probe.users * 1000:,.3f} ms",
                )

    finally:
        model.close()


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    experiment_started = perf_counter()

    print("=" * 70)
    print("DUWHAL-YT")
    print("=" * 70)
    print("Mode:", MODE)
    print()
    print("Loading KuaiRec...", flush=True)

    started = perf_counter()
    interactions = load_interactions(
        max_users=MAX_USERS,
        min_watch_ratio=MIN_WATCH_RATIO,
        sampling="random",
    )
    print(f"Dataset loaded in {perf_counter() - started:,.2f}s")

    started = perf_counter()
    train, test = temporal_split(
        interactions,
        test_fraction=TEST_FRACTION,
        min_interactions=5,
    )
    print(f"Temporal split in {perf_counter() - started:,.2f}s")

    print_dataset_summary(interactions, train, test)

    if MODE == "fast":
        run_fast(train, test)
    else:
        run_benchmark(train, test)

    print()
    print("=" * 70)
    print("HISTORICAL LATENCY REFERENCE")
    print("=" * 70)
    print()
    print(
        "Old Duwhal experiment:",
        f"{HISTORICAL_DUWHAL_QUERY_SECONDS * 1000:,.1f} ms/query",
    )
    print(
        "External CandidateIndex:",
        f"{EXTERNAL_CANDIDATE_INDEX_MEAN_SECONDS * 1000:,.3f} ms/query",
    )

    print()
    print("=" * 70)
    print(f"TOTAL EXPERIMENT TIME: {perf_counter() - experiment_started:,.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
