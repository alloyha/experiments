from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .model import (
    DuwhalVideoRecommender,
)


@dataclass
class EvaluationRun:
    strategy: str
    metrics: pd.DataFrame
    cold_query_seconds: float | None
    warmup_seconds: list[float]
    steady_state_seconds: list[float]


@dataclass
class BatchEvaluationRun:
    metrics: pd.DataFrame
    cold_batch_seconds: float | None
    warmup_batch_seconds: float | None
    batch_seconds: float
    batch_users: int


def future_items(
    test: pd.DataFrame,
    user_id: int,
    *,
    horizon: int = 20,
) -> set[int]:
    rows = test[test["user_id"] == user_id].sort_values("timestamp").head(horizon)

    return set(rows["video_id"].astype(int))


def evaluation_users(
    test: pd.DataFrame,
    *,
    max_users: int = 100,
) -> list[int]:
    users = sorted(test["user_id"].drop_duplicates().astype(int).tolist())

    return users[:max_users]


def _metrics_row(
    user_id: int,
    recommendations: pd.DataFrame,
    relevant: set[int],
    *,
    k: int,
    query_seconds: float | None,
) -> dict[str, float | int]:
    predicted = (
        recommendations["video_id"].astype(int).tolist()
        if not recommendations.empty
        else []
    )

    row: dict[str, float | int] = {
        "user_id": user_id,
        f"precision@{k}": precision_at_k(
            predicted,
            relevant,
            k,
        ),
        f"recall@{k}": recall_at_k(
            predicted,
            relevant,
            k,
        ),
        f"hit_rate@{k}": hit_rate_at_k(
            predicted,
            relevant,
            k,
        ),
        f"ndcg@{k}": ndcg_at_k(
            predicted,
            relevant,
            k,
        ),
        "returned": len(predicted),
    }

    if query_seconds is not None:
        row["query_seconds"] = query_seconds

    return row


def _recommend(
    model: DuwhalVideoRecommender,
    user_id: int,
    *,
    strategy: str,
    k: int,
    history_size: int,
    candidate_pool: int,
) -> pd.DataFrame:
    if strategy == "cf":
        return model.recommend_cf(
            user_id,
            k=k,
            history_size=history_size,
            candidate_pool=candidate_pool,
        )

    if strategy == "graph":
        return model.recommend_graph(
            user_id,
            k=k,
            history_size=history_size,
            candidate_pool=candidate_pool,
            return_paths=False,
        )

    if strategy == "popular":
        return model.recommend_popular(
            user_id,
            k=k,
        )

    if strategy == "hybrid":
        return model.recommend_hybrid(
            user_id,
            k=k,
            history_size=history_size,
            candidate_pool=candidate_pool,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_strategy(
    model: DuwhalVideoRecommender,
    test: pd.DataFrame,
    *,
    strategy: str,
    users: list[int],
    k: int = 10,
    future_horizon: int = 20,
    history_size: int = 5,
    candidate_pool: int = 200,
    warmup_runs: int = 2,
    verbose: bool = True,
) -> EvaluationRun:
    """
    Evaluate quality and steady-state serving latency.

    Timing phases:

    1. First query
       Captured separately as `cold_query_seconds`.

    2. Warm-up queries
       Repeated against the same first eligible user.
       Recorded for diagnostics but excluded from percentiles.

    3. Steady-state queries
       Every evaluation user is queried exactly once and these timings
       form p50/p95/p99.

    The first user therefore still participates in quality evaluation,
    but its quality-producing query occurs again after warm-up.
    """
    eligible_users = [
        user_id
        for user_id in users
        if future_items(
            test,
            user_id,
            horizon=future_horizon,
        )
    ]

    if not eligible_users:
        return EvaluationRun(
            strategy=strategy,
            metrics=pd.DataFrame(),
            cold_query_seconds=None,
            warmup_seconds=[],
            steady_state_seconds=[],
        )

    probe_user = eligible_users[0]

    # --------------------------------------------------------------
    # First query
    # --------------------------------------------------------------

    started = perf_counter()

    _recommend(
        model,
        probe_user,
        strategy=strategy,
        k=k,
        history_size=history_size,
        candidate_pool=candidate_pool,
    )

    cold_query_seconds = perf_counter() - started

    # --------------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------------

    warmup_seconds: list[float] = []

    for _ in range(
        max(
            int(warmup_runs),
            0,
        )
    ):
        started = perf_counter()

        _recommend(
            model,
            probe_user,
            strategy=strategy,
            k=k,
            history_size=history_size,
            candidate_pool=candidate_pool,
        )

        warmup_seconds.append(perf_counter() - started)

    # --------------------------------------------------------------
    # Steady-state evaluation
    # --------------------------------------------------------------

    rows = []

    steady_state_seconds: list[float] = []

    for index, user_id in enumerate(
        eligible_users,
        start=1,
    ):
        relevant = future_items(
            test,
            user_id,
            horizon=future_horizon,
        )

        started = perf_counter()

        recommendations = _recommend(
            model,
            user_id,
            strategy=strategy,
            k=k,
            history_size=history_size,
            candidate_pool=candidate_pool,
        )

        elapsed = perf_counter() - started

        steady_state_seconds.append(elapsed)

        rows.append(
            _metrics_row(
                user_id,
                recommendations,
                relevant,
                k=k,
                query_seconds=elapsed,
            )
        )

        if verbose:
            print(
                f"{strategy:<8} "
                f"{index:>3}/"
                f"{len(eligible_users):<3} "
                f"{elapsed * 1000:>9.3f} ms"
            )

    return EvaluationRun(
        strategy=strategy,
        metrics=pd.DataFrame(rows),
        cold_query_seconds=(cold_query_seconds),
        warmup_seconds=(warmup_seconds),
        steady_state_seconds=(steady_state_seconds),
    )


def evaluate_cf_batch(
    model: DuwhalVideoRecommender,
    test: pd.DataFrame,
    *,
    users: list[int],
    k: int = 10,
    future_horizon: int = 20,
    history_size: int = 5,
    candidate_pool: int = 250,
    warmup_users: int = 5,
) -> BatchEvaluationRun:
    """
    Evaluate vectorized ItemCF separately from scalar serving.

    A small batch is first executed twice:

    - first batch -> cold diagnostic
    - second batch -> warm-up diagnostic

    Then the complete requested user batch is measured.
    """
    eligible_users = [
        user_id
        for user_id in users
        if future_items(
            test,
            user_id,
            horizon=future_horizon,
        )
    ]

    if not eligible_users:
        return BatchEvaluationRun(
            metrics=pd.DataFrame(),
            cold_batch_seconds=None,
            warmup_batch_seconds=None,
            batch_seconds=0.0,
            batch_users=0,
        )

    warm_users = eligible_users[
        : max(
            1,
            min(
                int(warmup_users),
                len(eligible_users),
            ),
        )
    ]

    _, cold_batch_seconds = model.recommend_cf_batch(
        warm_users,
        k=k,
        history_size=history_size,
        candidate_pool=candidate_pool,
    )

    _, warmup_batch_seconds = model.recommend_cf_batch(
        warm_users,
        k=k,
        history_size=history_size,
        candidate_pool=candidate_pool,
    )

    recommendations, elapsed = model.recommend_cf_batch(
        eligible_users,
        k=k,
        history_size=history_size,
        candidate_pool=candidate_pool,
    )

    rows = []

    for user_id in eligible_users:
        relevant = future_items(
            test,
            user_id,
            horizon=future_horizon,
        )

        frame = recommendations.get(
            user_id,
            pd.DataFrame(),
        )

        rows.append(
            _metrics_row(
                user_id,
                frame,
                relevant,
                k=k,
                query_seconds=None,
            )
        )

    return BatchEvaluationRun(
        metrics=pd.DataFrame(rows),
        cold_batch_seconds=(cold_batch_seconds),
        warmup_batch_seconds=(warmup_batch_seconds),
        batch_seconds=elapsed,
        batch_users=len(eligible_users),
    )


def latency_summary(
    run: EvaluationRun,
) -> dict[str, float]:
    values = run.steady_state_seconds

    if not values:
        return {}

    milliseconds = (
        np.asarray(
            values,
            dtype=float,
        )
        * 1000.0
    )

    return {
        "p50_ms": float(
            np.percentile(
                milliseconds,
                50,
            )
        ),
        "p95_ms": float(
            np.percentile(
                milliseconds,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                milliseconds,
                99,
            )
        ),
        "mean_ms": float(milliseconds.mean()),
        "min_ms": float(milliseconds.min()),
        "max_ms": float(milliseconds.max()),
    }


def quality_summary(
    metrics: pd.DataFrame,
) -> pd.Series:
    if metrics.empty:
        return pd.Series(dtype=float)

    columns = [
        column
        for column in metrics.columns
        if column
        not in {
            "user_id",
            "query_seconds",
        }
    ]

    return metrics[columns].mean(numeric_only=True)
