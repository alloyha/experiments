from __future__ import annotations

from math import log2


def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    hits = set(recommended[:k]) & relevant

    return len(hits) / k


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    hits = set(recommended[:k]) & relevant

    return len(hits) / len(relevant)


def hit_rate_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(
        recommended[:k],
        start=1,
    ):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(
        len(relevant),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg
