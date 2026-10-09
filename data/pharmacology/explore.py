from __future__ import annotations

import argparse
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from duwhal.graph import InteractionGraph

from solver import (
    ClinicalModel,
    RegimenSolver,
    resolve_pathology,
)


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "patologias.duckdb"


# ============================================================
# AUDIT DATA
# ============================================================


@dataclass
class CandidateGenerationAudit:
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        name: str,
        seconds: float,
    ) -> None:
        self.timings[name] = (
            self.timings.get(name, 0.0)
            + seconds
        )

    def count(
        self,
        name: str,
        value: int,
    ) -> None:
        self.counts[name] = value

    @property
    def total_seconds(self) -> float:
        return self.timings.get("total", 0.0)


@dataclass
class DuwhalEngineAudit:
    """
    Lifecycle audit for one persistent InteractionGraph.

    Setup:
        graph_init
        load_interactions
        build_topology

    Query timings are accumulated separately.
    """

    setup_timings: dict[str, float] = field(
        default_factory=dict
    )

    query_timings: dict[str, float] = field(
        default_factory=dict
    )

    counts: dict[str, int] = field(
        default_factory=dict
    )

    def add_setup(
        self,
        name: str,
        seconds: float,
    ) -> None:
        self.setup_timings[name] = (
            self.setup_timings.get(name, 0.0)
            + seconds
        )

    def add_query(
        self,
        name: str,
        seconds: float,
    ) -> None:
        self.query_timings[name] = (
            self.query_timings.get(name, 0.0)
            + seconds
        )

    def increment(
        self,
        name: str,
        amount: int = 1,
    ) -> None:
        self.counts[name] = (
            self.counts.get(name, 0)
            + amount
        )

    @property
    def setup_total(self) -> float:
        return sum(
            self.setup_timings.values()
        )

    @property
    def query_total(self) -> float:
        return sum(
            self.query_timings.values()
        )


# ============================================================
# LOADERS
# ============================================================


def load_therapeutic_support(
    con: duckdb.DuckDBPyConnection,
) -> dict[int, set[int]]:
    rows = con.execute(
        """
        SELECT
            apresentacao_id,
            sintoma_id
        FROM apresentacao_alivia
        """
    ).fetchall()

    result: dict[int, set[int]] = {}

    for presentation_id, symptom_id in rows:
        result.setdefault(
            int(presentation_id),
            set(),
        ).add(
            int(symptom_id)
        )

    return result


def load_adverse_support(
    con: duckdb.DuckDBPyConnection,
) -> dict[int, set[int]]:
    rows = con.execute(
        """
        SELECT
            apresentacao_id,
            sintoma_id
        FROM apresentacao_evento_adverso
        """
    ).fetchall()

    result: dict[int, set[int]] = {}

    for presentation_id, symptom_id in rows:
        result.setdefault(
            int(presentation_id),
            set(),
        ).add(
            int(symptom_id)
        )

    return result


# ============================================================
# DUWHAL INTERACTION ADAPTER
# ============================================================


def build_duwhal_interactions(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    symptom_rows = con.execute(
        """
        SELECT id
        FROM sintoma
        ORDER BY id
        """
    ).fetchall()

    relief_rows = con.execute(
        """
        SELECT
            apresentacao_id,
            sintoma_id
        FROM apresentacao_alivia
        ORDER BY
            sintoma_id,
            apresentacao_id
        """
    ).fetchall()

    by_symptom: dict[
        int,
        list[int],
    ] = {}

    for presentation_id, symptom_id in relief_rows:
        by_symptom.setdefault(
            int(symptom_id),
            [],
        ).append(
            int(presentation_id)
        )

    records: list[
        dict[str, str]
    ] = []

    for (symptom_id_raw,) in symptom_rows:
        symptom_id = int(
            symptom_id_raw
        )

        context = (
            f"therapeutic:symptom:{symptom_id}"
        )

        # Keep every symptom represented,
        # including singleton contexts.
        records.append(
            {
                "context": context,
                "node": (
                    f"symptom:{symptom_id}"
                ),
            }
        )

        for presentation_id in (
            by_symptom.get(
                symptom_id,
                [],
            )
        ):
            records.append(
                {
                    "context": context,
                    "node": (
                        f"presentation:{presentation_id}"
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "context",
            "node",
        ],
    )


# ============================================================
# COVERAGE
# ============================================================


def candidate_coverage(
    presentation_id: int,
    unresolved: Iterable[int],
    therapeutic_support: dict[int, set[int]],
) -> tuple[
    frozenset[int],
    float,
]:
    unresolved_set = frozenset(
        int(value)
        for value in unresolved
    )

    covered = frozenset(
        therapeutic_support.get(
            int(presentation_id),
            set(),
        )
        & set(unresolved_set)
    )

    coverage = (
        len(covered)
        / len(unresolved_set)

        if unresolved_set
        else 0.0
    )

    return (
        covered,
        coverage,
    )


# ============================================================
# RESULT NORMALIZATION
# ============================================================


def _normalize_ranked_frame(
    ranked: Any,
) -> pd.DataFrame:
    if isinstance(
        ranked,
        pd.DataFrame,
    ):
        return ranked.copy()

    if hasattr(
        ranked,
        "to_pandas",
    ):
        return ranked.to_pandas()

    return pd.DataFrame(
        ranked
    )


def _filter_presentation_nodes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    if "node" not in frame.columns:
        raise KeyError(
            "Duwhal rank_nodes() result "
            "does not contain 'node'."
        )

    return frame[
        frame[
            "node"
        ]
        .astype(str)
        .str.startswith(
            "presentation:"
        )
    ].copy()


def _parse_presentation_frame(
    frame: pd.DataFrame,
    *,
    depth: int,
) -> pd.DataFrame:
    columns = [
        "presentation_id",
        "score",
        "hops",
        "reason",
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=columns
        )

    frame = frame.copy()

    if (
        "steps" in frame.columns
        and "hops" not in frame.columns
    ):
        frame = frame.rename(
            columns={
                "steps": "hops",
            }
        )

    if "score" not in frame.columns:
        raise KeyError(
            "Duwhal result does not "
            "contain 'score'."
        )

    frame[
        "presentation_id"
    ] = (
        frame[
            "node"
        ]
        .astype(str)
        .str.split(
            ":",
            n=1,
        )
        .str[1]
        .astype(int)
    )

    if "hops" not in frame.columns:
        frame[
            "hops"
        ] = depth

    if "reason" not in frame.columns:
        frame[
            "reason"
        ] = ""

    return frame[
        columns
    ].copy()


def _sort_candidate_frame(
    frame: pd.DataFrame,
    *,
    top: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(
            drop=True
        )

    return (
        frame
        .sort_values(
            by=[
                "score",
                "hops",
                "presentation_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .head(top)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PERSISTENT DUWHAL ENGINE
# ============================================================


class PersistentDuwhalCandidateEngine:
    """
    One graph lifecycle per search.

    Old lifecycle:

        query(U1):
            create
            load
            build
            rank

        query(U2):
            create
            load
            build
            rank

    New lifecycle:

        create
        load
        build

        query(U1): rank
        query(U2): rank
        ...
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        *,
        min_interactions: int = 1,
    ):
        self.interactions = interactions

        self.min_interactions = (
            min_interactions
        )

        self.audit = (
            DuwhalEngineAudit()
        )

        self.graph: (
            InteractionGraph
            | None
        ) = None

        self._closed = False

        self._setup()

    def _setup(
        self,
    ) -> None:
        started = (
            time.perf_counter()
        )

        self.graph = (
            InteractionGraph()
        )

        self.audit.add_setup(
            "graph_init",
            time.perf_counter()
            - started,
        )

        started = (
            time.perf_counter()
        )

        loaded = (
            self.graph
            .load_interactions(
                self.interactions,
                context_col="context",
                node_col="node",
            )
        )

        self.audit.add_setup(
            "load_interactions",
            time.perf_counter()
            - started,
        )

        if isinstance(
            loaded,
            int,
        ):
            self.audit.increment(
                "loaded_interactions",
                loaded,
            )

        started = (
            time.perf_counter()
        )

        self.graph.build_topology(
            min_interactions=(
                self.min_interactions
            )
        )

        self.audit.add_setup(
            "build_topology",
            time.perf_counter()
            - started,
        )

        self.audit.increment(
            "interaction_rows",
            len(self.interactions),
        )

        self.audit.increment(
            "engine_builds",
            1,
        )

    def generate(
        self,
        unresolved: Iterable[int],
        *,
        top: int = 10,
        depth: int = 1,
    ) -> pd.DataFrame:
        if self._closed:
            raise RuntimeError(
                "Duwhal engine is closed."
            )

        if self.graph is None:
            raise RuntimeError(
                "Duwhal engine was not initialized."
            )

        total_started = (
            time.perf_counter()
        )

        unresolved = frozenset(
            int(value)
            for value in unresolved
        )

        self.audit.increment(
            "query_calls",
            1,
        )

        self.audit.increment(
            "query_unresolved_symptoms",
            len(unresolved),
        )

        if not unresolved:
            result = pd.DataFrame(
                columns=[
                    "presentation_id",
                    "score",
                    "hops",
                    "reason",
                ]
            )

            self.audit.add_query(
                "query_total",
                time.perf_counter()
                - total_started,
            )

            return result

        # ----------------------------------------------------
        # PREPARE SEEDS
        # ----------------------------------------------------

        started = (
            time.perf_counter()
        )

        seeds = [
            f"symptom:{symptom_id}"
            for symptom_id
            in sorted(unresolved)
        ]

        graph_limit = max(
            top * 4,
            top + len(seeds),
            32,
        )

        self.audit.add_query(
            "prepare_seeds",
            time.perf_counter()
            - started,
        )

        self.audit.increment(
            "seed_count",
            len(seeds),
        )

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        started = (
            time.perf_counter()
        )

        ranked = self.graph.rank_nodes(
            seed_nodes=seeds,
            steps=depth,
            scoring="frequency",
            limit=graph_limit,
        )

        self.audit.add_query(
            "rank_nodes",
            time.perf_counter()
            - started,
        )

        # ----------------------------------------------------
        # NATIVE -> PANDAS
        # ----------------------------------------------------

        started = (
            time.perf_counter()
        )

        frame = _normalize_ranked_frame(
            ranked
        )

        self.audit.add_query(
            "native_to_pandas",
            time.perf_counter()
            - started,
        )

        self.audit.increment(
            "ranked_rows",
            len(frame),
        )

        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        started = (
            time.perf_counter()
        )

        frame = (
            _filter_presentation_nodes(
                frame
            )
        )

        self.audit.add_query(
            "filter_presentations",
            time.perf_counter()
            - started,
        )

        self.audit.increment(
            "presentation_rows",
            len(frame),
        )

        # ----------------------------------------------------
        # PARSE
        # ----------------------------------------------------

        started = (
            time.perf_counter()
        )

        frame = (
            _parse_presentation_frame(
                frame,
                depth=depth,
            )
        )

        self.audit.add_query(
            "parse_presentations",
            time.perf_counter()
            - started,
        )

        # ----------------------------------------------------
        # SORT / LIMIT
        # ----------------------------------------------------

        started = (
            time.perf_counter()
        )

        frame = (
            _sort_candidate_frame(
                frame,
                top=top,
            )
        )

        self.audit.add_query(
            "sort_limit",
            time.perf_counter()
            - started,
        )

        self.audit.increment(
            "returned_rows",
            len(frame),
        )

        self.audit.add_query(
            "query_total",
            time.perf_counter()
            - total_started,
        )

        return frame

    def close(
        self,
    ) -> None:
        if self._closed:
            return

        if self.graph is not None:
            try:
                self.graph.db.close()
            except Exception:
                pass

        self._closed = True

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        *_,
    ):
        self.close()


# ============================================================
# ORIGINAL STATELESS API
# ============================================================


def generate_candidates(
    interactions: pd.DataFrame,
    unresolved: Iterable[int],
    *,
    top: int = 10,
    depth: int = 1,
) -> pd.DataFrame:
    """
    Backward-compatible stateless API.

    New search code should prefer
    PersistentDuwhalCandidateEngine.
    """

    with (
        PersistentDuwhalCandidateEngine(
            interactions
        )
    ) as engine:
        return engine.generate(
            unresolved,
            top=top,
            depth=depth,
        )


# ============================================================
# OPTIONAL CLI
# ============================================================


def parse_regimen(
    value: str | None,
) -> tuple[int, ...]:
    if not value:
        return tuple()

    return tuple(
        sorted(
            {
                int(part.strip())
                for part
                in value.split(",")
                if part.strip()
            }
        )
    )


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
        "--top",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help=(
            "Number of repeated queries "
            "against one persistent graph."
        ),
    )

    return parser.parse_args()


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
            model
        )

        evaluation = (
            solver.evaluate(
                parse_regimen(
                    args.regimen
                )
            )
        )

        unresolved = (
            evaluation
            .symptom_space
            .unresolved_support
        )

        interactions = (
            build_duwhal_interactions(
                con
            )
        )

        print(
            f"Pathology            : "
            f"{model.pathology_name}"
        )

        print(
            f"Unresolved           : "
            f"{sorted(unresolved)}"
        )

        print(
            f"Interaction rows     : "
            f"{len(interactions)}"
        )

        print(
            f"Contexts             : "
            f"{interactions['context'].nunique()}"
        )

        with (
            PersistentDuwhalCandidateEngine(
                interactions
            )
        ) as engine:
            print()

            print(
                "ENGINE SETUP"
            )

            print(
                "-" * 60
            )

            for name, seconds in (
                engine
                .audit
                .setup_timings
                .items()
            ):
                print(
                    f"{name:<28}"
                    f"{seconds * 1000:>12.3f} ms"
                )

            print(
                f"{'setup_total':<28}"
                f"{engine.audit.setup_total * 1000:>12.3f} ms"
            )

            frame = None

            for index in range(
                args.repeat
            ):
                before = dict(
                    engine.audit.query_timings
                )

                frame = engine.generate(
                    unresolved,
                    top=args.top,
                    depth=args.depth,
                )

                after = (
                    engine.audit.query_timings
                )

                print()

                print(
                    f"QUERY {index + 1}"
                )

                print(
                    "-" * 60
                )

                for name in (
                    "prepare_seeds",
                    "rank_nodes",
                    "native_to_pandas",
                    "filter_presentations",
                    "parse_presentations",
                    "sort_limit",
                    "query_total",
                ):
                    delta = (
                        after.get(
                            name,
                            0.0,
                        )
                        - before.get(
                            name,
                            0.0,
                        )
                    )

                    print(
                        f"{name:<28}"
                        f"{delta * 1000:>12.3f} ms"
                    )

            print()

            print(
                frame.to_string(
                    index=False
                )
            )

    finally:
        con.close()


if __name__ == "__main__":
    main()