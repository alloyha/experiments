#!/usr/bin/env python3
"""Read-only, single-publication Gold graph SQL diagnostic (NOT a certificate).

Run the unmodified dbt graph projection and assertion SQL on temporary cycle-
scoped source views. The candidate fact is pinned to one explicit published
Parquet file; historical Parquets cannot accidentally be scanned. The DuckDB
warehouse is attached READ_ONLY and never changed. No dbt run/build happens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "tse_dbt/models/gold/graph"
TESTS = ROOT / "tse_dbt/tests"
SCOPED_SOURCES = (
    "dim_election_cycle", "dim_election", "dim_geography", "dim_candidate",
    "dim_party", "fact_party_votes",
)
ASSERTIONS = (
    "node_identity", "edge_identity", "edge_endpoints", "edge_measures",
    "edge_context", "affiliation_conservation",
    "candidate_vote_conservation", "party_vote_conservation",
)
CYCLE_PATTERN = re.compile(r"^(\d{4}):(general|municipal)$")


def quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render(path: Path, *, year: int, election_type: str) -> str:
    """Render only supported dbt expressions; StrictUndefined prevents silent omissions."""
    def predicate(year_column: str = "election_year",
                  type_column: str = "election_type") -> str:
        return (f"{year_column} in ({year}) and "
                f"{type_column} in ({quote_sql(election_type)})")
    env = Environment(undefined=StrictUndefined)
    return env.from_string(path.read_text(encoding="utf-8")).render(
        config=lambda **kwargs: "",
        ref=lambda name: name,
        incremental_partition_predicate=predicate,
    )


def selected_publication(root: Path, year: int, election_type: str) -> tuple[Path, dict]:
    """Resolve only the selected public partition; reject escape and missing bytes."""
    part = (root.resolve() / f"election_type={election_type}" / f"year={year}")
    public = part / "data.parquet"
    if not public.is_file():
        raise ValueError(f"No published Parquet for selected cycle: {public}")
    actual = public.resolve(strict=True)
    if actual.parent != part:
        raise ValueError(f"Published Parquet points outside partition: {public}")
    sha = hashlib.sha256()
    byte_count = 0
    with actual.open("rb") as stream:
        while block := stream.read(2 * 1024 * 1024):
            sha.update(block)
            byte_count += len(block)
    if not byte_count:
        raise ValueError("Published Parquet is empty")
    return actual, {"real_path": str(actual), "sha256": sha.hexdigest(),
                    "bytes": byte_count}


def setup_scoped_views(con, warehouse: Path, parquet: Path, year: int,
                       election_type: str) -> None:
    # In-memory connection: ATTACH READ_ONLY is the only warehouse access.
    con.execute(f"ATTACH {quote_sql(str(warehouse.resolve()))} AS historical (READ_ONLY)")
    for source in SCOPED_SOURCES:
        con.execute(f"CREATE VIEW {source} AS SELECT * FROM historical.main.{source} "
                    f"WHERE cycle_id = {quote_sql(f'{year}:{election_type}')}")
    # Explicit path (not wildcard) is the physical pruning guarantee.
    con.execute(f"""CREATE VIEW fact_candidate_votes AS
        SELECT *, CAST(election_year AS VARCHAR) || ':' || election_type AS cycle_id
        FROM read_parquet({quote_sql(str(parquet))},
                          hive_partitioning=true, union_by_name=true)
        WHERE election_year={year} AND year={year}
          AND election_type={quote_sql(election_type)}""")
    for name in ("graph_vertices", "graph_edges"):
        model = render(MODELS / f"{name}.sql", year=year, election_type=election_type)
        con.execute(f"CREATE VIEW {name} AS {model}")


def run(args) -> dict:
    match = CYCLE_PATTERN.fullmatch(args.cycle)
    if not match:
        raise ValueError("--cycle must be YYYY:general or YYYY:municipal")
    year, kind = int(match.group(1)), match.group(2)
    warehouse = args.db.resolve(strict=True)
    if not warehouse.is_file():
        raise ValueError("--db must identify an existing DuckDB file")
    parquet, publication = selected_publication(args.source_root, year, kind)
    selected = tuple(args.tests or ASSERTIONS)
    for test in selected:
        if test not in ASSERTIONS:
            raise ValueError(f"Unsupported assertion: {test}")
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"SET memory_limit = {quote_sql(args.memory_limit)}")
        con.execute("SET threads = 2")
        if args.temp_directory:
            args.temp_directory.mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory = {quote_sql(str(args.temp_directory.resolve()))}")
        setup_scoped_views(con, warehouse, parquet, year, kind)
        explanation = con.execute("""EXPLAIN SELECT count(*) FROM graph_edges
            WHERE edge_type='VOTED_FOR_CANDIDATE'""").fetchone()[1]
        if "READ_PARQUET" not in explanation.upper():
            raise ValueError("Expected candidate fact to depend on READ_PARQUET")
        report = {
            "status": "diagnostic_only_not_a_freshness_certificate",
            "cycle": args.cycle,
            "warehouse": str(warehouse),
            "source": publication,
            "physical_scope": "single selected Parquet file; no wildcard",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "assertions": [],
        }
        print(f"PASS: scoped read-only views; pinned physical path: {parquet}", flush=True)
        if not args.execute:
            print("PLAN-ONLY: SQL assertions were not executed", flush=True)
            return report
        for name in selected:
            sql = render(TESTS / f"assert_graph_{name}.sql", year=year, election_type=kind)
            started = time.monotonic()
            # A single violating row is sufficient for a dbt-style data assertion.
            # Derived SQL is trusted repository code, not user-supplied text.
            violation = con.execute(f"SELECT * FROM ({sql}) AS violations LIMIT 1").fetchone()
            elapsed = round(time.monotonic() - started, 3)
            report["assertions"].append({"name": name, "elapsed_seconds": elapsed,
                                         "pass": violation is None,
                                         "sample": repr(violation) if violation else None})
            print(f"{'PASS' if violation is None else 'FAIL'}: {name} ({elapsed}s)", flush=True)
            if violation is not None:
                raise AssertionError(f"{name}: invariant violation: {violation!r}")
        # Publication can change while tests execute. Reject even diagnostic PASS.
        _, after = selected_publication(args.source_root, year, kind)
        if after != publication:
            raise ValueError("Publication changed while the diagnostic ran")
        print("PASS: scoped graph assertions; not freshness/production-certified", flush=True)
        return report
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly run expensive SQL assertions. Default: plan only")
    parser.add_argument("--tests", nargs="+", choices=ASSERTIONS)
    parser.add_argument("--report", type=Path, help="New JSON evidence file; never overwrite")
    args = parser.parse_args()
    try:
        report = run(args)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
        return 0
    except (OSError, ValueError, AssertionError, duckdb.Error) as error:
        print(f"FAIL: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
