#!/usr/bin/env python3
"""Read-only eligibility gate for transferring a PREVIEW candidate cube.

This command NEVER writes to either DuckDB, never transfers rows, and never
issues a freshness certificate. It checks one cycle by reconstructing the
actual dbt semantic cube against physically pinned Parquet and the dimensions
currently deployed in the main warehouse. All 29 projected columns are compared
with exact duplicate-sensitive multiset equality (signed multiplicity grouping
or the original two-way EXCEPT ALL fallback).

Its PASS is a staging eligibility finding, NOT permission to copy into main.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.candidate_cube_freshness import publication_signature
from scripts.certify_candidate_cube import available_cycles
from scripts.graph_guarded_read import _shared_flock

MODEL = "cube_candidate_performance_municipality"
CYCLE_PATTERN = re.compile(r"^(20\d{2}):(general|municipal)$")
COLUMNS = (
    "cycle_id", "election_id", "election_year", "election_type",
    "election_scope", "election_code", "round_number", "uf",
    "municipality_code", "office_code", "is_transit_vote", "candidate_id",
    "nominal_valid_votes", "zone_rows", "zone_count",
    "observed_candidate_nominal_valid_votes", "municipal_candidate_rank",
    "municipality", "geography_matched", "candidate_matched",
    "candidate_number", "candidate_name", "ballot_name", "party_number",
    "party", "party_name", "office", "office_scope",
    "candidate_share_of_observed_nominal_votes",
)
MAIN_ALIAS = "tse_analytics"
PREVIEW_ALIAS = "tse_event_registry_preview"


class GateError(RuntimeError):
    pass


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cycle(value: str) -> tuple[int, str]:
    match = CYCLE_PATTERN.fullmatch(value)
    if not match:
        raise GateError("--cycle must be YYYY:general or YYYY:municipal (year >= 2000)")
    return int(match.group(1)), match.group(2)


def pinned_parquet(root: Path, cycle: str) -> Path:
    year, kind = parse_cycle(cycle)
    path = root / f"election_type={kind}" / f"year={year}" / "data.parquet"
    if path.is_symlink():
        target = os.readlink(path)
        if Path(target).name != target or target in (".", ".."):
            raise GateError(f"Unsafe physical publication pointer: {path}")
    if not path.is_file():
        raise GateError(f"Published Parquet missing: {path}")
    return path


def expected_cube_sql(path: Path, cycle: str) -> str:
    """Replicate the checked-in dbt model with one physically pinned input."""
    year, kind = parse_cycle(cycle)
    return f"""
    WITH vote_rollup AS (
      SELECT cast(election_year AS varchar) || ':' || election_type AS cycle_id,
        election_id, election_year, election_type, election_scope, election_code,
        round_number, uf, municipality_code, office_code, is_transit_vote,
        candidate_id, SUM(nominal_valid_votes) AS nominal_valid_votes,
        COUNT(*) AS zone_rows, COUNT(DISTINCT zone) AS zone_count
      FROM read_parquet({sql_literal(str(path))}, hive_partitioning=true,
                        union_by_name=true)
      WHERE election_year = {year} AND election_type = {sql_literal(kind)}
      GROUP BY ALL
    ), ranked AS (
      SELECT *,
        SUM(nominal_valid_votes) OVER (
          PARTITION BY election_id, round_number, uf, municipality_code,
                       office_code, is_transit_vote
        ) AS observed_candidate_nominal_valid_votes,
        ROW_NUMBER() OVER (
          PARTITION BY election_id, round_number, uf, municipality_code,
                       office_code, is_transit_vote
          ORDER BY nominal_valid_votes DESC, candidate_id
        ) AS municipal_candidate_rank
      FROM vote_rollup
    )
    SELECT v.*, g.municipality,
      (g.municipality_code IS NOT NULL) AS geography_matched,
      (d.candidate_id IS NOT NULL) AS candidate_matched,
      d.candidate_number, d.candidate_name, d.ballot_name,
      d.party_number, d.party, d.party_name, d.office, d.office_scope,
      CASE WHEN v.observed_candidate_nominal_valid_votes > 0
        THEN v.nominal_valid_votes::double /
             v.observed_candidate_nominal_valid_votes
      END AS candidate_share_of_observed_nominal_votes
    FROM ranked v
    LEFT JOIN {MAIN_ALIAS}.main.dim_candidate d
      ON d.election_id = v.election_id AND d.candidate_id = v.candidate_id
    LEFT JOIN {MAIN_ALIAS}.main.dim_geography g
      ON g.cycle_id = v.cycle_id AND g.uf = v.uf
     AND g.municipality_code = v.municipality_code
    """


def relation_type(con, alias: str, name: str) -> str | None:
    result = con.execute("""
        SELECT table_type FROM information_schema.tables
        WHERE table_catalog = ? AND table_schema = 'main' AND table_name = ?
    """, [alias, name]).fetchone()
    return result[0] if result else None


def describe(con, relation: str) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def multiset_comparison_sql(cycle: str, *, mode: str = "signed",
                            expected_relation: str = "expected_cube",
                            preview_relation: str = f"{PREVIEW_ALIAS}.main.{MODEL}") -> str:
    """Exact duplicate-sensitive equality; never collapse columns to a hash.

    Signed mode groups the UNION ALL exactly ON ALL 29 values, then sums +1
    and -1 multiplicities. This is mathematically equivalent to the two-way
    EXCEPT ALL result, including NULL groups and duplicate rows, with one
    aggregation rather than two expensive set-difference operations.

    Relation arguments are trusted internal SQL identifiers, not CLI input.
    """
    projection = ", ".join('"' + c + '"' for c in COLUMNS)
    lhs = f"SELECT {projection} FROM {expected_relation}"
    rhs = (f"SELECT {projection} FROM {preview_relation} "
           f"WHERE cycle_id = {sql_literal(cycle)}")
    if mode == "except-all":
        return f"""
          SELECT count(*) FROM (
            ({lhs} EXCEPT ALL {rhs})
            UNION ALL
            ({rhs} EXCEPT ALL {lhs})
          ) AS differences
        """
    if mode != "signed":
        raise GateError(f"Unknown comparison mode: {mode}")
    return f"""
      WITH signed_rows AS (
        SELECT {projection}, 1 AS _transfer_delta FROM {expected_relation}
        UNION ALL
        SELECT {projection}, -1 AS _transfer_delta FROM {preview_relation}
        WHERE cycle_id = {sql_literal(cycle)}
      ), balances AS (
        SELECT {projection}, SUM(_transfer_delta) AS _balance
        FROM signed_rows
        GROUP BY {projection}
        HAVING SUM(_transfer_delta) <> 0
      )
      SELECT COALESCE(SUM(ABS(_balance)), 0) FROM balances
    """


def diff_count(con, cycle: str, *, mode: str = "signed") -> int:
    """Return exact total multiset row discrepancy across all 29 fields."""
    return int(con.execute(multiset_comparison_sql(cycle, mode=mode)).fetchone()[0])


def progress(message: str, started: float) -> None:
    print(f"[cube-audit +{time.monotonic()-started:.1f}s] {message}",
          file=sys.stderr, flush=True)


def require_existing_lock(path: Path) -> None:
    """Read-only checker must not manufacture a sidecar lock file."""
    if path.is_symlink() or not path.is_file():
        raise GateError(f"Required stable lock missing or unsafe: {path}")


def validate_locations(main: Path, preview: Path, root: Path, cycle: str) -> None:
    parse_cycle(cycle)
    if main == preview or main.suffix != ".duckdb" or preview.suffix != ".duckdb":
        raise GateError("Two distinct explicit DuckDB paths are required")
    if main.is_symlink() or preview.is_symlink() or root.is_symlink():
        raise GateError("Refusing symlink warehouse or source root")
    if not main.is_file() or not preview.is_file() or not root.is_dir():
        raise GateError("Missing main, preview, or published physical fact root")
    if os.path.samefile(main, preview):
        raise GateError("Main and preview must be distinct physical database files")
    for db in (main, preview):
        wal = Path(str(db) + ".wal")
        if wal.exists() or wal.is_symlink():
            raise GateError(f"WAL exists; close writer before read-only audit: {wal}")
    pinned_parquet(root, cycle)


def audit(main: Path, preview: Path, root: Path, cycle: str, *,
          expected_main_sha256: str, lock_timeout: float, memory_limit: str,
          max_rows: int, compare_mode: str = "signed") -> dict:
    if compare_mode not in {"signed", "except-all"}:
        raise GateError("Unsupported --compare-mode")
    started = time.monotonic()
    if lock_timeout <= 0 or max_rows <= 0:
        raise GateError("Positive lock timeout and maximum cube row count required")
    if not re.fullmatch(r"[1-9]\d*(MB|GB)", memory_limit):
        raise GateError("Memory limit must be e.g. 4GB")
    validate_locations(main, preview, root, cycle)
    deadline = time.monotonic() + lock_timeout
    import duckdb
    # Coordinated cooperative locks; no warehouse writes and no certificate issuance.
    progress("acquiring cooperative read locks", started)
    with ExitStack() as locks:
        for db in sorted((main, preview)):
            lock = Path(str(db) + ".build.lock")
            require_existing_lock(lock)
            locks.enter_context(_shared_flock(lock, deadline))
        validate_locations(main, preview, root, cycle)
        physical = available_cycles(root)
        if cycle not in physical:
            raise GateError(f"Selected cycle is not published: {cycle}")
        # This is a SINGLE-CYCLE audit: hold the selected physical publisher
        # lock, not all historic partitions' locks. Older, already-published
        # partitions may legitimately predate the .publish.lock protocol.
        # Both warehouse locks remain shared across this entire operation;
        # cooperating candidate-fact publishers obtain the warehouse writer
        # lock before modifying the physical publication inventory.
        year, kind = parse_cycle(cycle)
        selected_lock = root / f"election_type={kind}" / f"year={year}" / ".publish.lock"
        require_existing_lock(selected_lock)
        locks.enter_context(_shared_flock(selected_lock, deadline))
        if physical != available_cycles(root):
            raise GateError("Published cycle inventory changed while acquiring locks")
        progress("locks acquired; hashing both warehouses and selected Parquet", started)
        original_main_hash = sha256(main)
        if original_main_hash != expected_main_sha256:
            raise GateError("Main warehouse SHA-256 differs from approved conformant Gold")
        preview_hash = sha256(preview)
        source_before = publication_signature(root, cycle)
        file = pinned_parquet(root, cycle)
        with duckdb.connect(":memory:") as con:
            con.execute(f"SET memory_limit={sql_literal(memory_limit)}")
            con.execute(f"ATTACH {sql_literal(str(main))} AS {MAIN_ALIAS} (READ_ONLY)")
            con.execute(f"ATTACH {sql_literal(str(preview))} AS {PREVIEW_ALIAS} (READ_ONLY)")
            if relation_type(con, MAIN_ALIAS, MODEL) is not None:
                raise GateError("Main warehouse already has candidate cube: never overwrite it")
            if relation_type(con, PREVIEW_ALIAS, MODEL) != "BASE TABLE":
                raise GateError("Preview cube is not a persistent BASE TABLE")
            existing = dict(con.execute(
                f"SELECT cycle_id, count(*) FROM {PREVIEW_ALIAS}.main.{MODEL} "
                "GROUP BY cycle_id").fetchall())
            if set(existing) != physical or any(not n for n in existing.values()):
                raise GateError(f"Preview/physical cycle coverage mismatch: {existing} vs {sorted(physical)}")
            if existing[cycle] > max_rows:
                raise GateError(f"Cycle has {existing[cycle]} rows; exceeds opt-in --max-rows {max_rows}")
            # The main Gold inputs are authoritative; the preview's descriptive
            # dimension snapshots are NOT assumed equal to those of main.
            progress("reconstructing expected 29-column cube from Parquet and main dimensions", started)
            con.execute("CREATE TEMP TABLE expected_cube AS " + expected_cube_sql(file, cycle))
            progress("expected cube materialized; validating row counts and schema", started)
            count, votes = con.execute(
                "SELECT count(*), sum(nominal_valid_votes) FROM expected_cube"
            ).fetchone()
            if count != existing[cycle] or votes is None:
                raise GateError(f"Derived row-count mismatch: expected={count}, preview={existing[cycle]}")
            expected_schema = describe(con, "expected_cube")
            actual_schema = describe(con, f"{PREVIEW_ALIAS}.main.{MODEL}")
            if expected_schema != actual_schema:
                raise GateError(f"Schema mismatch, expected={expected_schema}, preview={actual_schema}")
            if [name for name, _ in actual_schema] != list(COLUMNS):
                raise GateError("Semantic model no longer matches the 29-column canonical projection")
            progress(f"beginning {compare_mode} exact 29-column multiset comparison", started)
            comparison_start = time.monotonic()
            differences = diff_count(con, cycle, mode=compare_mode)
            comparison_seconds = round(time.monotonic() - comparison_start, 3)
            progress(f"multiset comparison complete: differences={differences}", started)
            if differences:
                raise GateError(f"Full 29-column multiset mismatch: {differences} differing rows")
        progress("verifying post-audit Parquet and warehouse SHA-256", started)
        source_after = publication_signature(root, cycle)
        if source_before != source_after or physical != available_cycles(root):
            raise GateError("Physical Parquet changed during audit")
        if sha256(main) != original_main_hash or sha256(preview) != preview_hash:
            raise GateError("Warehouse bytes changed during the read-only audit")
        return {
            "result": "PASS", "scope": "readonly_single_cycle_transfer_eligibility_not_freshness_certification",
            "cycle": cycle, "main": str(main), "preview": str(preview),
            "main_sha256": original_main_hash, "preview_sha256": preview_hash,
            "physical_parquet": str(file), "source_signature": source_after,
            "published_cycles": sorted(physical), "cube_rows_by_cycle": existing,
            "selected_cube_rows": count, "selected_nominal_votes": int(votes),
            "selected_schema_columns": len(COLUMNS), "multiset_mismatch_rows": 0,
            "comparison_mode": compare_mode,
            "comparison_seconds": comparison_seconds,
            "total_seconds": round(time.monotonic() - started, 3),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": "cooperative locks only; does not copy, certify or authorize promotion",
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-db", required=True, type=Path)
    p.add_argument("--preview-db", required=True, type=Path)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--cycle", required=True)
    p.add_argument("--main-sha256", required=True)
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--compare-mode", choices=("signed", "except-all"), default="signed",
                   help="Signed multiset grouping (default) or original two-way EXCEPT ALL")
    p.add_argument("--lock-timeout", type=float, default=300.0)
    p.add_argument("--max-rows", type=int, default=2_000_000)
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    try:
        result = audit(args.main_db.expanduser().absolute(), args.preview_db.expanduser().absolute(),
                       args.source_root.expanduser().absolute(), args.cycle,
                       expected_main_sha256=args.main_sha256, lock_timeout=args.lock_timeout,
                       memory_limit=args.memory_limit, max_rows=args.max_rows,
                       compare_mode=args.compare_mode)
        output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(output)
                handle.flush()
                os.fsync(handle.fileno())
        print(output, end="")
        return 0
    except (GateError, OSError, ValueError, TimeoutError, RuntimeError) as e:
        print(f"CUBE TRANSFER DENIED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
