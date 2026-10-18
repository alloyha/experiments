#!/usr/bin/env python3
"""Independent, read-only five-cycle candidate group/zone verifier.

This program never certifies graph freshness, changes canonical SQL, creates
sidecars beside Parquet, or modifies an existing Gold database. A PASS is
specific to exact pinned physical bytes, warehouse SHA, and lifecycle epoch.
Only cooperating POSIX-local publishers are fenced by existing flock inodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import consolidate_candidate_cube_transfer as locks
from scripts import certify_gold_graph_cycle as diagnostic
from scripts.warehouse_lifecycle import lifecycle_epoch

CYCLES = ("2018:general", "2020:municipal", "2022:general",
          "2024:municipal", "2026:general")
SCOPE = "historical_physical_candidate_zone_uniqueness_not_graph_freshness"
KEYS = ("election_id", "round_number", "uf", "municipality_code",
        "office_code", "is_transit_vote", "candidate_id", "zone")
COLUMNS = ("election_year", "election_type", *KEYS)
# Independent exact validator: DISTINCT over the complete typed tuple, rather
# than the publisher's per-key multiplicity CTE. This avoids relying on the
# same SQL implementation when validating historical source generations.
GUARD_SQL = """SELECT COUNT(*)::HUGEINT AS source_rows,
       COUNT(*) FILTER (WHERE zone IS NULL)::HUGEINT AS null_zone_rows,
       COUNT(*) FILTER (WHERE zone IS NOT NULL)::HUGEINT AS nonnull_rows,
       COUNT(DISTINCT (election_id, round_number, uf, municipality_code,
                       office_code, is_transit_vote, candidate_id, zone))
           FILTER (WHERE zone IS NOT NULL)::HUGEINT AS distinct_nonnull_keys
FROM read_parquet(?, hive_partitioning=false)"""
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HistoryGuardError(RuntimeError):
    pass


def expected_input(db: Path, root: Path, out: Path, *, expected_sha256: str) -> tuple[Path, Path, Path]:
    """Resolve safe inputs without modifying warehouse, lock inodes or Parquet."""
    if not SHA256.fullmatch(expected_sha256):
        raise HistoryGuardError("Explicit --expected-main-sha256 must be 64 lowercase hex digits")
    db = Path(db).expanduser().absolute()
    root = Path(root).expanduser().absolute()
    out = Path(out).expanduser().absolute()
    if db.is_symlink() or not db.is_file() or db.suffix != ".duckdb":
        raise HistoryGuardError("Expected an existing, non-symlink DuckDB file")
    if root.is_symlink() or not root.is_dir():
        raise HistoryGuardError("Expected existing, non-symlink physical fact root")
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise HistoryGuardError("Output must be absent and parent directory must exist")
    if out.parent.is_symlink() or out.resolve(strict=False).is_relative_to(db.parent.resolve()):
        raise HistoryGuardError("Report must be outside the live warehouse directory")
    if out.resolve(strict=False).is_relative_to(root.resolve()):
        raise HistoryGuardError("Report must be outside the live physical root")
    return db, root, out


def check_physical_scope(con, parquet: Path, year: int, election_type: str) -> int:
    """Every row must belong to the requested electoral cycle; no vacuous PASS."""
    columns = {column[0] for column in con.execute(
        "SELECT * FROM read_parquet(?, hive_partitioning=false) LIMIT 0",
        [str(parquet)],
    ).description}
    absent = sorted(set(COLUMNS) - columns)
    if absent:
        raise HistoryGuardError(f"Physical Parquet missing mandatory columns: {absent}")
    total, wrong = map(int, con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (
            WHERE election_year IS DISTINCT FROM ?
               OR election_type IS DISTINCT FROM ?)
        FROM read_parquet(?, hive_partitioning=false)
    """, [year, election_type, str(parquet)]).fetchone())
    if total <= 0 or wrong:
        raise HistoryGuardError(
            f"Physical source scope invalid: rows={total}, out_of_cycle={wrong}")
    return total


def check_exact_uniqueness(con, parquet: Path, *, expected_rows: int) -> dict:
    """Run the publisher's FULL multiplicity SQL, not a sampled/hashed proxy."""
    values = con.execute(GUARD_SQL, [str(parquet)]).fetchone()
    if values is None or any(v is None for v in values):
        raise HistoryGuardError("No complete uniqueness result")
    scanned, nulls, nonnull, unique_nonnull = map(int, values)
    duplicates = nonnull - unique_nonnull
    if (scanned != expected_rows or scanned <= 0 or
        nonnull + nulls != scanned or unique_nonnull < 0 or
        duplicates < 0 or unique_nonnull > nonnull):
        raise HistoryGuardError("Physical group/zone inventory mismatch")
    if nulls or duplicates:
        raise HistoryGuardError(
            f"Invalid group/zone multiplicities: null_zone_rows={nulls}, "
            f"duplicate_nonnull_zone_rows={duplicates}")
    return {"source_rows": scanned, "distinct_group_zone_keys": unique_nonnull,
            "null_zone_rows": nulls, "duplicate_nonnull_zone_rows": duplicates}


def verify_cycle(db: Path, root: Path, cycle: str, *,
                 memory_limit: str, threads: int, max_temp_gb: int) -> dict:
    """Proof is independent of the publisher's optional sidecar evidence."""
    import duckdb
    year_text, election_type = cycle.split(":")
    year = int(year_text)
    selected, before = diagnostic.selected_publication(root, year, election_type)
    if selected.is_symlink() or not selected.is_file():
        raise HistoryGuardError("Resolved Parquet must be a real, existing generation file")
    with tempfile.TemporaryDirectory(prefix="tse-historical-zone-spill-") as scratch:
        con = duckdb.connect(":memory:")
        try:
            con.execute("SET memory_limit = " + diagnostic.quote_sql(memory_limit))
            con.execute(f"SET threads = {threads}")
            con.execute("SET temp_directory = " + diagnostic.quote_sql(scratch))
            con.execute("SET max_temp_directory_size = " +
                        diagnostic.quote_sql(f"{max_temp_gb}GB"))
            rows = check_physical_scope(con, selected, year, election_type)
            inventory = check_exact_uniqueness(con, selected, expected_rows=rows)
        finally:
            con.close()
    _, after = diagnostic.selected_publication(root, year, election_type)
    if before != after:
        raise HistoryGuardError(f"Physical publication changed while verifying {cycle}")
    return {"cycle": cycle, "result": "PASS", "publication": after,
            "guard_sql_sha256": hashlib.sha256(GUARD_SQL.encode("utf-8")).hexdigest(),
            **inventory}


def audit(*, db: Path, root: Path, out: Path, expected_sha256: str,
          memory_limit: str = "4GB", threads: int = 2,
          max_temp_gb: int = 6, lock_timeout: float = 0) -> dict:
    """Atomically preserve one create-only report only after all five proofs pass."""
    db, root, out = expected_input(db, root, out, expected_sha256=expected_sha256)
    if threads < 1 or threads > 16 or max_temp_gb < 1 or lock_timeout < 0:
        raise HistoryGuardError("Invalid resource limits")
    if Path(str(db) + ".wal").exists() or Path(str(db) + ".wal").is_symlink():
        raise HistoryGuardError("Live DuckDB WAL exists; cannot establish closed snapshot")
    lock_paths = [locks.lock_path(db)]
    lock_paths.extend(locks.partition_for(root, cycle) / ".publish.lock" for cycle in CYCLES)
    with locks.noncreating_shared_locks(lock_paths, timeout=lock_timeout):
        actual_sha = locks.digest(db)
        if actual_sha != expected_sha256:
            raise HistoryGuardError("Warehouse SHA differs from reviewed expectation")
        epoch = lifecycle_epoch(db)
        cycles = [verify_cycle(db, root, cycle, memory_limit=memory_limit,
                               threads=threads, max_temp_gb=max_temp_gb)
                  for cycle in CYCLES]
        if (locks.digest(db) != actual_sha or lifecycle_epoch(db) != epoch or
            Path(str(db) + ".wal").exists()):
            raise HistoryGuardError("Warehouse SHA/epoch/WAL changed during historical audit")
        # One report, written only after full verification. Never publish partial PASS.
        report = {"schema_version": 1, "result": "PASS", "scope": SCOPE,
                  "all_five_cycles_verified": True,
                  "warehouse": str(db), "warehouse_sha256": actual_sha,
                  "lifecycle_epoch": epoch, "source_root": str(root),
                  "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "guard_implementation_sha256": locks.digest(Path(__file__)),
                  "cycles": cycles,
                  "limitations": [
                      "Read-only evidence, not an active graph/cube freshness certificate",
                      "Cooperating POSIX-local writers only; not protection from uncoordinated mutations",
                      "Independent full scan of exact selected physical Parquet bytes",
                      "Published source remains unchanged; no historical sidecars fabricated",
                  ]}
        # Create-only prevents accidental report replacement.
        with out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-main-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New file outside live warehouse; never overwritten")
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-temp-gb", type=int, default=6)
    parser.add_argument("--lock-timeout", type=float, default=0)
    args = parser.parse_args(argv)
    try:
        report = audit(db=args.db, root=args.source_root, out=args.output,
                       expected_sha256=args.expected_main_sha256,
                       memory_limit=args.memory_limit, threads=args.threads,
                       max_temp_gb=args.max_temp_gb, lock_timeout=args.lock_timeout)
        print(json.dumps({"result": report["result"],
                          "all_five_cycles_verified": True,
                          "report": str(args.output)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"HISTORICAL ZONE AUDIT DENIED/FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
