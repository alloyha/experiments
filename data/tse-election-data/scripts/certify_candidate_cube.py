#!/usr/bin/env python3
"""Audit candidate-cube coverage/freshness or certify one partition-replace load.

No raw ingestion and no implicit full-refresh. The caller explicitly selects the
DuckDB file and published candidate-fact Parquet root. Bare coverage results
are not proof of freshness. Full freshness requires per-cycle attestations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.candidate_cube_freshness import (
    FreshnessError, audit_freshness, input_signature, output_signature,
    issue_certificate, revoke_certificate,
)

MODEL = "cube_candidate_performance_municipality"
TESTS = (
    "assert_cube_candidate_vote_conservation",
    "assert_cube_candidate_share_and_rank",
)
PROJECT = Path(__file__).resolve().parents[1] / "tse_dbt"

# Key/measure fingerprint for *unselected* cycles. Reusable within a DuckDB
# version and execution, not a collision-free proof of all descriptive columns.
SNAPSHOT_SQL = f"""
    select cycle_id, uf,
           count(*) as rows,
           sum(nominal_valid_votes) as votes,
           sum(zone_rows) as source_rows,
           bit_xor(hash(
               election_id, round_number, uf, municipality_code,
               office_code, is_transit_vote, candidate_id,
               nominal_valid_votes, zone_rows, zone_count
           )) as fingerprint
    from main.{MODEL}
    where cycle_id <> ?
    group by cycle_id, uf
    order by cycle_id, uf
"""


class CertificationError(RuntimeError):
    pass


def available_cycles(root: Path) -> set[str]:
    """Discover only successfully published candidate-vote data.parquet paths."""
    if not root.is_dir():
        raise CertificationError(f"Physical Parquet root not found: {root}")
    result: set[str] = set()
    for parquet in root.glob("election_type=*/year=*/data.parquet"):
        if not parquet.is_file():
            raise CertificationError(f"Broken or invalid published partition: {parquet}")
        parts = parquet.relative_to(root).parts
        kind = parts[0].removeprefix("election_type=")
        year_raw = parts[1].removeprefix("year=")
        if kind not in {"general", "municipal"} or not year_raw.isdigit():
            raise CertificationError(f"Unexpected physical partition: {parquet}")
        cycle = f"{int(year_raw)}:{kind}"
        if cycle in result:
            raise CertificationError(f"Duplicate physical partition: {cycle}")
        result.add(cycle)
    if not result:
        raise CertificationError(f"No published partitions found in {root}")
    return result


def require_cycle(root: Path, cycle: str) -> None:
    if cycle not in available_cycles(root):
        raise CertificationError(f"Cannot load missing published Parquet: {cycle}")


def validate_coverage(physical: set[str], materialized: set[str]) -> None:
    missing = sorted(physical - materialized)
    unexpected = sorted(materialized - physical)
    if missing or unexpected:
        raise CertificationError(
            f"History incomplete: missing={missing}; unexpected={unexpected}"
        )


@contextmanager
def guarded_cube_audit_locks(db_path: Path, root: Path, *, timeout: float = 30.0):
    """Hold shared warehouse and all published partition locks for an audit.

    The warehouse lock is acquired BEFORE discovering the physical cycles,
    preventing a cooperative publisher from introducing/removing cycles between
    discovery and certification. Partition locks are then acquired in sorted
    cycle order. The whole operation shares one monotonic timeout deadline.
    No missing publication directory is created. POSIX flock is cooperative.
    """
    # Reuse the existing GG-009 shared-fcntl mechanism, not a second lock API.
    from scripts.graph_guarded_read import _shared_flock

    if timeout < 0:
        raise CertificationError("--lock-timeout must be nonnegative")
    db_path = Path(db_path).resolve(strict=False)
    root = Path(root).resolve(strict=False)
    if not db_path.parent.is_dir():
        raise CertificationError(f"Warehouse directory does not exist: {db_path.parent}")
    deadline = time.monotonic() + timeout
    with ExitStack() as held:
        def acquire(path: Path) -> None:
            try:
                held.enter_context(_shared_flock(path, deadline))
            except TimeoutError as exc:
                raise TimeoutError(f"Timed out acquiring candidate audit lock: {path}") from exc

        acquire(db_path.with_name(db_path.name + ".build.lock"))
        # Destructive cleanup is now fenced. Recheck existence UNDER the lock.
        if not db_path.is_file():
            raise CertificationError(f"DuckDB path does not exist: {db_path}")
        physical = available_cycles(root)
        for cycle in sorted(physical):
            year, kind = cycle.split(":")
            partition = root / f"election_type={kind}" / f"year={year}"
            if not partition.is_dir():
                raise CertificationError(f"Published partition does not exist: {partition}")
            acquire(partition / ".publish.lock")
        if physical != available_cycles(root):
            raise CertificationError("Physical cycles changed during audit lock acquisition")
        yield physical


def db_connection(path: Path):
    import duckdb  # imported at runtime, not required for pure unit tests
    return duckdb.connect(str(path), read_only=True)


def materialized_cycles(db_path: Path) -> dict[str, tuple[int, int]]:
    with db_connection(db_path) as con:
        kind = con.execute("""
            select table_type from information_schema.tables
            where table_schema = 'main' and table_name = ?
        """, [MODEL]).fetchone()
        if kind != ("BASE TABLE",):
            raise CertificationError(f"Candidate cube is not a BASE TABLE: {kind}")
        rows = con.execute(f"""
            select cycle_id, count(*) as rows, sum(nominal_valid_votes) as votes
            from main.{MODEL} group by cycle_id order by cycle_id
        """).fetchall()
    if not rows:
        raise CertificationError("Materialized candidate cube is empty")
    cycles: dict[str, tuple[int, int]] = {}
    for cycle, count, votes in rows:
        if not cycle or cycle in cycles or count <= 0 or votes is None:
            raise CertificationError(f"Invalid materialized cycle: {cycle}")
        cycles[cycle] = (count, votes)
    return cycles


def snapshot_other_cycles(db_path: Path, exclude: str) -> list[tuple]:
    with db_connection(db_path) as con:
        kind = con.execute("""
            select table_type from information_schema.tables
            where table_schema = 'main' and table_name = ?
        """, [MODEL]).fetchone()
        if kind is None:  # first-ever incremental load
            return []
        if kind != ("BASE TABLE",):
            raise CertificationError(f"Refusing to replace non-table relation: {kind}")
        return con.execute(SNAPSHOT_SQL, [exclude]).fetchall()


def dbt_call(dbt: str, operation: str, db_path: Path, memory: str,
             vars_string: str, timeout_seconds: int, *, lock_fd: int) -> float:
    """Execute dbt while its process tree inherits the warehouse writer lock."""
    if lock_fd < 0:
        raise CertificationError("An acquired warehouse lock descriptor is required")
    env = os.environ.copy()
    env["TSE_DUCKDB_PATH"] = str(db_path)
    env["TSE_DUCKDB_MEMORY_LIMIT"] = memory
    env.pop("TSE_DUCKDB_TEMP_DIRECTORY", None)
    selection = [MODEL] if operation == "run" else list(TESTS)
    command = [
        "timeout", "--signal=INT", "--kill-after=30s", f"{timeout_seconds}s",
        dbt, operation,
        "--project-dir", str(PROJECT),
        "--profiles-dir", str(PROJECT),
        "--select", *selection,
        "--vars", vars_string,
    ]
    print("Executing:", " ".join(command), flush=True)
    start = time.monotonic()
    status = subprocess.run(command, env=env, check=False,
                            pass_fds=(lock_fd,), start_new_session=True).returncode
    duration = round(time.monotonic() - start, 2)
    if status != 0:
        raise CertificationError(
            f"dbt {operation} failed (exit={status}, elapsed={duration}s). "
            "No subsequent certification steps were executed."
        )
    return duration


def report_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwriting of past evidence.
    with path.open("x", encoding="utf8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def audit(db_path: Path, root: Path) -> dict:
    physical = available_cycles(root)
    materialized = materialized_cycles(db_path)
    validate_coverage(physical, set(materialized))
    print("PASS: complete candidate-cube cycle coverage")
    for cycle, (count, votes) in sorted(materialized.items()):
        print(f"  {cycle}: {count:,} rows; {votes:,} nominal votes")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "db": str(db_path), "source_root": str(root),
        "coverage": "complete", "cycles": materialized,
        "scope": "physical-publication-versus-materialization; not source freshness",
    }


def guarded_freshness_audit(db_path: Path, root: Path, attestations_dir: Path,
                            *, timeout: float = 30.0) -> dict:
    """Authorize all published cube cycles from one cooperatively fenced view."""
    with guarded_cube_audit_locks(db_path, root, timeout=timeout) as physical:
        coverage = audit(db_path, root)
        if set(coverage["cycles"]) != physical:
            raise CertificationError("Published cycle coverage changed during audit")
        freshness = audit_freshness(db_path, root, attestations_dir, physical)
        if physical != available_cycles(root):
            raise CertificationError("Physical cycles changed during freshness audit")
        print("PASS: all published candidate cycles have current attestation")
        return {**coverage, **freshness}


def certify(args: argparse.Namespace, db_path: Path, root: Path) -> dict:
    if args.year <= 0:
        raise CertificationError("Year must be positive")
    # A mutating certification must never bypass revocation/reattestation.
    # Otherwise a certificate issued by an earlier run could remain active.
    if getattr(args, "attestations_dir", None) is None:
        raise CertificationError("--attestations-dir is required for certify")
    timeout = getattr(args, "lock_timeout", 300.0)
    if timeout < 0:
        raise CertificationError("--lock-timeout must be nonnegative")

    # Capture, revocation, dbt run/test and final proof share ONE exclusive
    # lock; never nest the coordinated dbt gateway here (it would deadlock).
    from scripts.build_candidate_fact_partition import warehouse_write_lock
    with warehouse_write_lock(db_path, timeout=timeout) as handle:
        return _certify_locked(args, db_path, root, lock_fd=handle.fileno())


def _certify_locked(args: argparse.Namespace, db_path: Path, root: Path,
                    *, lock_fd: int) -> dict:
    cycle = f"{args.year}:{args.election_type}"
    require_cycle(root, cycle)
    before = snapshot_other_cycles(db_path, cycle)
    # Capture the publication and both durable dimensions before any write.
    # A failed/interrupted refresh must not leave an active certificate behind.
    before_inputs = (
        input_signature(db_path, root, cycle)
        if getattr(args, "attestations_dir", None) is not None else None
    )
    if before_inputs is not None:
        revoke_certificate(args.attestations_dir, cycle)
    vars_string = (
        "{incremental_years: [" + str(args.year) + "], "
        "incremental_election_types: [" + args.election_type + "]}"
    )
    print(f"Certifying {cycle} in {db_path}", flush=True)
    run_seconds = dbt_call(
        args.dbt, "run", db_path, args.memory_limit, vars_string,
        args.run_timeout, lock_fd=lock_fd,
    )
    after = snapshot_other_cycles(db_path, cycle)
    if before != after:
        raise CertificationError("Unselected cycles changed during incremental load")
    cycles = materialized_cycles(db_path)
    if cycle not in cycles:
        raise CertificationError(f"Loaded cycle missing or empty: {cycle}")
    # Ensure the model did not create an unexpected extra cycle.
    other_before = {row[0] for row in before}
    if set(cycles) - {cycle} != other_before:
        raise CertificationError("Unselected cycle set changed unexpectedly")
    print(f"PASS: other cycles preserved; {cycle}: {cycles[cycle]}", flush=True)
    before_output = output_signature(db_path, cycle) if before_inputs is not None else None
    test_seconds = dbt_call(
        args.dbt, "test", db_path, args.memory_limit, vars_string,
        args.test_timeout, lock_fd=lock_fd,
    )
    if snapshot_other_cycles(db_path, cycle) != before:
        raise CertificationError("Unselected cycles changed during tests")
    current = materialized_cycles(db_path)
    if current.get(cycle) != cycles[cycle]:
        raise CertificationError("Selected cycle totals changed during tests")
    if before_inputs is not None:
        if input_signature(db_path, root, cycle) != before_inputs:
            raise CertificationError("Source Parquet or dimensions changed during certification")
        if output_signature(db_path, cycle) != before_output:
            raise CertificationError("Selected cube changed during certification tests")
        issue_certificate(args.attestations_dir, db_path, root, cycle,
                          before_inputs, before_output)
    report = {
        "certified_at_utc": datetime.now(timezone.utc).isoformat(),
        "db": str(db_path), "source_root": str(root),
        "cycle": cycle, "rows": current[cycle][0], "nominal_votes": current[cycle][1],
        "run_seconds": run_seconds, "test_seconds": test_seconds,
        "tests": list(TESTS), "unselected_cycles_preserved": True,
        "scope": ("dbt cycle-scoped contracts with freshness attestation" if before_inputs
                  is not None else "dbt cycle-scoped contracts; not source freshness"),
        "freshness_attested": before_inputs is not None,
    }
    print(f"PASS: {cycle} certified; run={run_seconds}s test={test_seconds}s")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True,
                        help="EXPLICIT canary DuckDB path; no production default")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Published Parquet root containing election_type=*/year=*/data.parquet")
    parser.add_argument("--report", type=Path,
                        help="Write structured evidence to a NEW JSON path")
    parser.add_argument("--attestations-dir", type=Path,
                        help="Directory for per-cycle freshness certificates; required for audit-freshness")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("audit", help="Read-only: check every physical cycle has a nonempty table partition")
    freshness_audit = sub.add_parser(
        "audit-freshness", help="Cooperatively guarded freshness of all published cube cycles"
    )
    freshness_audit.add_argument("--lock-timeout", type=float, default=30.0)
    one = sub.add_parser("certify", help="Mutates selected cube cycle and runs exactly two dbt tests")
    one.add_argument("--year", type=int, required=True)
    one.add_argument("--election-type", choices=("general", "municipal"), required=True)
    one.add_argument("--dbt", default="dbt")
    one.add_argument("--memory-limit", default="4GB")
    one.add_argument("--run-timeout", type=int, default=600)
    one.add_argument("--test-timeout", type=int, default=600)
    one.add_argument("--lock-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    db_path = args.db.expanduser().resolve(strict=False)
    root = args.source_root.expanduser().resolve(strict=False)
    try:
        if not db_path.is_file():
            raise CertificationError(f"DuckDB path does not exist: {db_path}")
        if args.action == "audit":
            result = audit(db_path, root)
        elif args.action == "audit-freshness":
            if args.attestations_dir is None:
                raise CertificationError("--attestations-dir is required for audit-freshness")
            result = guarded_freshness_audit(
                db_path, root, args.attestations_dir, timeout=args.lock_timeout
            )
        else:
            if args.run_timeout <= 0 or args.test_timeout <= 0:
                raise CertificationError("Timeouts must be positive")
            result = certify(args, db_path, root)
        if args.report:
            report_json(args.report, result)
            print(f"Evidence: {args.report}")
        return 0
    except (CertificationError, FreshnessError, OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
