#!/usr/bin/env python3
"""Publish one candidate fact partition with POSIX writer fencing.

The public ``data.parquet`` path is an atomically replaced symlink to a durable,
immutable generation file in the same directory. This keeps the existing dbt
read_parquet glob compatible and makes the active generation self-describing.

Requires a local/POSIX filesystem with advisory flock, atomic rename and fsync.
All publishers must cooperate with the lock; readers need no lock.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# Direct CLI execution adds scripts/ to sys.path rather than the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_TABLE = "main.int_candidate_fact_partition"
DEFAULT_LOCK_TIMEOUT = 30.0


def current_generation(partition: Path) -> str:
    """Identify the currently published generation without a sidecar manifest.

    ``none`` and ``legacy`` cover an absent publication and the PR35 regular-file
    layout. A generation string is the basename of the immutable Parquet target.
    """
    path = partition / "data.parquet"
    if path.is_symlink():
        link = os.readlink(path)
        if Path(link).is_absolute() or Path(link).name != link:
            raise RuntimeError(f"Invalid published generation pointer: {link}")
        if not (partition / link).is_file():
            raise RuntimeError(f"Published generation is missing: {link}")
        return link
    if path.exists():
        if not path.is_file():
            raise RuntimeError(f"Published partition is not a regular file: {path}")
        return "legacy"
    return "none"


@contextmanager
def partition_write_lock(partition: Path, *, timeout: float):
    """Obtain an OS-released, process-scoped advisory lock for this partition."""
    if timeout < 0:
        raise ValueError("lock timeout must be nonnegative")
    partition.mkdir(parents=True, exist_ok=True)
    lock_file = partition / ".publish.lock"
    deadline = time.monotonic() + timeout
    with lock_file.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring partition lock: {lock_file}")
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def clear_abandoned_staging(partition: Path) -> None:
    """Only delete unpublished temporary objects while holding the lock.

    Old immutable *.parquet files are intentionally retained: a reader could
    already have resolved an older generation before the pointer was switched.
    """
    for pattern in (".generation-*.tmp", ".data-parquet-*.link",
                    ".zone-proof-*.tmp"):
        for path in partition.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()


def validate_staged_parquet(
    con: duckdb.DuckDBPyConnection, staged: Path, expected_rows: int
) -> None:
    """Reject incomplete Parquet exports before publication."""
    if not staged.is_file():
        raise RuntimeError("Temporary parquet was not created")

    rows = con.execute(
        "select count(*) from read_parquet(?, hive_partitioning = false)",
        [str(staged)],
    ).fetchone()[0]
    if rows != expected_rows:
        raise RuntimeError(
            f"Staged parquet row-count mismatch: source={expected_rows}, parquet={rows}"
        )

    source_schema = con.execute(f"select * from {SOURCE_TABLE} limit 0").description
    staged_schema = con.execute(
        "select * from read_parquet(?, hive_partitioning = false) limit 0",
        [str(staged)],
    ).description
    source_fields = [(col[0], str(col[1])) for col in source_schema]
    staged_fields = [(col[0], str(col[1])) for col in staged_schema]
    if source_fields != staged_fields:
        raise RuntimeError(
            f"Staged parquet schema mismatch: source={source_fields}, parquet={staged_fields}"
        )


# Scoped physical partitions have exactly one cycle, so cycle_id is constant.
# The remaining seven candidate-rollup keys plus zone are required verbatim.
CANDIDATE_ZONE_KEYS = (
    "election_id", "round_number", "uf", "municipality_code",
    "office_code", "is_transit_vote", "candidate_id", "zone",
)
ZONE_UNIQUENESS_SQL = """WITH occurrences AS (
    SELECT election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote, candidate_id, zone,
           COUNT(*)::HUGEINT AS n
    FROM read_parquet(?, hive_partitioning=false)
    GROUP BY ALL
)
SELECT COALESCE(SUM(CASE WHEN zone IS NULL THEN n ELSE 0 END), 0)::HUGEINT,
       COALESCE(SUM(CASE WHEN zone IS NOT NULL THEN n - 1 ELSE 0 END), 0)::HUGEINT,
       COALESCE(SUM(n), 0)::HUGEINT,
       COUNT(*)::HUGEINT
FROM occurrences"""


def prove_staged_zone_uniqueness(
    con: duckdb.DuckDBPyConnection, staged: Path, expected_rows: int,
) -> dict:
    """Exact candidate group/zone gate on staged physical bytes, before swap.

    Fail closed for absent fields, null zones, duplicate group/zone keys, or
    inconsistent row inventory. Never use COUNT(*) as a fallback for bad data.
    """
    columns = {field[0] for field in con.execute(
        "SELECT * FROM read_parquet(?, hive_partitioning=false) LIMIT 0",
        [str(staged)],
    ).description}
    missing = sorted(set(CANDIDATE_ZONE_KEYS) - columns)
    if missing:
        raise RuntimeError(f"Candidate zone publication schema missing: {missing}")
    null_rows, duplicate_rows, total_rows, group_zone_keys = map(int, con.execute(
        ZONE_UNIQUENESS_SQL, [str(staged)],
    ).fetchone())
    if total_rows != expected_rows or total_rows <= 0 or group_zone_keys <= 0:
        raise RuntimeError("Candidate zone publication inventory mismatch")
    if null_rows or duplicate_rows:
        raise RuntimeError(
            "Candidate zone uniqueness rejected publication: "
            f"null_zone_rows={null_rows}, duplicate_nonnull_zone_rows={duplicate_rows}"
        )
    return {
        "source_rows": total_rows,
        "distinct_group_zone_keys": group_zone_keys,
        "null_zone_rows": null_rows,
        "duplicate_nonnull_zone_rows": duplicate_rows,
    }


def persist_immutable_zone_proof(
    partition: Path, immutable: Path, *, year: int, election_type: str,
    inventory: dict,
) -> Path:
    """Create-only, fsynced evidence bound to one immutable Parquet generation.

    This is a source-publishing prerequisite, NOT graph/cube certification.
    The sidecar is created *before* the active symlink is atomically switched.
    An interrupted publisher can leave an orphan generation and sidecar, never
    a half-validated active publication.
    """
    digest = hashlib.sha256()
    with immutable.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    record = {
        "schema_version": 1,
        "scope": "candidate_physical_zone_uniqueness_not_graph_freshness",
        "result": "PASS",
        "cycle": f"{year}:{election_type}",
        "generation": immutable.name,
        "parquet_sha256": digest.hexdigest(),
        "guard_sql_sha256": hashlib.sha256(ZONE_UNIQUENESS_SQL.encode()).hexdigest(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        **inventory,
    }
    target = immutable.with_suffix(".zone-uniqueness.json")
    temporary = partition / f".zone-proof-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # create-only: never overwrite a proof
        fsync_directory(partition)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _publish_partition_locked(
    db_path: Path,
    root: Path,
    year: int,
    election_type: str,
    *,
    expected_generation: str | None = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Path:
    """Publish under a per-partition lock and compare-and-swap generation gate.

    If an expected generation is not supplied, capture one before acquiring
    the lock. Explicit expected_generation can be captured by an orchestrator
    *before* preparing a source snapshot to detect a stale later publication.
    """
    if year < 1 or election_type not in {"general", "municipal"}:
        raise ValueError("A positive year and general/municipal election type are required")
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB warehouse not found: {db_path}")

    partition = root / f"election_type={election_type}" / f"year={year}"
    observed_generation = (
        current_generation(partition)
        if expected_generation is None
        else expected_generation
    )
    final_path = partition / "data.parquet"
    token = uuid.uuid4().hex
    staged = partition / f".generation-{token}.tmp"
    immutable = partition / f"generation-{token}.parquet"
    link_tmp = partition / f".data-parquet-{token}.link"

    with partition_write_lock(partition, timeout=lock_timeout):
        active = current_generation(partition)
        if active != observed_generation:
            raise RuntimeError(
                "Published generation changed before lock acquisition: "
                f"expected={observed_generation}, actual={active}"
            )
        clear_abandoned_staging(partition)
        try:
            with duckdb.connect(str(db_path)) as con:
                try:
                    cycles = con.execute(f"""
                        select distinct election_year, election_type
                        from {SOURCE_TABLE}
                        order by 1, 2
                    """).fetchall()
                except duckdb.CatalogException as exc:
                    raise RuntimeError(
                        f"Missing dbt physical source table {SOURCE_TABLE}; "
                        "run the enabled int_candidate_fact_partition model first"
                    ) from exc

                row_count = con.execute(
                    f"select count(*) from {SOURCE_TABLE}"
                ).fetchone()[0]
                if not row_count:
                    raise RuntimeError("Refusing to publish an empty candidate fact partition")
                expected = [(year, election_type)]
                if cycles != expected:
                    raise RuntimeError(
                        f"{SOURCE_TABLE} is not scoped to the requested partition: "
                        f"expected={expected}, actual={cycles}"
                    )

                escaped = str(staged).replace("'", "''")
                con.execute(f"""
                    copy (select * from {SOURCE_TABLE})
                    to '{escaped}' (format parquet, compression zstd)
                """)
                validate_staged_parquet(con, staged, row_count)
                # Mandatory proof on the actual exported bytes; any failure
                # leaves the previous public pointer untouched.
                zone_proof = prove_staged_zone_uniqueness(con, staged, row_count)

            fsync_file(staged)
            os.replace(staged, immutable)
            fsync_directory(partition)
            persist_immutable_zone_proof(
                partition, immutable, year=year,
                election_type=election_type, inventory=zone_proof,
            )

            # Preserve the PR35 regular file as an immutable, hard-linked
            # generation while migrating its name to the new pointer layout.
            if active == "legacy":
                os.link(final_path, partition / f"generation-legacy-{uuid.uuid4().hex}.parquet")
                fsync_directory(partition)

            link_tmp.symlink_to(immutable.name)
            fsync_directory(partition)
            os.replace(link_tmp, final_path)
            fsync_directory(partition)
        finally:
            staged.unlink(missing_ok=True)
            link_tmp.unlink(missing_ok=True)

    print(f"candidate fact partition replaced: {year}/{election_type}")
    print(f"rows: {row_count}")
    print(f"generation: {immutable.name}")
    print(f"path: {final_path}")
    return final_path


def publish_partition(
    db_path: Path,
    root: Path,
    year: int,
    election_type: str,
    *,
    expected_generation: str | None = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    warehouse_lock_timeout: float = 300.0,
    borrowed_warehouse_lock_fd: int | None = None,
) -> Path:
    """Public publication gate: warehouse -> partition, never the reverse.

    ``borrowed_warehouse_lock_fd`` is an internal coordinator capability. The
    FD must refer to this warehouse's existing lock inode, held by the caller.
    It must NEVER be used by standalone callers to bypass coordination.
    """
    db_path = Path(db_path).resolve()
    if year < 1 or election_type not in {"general", "municipal"}:
        raise ValueError("A positive year and general/municipal election type are required")
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB warehouse not found: {db_path}")
    if warehouse_lock_timeout < 0:
        raise ValueError("warehouse lock timeout must be nonnegative")

    def publish() -> Path:
        return _publish_partition_locked(
            db_path, root, year, election_type,
            expected_generation=expected_generation, lock_timeout=lock_timeout,
        )

    if borrowed_warehouse_lock_fd is not None:
        lock_path = db_path.with_name(db_path.name + ".build.lock")
        try:
            actual_inode = os.fstat(borrowed_warehouse_lock_fd)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid borrowed warehouse lock descriptor") from exc
        try:
            expected_inode = lock_path.stat()
        except FileNotFoundError as exc:
            # A borrowed FD for some other inode must not manufacture a lock.
            raise RuntimeError("Borrowed warehouse lock does not match DuckDB path") from exc
        if not os.path.samestat(actual_inode, expected_inode):
            raise RuntimeError("Borrowed warehouse lock does not match DuckDB path")
        return publish()

    # Import at invocation time to avoid circular imports: the build coordinator
    # itself imports this publisher and calls it with its already-held FD.
    from scripts.build_candidate_fact_partition import warehouse_write_lock
    with warehouse_write_lock(db_path, timeout=warehouse_lock_timeout):
        return publish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/warehouse/tse_analytics.duckdb")
    parser.add_argument("--root", default="data/warehouse/fact_candidate_votes")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--election-type", choices=["general", "municipal"], required=True
    )
    parser.add_argument(
        "--expected-generation",
        help="Generation observed before source preparation (none/legacy/file basename)",
    )
    parser.add_argument("--lock-timeout", type=float, default=DEFAULT_LOCK_TIMEOUT)
    parser.add_argument("--warehouse-lock-timeout", type=float, default=300.0)
    parser.add_argument(
        "--show-generation", action="store_true",
        help="Print current publication token without reading or publishing the DuckDB source",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.show_generation:
        print(current_generation(root / f"election_type={args.election_type}" / f"year={args.year}"))
        return
    publish_partition(
        Path(args.db).resolve(), root, args.year, args.election_type,
        expected_generation=args.expected_generation,
        lock_timeout=args.lock_timeout,
        warehouse_lock_timeout=args.warehouse_lock_timeout,
    )


if __name__ == "__main__":
    main()
