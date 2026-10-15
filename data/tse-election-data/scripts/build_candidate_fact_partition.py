#!/usr/bin/env python3
"""Coordinate a single DuckDB CTAS -> candidate partition publication.

A warehouse-scoped POSIX flock spans preparation and publication. The lock's
file descriptor is inherited by the dbt subprocess so SIGKILL of this parent
cannot release the lock while that subprocess still writes the warehouse.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import write_candidate_fact_partition as publisher

DEFAULT_WAREHOUSE_LOCK_TIMEOUT = 300.0


@contextmanager
def warehouse_write_lock(db_path: Path, *, timeout: float):
    """Exclusive cooperative lock keyed to the canonical DuckDB filename.

    Use a separate lock inode rather than the DuckDB file: dbt may create or
    replace the database, and a lock on that inode could become stale.
    """
    if timeout < 0:
        raise ValueError("warehouse lock timeout must be nonnegative")
    db_path = db_path.resolve()
    lock_path = db_path.with_name(db_path.name + ".build.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with lock_path.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring warehouse lock: {lock_path}")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_and_publish(
    *,
    db_path: Path,
    fact_root: Path,
    raw_root: Path,
    project_dir: Path,
    profiles_dir: Path,
    dbt_executable: Path,
    year: int,
    election_type: str,
    warehouse_lock_timeout: float = DEFAULT_WAREHOUSE_LOCK_TIMEOUT,
    publication_lock_timeout: float = publisher.DEFAULT_LOCK_TIMEOUT,
    artifacts_root: Path | None = None,
) -> Path:
    """Serialize coordinated jobs on one DuckDB file, not globally.

    No cross-store ACID transaction is promised. Failures before the publication
    step leave the active physical generation untouched.
    """
    if year < 1 or election_type not in {"general", "municipal"}:
        raise ValueError("year must be positive and election type general/municipal")
    db_path = Path(db_path).resolve()
    fact_root = Path(fact_root).resolve()
    raw_root = Path(raw_root).resolve()
    project_dir = Path(project_dir).resolve()
    profiles_dir = Path(profiles_dir).resolve()
    dbt_executable = Path(dbt_executable).resolve()
    artifacts_root = (
        Path(artifacts_root).resolve() if artifacts_root is not None
        else db_path.with_name(f".{db_path.name}.dbt-runs")
    )
    if not dbt_executable.is_file() or not os.access(dbt_executable, os.X_OK):
        raise FileNotFoundError(f"dbt executable missing or not executable: {dbt_executable}")

    # Unique directories prevent concurrent jobs against independent warehouses
    # from sharing dbt manifests, compiled SQL, logs or DuckDB spill files.
    artifacts_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts_root / uuid.uuid4().hex
    run_dir.mkdir(mode=0o700)
    target_dir = run_dir / "target"
    log_dir = run_dir / "logs"
    # DuckDB uses <db_path>.tmp by default. Its warehouse lock serializes
    # writers to one database; dbt artifacts remain isolated per run.
    for directory in (target_dir, log_dir):
        directory.mkdir()

    partition = fact_root / f"election_type={election_type}" / f"year={year}"
    variables = {
        "tse_raw_root": str(raw_root),
        "candidate_fact_root": str(fact_root),
        "election_years": [year],
        "election_types": [election_type],
        "incremental_years": [year],
        "incremental_election_types": [election_type],
        "use_prepared_candidate_votes": True,
        "build_candidate_fact_partition": True,
    }
    command = [
        str(dbt_executable), "run",
        "--project-dir", str(project_dir),
        "--profiles-dir", str(profiles_dir),
        "--select", "+int_candidate_fact_partition",
        "--target-path", str(target_dir),
        "--log-path", str(log_dir),
        "--vars", json.dumps(variables, separators=(",", ":")),
    ]
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["TSE_DUCKDB_PATH"] = str(db_path)
    # Do not expose a misleading override now that the dbt profile relies
    # on DuckDB's per-database default; avoids cursor SET failures on spill.
    env.pop("TSE_DUCKDB_TEMP_DIRECTORY", None)

    with warehouse_write_lock(db_path, timeout=warehouse_lock_timeout) as lock:
        # Observe the active generation only after acquiring the warehouse lock.
        expected_generation = publisher.current_generation(partition)
        # POSIX fd inheritance is essential: if the coordinator is SIGKILLed,
        # dbt retains the flock until it exits. Execute the dbt venv binary
        # directly; uv wrappers may close unrecognized descriptors.
        subprocess.run(
            command, check=True, cwd=str(project_dir.parent), env=env,
            pass_fds=(lock.fileno(),), start_new_session=True,
        )
        return publisher.publish_partition(
            db_path, fact_root, year, election_type,
            expected_generation=expected_generation,
            lock_timeout=publication_lock_timeout,
            borrowed_warehouse_lock_fd=lock.fileno(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--dbt-executable", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--election-type", choices=["general", "municipal"], required=True)
    parser.add_argument("--warehouse-lock-timeout", type=float, default=DEFAULT_WAREHOUSE_LOCK_TIMEOUT)
    parser.add_argument("--publication-lock-timeout", type=float, default=publisher.DEFAULT_LOCK_TIMEOUT)
    args = parser.parse_args()
    build_and_publish(
        db_path=args.db, fact_root=args.root, raw_root=args.raw_root,
        project_dir=args.project_dir, profiles_dir=args.profiles_dir,
        dbt_executable=args.dbt_executable, year=args.year,
        election_type=args.election_type,
        warehouse_lock_timeout=args.warehouse_lock_timeout,
        publication_lock_timeout=args.publication_lock_timeout,
        artifacts_root=args.artifacts_root,
    )


if __name__ == "__main__":
    main()
