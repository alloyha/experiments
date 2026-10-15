#!/usr/bin/env python3
"""Run warehouse-writing dbt commands under the existing POSIX warehouse lock.

This is the Makefile's dbt gateway. Writers share the same stable lock inode
used by build_candidate_fact_partition and graph_guarded_read. The lock FD is
inherited by dbt so the parent dying cannot let readers pass while dbt writes.

Cooperative guarantee only: invoking dbt/DuckDB outside this gateway bypasses
flock. GG-009 therefore remains planned until all write surfaces are fenced.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# Direct CLI execution sets sys.path[0] to scripts/, not its parent.
# Resolve imports independently of the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# dbt deps/parse/compile do not mutate warehouse tables. A command not on this
# list is treated as a potential writer, including dbt run-operation.
NO_WAREHOUSE_WRITE = frozenset({"deps", "parse", "compile", "debug", "ls", "list"})


def _latest_results(artifact_root: Path, run_target: Path) -> None:
    """Publish only this warehouse's most recent run_results atomically."""
    latest = artifact_root / "latest_run_results.json"
    latest.unlink(missing_ok=True)  # never serve stale results after a failed run
    results = run_target / "run_results.json"
    if results.is_file():
        temporary = artifact_root / f".latest-{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(results, temporary)
            os.replace(temporary, latest)
        finally:
            temporary.unlink(missing_ok=True)


def coordinated_dbt(*, db: Path, dbt_executable: Path, artifacts_root: Path,
                    dbt_args: list[str], lock_timeout: float = 300.0) -> int:
    """Preserve dbt arguments while preventing same-warehouse writer overlap."""
    if not dbt_args or dbt_args[0].startswith("-"):
        raise ValueError("A dbt subcommand must be supplied after --")
    if lock_timeout < 0:
        raise ValueError("--lock-timeout must be nonnegative")
    executable = Path(dbt_executable).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"dbt executable is not executable: {executable}")
    warehouse = Path(db).resolve()
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["TSE_DUCKDB_PATH"] = str(warehouse)
    # Remove obsolete custom DuckDB spill override; keep DuckDB's own default.
    env.pop("TSE_DUCKDB_TEMP_DIRECTORY", None)

    if dbt_args[0] in NO_WAREHOUSE_WRITE:
        return subprocess.run([str(executable), *dbt_args], env=env, check=False).returncode

    # A caller cannot accidentally redirect target/logs into another worker's
    # output or reuse a stale manifest; the wrapper alone owns these paths.
    if any(arg in {"--target-path", "--log-path"} or
           arg.startswith(("--target-path=", "--log-path=")) for arg in dbt_args):
        raise ValueError("Writer target/log paths are owned by the coordinated gateway")
    from scripts.build_candidate_fact_partition import warehouse_write_lock
    root = Path(artifacts_root).resolve()
    with warehouse_write_lock(warehouse, timeout=lock_timeout) as handle:
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / uuid.uuid4().hex
        run_dir.mkdir(mode=0o700)
        target = run_dir / "target"
        logs = run_dir / "logs"
        target.mkdir()
        logs.mkdir()
        (root / "latest_run_results.json").unlink(missing_ok=True)
        command = [str(executable), *dbt_args,
                   "--target-path", str(target), "--log-path", str(logs)]
        completed = subprocess.run(command, env=env, check=False,
                                   pass_fds=(handle.fileno(),),
                                   start_new_session=True)
        # Failed runs retain their UUID directory but never become the latest
        # performance baseline. The old pointer was revoked at run start.
        if completed.returncode == 0:
            _latest_results(root, target)
        return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--dbt-executable", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument("dbt_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.dbt_args[1:] if args.dbt_args[:1] == ["--"] else args.dbt_args
    try:
        return coordinated_dbt(db=args.db, dbt_executable=args.dbt_executable,
                               artifacts_root=args.artifacts_root,
                               dbt_args=command, lock_timeout=args.lock_timeout)
    except (ValueError, OSError, TimeoutError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
