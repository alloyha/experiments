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
import hashlib
import json
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



# C3_ENFORCEMENT_V2: resolve the actual model graph before any dbt writer.
C3_MODELS = frozenset({"silver_tally_munzona", "fact_tally_munzona"})
SELECTION_FLAGS = frozenset({"--select", "-s", "--exclude", "--selector", "--indirect-selection"})
VALUE_FLAGS = frozenset({"--project-dir", "--profiles-dir", "--vars", "--target"}) | SELECTION_FLAGS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_reference_evidence(evidence_path: Path, reference_db: Path, raw_root: Path) -> None:
    """Authenticate recorded bytes and active source identity; not a signed build attestation."""
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != 1:
        raise ValueError("C3 evidence schema_version must be 1")
    if Path(evidence.get("reference_database", "")).resolve() != reference_db.resolve():
        raise ValueError("C3 evidence reference_database mismatch")
    if _sha256(reference_db) != evidence.get("reference_sha256"):
        raise ValueError("C3 reference database SHA-256 mismatch")
    # The verifier's return contract is kept separate from this gateway: rerun
    # the project's validated CLI (and require the JSON PASS status).
    verifier = Path(__file__).with_name("verify_c3_sources.py")
    result = subprocess.run([sys.executable, str(verifier), "--root", str(raw_root)],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError("C3 source verification failed: " + result.stderr[-500:])
    report = json.loads(result.stdout)
    if report.get("status") != "PASS":
        raise ValueError("C3 source integrity report is not PASS")
    declared = evidence.get("cycles")
    actual = {r["cycle"]: r["sha256"] for r in report["cycles"]}
    if not isinstance(declared, dict) or set(actual) != set(declared):
        raise ValueError("C3 evidence cycle coverage mismatch")
    if any(declared[k].get("source_sha256") != v for k, v in actual.items()):
        raise ValueError("C3 evidence source SHA-256 mismatch")


def _resolve_selected_models(executable: Path, args: list[str], env: dict) -> set[str]:
    """Ask dbt itself to resolve selectors; never parse tags/graph operators as names."""
    if args[0] not in {"run", "build"}:
        raise ValueError("Uncertified dbt writer subcommand: " + args[0])
    forwarded = []
    i = 1
    while i < len(args):
        flag = args[i]
        name = flag.split("=", 1)[0]
        if name in {"--select", "-s", "--exclude"}:
            # dbt supports multiple space-separated selectors per flag.
            forwarded.append(flag)
            if "=" not in flag:
                initial = i
                while i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 1
                    forwarded.append(args[i])
                if i == initial:
                    raise ValueError("Missing dbt option value: " + name)
            elif not flag.split("=", 1)[1]:
                raise ValueError("Missing dbt option value: " + name)
        elif name in VALUE_FLAGS:
            if "=" in flag:
                if not flag.split("=", 1)[1]:
                    raise ValueError("Missing dbt option value: " + name)
                forwarded.append(flag)
            else:
                if i + 1 >= len(args) or args[i + 1].startswith("-"):
                    raise ValueError("Missing dbt option value: " + name)
                forwarded.extend((flag, args[i + 1]))
                i += 1
        elif flag in {"--full-refresh", "-f"}:
            raise ValueError("Full refresh requires separate C3 certification policy")
        elif flag in {"--fail-fast", "--store-failures"}:
            pass
        elif flag.startswith("-"):
            # Unsupported flags might change graph resolution. Deny, do not guess.
            raise ValueError("Unsupported dbt writer option for C3 resolution: " + flag)
        else:
            raise ValueError("Unexpected dbt argument: " + flag)
        i += 1
    command = [str(executable), "ls", "--resource-type", "model", "--output", "json", *forwarded]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError("dbt ls could not resolve the publication graph: " + result.stderr[-600:])
    selected = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("resource_type") == "model" and isinstance(row.get("name"), str):
            selected.add(row["name"])
    if not selected:
        raise ValueError("dbt ls resolved no models or returned no machine-readable selection")
    return selected


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
                    dbt_args: list[str], lock_timeout: float = 300.0,
                    c3_reference_db: Path | None = None,
                    c3_reference_raw_root: Path | None = None,
                    c3_candidate_raw_root: Path | None = None,
                    c3_evidence: Path | None = None) -> int:
    """Preserve dbt arguments while preventing same-warehouse writer overlap."""
    if not dbt_args or dbt_args[0].startswith("-"):
        raise ValueError("A dbt subcommand must be supplied after --")
    if lock_timeout < 0:
        raise ValueError("--lock-timeout must be nonnegative")
    executable = Path(dbt_executable).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"dbt executable is not executable: {executable}")
    c3_fields = (c3_reference_db, c3_reference_raw_root, c3_candidate_raw_root, c3_evidence)
    if any(x is not None for x in c3_fields) and not all(x is not None for x in c3_fields):
        raise ValueError("C3 preflight requires reference DB, both raw roots and evidence")
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
    # Keep the generic GG-009 writer gateway compatible with unrelated dbt
    # projects. Resolve C3 only for this repository's TSE dbt project, or
    # when the caller explicitly supplies C3 evidence. A missing --project-dir
    # uses dbt's default project resolution; treat it as TSE conservatively.
    tse_project = (Path(__file__).resolve().parents[1] / "tse_dbt").resolve()
    explicit_project = None
    for index, argument in enumerate(dbt_args[1:], 1):
        if argument == "--project-dir":
            if index + 1 >= len(dbt_args):
                raise ValueError("Missing --project-dir value")
            explicit_project = Path(dbt_args[index + 1]).resolve()
            break
        if argument.startswith("--project-dir="):
            explicit_project = Path(argument.partition("=")[2]).resolve()
            break
    resolve_c3 = explicit_project is None or explicit_project == tse_project or c3_reference_db is not None
    selected = _resolve_selected_models(executable, dbt_args, env) if resolve_c3 else set()
    targets_c3 = bool(selected & C3_MODELS)
    if targets_c3 and c3_reference_db is None:
        raise ValueError("C3 protected model selected without mandatory reference evidence")
    if c3_reference_db is not None and not targets_c3:
        raise ValueError("C3 evidence provided to a non-C3 publication")
    from scripts.build_candidate_fact_partition import warehouse_write_lock
    root = Path(artifacts_root).resolve()
    with warehouse_write_lock(warehouse, timeout=lock_timeout) as handle:
        if targets_c3:
            assert c3_evidence is not None and c3_reference_db is not None
            _verify_reference_evidence(Path(c3_evidence), Path(c3_reference_db),
                                       Path(c3_reference_raw_root))
            # C3_SCOPE_BIND_V4: pin actual dbt vars to certified source/cycles.
            from scripts.c3_scope_policy import validate_c3_scope
            validate_c3_scope(
                dbt_args,
                evidence_path=Path(c3_evidence),
                raw_root=Path(c3_candidate_raw_root),
                calendar_csv=Path(__file__).resolve().parents[1]
                / "tse_dbt/seeds/election_calendar.csv",
            )
            # Preflight runs under the SAME lock as the dbt writer. Its inputs are
            # explicitly experimental until reference provenance is certified.
            certifier = Path(__file__).with_name("certify_tally_bronze.py")
            preflight = subprocess.run([
                sys.executable, str(certifier),
                "--reference-db", str(Path(c3_reference_db).resolve()),
                "--candidate-db", str(warehouse),
                "--reference-raw-root", str(Path(c3_reference_raw_root).resolve()),
                "--candidate-raw-root", str(Path(c3_candidate_raw_root).resolve()),
            ], check=False, pass_fds=(handle.fileno(),))
            if preflight.returncode != 0:
                print("FAIL: C3 Bronze parity preflight rejected dbt run", file=sys.stderr)
                return preflight.returncode
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
    parser.add_argument("--c3-reference-db", type=Path)
    parser.add_argument("--c3-reference-raw-root", type=Path)
    parser.add_argument("--c3-candidate-raw-root", type=Path)
    parser.add_argument("--c3-evidence", type=Path)
    parser.add_argument("dbt_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.dbt_args[1:] if args.dbt_args[:1] == ["--"] else args.dbt_args
    try:
        return coordinated_dbt(db=args.db, dbt_executable=args.dbt_executable,
                               artifacts_root=args.artifacts_root,
                               dbt_args=command, lock_timeout=args.lock_timeout,
                               c3_reference_db=args.c3_reference_db,
                               c3_reference_raw_root=args.c3_reference_raw_root,
                               c3_candidate_raw_root=args.c3_candidate_raw_root,
                               c3_evidence=args.c3_evidence)
    except (ValueError, OSError, TimeoutError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
