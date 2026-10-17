#!/usr/bin/env python3
"""Read-only offline contract audit of a previously verified candidate-cube clone.

Runs the exact two repository dbt data tests for every pinned election cycle,
plus materialized-table, key, coverage and stage-integrity checks. No dbt run,
no writes to any DuckDB and NO issuance of production freshness certificates.

POSIX locks only cover cooperating publishers. The offline clone has no build
lock created by the staging command; --ack-quiescent is mandatory and its bytes
and WAL are checked before and after the read-only verification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import stage_candidate_cube as staging

PROJECT = Path(__file__).resolve().parents[1]
TESTS = (
    "assert_cube_candidate_vote_conservation",
    "assert_cube_candidate_share_and_rank",
)
MODEL = staging.TABLE
SCOPE = "offline_clone_readonly_dbt_contracts_not_production_freshness"
MEMORY_RE = re.compile(r"[1-9][0-9]*(?:MB|GB)\Z")
KEYS = (
    "election_id", "round_number", "uf", "municipality_code",
    "office_code", "is_transit_vote", "candidate_id",
)
NOT_NULL = ("cycle_id", *KEYS, "nominal_valid_votes", "candidate_matched", "geography_matched")


class OfflineAuditError(RuntimeError):
    pass


def canonical_path(value: Path) -> Path:
    return value.expanduser().absolute()


def load_json(path: Path) -> dict:
    evidence.regular_file(path)
    result = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(result, dict):
        raise OfflineAuditError(f"not a JSON object: {path}")
    return result


def validate_metadata(*, main: Path, preview: Path, clone: Path, root: Path,
                      manifest_path: Path, manifest_hash: str, stage_path: Path,
                      clone_hash: str, report: Path) -> tuple[dict, dict]:
    """Reject stale or cross-warehouse artifacts BEFORE running costly SQL."""
    if len({main, preview, clone, manifest_path, stage_path, report}) != 6:
        raise OfflineAuditError("database, evidence, and report paths must be distinct")
    if main.parent != preview.parent or clone.parent != main.parent:
        raise OfflineAuditError("all three DuckDBs must share the warehouse directory")
    if any(db.suffix != ".duckdb" for db in (main, preview, clone)):
        raise OfflineAuditError("all three database paths must have .duckdb suffix")
    for db in (main, preview, clone):
        evidence.regular_file(db)
        staging.assert_no_wal(db)
    if root.is_symlink() or not root.is_dir():
        raise OfflineAuditError("published Parquet root unavailable or symlinked")
    if not evidence.SHA256_RE.fullmatch(manifest_hash):
        raise OfflineAuditError("--manifest-sha256 must be a full SHA-256")
    if not evidence.SHA256_RE.fullmatch(clone_hash):
        raise OfflineAuditError("--clone-sha256 must be a full SHA-256")
    if evidence.digest(manifest_path) != manifest_hash:
        raise OfflineAuditError("manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    staging.check_identity(main, preview, root, manifest)
    stage = load_json(stage_path)
    if (stage.get("result") != "PASS" or stage.get("scope") != staging.STAGE_SCOPE
        or stage.get("main") != str(main) or stage.get("preview") != str(preview)
        or stage.get("source_root") != str(root)
        or stage.get("manifest") != str(manifest_path)
        or stage.get("manifest_sha256") != manifest_hash
        or stage.get("main_sha256") != manifest["main_sha256"]
        or stage.get("preview_sha256") != manifest["preview_sha256"]
        or stage.get("output_db") != str(clone)
        or stage.get("output_sha256") != clone_hash
        or stage.get("schema_columns") != 29
        or stage.get("total_rows") != manifest["total_rows"]
        or stage.get("total_nominal_votes") != manifest["total_nominal_votes"]
        or not isinstance(stage.get("cycles"), dict)
        or set(stage["cycles"]) != set(evidence.CYCLES)):
        raise OfflineAuditError("offline stage report identity/coverage mismatch")
    for cycle in evidence.CYCLES:
        observed, expected = stage["cycles"][cycle], manifest["cycles"][cycle]
        if (not isinstance(observed, dict)
            or observed.get("rows") != expected["rows"]
            or observed.get("nominal_votes") != expected["nominal_votes"]
            or observed.get("missing_rows") != 0
            or observed.get("extra_rows") != 0):
            raise OfflineAuditError(f"offline stage multiset mismatch: {cycle}")
    if report.exists() or report.is_symlink():
        raise OfflineAuditError(f"refusing overwrite of existing report: {report}")
    if not report.parent.is_dir() or report.parent.is_symlink():
        raise OfflineAuditError("report parent must be an existing real directory")
    if evidence.digest(clone) != clone_hash:
        raise OfflineAuditError("clone SHA-256 mismatch")
    return manifest, stage


def render_test(name: str, cycle: str) -> tuple[str, str]:
    """Render existing dbt assertions, not a hand-copied variant."""
    if name not in TESTS or cycle not in evidence.CYCLES:
        raise OfflineAuditError("unsupported dbt contract/cycle")
    from jinja2 import Environment, StrictUndefined

    year, kind = cycle.split(":")
    path = PROJECT / "tse_dbt" / "tests" / f"{name}.sql"
    evidence.regular_file(path)
    body = path.read_text(encoding="utf8")

    def ref(model: str) -> str:
        if model not in {MODEL, "fact_candidate_votes"}:
            raise OfflineAuditError(f"unapproved dbt dependency: {model}")
        return f'"main"."{model}"'

    def predicate(year_col: str = "election_year", type_col: str = "election_type") -> str:
        if year_col not in {"election_year", "year"} or type_col != "election_type":
            raise OfflineAuditError("unapproved incremental partition predicate")
        return f"{year_col} = {int(year)} AND {type_col} = '{kind}'"

    sql = Environment(undefined=StrictUndefined).from_string(body).render(
        ref=ref, incremental_partition_predicate=predicate,
    ).strip().rstrip(";")
    if not sql:
        raise OfflineAuditError(f"empty compiled SQL test: {name}")
    return sql, evidence.digest(path)


def inspect_clone(con, manifest: dict) -> None:
    kind = con.execute("""
        select table_type from information_schema.tables
        where table_catalog=current_database()
          and table_schema='main' and table_name=?
    """, [MODEL]).fetchone()
    if kind != ("BASE TABLE",):
        raise OfflineAuditError(f"candidate cube must be a persisted BASE TABLE: {kind}")
    columns = con.execute(f"DESCRIBE SELECT * FROM main.{MODEL}").fetchall()
    names = [row[0] for row in columns]
    if len(names) != 29 or len(names) != len(set(names)) or not set(NOT_NULL).issubset(names):
        raise OfflineAuditError("29-column schema or mandatory fields mismatch")
    rows = con.execute(f"""
        select cycle_id, count(*), sum(nominal_valid_votes)
        from main.{MODEL} group by cycle_id order by cycle_id
    """).fetchall()
    observed = {cycle: (n, votes) for cycle, n, votes in rows}
    expected = staging.expected_counts(manifest)
    if len(rows) != len(expected) or observed != expected:
        raise OfflineAuditError(f"cube inventory mismatch: {observed}")


def run_cycle(con, cycle: str, *, started: float) -> dict:
    def log(message: str):
        print(f"[cube-offline +{time.monotonic()-started:.1f}s] {cycle}: {message}",
              file=sys.stderr, flush=True)

    log("checking key nullability")
    keys_null = " or ".join(f'"{key}" is null' for key in NOT_NULL)
    violation = con.execute(
        f"SELECT 1 FROM main.{MODEL} WHERE cycle_id=? AND ({keys_null}) LIMIT 1",
        [cycle],
    ).fetchone()
    if violation:
        raise OfflineAuditError(f"non-null key/measure contract failed: {cycle}")
    log("checking uniqueness of complete candidate municipality grain")
    grouping = ", ".join(f'"{key}"' for key in KEYS)
    violation = con.execute(f"""
        select 1 from main.{MODEL} where cycle_id=?
        group by {grouping} having count(*) > 1 limit 1
    """, [cycle]).fetchone()
    if violation:
        raise OfflineAuditError(f"grain uniqueness failed: {cycle}")
    results = []
    for name in TESTS:
        sql, checksum = render_test(name, cycle)
        log(f"running exact dbt assertion {name}")
        start = time.monotonic()
        violation = con.execute(f"SELECT * FROM ({sql}) AS violations LIMIT 1").fetchone()
        elapsed = round(time.monotonic() - start, 3)
        if violation is not None:
            raise OfflineAuditError(f"dbt assertion failed: {cycle} {name}: {violation!r}")
        results.append({"test": name, "sha256": checksum, "seconds": elapsed, "pass": True})
        log(f"PASS {name} ({elapsed}s)")
    return {"cycle": cycle, "not_null": "PASS", "grain_unique": "PASS", "tests": results}


def verify_again(*, main: Path, preview: Path, clone: Path, root: Path,
                 manifest_path: Path, manifest_hash: str, clone_hash: str,
                 stage_path: Path, stage_hash: str, manifest: dict):
    staging.verify_sources(main, preview, root, manifest)
    for path, expected, label in (
        (clone, clone_hash, "clone"),
        (manifest_path, manifest_hash, "manifest"),
        (stage_path, stage_hash, "stage report"),
    ):
        evidence.regular_file(path)
        if evidence.digest(path) != expected:
            raise OfflineAuditError(f"{label} changed during verification")
    staging.assert_no_wal(clone)


def audit(*, main: Path, preview: Path, clone: Path, root: Path,
          manifest_path: Path, manifest_hash: str, stage_path: Path,
          clone_hash: str, report: Path, memory: str, timeout: float,
          ack_quiescent: bool) -> dict:
    main, preview, clone, root, manifest_path, stage_path, report = map(
        canonical_path, (main, preview, clone, root, manifest_path, stage_path, report))
    if not ack_quiescent:
        raise OfflineAuditError("must explicitly acknowledge externally enforced writer quiescence")
    if not MEMORY_RE.fullmatch(memory):
        raise OfflineAuditError("--memory-limit requires positive MB or GB value")
    manifest, stage = validate_metadata(
        main=main, preview=preview, clone=clone, root=root,
        manifest_path=manifest_path, manifest_hash=manifest_hash,
        stage_path=stage_path, clone_hash=clone_hash, report=report,
    )
    started = time.monotonic()
    stage_hash = evidence.digest(stage_path)
    locks = staging.locks_for(main, preview, root)
    print("[cube-offline] acquiring all existing shared writer-coordination locks",
          file=sys.stderr, flush=True)
    with evidence.noncreating_shared_locks(locks, timeout):
        contracts = {name: evidence.digest(PROJECT / "tse_dbt/tests" / f"{name}.sql")
                     for name in TESTS}
        # Main and preview must match the manifest *within* the locked scope.
        verify_again(main=main, preview=preview, clone=clone, root=root,
                     manifest_path=manifest_path, manifest_hash=manifest_hash,
                     clone_hash=clone_hash, stage_path=stage_path,
                     stage_hash=stage_hash, manifest=manifest)
        import duckdb
        try:
            con = duckdb.connect(str(clone), read_only=True)
        except duckdb.Error as exc:
            raise OfflineAuditError(f"cannot open offline clone read-only: {exc}") from exc
        try:
            con.execute("SET memory_limit = '" + memory + "'")
            con.execute("SET threads = 2")
            try:
                inspect_clone(con, manifest)
                cycles = [run_cycle(con, c, started=started) for c in evidence.CYCLES]
            except duckdb.Error as exc:
                raise OfflineAuditError(f"DuckDB contract execution failed: {exc}") from exc
        finally:
            con.close()
        verify_again(main=main, preview=preview, clone=clone, root=root,
                     manifest_path=manifest_path, manifest_hash=manifest_hash,
                     clone_hash=clone_hash, stage_path=stage_path,
                     stage_hash=stage_hash, manifest=manifest)
        if ({name: evidence.digest(PROJECT / "tse_dbt/tests" / f"{name}.sql")
             for name in TESTS} != contracts
            or any(item["sha256"] != contracts[item["test"]]
                   for cycle in cycles for item in cycle["tests"])):
            raise OfflineAuditError("dbt SQL contract source changed during audit")
        if report.exists() or report.is_symlink():
            raise OfflineAuditError("report output appeared during audit")
        record = {
            "result": "PASS", "scope": SCOPE,
            "main": str(main), "preview": str(preview), "clone": str(clone),
            "main_sha256": manifest["main_sha256"],
            "preview_sha256": manifest["preview_sha256"],
            "clone_sha256": clone_hash,
            "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
            "stage_report": str(stage_path), "stage_report_sha256": stage_hash,
            "source_root": str(root), "cycles": cycles,
            "total_rows": manifest["total_rows"],
            "total_nominal_votes": manifest["total_nominal_votes"],
            "duckdb_version": duckdb.__version__,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic()-started, 3),
            "limitations": ("offline read-only SQL contract evidence, not a live-path graph "
                           "certificate, not the candidate-cube freshness format, not promotion "
                           "authority; locks are cooperative and quiescence is operator-enforced"),
        }
        evidence.write_create_only(report, record)
        return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-db", required=True, type=Path)
    p.add_argument("--preview-db", required=True, type=Path)
    p.add_argument("--clone-db", required=True, type=Path)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--stage-report", required=True, type=Path)
    p.add_argument("--clone-sha256", required=True)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--memory-limit", default="4GB")
    p.add_argument("--lock-timeout", type=float, default=300.0)
    p.add_argument("--ack-quiescent", action="store_true")
    a = p.parse_args(argv)
    try:
        result = audit(main=a.main_db, preview=a.preview_db,
                       clone=a.clone_db, root=a.source_root,
                       manifest_path=a.manifest, manifest_hash=a.manifest_sha256,
                       stage_path=a.stage_report, clone_hash=a.clone_sha256,
                       report=a.report, memory=a.memory_limit,
                       timeout=a.lock_timeout, ack_quiescent=a.ack_quiescent)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OfflineAuditError, evidence.EvidenceError, staging.StageError,
            ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"OFFLINE CUBE CONTRACT AUDIT DENIED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
