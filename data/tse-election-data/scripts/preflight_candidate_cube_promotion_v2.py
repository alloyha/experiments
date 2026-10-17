#!/usr/bin/env python3
"""Read-only approval-input gate for a SECOND (versioned) TSE Gold promotion.

This command does not modify DuckDB, lifecycle epochs, journals, or certificates.
It checks the existing v1 active lineage, the physical five-cycle sources,
all three warehouse bytes, and the separate offline contract proof. The output
is create-only evidence, NOT permission to perform an in-place file swap.

A future v2 activation/recovery implementation must independently revalidate
these assumptions under EXCLUSIVE locks before making any changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import stage_candidate_cube as staging
from scripts import audit_staged_candidate_cube_contracts as offline
from scripts.warehouse_lifecycle import lifecycle_epoch

SCOPE = "candidate_cube_v2_readonly_promotion_preflight_not_activation"
V1_ACTIVE = "ACTIVE_UNCERTIFIED"
V2_STATE_SUFFIX = ".promotion-v2"


class PreflightError(RuntimeError):
    pass


def existing_json(path: Path) -> dict:
    evidence.regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PreflightError(f"malformed report or journal {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"expected JSON object at {path}")
    return value


def full_sha(value: object) -> bool:
    return isinstance(value, str) and evidence.SHA256_RE.fullmatch(value) is not None


def safe_report_path(path: Path, critical: set[Path]) -> None:
    if path in critical or path.exists() or path.is_symlink():
        raise PreflightError(f"report already exists or unsafe: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PreflightError("report parent must already be a real directory")


def legacy_anchor(main: Path, approved_hash: str) -> dict:
    """Prove that historical v1 ACTIVE_UNCERTIFIED is consistent and restorable."""
    journal_path = main.with_name(main.name + ".promotion.json")
    journal = existing_json(journal_path)
    if (journal.get("version") != 1
        or journal.get("original") != str(main)
        or journal.get("phase") != V1_ACTIVE
        or journal.get("candidate_sha256") != approved_hash
        or not full_sha(journal.get("original_sha256"))
        or not isinstance(journal.get("lifecycle_epoch"), str)):
        raise PreflightError("v1 promotion journal is not the expected active baseline")
    if journal["lifecycle_epoch"] != lifecycle_epoch(main):
        raise PreflightError("v1 lifecycle epoch differs from current identity")
    if journal.get("id") is None:
        raise PreflightError("v1 journal lacks immutable promotion id")
    backup = Path(str(journal.get("backup", "")))
    if backup.is_symlink() or backup.parent != main.parent or backup == main:
        raise PreflightError("legacy backup path is unsafe")
    evidence.regular_file(backup)
    previous_hash = evidence.digest(backup)
    if previous_hash != journal["original_sha256"]:
        raise PreflightError("legacy backup SHA-256 has drifted")
    if evidence.digest(main) != approved_hash:
        raise PreflightError("main differs from current v1 activated bytes")
    reserved = main.with_name(main.name + V2_STATE_SUFFIX)
    if reserved.exists() or reserved.is_symlink():
        raise PreflightError("v2 state already exists; use v2 status/recovery, not initial preflight")
    return {
        "journal": str(journal_path),
        "journal_sha256": evidence.digest(journal_path),
        "legacy_promotion_id": journal["id"],
        "phase": V1_ACTIVE,
        "lifecycle_epoch": journal["lifecycle_epoch"],
        "main_sha256": approved_hash,
        "legacy_backup": str(backup),
        "legacy_backup_sha256": previous_hash,
        "reserved_v2_state": str(reserved),
    }


def validate_offline_report(*, record: dict, main: Path, preview: Path, clone: Path,
                            root: Path, manifest_path: Path, manifest_hash: str,
                            stage_path: Path, stage_hash: str,
                            manifest: dict) -> dict[str, str]:
    if (record.get("result") != "PASS" or record.get("scope") != offline.SCOPE
        or record.get("main") != str(main) or record.get("preview") != str(preview)
        or record.get("clone") != str(clone)
        or record.get("source_root") != str(root)
        or record.get("main_sha256") != manifest["main_sha256"]
        or record.get("preview_sha256") != manifest["preview_sha256"]
        or record.get("clone_sha256") is None
        or record.get("manifest") != str(manifest_path)
        or record.get("manifest_sha256") != manifest_hash
        or record.get("stage_report") != str(stage_path)
        or record.get("stage_report_sha256") != stage_hash
        or record.get("total_rows") != manifest["total_rows"]
        or record.get("total_nominal_votes") != manifest["total_nominal_votes"]):
        raise PreflightError("offline contract proof has conflicting identities or coverage")
    cycles = record.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != len(evidence.CYCLES):
        raise PreflightError("missing offline cycle proofs")
    hashes: dict[str, str] = {}
    observed = set()
    for item in cycles:
        if not isinstance(item, dict):
            raise PreflightError("malformed offline cycle proof")
        cycle = item.get("cycle")
        if cycle not in evidence.CYCLES or cycle in observed:
            raise PreflightError("duplicate or unexpected offline proof cycle")
        observed.add(cycle)
        if item.get("not_null") != "PASS" or item.get("grain_unique") != "PASS":
            raise PreflightError(f"key contracts not certified in offline proof: {cycle}")
        tests = item.get("tests")
        if not isinstance(tests, list) or len(tests) != len(offline.TESTS):
            raise PreflightError(f"incomplete SQL assertions for {cycle}")
        seen = set()
        for test in tests:
            if not isinstance(test, dict):
                raise PreflightError("bad offline SQL test record")
            name = test.get("test")
            if (name not in offline.TESTS or name in seen or test.get("pass") is not True
                or not full_sha(test.get("sha256"))):
                raise PreflightError(f"invalid SQL proof for {cycle}")
            seen.add(name)
            if name in hashes and hashes[name] != test["sha256"]:
                raise PreflightError("different SQL code hashes between cycle proofs")
            hashes[name] = test["sha256"]
        if seen != set(offline.TESTS):
            raise PreflightError("not all required SQL assertions passed")
    if observed != set(evidence.CYCLES):
        raise PreflightError("incomplete five-cycle SQL assertion coverage")
    for name in offline.TESTS:
        sql_path = offline.PROJECT / "tse_dbt" / "tests" / f"{name}.sql"
        evidence.regular_file(sql_path)
        if evidence.digest(sql_path) != hashes[name]:
            raise PreflightError(f"SQL test changed after offline execution: {name}")
    return hashes


def preflight(*, main: Path, preview: Path, clone: Path, root: Path,
              manifest_path: Path, manifest_hash: str, stage_path: Path,
              stage_hash: str, contracts_path: Path, contracts_hash: str,
              clone_hash: str, report_path: Path, lock_timeout: float) -> dict:
    paths = [main, preview, clone, root, manifest_path, stage_path,
             contracts_path, report_path]
    main, preview, clone, root, manifest_path, stage_path, contracts_path, report_path = (
        p.expanduser().absolute() for p in paths
    )
    critical = {main, preview, clone, manifest_path, stage_path, contracts_path}
    if len(critical) != 6 or len({main, preview, clone}) != 3:
        raise PreflightError("paths alias another protected file")
    if main.parent != preview.parent or clone.parent != main.parent:
        raise PreflightError("all DuckDB files must share a filesystem and parent")
    if any(path.suffix != ".duckdb" for path in (main, preview, clone)):
        raise PreflightError("all warehouses must be .duckdb files")
    if any(not full_sha(h) for h in (manifest_hash, stage_hash, contracts_hash, clone_hash)):
        raise PreflightError("all explicitly pinned SHA-256 arguments must be complete")
    safe_report_path(report_path, critical)
    for p in (main, preview, clone):
        evidence.regular_file(p)
        staging.assert_no_wal(p)
    for path, expected in ((manifest_path, manifest_hash), (stage_path, stage_hash),
                           (contracts_path, contracts_hash), (clone, clone_hash)):
        evidence.regular_file(path)
        if evidence.digest(path) != expected:
            raise PreflightError(f"SHA mismatch: {path}")
    manifest = existing_json(manifest_path)
    staging.check_identity(main, preview, root, manifest)
    stage = existing_json(stage_path)
    stage_anchor = offline.validate_metadata(
        main=main, preview=preview, clone=clone, root=root,
        manifest_path=manifest_path, manifest_hash=manifest_hash,
        stage_path=stage_path, clone_hash=clone_hash, report=report_path,
    )
    if stage_anchor[0] != manifest or stage_anchor[1] != stage:
        raise PreflightError("inconsistent stage proof")
    record = existing_json(contracts_path)
    hashes = validate_offline_report(
        record=record, main=main, preview=preview, clone=clone, root=root,
        manifest_path=manifest_path, manifest_hash=manifest_hash,
        stage_path=stage_path, stage_hash=stage_hash, manifest=manifest,
    )
    # Entire evidence inspection is fenced against cooperative source writers.
    with evidence.noncreating_shared_locks(staging.locks_for(main, preview, root), lock_timeout):
        staging.verify_sources(main, preview, root, manifest)
        staging.assert_no_wal(clone)
        if evidence.digest(clone) != clone_hash:
            raise PreflightError("clone bytes changed")
        anchor = legacy_anchor(main, manifest["main_sha256"])
        # There must be space for a copy of the current Gold and staged candidate.
        # A future mutating promoter must check this again under exclusive locks.
        required = main.stat().st_size + clone.stat().st_size + 512 * 1024 * 1024
        available = shutil.disk_usage(main.parent).free
        if available < required:
            raise PreflightError(f"insufficient disk for two durable promotion copies: {required}")
        staging.verify_sources(main, preview, root, manifest)
        for path, expected in ((manifest_path, manifest_hash), (stage_path, stage_hash),
                               (contracts_path, contracts_hash), (clone, clone_hash)):
            if evidence.digest(path) != expected:
                raise PreflightError(f"approved evidence changed under locks: {path}")
        if anchor != legacy_anchor(main, manifest["main_sha256"]):
            raise PreflightError("historical promotion anchor changed under locks")
        safe_report_path(report_path, critical)
        return {
            "result": "PASS", "scope": SCOPE,
            "main": str(main), "preview": str(preview), "clone": str(clone),
            "main_sha256": manifest["main_sha256"],
            "preview_sha256": manifest["preview_sha256"],
            "clone_sha256": clone_hash, "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash, "stage_report": str(stage_path),
            "stage_report_sha256": stage_hash, "offline_contract_report": str(contracts_path),
            "offline_contract_report_sha256": contracts_hash,
            "source_root": str(root),
            "required_free_bytes_at_audit": required,
            "observed_free_bytes_at_audit": available,
            "legacy_anchor": anchor,
            "contract_sql_sha256": hashes,
            "cycle_count": len(evidence.CYCLES),
            "total_rows": manifest["total_rows"],
            "total_nominal_votes": manifest["total_nominal_votes"],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": (
                "read-only cooperative preflight only, not a production promotion or "
                "freshness certificate; revalidate every source with exclusive writer locks "
                "and explicit operator quiescence before future activation"
            ),
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("main-db", "preview-db", "clone-db", "source-root", "manifest",
                 "stage-report", "offline-contract-report", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("manifest-sha256", "stage-report-sha256",
                 "offline-contract-report-sha256", "clone-sha256"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--lock-timeout", type=float, default=300.0)
    a = p.parse_args(argv)
    try:
        result = preflight(
            main=a.main_db, preview=a.preview_db, clone=a.clone_db,
            root=a.source_root, manifest_path=a.manifest,
            manifest_hash=a.manifest_sha256,
            stage_path=a.stage_report, stage_hash=a.stage_report_sha256,
            contracts_path=a.offline_contract_report,
            contracts_hash=a.offline_contract_report_sha256,
            clone_hash=a.clone_sha256, report_path=a.report,
            lock_timeout=a.lock_timeout,
        )
        evidence.write_create_only(a.report.expanduser().absolute(), result)
        print(json.dumps({"result": "PASS", "scope": SCOPE,
                          "report": str(a.report.expanduser().absolute()),
                          "eligible_cycles": result["cycle_count"],
                          "total_rows": result["total_rows"]}, indent=2))
        return 0
    except (PreflightError, evidence.EvidenceError, staging.StageError,
            offline.OfflineAuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"CUBE PROMOTION PREFLIGHT DENIED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
