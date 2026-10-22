"""Coordinated C5-P migration with durable intent and fail-closed GG authority.

Not a production rollout CLI. Requires cooperating writers/readers.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import duckdb

from scripts.build_candidate_fact_partition import warehouse_write_lock
from scripts.write_candidate_fact_partition import (
    current_generation,
    fsync_directory,
    partition_write_lock,
    prove_staged_zone_uniqueness,
)
from scripts.certify_gold_graph_cycle import selected_publication
from scripts import graph_freshness
from scripts.migrate_candidate_legacy import _migrate_locked


JOURNAL_NAME = ".legacy-migration-intent.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_journal(path: Path) -> dict | None:
    if path.is_symlink():
        raise RuntimeError("Symlink journal rejected")
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError("Invalid journal")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Invalid journal object")
    return value


def _create_journal(path: Path, record: dict) -> None:
    """Durable create-only journal, never replace an existing intent."""
    temp = path.with_name(f".legacy-intent-{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path, follow_symlinks=False)
        fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)



def _verify_revocation_evidence(
    attestations_dir: Path,
    cycle: str,
    certificate_sha256: str,
    lifecycle_epoch: str,
) -> None:
    """Require the exact archived GG authority before recovery proceeds."""
    if (
        len(certificate_sha256) != 64
        or any(c not in "0123456789abcdef" for c in certificate_sha256)
    ):
        raise RuntimeError("Invalid journal certificate digest")

    archive = attestations_dir / "revoked"
    if not archive.is_dir() or archive.is_symlink():
        raise RuntimeError("Missing GG revocation evidence directory")

    matches = []

    for path in archive.glob(f"{cycle.replace(':', '-')}-*.json"):
        if path.is_symlink() or not path.is_file():
            continue

        raw = path.read_bytes()
        if _sha256(raw) != certificate_sha256:
            continue

        try:
            record = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(
                "Unreadable matching GG revocation evidence"
            ) from exc

        if (
            not isinstance(record, dict)
            or record.get("cycle") != cycle
            or record.get("lifecycle_epoch") != lifecycle_epoch
        ):
            raise RuntimeError(
                "GG revocation evidence identity mismatch"
            )

        matches.append(path)

    if len(matches) != 1:
        raise RuntimeError(
            "Missing or ambiguous GG revocation evidence: "
            f"matching_certificates={len(matches)}"
        )

def coordinate(
    *,
    db: Path,
    root: Path,
    attestations_dir: Path,
    year: int,
    election_type: str,
    expected_sha256: str,
    lock_timeout: float = 30.0,
    fault: str | None = None,
) -> dict:
    """Migrate one cycle; leave GG revoked pending separate certification."""
    if year <= 0 or election_type not in ("general", "municipal"):
        raise ValueError("Invalid cycle")
    if len(expected_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_sha256
    ):
        raise ValueError("Invalid expected SHA-256")

    db = Path(db).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    attestations_dir = Path(attestations_dir).resolve(strict=True)

    cycle = f"{year}:{election_type}"
    partition = root / f"election_type={election_type}" / f"year={year}"
    if not partition.is_dir():
        raise FileNotFoundError(partition)

    journal_path = partition / JOURNAL_NAME
    cert_path = graph_freshness.certificate_path(
        attestations_dir, cycle
    )

    # Consistent global lock order: warehouse, then partition.
    with warehouse_write_lock(db, timeout=lock_timeout):
        with partition_write_lock(partition, timeout=lock_timeout):
            _, publication = selected_publication(
                root, year, election_type
            )
            if publication["sha256"] != expected_sha256:
                raise RuntimeError("Physical SHA-256 mismatch")

            active_generation = current_generation(partition)
            target_generation = (
                f"generation-legacy-{expected_sha256}.parquet"
            )
            if active_generation not in ("legacy", target_generation):
                raise RuntimeError("Unexpected physical generation")

            existing = _load_journal(journal_path)

            if existing is None:
                if active_generation != "legacy":
                    raise RuntimeError(
                        "Migrated generation without migration journal"
                    )

                if not cert_path.is_file() or cert_path.is_symlink():
                    raise RuntimeError("Missing active GG certificate")

                cert_bytes = cert_path.read_bytes()
                certificate = json.loads(cert_bytes)

                required = graph_freshness.identity(
                    db, root, cycle
                )
                if any(
                    certificate.get(key) != value
                    for key, value in required.items()
                ):
                    raise RuntimeError("GG certificate identity mismatch")

                # Validate the complete live graph fingerprint.
                current = graph_freshness.snapshot(
                    db, root, cycle, memory_limit="2GB"
                )
                graph_freshness.audit_certificate(
                    attestations_dir, db, root, cycle, current
                )
                if current["publication"] != publication:
                    raise RuntimeError("Publication changed during audit")

                # Validate physical zone uniqueness BEFORE revocation.
                with duckdb.connect(":memory:") as con:
                    con.execute("SET threads=2")
                    con.execute("SET memory_limit='2GB'")
                    physical_path = Path(publication["real_path"])
                    count = con.execute(
                        "SELECT COUNT(*) FROM read_parquet("
                        "?, hive_partitioning=false)",
                        [str(physical_path)],
                    ).fetchone()[0]
                    prove_staged_zone_uniqueness(
                        con, physical_path, count
                    )

                existing = {
                    "version": 1,
                    "cycle": cycle,
                    "warehouse": str(db),
                    "source_root": str(root),
                    "attestations_dir": str(attestations_dir),
                    "lifecycle_epoch": required["lifecycle_epoch"],
                    "expected_sha256": expected_sha256,
                    "legacy_publication": publication,
                    "certificate_sha256": _sha256(cert_bytes),
                    "target_generation": target_generation,
                    "scope": "C5-P_physical_migration_not_GG",
                }
                _create_journal(journal_path, existing)

                if fault == "after_journal":
                    raise RuntimeError("Injected failure: after_journal")

            # Recover only the exact same operation.
            required_fields = {
                "version": 1,
                "cycle": cycle,
                "warehouse": str(db),
                "source_root": str(root),
                "attestations_dir": str(attestations_dir),
                "lifecycle_epoch": graph_freshness.identity(
                    db, root, cycle
                )["lifecycle_epoch"],
                "expected_sha256": expected_sha256,
                "target_generation": target_generation,
                "scope": "C5-P_physical_migration_not_GG",
            }
            if any(
                existing.get(key) != value
                for key, value in required_fields.items()
            ):
                raise RuntimeError("Migration journal identity mismatch")

            if not isinstance(
                existing.get("legacy_publication"), dict
            ):
                raise RuntimeError("Missing original publication")
            if existing["legacy_publication"]["sha256"] != expected_sha256:
                raise RuntimeError("Journal source SHA-256 mismatch")
            if not isinstance(existing.get("certificate_sha256"), str):
                raise RuntimeError("Missing certificate identity")

            # A journal alone is not authority to erase a certificate.
            if cert_path.exists() or cert_path.is_symlink():
                if active_generation != "legacy":
                    raise RuntimeError(
                        "Active GG certificate after pointer migration"
                    )
                if cert_path.is_symlink() or not cert_path.is_file():
                    raise RuntimeError("Invalid active certificate")
                if _sha256(cert_path.read_bytes()) != existing[
                    "certificate_sha256"
                ]:
                    raise RuntimeError(
                        "GG certificate changed since preparation"
                    )

                # A resumed PREPARED operation must also recheck freshness.
                current = graph_freshness.snapshot(
                    db, root, cycle, memory_limit="2GB"
                )
                graph_freshness.audit_certificate(
                    attestations_dir, db, root, cycle, current
                )
                if current["publication"] != existing["legacy_publication"]:
                    raise RuntimeError("Stale prepared publication")

                graph_freshness.revoke_certificate(
                    attestations_dir, cycle
                )

            if cert_path.exists() or cert_path.is_symlink():
                raise RuntimeError("GG revocation incomplete")

            _verify_revocation_evidence(
                attestations_dir,
                cycle,
                existing["certificate_sha256"],
                existing["lifecycle_epoch"],
            )

            if fault == "after_revoke":
                raise RuntimeError("Injected failure: after_revoke")

            # Idempotent when already published; physical evidence is
            # revalidated by the existing migrator on each invocation.
            record = _migrate_locked(
                root, year, election_type,
                expected_sha256=expected_sha256,
                lock_timeout=lock_timeout,
                fault=fault,
            )

            _, after = selected_publication(
                root, year, election_type
            )
            if after["sha256"] != expected_sha256:
                raise RuntimeError("Post-migration hash mismatch")
            if Path(after["real_path"]).name != target_generation:
                raise RuntimeError("Wrong published generation")
            if cert_path.exists() or cert_path.is_symlink():
                raise RuntimeError("GG certificate unexpectedly active")

            return {
                "cycle": cycle,
                "status": "AWAITING_GG_CERTIFICATION",
                "journal": str(journal_path),
                "migration": record,
            }
