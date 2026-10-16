#!/usr/bin/env python3
"""Scoped Gold graph freshness evidence; not a concurrency-safe publication gate.

Evidence binds an exact published candidate Parquet generation, every scoped
Gold relation used by the graph and both derived live graph views. Full
assertions must be run by the certifying command; proof issuance alone is not
certification. No persistent warehouse mutation occurs.

IMPORTANT: DuckDB's hash() is *not cryptographic*. This is a change detector,
not a signed/authenticity proof. GG-009/GG-012 cover cooperating
POSIX-local producers and guarded consumers only. Unguarded audit is diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.warehouse_lifecycle import lifecycle_epoch
from scripts.write_candidate_fact_partition import fsync_directory
from scripts.certify_gold_graph_cycle import (
    ASSERTIONS, SCOPED_SOURCES, CYCLE_PATTERN, quote_sql,
    ROOT, MODELS, TESTS, selected_publication, setup_scoped_views,
    run as run_diagnostic,
)

VERSION = 1
OUTPUTS = ("graph_vertices", "graph_edges")


class GraphFreshnessError(RuntimeError):
    pass


def identity(db: Path, source_root: Path, cycle: str) -> dict:
    if CYCLE_PATTERN.fullmatch(cycle) is None:
        raise GraphFreshnessError(f"Invalid cycle: {cycle!r}")
    return {
        "warehouse": str(db.resolve(strict=True)),
        "source_root": str(source_root.resolve(strict=True)),
        "cycle": cycle,
        # A cleaning epoch remains outside the DuckDB file and lock inode.
        # Even identical re-created bytes at the same path cannot resurrect a proof.
        "lifecycle_epoch": lifecycle_epoch(db),
    }


def certificate_path(directory: Path, cycle: str) -> Path:
    if CYCLE_PATTERN.fullmatch(cycle) is None:
        raise GraphFreshnessError(f"Invalid cycle: {cycle!r}")
    return Path(directory) / f"{cycle.replace(':', '-')}.json"


def _relation_fingerprint(con, name: str, cycle: str) -> dict:
    # Fixed allowlisted relation names; never interpolate user-controlled SQL.
    if name not in (*SCOPED_SOURCES, *OUTPUTS):
        raise GraphFreshnessError(f"Not a graph relation: {name}")
    info = con.execute(f"PRAGMA table_info('{name}')").fetchall()
    # Canonical JSON: tuples become lists on disk, so use lists in memory.
    # Otherwise an independent audit rejects unchanged column schemas.
    columns = [[row[1], str(row[2])] for row in info]
    if not columns or "cycle_id" not in {c for c, _ in columns}:
        raise GraphFreshnessError(f"Missing cycle_id or relation: {name}")
    fields = ", ".join('"' + column.replace('"', '""') + '"' for column, _ in columns)
    count, xordigest, sumdigest = con.execute(
        f"SELECT count(*), bit_xor(hash({fields})), "
        f"sum(cast(hash({fields}) as hugeint)) FROM {name} WHERE cycle_id = ?",
        [cycle],
    ).fetchone()
    return {"columns": columns, "rows": count, "xor": str(xordigest),
            "sum": str(sumdigest)}


def contract_fingerprint() -> dict:
    """Bind proof to the exact graph projection and assertion implementation."""
    paths = [MODELS / f"{name}.sql" for name in OUTPUTS]
    paths += [TESTS / f"assert_graph_{name}.sql" for name in ASSERTIONS]
    paths += [ROOT / "scripts/certify_gold_graph_cycle.py", Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def snapshot(db: Path, source_root: Path, cycle: str, *,
             memory_limit: str = "4GB", temp_directory: Path | None = None) -> dict:
    """Compute source + Gold inputs + actual derived outputs in one read-only run."""
    match = CYCLE_PATTERN.fullmatch(cycle)
    if not match:
        raise GraphFreshnessError(f"Invalid cycle: {cycle!r}")
    year, kind = int(match.group(1)), match.group(2)
    warehouse = db.resolve(strict=True)
    parquet, first_publication = selected_publication(source_root, year, kind)
    import duckdb
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET memory_limit = " + quote_sql(memory_limit))
        con.execute("SET threads = 2")
        if temp_directory is not None:
            temp_directory.mkdir(parents=True, exist_ok=True)
            con.execute("SET temp_directory = " + quote_sql(str(temp_directory.resolve())))
        setup_scoped_views(con, warehouse, parquet, year, kind)
        signatures = {name: _relation_fingerprint(con, name, cycle)
                      for name in (*SCOPED_SOURCES, *OUTPUTS)}
        # Detect a switched generation or in-place content change during scan.
        _, second_publication = selected_publication(source_root, year, kind)
        if first_publication != second_publication:
            raise GraphFreshnessError("Source publication changed while fingerprinting")
        return {"duckdb_version": duckdb.__version__,
                "contract": contract_fingerprint(),
                "publication": first_publication,
                "gold_sources": {k: signatures[k] for k in SCOPED_SOURCES},
                "graph": {k: signatures[k] for k in OUTPUTS}}
    finally:
        con.close()


@contextmanager
def graph_certification_locks(db: Path, source_root: Path, cycle: str,
                              *, timeout: float = 300.0):
    """Exclusive pair matching existing CTAS/publisher and guarded-reader locks.

    The entire revoke -> fingerprints -> SQL checks -> issue boundary is fenced.
    This is cooperative POSIX coordination, NOT isolation from ad-hoc SQL.
    """
    if timeout < 0:
        raise ValueError("lock timeout must be nonnegative")
    match = CYCLE_PATTERN.fullmatch(cycle)
    if match is None:
        raise GraphFreshnessError(f"Invalid cycle: {cycle!r}")
    warehouse = Path(db).resolve(strict=True)
    if not warehouse.is_file():
        raise GraphFreshnessError("Warehouse must be an existing DuckDB file")
    year, election_type = match.groups()
    part = (Path(source_root).resolve(strict=True) /
            f"election_type={election_type}" / f"year={year}")
    if not part.is_dir():
        raise GraphFreshnessError(f"Published partition does not exist: {part}")

    # Reuse the actual public-writer implementations and their lock inodes.
    # Order is warehouse -> partition, exactly as in the CTAS coordinator.
    from scripts import build_candidate_fact_partition as builder
    from scripts import write_candidate_fact_partition as publisher
    deadline = time.monotonic() + timeout
    with builder.warehouse_write_lock(warehouse, timeout=timeout):
        remaining = max(0.0, deadline - time.monotonic())
        with publisher.partition_write_lock(part, timeout=remaining):
            yield


def revoke_certificate(directory: Path, cycle: str) -> None:
    path = certificate_path(directory, cycle)
    if not path.exists() and not path.is_symlink():
        return
    archive = path.parent / "revoked"
    archive.mkdir(parents=True, exist_ok=True)
    os.replace(path, archive / f"{path.stem}-{uuid.uuid4().hex}.json")
    # Persist removal of active authority before persisting the archive.
    # A worker dying after rename must not leave recoverable old proof.
    fsync_directory(path.parent)
    fsync_directory(archive)


def issue_certificate(directory: Path, db: Path, source_root: Path,
                      cycle: str, evidence: dict) -> Path:
    """Create-only publication; never replace an existing active certificate."""
    path = certificate_path(directory, cycle)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise GraphFreshnessError(f"Active certificate exists: {path}")
    record = {"version": VERSION, **identity(db, source_root, cycle),
              "certified_at_utc": datetime.now(timezone.utc).isoformat(),
              "assertions": list(ASSERTIONS), "snapshot": evidence,
              "limitations": "not concurrency-safe; unsigned DuckDB hash evidence"}
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # An atomic hard-link publication is create-only on POSIX systems.
        os.link(tmp, path, follow_symlinks=False)
        # Hard-link creation is the publication linearization point.
        # Durably sync before reporting issuance success. A failed fsync
        # leaves the publication outcome uncertain: retract visible authority
        # while still under the issuer's exclusive cooperative locks.
        try:
            fsync_directory(path.parent)
        except OSError:
            path.unlink(missing_ok=True)
            fsync_directory(path.parent)
            raise
    finally:
        tmp.unlink(missing_ok=True)
    return path


def audit_certificate(directory: Path, db: Path, source_root: Path,
                      cycle: str, current: dict) -> dict:
    path = certificate_path(directory, cycle)
    if not path.is_file() or path.is_symlink():
        raise GraphFreshnessError(f"Missing/invalid graph attestation: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise GraphFreshnessError("Unreadable graph attestation") from exc
    if not isinstance(record, dict) or record.get("version") != VERSION:
        raise GraphFreshnessError("Unknown graph certificate version")
    required = identity(db, source_root, cycle)
    if any(record.get(k) != v for k, v in required.items()):
        raise GraphFreshnessError("Graph certificate bound to different warehouse/root/cycle")
    if record.get("assertions") != list(ASSERTIONS):
        raise GraphFreshnessError("Graph certificate lacks complete invariant assertion set")
    if record.get("snapshot") != current:
        raise GraphFreshnessError("Graph freshness failure: source/Gold/graph changed")
    return record


def certify_with_locks(args: argparse.Namespace) -> int:
    # No active proof may be revoked before obtaining exclusive coordination.
    with graph_certification_locks(args.db, args.source_root, args.cycle,
                                   timeout=args.lock_timeout):
        # Revoke BEFORE even reading a snapshot: an interrupted or failed
        # attempt must never leave an older certificate active.
        revoke_certificate(args.attestations_dir, args.cycle)
        before = snapshot(args.db, args.source_root, args.cycle,
                          memory_limit=args.memory_limit, temp_directory=args.temp_directory)
        diagnostic_args = argparse.Namespace(
            db=args.db, source_root=args.source_root, cycle=args.cycle,
            memory_limit=args.memory_limit, temp_directory=args.temp_directory,
            execute=True, tests=list(ASSERTIONS),
        )
        result = run_diagnostic(diagnostic_args)
        if len(result.get("assertions", [])) != len(ASSERTIONS) or not all(
            item.get("pass") is True for item in result["assertions"]
        ):
            raise GraphFreshnessError("Incomplete graph invariant diagnostics")
        after = snapshot(args.db, args.source_root, args.cycle,
                         memory_limit=args.memory_limit, temp_directory=args.temp_directory)
        if before != after:
            raise GraphFreshnessError("Graph sources or output changed during certification")
        cert = issue_certificate(args.attestations_dir, args.db, args.source_root,
                                 args.cycle, after)
        print(f"PASS: graph proof created at {cert}; NOT concurrency/production-certified")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--attestations-dir", type=Path, required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--lock-timeout", type=float, default=300.0,
                        help="POSIX lock wait for coordinated graph certification")
    parser.add_argument("--temp-directory", type=Path)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("audit", help="Recompute proof from live Gold graph, fail closed")
    actions.add_parser("certify", help="Run ALL 8 graph assertions and issue create-only proof")
    args = parser.parse_args(argv)
    try:
        if args.action == "audit":
            # Avoid an expensive graph scan if no proof exists at all.
            active = certificate_path(args.attestations_dir, args.cycle)
            if not active.is_file() or active.is_symlink():
                raise GraphFreshnessError(f"Missing/invalid graph attestation: {active}")
            now = snapshot(args.db, args.source_root, args.cycle,
                           memory_limit=args.memory_limit, temp_directory=args.temp_directory)
            audit_certificate(args.attestations_dir, args.db, args.source_root,
                              args.cycle, now)
            print(f"PASS: {args.cycle} graph freshness matches certificate; NOT concurrency-certified")
            return 0
        return certify_with_locks(args)
    except (GraphFreshnessError, OSError, ValueError, AssertionError) as err:
        print(f"FAIL: {err}")
        return 1
    except Exception as err:
        # DuckDB/Jinja exceptions also fail closed without a traceback by default.
        print(f"FAIL: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
