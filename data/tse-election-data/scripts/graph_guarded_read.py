#!/usr/bin/env python3
"""Cooperatively guarded, cycle-scoped Gold graph reader (GG-009 prototype).

All *cooperating* writers must use the existing warehouse and partition lock
inodes, in that order. This is not protection from direct/uncoordinated SQL.
Nothing is served without valid GG-010 freshness evidence both before and after
reading the same scoped graph. This command never mutates the warehouse.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import graph_freshness as freshness
from scripts.certify_gold_graph_cycle import (
    CYCLE_PATTERN, quote_sql, selected_publication, setup_scoped_views,
)

MAX_ROWS = 1000
RELATIONS = {"vertices": ("graph_vertices", "node_type"),
             "edges": ("graph_edges", "edge_type")}


class GuardError(RuntimeError):
    pass


@contextmanager
def _shared_flock(lock_path: Path, deadline: float):
    """Shared POSIX flock on the same stable inode used by existing writers."""
    if not lock_path.parent.is_dir():
        raise GuardError(f"Lock parent is missing: {lock_path.parent}")
    with lock_path.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring graph read lock: {lock_path}") from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def graph_read_locks(db: Path, root: Path, cycle: str, *, timeout: float = 30.0):
    """Order matches build_candidate_fact_partition: warehouse then partition.

    Never use a lock on the DuckDB inode itself. Do not create missing source
    partitions: a missing publication must fail without manufacturing a path.
    """
    if timeout < 0:
        raise ValueError("Lock timeout must be nonnegative")
    match = CYCLE_PATTERN.fullmatch(cycle)
    if match is None:
        raise ValueError("Invalid election cycle")
    if not db.is_file():
        raise GuardError(f"DuckDB warehouse is missing: {db}")
    year, election_type = match.groups()
    part = root.resolve() / f"election_type={election_type}" / f"year={year}"
    if not part.is_dir():
        raise GuardError(f"Published partition does not exist: {part}")
    canonical = db.resolve(strict=True)
    db_lock = canonical.with_name(canonical.name + ".build.lock")
    partition_lock = part / ".publish.lock"
    deadline = time.monotonic() + timeout
    with ExitStack() as stack:
        stack.enter_context(_shared_flock(db_lock, deadline))
        stack.enter_context(_shared_flock(partition_lock, deadline))
        yield


def read_graph(*, db: Path, source_root: Path, attestations_dir: Path,
               cycle: str, relation: str, kind: str | None = None,
               limit: int = 25, lock_timeout: float = 30.0,
               memory_limit: str = "4GB", temp_directory: Path | None = None) -> dict:
    """Return ONLY allowlisted, bounded rows after two freshness audits.

    The same POSIX shared locks remain held through source fingerprinting,
    query execution, post-read fingerprinting and certificate identity checks.
    """
    if relation not in RELATIONS:
        raise ValueError("Only vertices and edges may be queried")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_ROWS}")
    if kind is not None and (not isinstance(kind, str) or not kind or len(kind) > 80):
        raise ValueError("Invalid graph relation type")
    model, kind_column = RELATIONS[relation]
    with graph_read_locks(db, source_root, cycle, timeout=lock_timeout):
        cert_path = freshness.certificate_path(attestations_dir, cycle)
        if not cert_path.is_file() or cert_path.is_symlink():
            raise GuardError(f"Missing active graph certificate: {cert_path}")
        # Digest detects a revocation/reissue by non-cooperating certificate
        # writers even when the underlying graph happens to be unchanged.
        import hashlib
        certificate_before = hashlib.sha256(cert_path.read_bytes()).hexdigest()
        before = freshness.snapshot(db, source_root, cycle, memory_limit=memory_limit,
                                    temp_directory=temp_directory)
        freshness.audit_certificate(attestations_dir, db, source_root, cycle, before)
        # Use the same single physical publication as the GG-010 diagnostic.
        match = CYCLE_PATTERN.fullmatch(cycle)
        assert match is not None
        year, election_type = int(match.group(1)), match.group(2)
        pinned_path, first_source = selected_publication(source_root, year, election_type)
        import duckdb
        with duckdb.connect(":memory:") as con:
            con.execute("SET memory_limit = " + quote_sql(memory_limit))
            con.execute("SET threads = 2")
            if temp_directory:
                temp_directory.mkdir(parents=True, exist_ok=True)
                con.execute("SET temp_directory = " + quote_sql(str(temp_directory.resolve())))
            setup_scoped_views(con, db, pinned_path, year, election_type)
            sql = f"SELECT * FROM {model} WHERE cycle_id = ?"
            params: list = [cycle]
            if kind is not None:
                sql += f" AND {kind_column} = ?"
                params.append(kind)
            sql += " ORDER BY 1 LIMIT ?"
            params.append(limit)
            cursor = con.execute(sql, params)
            names = [column[0] for column in cursor.description]
            rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        # Fail closed on changed publication, Gold snapshot, graph projection
        # or authority; do not expose provisional rows before these checks.
        _, final_source = selected_publication(source_root, year, election_type)
        if first_source != final_source:
            raise GuardError("Published source changed while reading the graph")
        after = freshness.snapshot(db, source_root, cycle, memory_limit=memory_limit,
                                   temp_directory=temp_directory)
        freshness.audit_certificate(attestations_dir, db, source_root, cycle, after)
        if before != after:
            raise GuardError("Graph changed between authorization and consumption")
        if not cert_path.is_file() or cert_path.is_symlink():
            raise GuardError("Graph certificate was revoked during consumption")
        if certificate_before != hashlib.sha256(cert_path.read_bytes()).hexdigest():
            raise GuardError("Graph certificate changed during consumption")
        return {"cycle": cycle, "relation": relation,
                "type_filter": kind, "rows": rows,
                "status": "cooperatively_guarded_not_production_certified"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--attestations-dir", required=True, type=Path)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--relation", choices=sorted(RELATIONS), required=True)
    parser.add_argument("--type", dest="kind", help="Exact node_type or edge_type")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--lock-timeout", type=float, default=30.0)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--temp-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        result = read_graph(db=args.db, source_root=args.source_root,
                            attestations_dir=args.attestations_dir, cycle=args.cycle,
                            relation=args.relation, kind=args.kind, limit=args.limit,
                            lock_timeout=args.lock_timeout,
                            memory_limit=args.memory_limit,
                            temp_directory=args.temp_directory)
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
