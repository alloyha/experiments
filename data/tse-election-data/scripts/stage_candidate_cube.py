#!/usr/bin/env python3
"""Stage a fully verified candidate cube in a NEW offline clone, never main.

No data file on main/preview/Parquet paths is opened for writing. The snapshot
is permitted only by an exact five-cycle manifest and its full SHA-256.
The stage path is private until all five cycles pass exact multiset checks.
No production promotion, certificate issuance, or journal lifecycle mutation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import consolidate_candidate_cube_transfer as evidence

TABLE = "cube_candidate_performance_municipality"
READ_SCOPE = "five_cycle_readonly_transfer_eligibility_not_freshness_or_publication"
STAGE_SCOPE = "offline_cube_clone_exact_transfer_not_certified_not_promoted"
MEMORY = re.compile(r"[1-9][0-9]*(?:MB|GB)\Z")


class StageError(RuntimeError):
    pass


def log(message: str, *, start: float) -> None:
    print(f"[cube-stage +{time.monotonic() - start:.1f}s] {message}", file=sys.stderr, flush=True)


def normalize(path: Path) -> Path:
    return path.expanduser().absolute()


def assert_no_wal(db: Path) -> None:
    wal = Path(str(db) + ".wal")
    if wal.exists() or wal.is_symlink():
        raise StageError(f"WAL present: {wal}")


def check_identity(main: Path, preview: Path, source: Path, manifest: dict) -> None:
    if (manifest.get("result") != "PASS" or manifest.get("scope") != READ_SCOPE
        or manifest.get("main") != str(main)
        or manifest.get("preview") != str(preview)
        or manifest.get("source_root") != str(source)):
        raise StageError("manifest identity/scope mismatch")
    if main == preview or main.parent != preview.parent:
        raise StageError("main and preview must be distinct in the same directory")
    if source.is_symlink() or not source.is_dir():
        raise StageError("source root is missing or symlinked")
    if (manifest.get("cycle_count") != 5
        or not isinstance(manifest.get("cycles"), dict)
        or set(manifest["cycles"]) != set(evidence.CYCLES)):
        raise StageError("incomplete five-cycle manifest")
    rows = votes = 0
    for cycle in evidence.CYCLES:
        entry = manifest["cycles"][cycle]
        if (not isinstance(entry, dict)
            or type(entry.get("rows")) is not int or entry["rows"] <= 0
            or type(entry.get("nominal_votes")) is not int or entry["nominal_votes"] <= 0
            or entry.get("mismatch_rows") != 0 or entry.get("columns") != 29
            or not isinstance(entry.get("source_signature"), dict)):
            raise StageError(f"bad manifest cycle: {cycle}")
        rows += entry["rows"]
        votes += entry["nominal_votes"]
    if (type(manifest.get("total_rows")) is not int or rows != manifest["total_rows"]
        or type(manifest.get("total_nominal_votes")) is not int
        or votes != manifest["total_nominal_votes"]):
        raise StageError("manifest totals inconsistent")
    if (not evidence.SHA256_RE.fullmatch(str(manifest.get("main_sha256", "")))
        or not evidence.SHA256_RE.fullmatch(str(manifest.get("preview_sha256", "")))):
        raise StageError("manifest contains malformed source hashes")


def locks_for(main: Path, preview: Path, root: Path) -> list[Path]:
    return ([evidence.lock_path(db) for db in sorted((main, preview))]
            + [evidence.partition_for(root, c) / ".publish.lock"
               for c in sorted(evidence.CYCLES)])


def verify_sources(main: Path, preview: Path, root: Path, manifest: dict) -> None:
    for db, field in ((main, "main_sha256"), (preview, "preview_sha256")):
        evidence.regular_file(db)
        assert_no_wal(db)
        if evidence.digest(db) != manifest[field]:
            raise StageError(f"warehouse SHA-256 differs from manifest: {db}")
    for cycle in evidence.CYCLES:
        current = evidence.publication_signature(evidence.partition_for(root, cycle))
        if current != manifest["cycles"][cycle]["source_signature"]:
            raise StageError(f"Parquet publication drift: {cycle}")


def assert_destination(main: Path, preview: Path, output: Path, report: Path,
                       manifest_path: Path) -> None:
    if (output.suffix != ".duckdb" or output.parent != main.parent
        or output in (main, preview) or output == manifest_path
        or output == report):
        raise StageError("output must be a distinct .duckdb in the warehouse directory")
    if output.exists() or output.is_symlink() or Path(str(output) + ".wal").exists():
        raise StageError(f"output already exists or has WAL: {output}")
    if report.exists() or report.is_symlink():
        raise StageError(f"report already exists: {report}")
    if report.parent != output.parent and not report.parent.is_dir():
        raise StageError("report parent directory missing")
    if not output.parent.is_dir():
        raise StageError("output parent directory missing")


def durable_copy(source: Path, destination: Path, approved: str) -> None:
    evidence.regular_file(source)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if evidence.digest(destination) != approved:
        raise StageError("staged original failed SHA verification")
    fsync_file(destination)
    fsync_dir(destination.parent)


def fsync_file(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def table_schema(con, name: str) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {name}").fetchall()]


def expected_counts(manifest: dict) -> dict:
    return {c: (manifest["cycles"][c]["rows"], manifest["cycles"][c]["nominal_votes"])
             for c in evidence.CYCLES}


def exact_delta_query(lhs: str, rhs: str) -> str:
    """Compute the cardinality of the duplicate-sensitive multiset difference."""
    return ("SELECT count(*) FROM ("
            f"SELECT * FROM {lhs} WHERE cycle_id = ? "
            "EXCEPT ALL "
            f"SELECT * FROM {rhs} WHERE cycle_id = ?"
            ")")


def import_and_verify(stage: Path, preview: Path, manifest: dict,
                      *, memory_limit: str, start: float) -> tuple[str, dict]:
    """Only stage DB is writable. Exact cross-catalog checks for every cycle."""
    import duckdb
    source = f"preview_source.main.{TABLE}"
    target = f"main.{TABLE}"
    counts: dict[str, dict] = {}
    log("opening new clone and attaching preview READ_ONLY", start=start)
    con = duckdb.connect(str(stage), read_only=False)
    attached = False
    try:
        con.execute(f"SET memory_limit = {quote_sql_string(memory_limit)}")
        con.execute(f"ATTACH {quote_sql_string(str(preview))} AS preview_source (READ_ONLY)")
        attached = True
        try:
            found = con.execute("""
                SELECT table_type FROM information_schema.tables
                WHERE table_catalog = current_database()
                AND table_schema = 'main' AND table_name = ?
            """, [TABLE]).fetchall()
            if found:
                raise StageError("Gold clone already has a relation named candidate cube")
            preview_schema = table_schema(con, source)
            if len(preview_schema) != 29:
                raise StageError(f"preview has {len(preview_schema)} columns; expected 29")
            log("copying persisted candidate-cube table into clone", start=start)
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(f"CREATE TABLE {target} AS SELECT * FROM {source}")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            if table_schema(con, target) != preview_schema:
                raise StageError("copy schema differs from source including types/order")
            rows = con.execute(
                f"SELECT cycle_id, count(*), sum(nominal_valid_votes) "
                f"FROM {target} GROUP BY cycle_id").fetchall()
            observed = {cycle: (count, int(votes)) for cycle, count, votes in rows}
            if observed != expected_counts(manifest):
                raise StageError(f"clone cycle inventory differs: {observed}")
            # Idempotence/equivalence of the *copy* is independent from the
            # earlier model-vs-Parquet eligibility proof.
            for cycle in evidence.CYCLES:
                log(f"exact 29-column comparison: {cycle}", start=start)
                missing = con.execute(exact_delta_query(source, target), [cycle, cycle]).fetchone()[0]
                extra = con.execute(exact_delta_query(target, source), [cycle, cycle]).fetchone()[0]
                if missing or extra:
                    raise StageError(f"staged cube differs from preview: {cycle}, {(missing, extra)}")
                counts[cycle] = {"rows": observed[cycle][0], "nominal_votes": observed[cycle][1],
                                 "missing_rows": int(missing), "extra_rows": int(extra)}
            log("all five exact multiset comparisons passed", start=start)
        finally:
            if attached:
                con.execute("DETACH preview_source")
    finally:
        con.close()
    assert_no_wal(stage)
    return duckdb.__version__, counts


def stage_cube(*, main: Path, preview: Path, root: Path, manifest_path: Path,
               manifest_sha256: str, output: Path, report: Path,
               memory_limit: str, lock_timeout: float,
               acknowledgements: bool) -> dict:
    start = time.monotonic()
    main, preview, root, manifest_path, output, report = map(
        normalize, (main, preview, root, manifest_path, output, report))
    if not acknowledgements:
        raise StageError("explicit --ack-quiescent required; external writers must be stopped")
    if not MEMORY.fullmatch(memory_limit):
        raise StageError("memory limit format must be a positive integer + MB or GB")
    evidence.regular_file(manifest_path)
    if not evidence.SHA256_RE.fullmatch(manifest_sha256):
        raise StageError("full manifest SHA-256 required")
    if evidence.digest(manifest_path) != manifest_sha256:
        raise StageError("manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise StageError("manifest must be a JSON object")
    check_identity(main, preview, root, manifest)
    assert_destination(main, preview, output, report, manifest_path)
    # Intentionally conservatively reserve preview size, main size, and 1GiB.
    required = preview.stat().st_size + main.stat().st_size + 1024**3
    if shutil.disk_usage(output.parent).free < required:
        raise StageError(f"disk capacity insufficient; require {required} free bytes")
    # Fail early on missing DuckDB before staging bytes.
    import duckdb  # noqa: F401
    stage = output.with_name("." + output.stem + ".incoming-" + uuid.uuid4().hex + ".duckdb")
    log("acquiring cooperative shared locks for BOTH warehouses and five partitions", start=start)
    with evidence.noncreating_shared_locks(locks_for(main, preview, root), lock_timeout):
        assert_destination(main, preview, output, report, manifest_path)
        verify_sources(main, preview, root, manifest)
        log("manifest and live source identities verified; cloning Gold", start=start)
        # If SIGKILL occurs, only a hidden orphan stage may remain. The main,
        # preview, physical Parquets, promotion journal and graph cert are untouched.
        durable_copy(main, stage, manifest["main_sha256"])
        version, cycles = import_and_verify(stage, preview, manifest,
                                           memory_limit=memory_limit, start=start)
        verify_sources(main, preview, root, manifest)
        evidence.regular_file(manifest_path)
        if evidence.digest(manifest_path) != manifest_sha256:
            raise StageError("manifest changed while staging")
        fsync_file(stage)
        fsync_dir(stage.parent)
        staged_hash = evidence.digest(stage)
        # Create-only publication, never replace another stage or the main.
        # A SIGKILL between linking and writing the report leaves an
        # untrusted candidate: do not treat output DB alone as certification.
        if output.exists() or output.is_symlink():
            raise StageError("output appeared during validation")
        os.link(stage, output, follow_symlinks=False)
        fsync_dir(output.parent)
        stage.unlink()
        fsync_dir(output.parent)
        log("published verified OFFLINE CLONE (not the main warehouse)", start=start)
        result = {
            "result": "PASS", "scope": STAGE_SCOPE,
            "main": str(main), "preview": str(preview),
            "source_root": str(root), "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256, "main_sha256": manifest["main_sha256"],
            "preview_sha256": manifest["preview_sha256"],
            "output_db": str(output), "output_sha256": staged_hash,
            "duckdb_version": version, "schema_columns": 29,
            "cycles": cycles, "total_rows": sum(x["rows"] for x in cycles.values()),
            "total_nominal_votes": sum(x["nominal_votes"] for x in cycles.values()),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": ("offline clone only; cooperative locks and operator quiescence; "
                            "no production promotion, cube freshness certificates, or graph recertification"),
        }
        evidence.write_create_only(report, result)
        log("created separate create-only stage report", start=start)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-db", type=Path, required=True)
    ap.add_argument("--preview-db", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--output-db", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--memory-limit", default="4GB")
    ap.add_argument("--lock-timeout", type=float, default=300.0)
    ap.add_argument("--ack-quiescent", action="store_true")
    args = ap.parse_args(argv)
    try:
        record = stage_cube(
            main=args.main_db, preview=args.preview_db,
            root=args.source_root, manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            output=args.output_db, report=args.report,
            memory_limit=args.memory_limit, lock_timeout=args.lock_timeout,
            acknowledgements=args.ack_quiescent,
        )
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    except (StageError, evidence.EvidenceError, OSError, ValueError,
            KeyError, TypeError, ImportError) as err:
        print(f"CUBE STAGE DENIED: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
