#!/usr/bin/env python3
"""Read-only conformance evidence closeout for five historic candidate-cube cycles.

Validates five *existing* independent audit reports against their actual current
warehouse and Parquet bytes under cooperative shared read locks. Does not
materialize or certify any cube and never mutates the two DuckDB files.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

CYCLES = ("2018:general", "2020:municipal", "2022:general", "2024:municipal", "2026:general")
SCOPE = "readonly_single_cycle_transfer_eligibility_not_freshness_certification"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for batch in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(batch)
    return h.hexdigest()


def regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"missing, symlinked, or nonregular file: {path}")


def lock_path(db: Path) -> Path:
    return db.with_name(db.name + ".build.lock")


def partition_for(root: Path, cycle: str) -> Path:
    year, election_type = cycle.split(":")
    return root / f"election_type={election_type}" / f"year={year}"


@contextmanager
def noncreating_shared_locks(paths: list[Path], timeout: float):
    """Existing lock inode only, same warehouse-before-partition order as writers."""
    if timeout < 0:
        raise EvidenceError("lock timeout must be nonnegative")
    deadline = time.monotonic() + timeout
    with ExitStack() as stack:
        for path in paths:
            regular_file(path)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise EvidenceError(f"cannot open existing lock: {path}: {exc}") from exc
            stream = stack.enter_context(os.fdopen(fd, "rb"))
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise EvidenceError(f"lock not regular: {path}")
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise EvidenceError(f"timed out waiting for lock: {path}") from exc
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            stack.enter_context(_hold(stream))
        yield


@contextmanager
def _hold(stream):
    """Stream stays alive through lock scope, then is unlocked and closed."""
    try:
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        # Note: fdopen is closed by its with block, not here.


def publication_signature(partition: Path) -> dict:
    pointer = partition / "data.parquet"

    def generation() -> str:
        if pointer.is_symlink():
            name = os.readlink(pointer)
            if Path(name).name != name or name in {".", ".."}:
                raise EvidenceError(f"unsafe Parquet generation pointer: {pointer}")
            regular_file(partition / name)
            return name
        regular_file(pointer)
        return "legacy-regular-file"

    before = generation()
    size = pointer.stat().st_size
    sha = digest(pointer)
    if before != generation() or pointer.stat().st_size != size:
        raise EvidenceError(f"Parquet publication changed during hash: {pointer}")
    return {"generation": before, "sha256": sha, "bytes": size}


def read_report(path: Path) -> tuple[dict, str]:
    regular_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvidenceError(f"report must be JSON object: {path}")
    return data, digest(path)


def _positive_number(value: object) -> bool:
    return type(value) is int and value > 0


def _as_string_path(value: object, *, expected: Path) -> bool:
    return isinstance(value, str) and value == str(expected)


def consolidate(*, main: Path, preview: Path, root: Path, report_paths: list[Path],
                main_sha256: str, preview_sha256: str, lock_timeout: float) -> dict:
    main, preview, root = (p.expanduser().absolute() for p in (main, preview, root))
    if main == preview or main.parent != preview.parent:
        raise EvidenceError("main and preview must be distinct DuckDB files in the same directory")
    if main.suffix != ".duckdb" or preview.suffix != ".duckdb":
        raise EvidenceError("warehouse files must have .duckdb suffix")
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError("source root must be a real directory")
    for p in (main, preview):
        regular_file(p)
        wal = Path(str(p) + ".wal")
        if wal.exists() or wal.is_symlink():
            raise EvidenceError(f"WAL present, decline read-only evidence closeout: {wal}")
    if not SHA256_RE.fullmatch(main_sha256) or not SHA256_RE.fullmatch(preview_sha256):
        raise EvidenceError("explicit full SHA-256 hashes are required")
    if len(report_paths) != len(CYCLES):
        raise EvidenceError(f"exactly {len(CYCLES)} explicit report files required")
    loaded = [(*read_report(path.expanduser().absolute()), path.expanduser().absolute())
              for path in report_paths]
    by_cycle: dict[str, tuple[dict, str, Path]] = {}
    for record, report_digest, path in loaded:
        cycle = record.get("cycle")
        if cycle not in CYCLES or cycle in by_cycle:
            raise EvidenceError(f"unexpected or duplicate cycle in reports: {cycle}")
        by_cycle[cycle] = (record, report_digest, path)
    if set(by_cycle) != set(CYCLES):
        raise EvidenceError("missing cycle evidence")

    # Writers lock the warehouse first, then the relevant physical partition.
    paths = [lock_path(p) for p in sorted((main, preview))]
    paths.extend(partition_for(root, cycle) / ".publish.lock" for cycle in sorted(CYCLES))
    with noncreating_shared_locks(paths, timeout=lock_timeout):
        for p in (main, preview):
            regular_file(p)
            wal = Path(str(p) + ".wal")
            if wal.exists() or wal.is_symlink():
                raise EvidenceError(f"active warehouse WAL: {wal}")
        if digest(main) != main_sha256 or digest(preview) != preview_sha256:
            raise EvidenceError("live warehouse hash does not match approved identity")
        expected_rows: dict[str, int] | None = None
        result = {}
        for cycle in CYCLES:
            record, report_digest, path = by_cycle[cycle]
            expected = partition_for(root, cycle) / "data.parquet"
            if (record.get("result") != "PASS" or record.get("scope") != SCOPE
                or not _as_string_path(record.get("main"), expected=main)
                or not _as_string_path(record.get("preview"), expected=preview)
                or not _as_string_path(record.get("physical_parquet"), expected=expected)
                or record.get("main_sha256") != main_sha256
                or record.get("preview_sha256") != preview_sha256):
                raise EvidenceError(f"invalid eligibility report identity or result: {cycle}")
            if (record.get("selected_schema_columns") != 29
                or type(record.get("multiset_mismatch_rows")) is not int
                or record["multiset_mismatch_rows"] != 0
                or not _positive_number(record.get("selected_cube_rows"))
                or not _positive_number(record.get("selected_nominal_votes"))):
                raise EvidenceError(f"invalid 29-column multiset evidence: {cycle}")
            inv = record.get("published_cycles")
            if not isinstance(inv, list) or set(inv) != set(CYCLES) or len(inv) != len(CYCLES):
                raise EvidenceError(f"published-cycle inventory mismatch: {cycle}")
            rows = record.get("cube_rows_by_cycle")
            if not isinstance(rows, dict) or set(rows) != set(CYCLES) or not all(
                _positive_number(n) for n in rows.values()
            ) or rows[cycle] != record["selected_cube_rows"]:
                raise EvidenceError(f"cube-row inventory mismatch: {cycle}")
            if expected_rows is None:
                expected_rows = rows
            elif rows != expected_rows:
                raise EvidenceError(f"cross-report cube-row inventory changed: {cycle}")
            observed = publication_signature(partition_for(root, cycle))
            if record.get("source_signature") != observed:
                raise EvidenceError(f"published Parquet changed or differs from report: {cycle}")
            result[cycle] = {
                "report_path": str(path), "report_sha256": report_digest,
                "rows": record["selected_cube_rows"],
                "nominal_votes": record["selected_nominal_votes"],
                "columns": record["selected_schema_columns"],
                "mismatch_rows": 0,
                "source_signature": observed,
                "comparison_mode": record.get("comparison_mode", "except-all"),
                "evaluated_at_utc": record.get("evaluated_at_utc"),
            }
        # Revalidate physical identity at the end of the shared-lock window.
        if digest(main) != main_sha256 or digest(preview) != preview_sha256:
            raise EvidenceError("warehouse drift during evidence closeout")
        for cycle in CYCLES:
            if publication_signature(partition_for(root, cycle)) != result[cycle]["source_signature"]:
                raise EvidenceError(f"publication drift during evidence closeout: {cycle}")
    return {
        "result": "PASS", "scope": "five_cycle_readonly_transfer_eligibility_not_freshness_or_publication",
        "main": str(main), "preview": str(preview), "source_root": str(root),
        "main_sha256": main_sha256, "preview_sha256": preview_sha256,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycles": result,
        "cycle_count": len(CYCLES),
        "total_rows": sum(item["rows"] for item in result.values()),
        "total_nominal_votes": sum(item["nominal_votes"] for item in result.values()),
        "limitations": "cooperative locks only; confirms historical reports against present bytes; does not certify freshness, create or authorize cube transfer",
    }


def write_create_only(path: Path, manifest: dict) -> None:
    """Evidence export is create-only; never overwrite previous decisions."""
    if path.is_symlink() or path.exists():
        raise EvidenceError(f"output already exists or unsafe: {path}")
    if not path.parent.is_dir():
        raise EvidenceError(f"output directory missing: {path.parent}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    dir_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-db", type=Path, required=True)
    parser.add_argument("--preview-db", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha256", required=True)
    parser.add_argument("--preview-sha256", required=True)
    parser.add_argument("--report", type=Path, action="append", required=True,
                        help="Exactly one existing eligibility JSON for each of the five cycles")
    parser.add_argument("--output", type=Path, required=True,
                        help="New create-only JSON manifest path")
    parser.add_argument("--lock-timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        outcome = consolidate(main=args.main_db, preview=args.preview_db,
                              root=args.source_root, report_paths=args.report,
                              main_sha256=args.main_sha256,
                              preview_sha256=args.preview_sha256,
                              lock_timeout=args.lock_timeout)
        write_create_only(args.output.expanduser().absolute(), outcome)
        print(json.dumps({"result": "PASS", "manifest": str(args.output.expanduser().absolute()),
                          "cycle_count": outcome["cycle_count"],
                          "total_rows": outcome["total_rows"],
                          "total_nominal_votes": outcome["total_nominal_votes"]}, indent=2))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"CUBE EVIDENCE DENIED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
