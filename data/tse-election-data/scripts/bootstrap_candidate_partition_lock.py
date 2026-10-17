#!/usr/bin/env python3
"""Explicit, durable bootstrap of one missing legacy candidate-fact .publish.lock.

This maintenance action creates ONLY a stable sidecar inode. It never writes to
DuckDB, Parquet, lifecycle markers, certificates or the promotion journal.
It is *not* a way to prove quiescence; operators must stop all other publishers,
including those using an unrelated warehouse path, before invoking bootstrap.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.candidate_cube_freshness import publication_signature, FreshnessError
from scripts.warehouse_lifecycle import maintenance_lock, lifecycle_epoch
from scripts.promote_conformed_gold import sha256, assert_no_wal, PromotionError

CYCLE_RE = re.compile(r"^(20\d{2}):(general|municipal)$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class BootstrapError(RuntimeError):
    pass


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"Missing or unsafe regular file: {path}")


def _partition(root: Path, cycle: str) -> Path:
    match = CYCLE_RE.fullmatch(cycle)
    if not match:
        raise BootstrapError("Expected explicit YYYY:general or YYYY:municipal")
    part = root / f"election_type={match[2]}" / f"year={match[1]}"
    if root.is_symlink() or part.is_symlink() or part.parent.is_symlink():
        raise BootstrapError("Symlinked source directories are not permitted")
    if not root.is_dir() or not part.is_dir():
        raise BootstrapError(f"Missing existing published partition: {part}")
    published = part / "data.parquet"
    if not published.is_file():
        raise BootstrapError(f"Published Parquet not found: {published}")
    return part


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_partition_lock(path: Path, timeout: float) -> tuple[int, bool]:
    """Open existing stable lock, or atomically create missing one, then flock it."""
    if path.is_symlink():
        raise BootstrapError(f"Refusing symlinked publication lock: {path}")
    created = False
    flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        if path.is_symlink():
            raise BootstrapError(f"Refusing symlinked publication lock: {path}")
        fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BootstrapError(f"Publication lock is not a regular file: {path}")
        if not os.path.samestat(os.fstat(fd), path.stat()):
            raise BootstrapError(f"Publication lock inode changed: {path}")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise BootstrapError(f"Timed out acquiring publication lock: {path}")
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        if created:
            os.fsync(fd)
            _fsync_directory(path.parent)
        return fd, created
    except BaseException:
        os.close(fd)
        raise


def bootstrap(main: Path, preview: Path, root: Path, cycle: str,
              main_hash: str, preview_hash: str, *, timeout: float) -> dict:
    """Requires affirmative CLI acknowledgements and an actual maintenance window."""
    if timeout <= 0 or not SHA_RE.fullmatch(main_hash) or not SHA_RE.fullmatch(preview_hash):
        raise BootstrapError("Positive timeout and two exact 64-char SHA-256 values required")
    if main == preview or not main.is_absolute() or not preview.is_absolute() or not root.is_absolute():
        raise BootstrapError("Distinct absolute paths required")
    if main.parent != preview.parent or root.parent != main.parent or root.name != "fact_candidate_votes":
        raise BootstrapError("Warehouse/physical-root directory layout differs from approved deployment")
    for db in (main, preview):
        _regular(db)
        assert_no_wal(db)
        _regular(Path(str(db) + ".build.lock"))
    if os.path.samefile(main, preview):
        raise BootstrapError("Main/preview must be distinct physical files")
    part = _partition(root, cycle)
    lock_path = part / ".publish.lock"
    if lock_path.is_symlink():
        raise BootstrapError("Refusing symlink publication lock")

    # Match the real publisher's lock order: warehouse(s) -> physical partition.
    # The selected publisher's partition lock is also held while verifying bytes.
    with ExitStack() as stack:
        for db in sorted((main, preview)):
            stack.enter_context(maintenance_lock(db, timeout=timeout))
        for db in (main, preview):
            assert_no_wal(db)
        if sha256(main) != main_hash or sha256(preview) != preview_hash:
            raise BootstrapError("Warehouse SHA-256 differs from approved hashes")
        epochs_before = [lifecycle_epoch(db) for db in (main, preview)]
        source_before = publication_signature(root, cycle)
        fd, created = _exclusive_partition_lock(lock_path, timeout)
        stack.callback(os.close, fd)
        stack.callback(fcntl.flock, fd, fcntl.LOCK_UN)
        source_after = publication_signature(root, cycle)
        if (source_before != source_after or sha256(main) != main_hash or
                sha256(preview) != preview_hash or
                epochs_before != [lifecycle_epoch(db) for db in (main, preview)]):
            raise BootstrapError("Source or warehouse changed during lock initialization")
        if not os.path.samestat(os.fstat(fd), lock_path.stat()):
            raise BootstrapError("Stable publication lock inode changed during initialization")
        return {
            "result": "PASS", "scope": "cooperative_lock_sidecar_bootstrap_only",
            "cycle": cycle, "lock": str(lock_path), "created": created,
            "source_signature": source_after, "main_sha256": main_hash,
            "preview_sha256": preview_hash,
            "warehouse_epoch_unchanged": True,
            "limitations": "Requires externally guaranteed quiescence; no Parquet/DB changes or certification",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-db", type=Path, required=True)
    parser.add_argument("--preview-db", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--main-sha256", required=True)
    parser.add_argument("--preview-sha256", required=True)
    parser.add_argument("--lock-timeout", type=float, default=60.0)
    parser.add_argument("--ack-quiescent", action="store_true")
    parser.add_argument("--confirm-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    if not (args.ack_quiescent and args.confirm_bootstrap):
        print("LOCK BOOTSTRAP DENIED: --ack-quiescent and --confirm-bootstrap required", file=sys.stderr)
        return 2
    try:
        result = bootstrap(args.main_db.expanduser().absolute(),
                           args.preview_db.expanduser().absolute(),
                           args.source_root.expanduser().absolute(), args.cycle,
                           args.main_sha256, args.preview_sha256,
                           timeout=args.lock_timeout)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (BootstrapError, FreshnessError, PromotionError, ValueError, OSError, TimeoutError) as exc:
        print(f"LOCK BOOTSTRAP DENIED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
