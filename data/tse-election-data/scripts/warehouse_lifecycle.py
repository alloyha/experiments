#!/usr/bin/env python3
"""Cooperatively fenced warehouse maintenance and durable proof invalidation.

Pure standard library: `make clean-all` must work even without an active venv.
Never delete the stable .build.lock or .lifecycle sidecars. The lock protocol
matches the existing warehouse writer: exclusive flock on <db>.build.lock.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

BOOTSTRAP_EPOCH = "unrotated"


def _db_path(value: Path) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink():
        raise ValueError(f"Refusing symlink warehouse: {path}")
    return path


def epoch_path(db: Path) -> Path:
    db = _db_path(db)
    return db.with_name(db.name + ".lifecycle")


def lifecycle_epoch(db: Path) -> str:
    """Fail closed on malformed/symlink identity state; no side effects on read."""
    marker = epoch_path(db)
    if marker.is_symlink():
        raise ValueError(f"Warehouse lifecycle marker is a symlink: {marker}")
    if not marker.exists():
        return BOOTSTRAP_EPOCH
    if not marker.is_file():
        raise ValueError(f"Invalid warehouse lifecycle marker: {marker}")
    content = marker.read_text(encoding="ascii").strip()
    if len(content) != 32 or any(ch not in "0123456789abcdef" for ch in content):
        raise ValueError(f"Malformed warehouse lifecycle marker: {marker}")
    return content


@contextmanager
def maintenance_lock(db: Path, *, timeout: float):
    """Same stable inode as coordinated writers; never replace/unlink it."""
    if timeout < 0:
        raise ValueError("Lock timeout must be nonnegative")
    lock = db.with_name(db.name + ".build.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_symlink():
        raise ValueError(f"Warehouse lock is a symlink: {lock}")
    deadline = time.monotonic() + timeout
    with lock.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring warehouse lock: {lock}") from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def rotate_epoch(db: Path) -> str:
    """Called ONLY with the exclusive warehouse lock held, before deletion."""
    marker = epoch_path(db)
    if marker.is_symlink():
        raise ValueError(f"Warehouse lifecycle marker is a symlink: {marker}")
    token = uuid.uuid4().hex
    temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        parent_fd = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return token


def _remove(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink for destructive cleanup: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _inside(project: Path, path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    # A symlink in an INTERMEDIATE path component is also unsafe, even if it
    # resolves to another in-project directory containing source code.
    resolved = path.resolve(strict=False)
    if resolved != path or not resolved.is_relative_to(project):
        raise ValueError(f"Refusing cleanup path outside project or via symlink: {path}")
    if path == project:
        raise ValueError("Refusing to delete project root")
    return path


def _permitted_warehouse_path(project: Path, db: Path) -> None:
    """A Makefile-cleanable warehouse must be a DuckDB under generated data/."""
    data_root = project / "data"
    if not db.is_relative_to(data_root) or db == data_root or db.suffix != ".duckdb":
        raise ValueError(f"Unsafe warehouse cleanup path: {db}")


def _permitted_clean_all_paths(project: Path, *, db: Path, dbt_dir: Path,
                                raw: Path, facts: Path, artifacts: Path,
                                venv: Path) -> None:
    """Only derived data/ and fixed local tool directories can be removed.

    A path merely being inside the project is NOT sufficient: scripts/, .git/
    and source packages are inside the project too.  This validation happens
    before lock acquisition, epoch rotation, or any destructive operation.
    """
    data_root = project / "data"
    for label, path in (("warehouse", db), ("raw", raw),
                        ("facts", facts), ("dbt artifacts", artifacts)):
        if not path.is_relative_to(data_root) or path == data_root:
            raise ValueError(f"Unsafe clean-all {label} path outside generated data/: {path}")
    _permitted_warehouse_path(project, db)
    if dbt_dir != project / "tse_dbt":
        raise ValueError(f"Unsafe clean-all dbt directory: {dbt_dir}")
    if venv != project / ".venv":
        raise ValueError(f"Unsafe clean-all virtual environment: {venv}")


def clean_warehouse(db: Path, *, timeout: float = 30.0,
                    extra_paths: tuple[Path, ...] = ()) -> None:
    """Rotate BEFORE destructive action and keep the lock through every unlink.

    ``extra_paths`` is private to validated `clean-all` workflow. The public
    clean-warehouse operation only removes the DuckDB, WAL and spill directory.
    """
    db = _db_path(db)
    spill = Path(str(db) + ".tmp")
    wal = Path(str(db) + ".wal")
    for item in (db, spill, wal, *extra_paths):
        if item.is_symlink():
            raise ValueError(f"Refusing symlink for destructive cleanup: {item}")
    # A broad clean-all root must not contain the stable lock/epoch sidecars.
    # Otherwise deleting a parent directory would manufacture a fresh inode
    # after the lock is released, and old certificates might be resurrected.
    protected = (db, db.with_name(db.name + ".build.lock"), epoch_path(db))
    for item in extra_paths:
        if any(anchor == item or anchor.is_relative_to(item) for anchor in protected):
            raise ValueError(f"Cleanup path would remove warehouse lock/epoch: {item}")
    with maintenance_lock(db, timeout=timeout):
        # This marker is deliberately retained outside the DuckDB; an identical
        # replacement at the same path cannot reuse pre-clean certificates.
        rotate_epoch(db)
        for item in (db, wal, spill, *extra_paths):
            _remove(item)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("warehouse", "all"))
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--lock-timeout", type=float, default=30.0)
    p.add_argument("--project-root", type=Path)
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--fact-root", type=Path)
    p.add_argument("--dbt-dir", type=Path)
    p.add_argument("--venv", type=Path)
    p.add_argument("--artifacts-root", type=Path)
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args(argv)
    try:
        extras: tuple[Path, ...] = ()
        if args.mode == "all":
            if not args.confirm:
                raise ValueError("clean-all requires --confirm")
            if any(getattr(args, key) is None for key in
                   ("project_root", "raw_root", "fact_root", "dbt_dir", "venv", "artifacts_root")):
                raise ValueError("clean-all requires all scoped directory arguments")
            project = args.project_root.expanduser().resolve(strict=True)
            db = _inside(project, args.db)
            dbt_dir = _inside(project, args.dbt_dir)
            raw, facts, artifacts, venv = (_inside(project, pth) for pth in
                                           (args.raw_root, args.fact_root,
                                            args.artifacts_root, args.venv))
            _permitted_clean_all_paths(project, db=db, dbt_dir=dbt_dir,
                                       raw=raw, facts=facts, artifacts=artifacts,
                                       venv=venv)
            extras = (raw, facts, artifacts,
                      _inside(project, dbt_dir / "target"),
                      _inside(project, dbt_dir / "dbt_packages"),
                      _inside(project, dbt_dir / "logs"), venv)
        else:
            # The Makefile passes its project root so DUCKDB_PATH overrides
            # cannot delete source code. Direct callers without this optional
            # scope still select an explicit DuckDB file themselves.
            if args.project_root is not None:
                project = args.project_root.expanduser().resolve(strict=True)
                db = _inside(project, args.db)
                _permitted_warehouse_path(project, db)
            else:
                db = _db_path(args.db)
        clean_warehouse(db, timeout=args.lock_timeout, extra_paths=extras)
        print(f"PASS: {args.mode} cleanup fenced; warehouse certificate epoch rotated")
        return 0
    except (ValueError, OSError, TimeoutError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
