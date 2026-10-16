#!/usr/bin/env python3
"""Explicit, recoverable, cooperatively fenced promotion of an offline TSE Gold clone.

Never automatically certifies a promoted graph. Never alters published Parquet,
raw assets, source candidate, or the stable warehouse .build.lock.

A durable promotion journal survives SIGKILL. An operator can run `status` then
explicit `rollback`, even when power failed between epoch rotation and rename.

This is NOT protection against uncoordinated direct DuckDB writers or readers.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

# Direct execution from scripts/ must also resolve the sibling scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.warehouse_lifecycle import lifecycle_epoch, maintenance_lock, rotate_epoch


class PromotionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f'expected non-symlink regular file: {path}')


def assert_no_wal(db: Path) -> None:
    wal = Path(str(db) + '.wal')
    if wal.exists() or wal.is_symlink():
        raise PromotionError(f'WAL exists; stop DuckDB writers and checkpoint: {wal}')


def journal_path(original: Path) -> Path:
    return original.with_name(original.name + '.promotion.json')


def read_journal(original: Path) -> dict:
    path = journal_path(original)
    regular_file(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    required = {'version', 'original', 'candidate', 'backup', 'stage',
                'original_sha256', 'candidate_sha256', 'phase'}
    if data.get('version') != 1 or not required.issubset(data):
        raise PromotionError('unrecognized or incomplete journal; do not attempt rollback')
    if data['original'] != str(original):
        raise PromotionError('journal belongs to another warehouse')
    for name in ('candidate', 'backup', 'stage'):
        item = Path(data[name])
        if item.parent != original.parent or item == original or item.is_symlink():
            raise PromotionError(f'unsafe journal {name}: {item}')
    return data


def write_journal(original: Path, record: dict, *, create: bool = False) -> None:
    path = journal_path(original)
    if path.is_symlink():
        raise PromotionError(f'journal path is a symlink: {path}')
    if create and path.exists():
        raise PromotionError(f'promotion journal already exists: {path}')
    if not create:
        regular_file(path)
    temp = original.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if create:
            # Use hard link to guarantee no overwrite even with a concurrent caller.
            os.link(temp, path)
            temp.unlink()
        else:
            os.replace(temp, path)
        fsync_directory(original.parent)
    finally:
        temp.unlink(missing_ok=True)


def durable_copy(source: Path, destination: Path, expected_sha256: str) -> None:
    regular_file(source)
    if destination.exists() or destination.is_symlink():
        raise PromotionError(f'refusing to overwrite: {destination}')
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if sha256(destination) != expected_sha256:
        raise PromotionError(f'copy hash mismatch: {destination}')
    fsync_directory(destination.parent)


def assert_paths(original: Path, candidate: Path) -> None:
    if original == candidate or original.suffix != '.duckdb' or candidate.suffix != '.duckdb':
        raise PromotionError('source and target must be different DuckDB files')
    if original.parent != candidate.parent:
        raise PromotionError('promotion requires candidate in the same directory as target')
    regular_file(original)
    regular_file(candidate)
    assert_no_wal(original)
    assert_no_wal(candidate)
    if original.stat().st_dev != candidate.stat().st_dev:
        raise PromotionError('target/candidate are on different filesystems')


def preflight(original: Path, candidate: Path,
              original_hash: str, candidate_hash: str) -> dict:
    assert_paths(original, candidate)
    if journal_path(original).exists() or journal_path(original).is_symlink():
        raise PromotionError('unfinished or archived journal exists: inspect status before promotion')
    actual_original, actual_candidate = sha256(original), sha256(candidate)
    if actual_original != original_hash or actual_candidate != candidate_hash:
        raise PromotionError('SHA-256 drift from explicitly approved files')
    lifecycle_epoch(original)
    # Reserve space for backup + staged copy; do not depend on CoW/reflinks.
    required = original.stat().st_size + candidate.stat().st_size + 512 * 1024 * 1024
    if shutil.disk_usage(original.parent).free < required:
        raise PromotionError(f'insufficient free space: need {required} bytes')
    return {'original': str(original), 'candidate': str(candidate),
            'original_sha256': actual_original, 'candidate_sha256': actual_candidate,
            'required_free_bytes': required, 'mode': 'read_only_preflight'}


def locked_pair(original: Path, candidate: Path, timeout: float) -> ExitStack:
    # Caller must enter ExitStack using `with` and must never remove lock files.
    held = ExitStack()
    try:
        for db in sorted((original, candidate)):
            held.enter_context(maintenance_lock(db, timeout=timeout))
        return held
    except BaseException:
        held.close()
        raise


def validate_promoted(original: Path, *, expected_events: int,
                      expected_cycles: int, contracts: Path) -> None:
    """Small live-path smoke + frozen cycle regression; a failed check rolls back."""
    import duckdb
    from scripts.check_cycle_regressions import check_cycle
    with duckdb.connect(str(original), read_only=True) as con:
        count, no_evidence = con.execute('''
            select count(*), count(*) filter (
                where evidence_sources is null or trim(evidence_sources) = '')
            from main.dim_election''').fetchone()
        cycles = con.execute('select count(*) from main.dim_election_cycle').fetchone()[0]
        if (count, no_evidence, cycles) != (expected_events, 0, expected_cycles):
            raise PromotionError(f'catalog smoke mismatch: {(count, no_evidence, cycles)}')
        for relation in ('fact_candidate_votes', 'party_tally_coverage_gaps'):
            con.execute(f'describe select * from main.{relation}')
        for cycle in json.loads(contracts.read_text(encoding='utf-8'))['cycles']:
            failures = check_cycle(con, cycle)
            if failures:
                raise PromotionError(f'historical regression: {failures}')


def _rollback_locked(original: Path, record: dict) -> dict:
    backup = Path(record['backup'])
    regular_file(backup)
    if sha256(backup) != record['original_sha256']:
        raise PromotionError('backup checksum mismatch: refusing unsafe rollback')
    regular_file(original)
    current = sha256(original)
    if current not in (record['original_sha256'], record['candidate_sha256']):
        raise PromotionError('target changed since activation: refusing unsafe rollback')
    # Rotate first, even if promotion never swapped: no certificate survives recovery.
    rotate_epoch(original)
    if current != record['original_sha256']:
        recovery = original.with_name('.' + original.stem + '-rollback-' + uuid.uuid4().hex + '.duckdb')
        durable_copy(backup, recovery, record['original_sha256'])
        os.replace(recovery, original)
        fsync_directory(original.parent)
    if sha256(original) != record['original_sha256']:
        raise PromotionError('rollback hash verification failed')
    record['phase'] = 'ROLLED_BACK'
    record['restored_sha256'] = record['original_sha256']
    record['lifecycle_epoch'] = lifecycle_epoch(original)
    write_journal(original, record)
    return record


def promote(original: Path, candidate: Path, original_hash: str, candidate_hash: str,
            *, timeout: float, expected_events: int, expected_cycles: int,
            contracts: Path, post_validator=None) -> dict:
    # Ensure the verification environment exists BEFORE any irreversible change.
    if post_validator is None:
        import duckdb  # noqa: F401
        if not contracts.is_file():
            raise PromotionError(f'missing cycle contracts: {contracts}')
    with locked_pair(original, candidate, timeout):
        preflight(original, candidate, original_hash, candidate_hash)
        ident = uuid.uuid4().hex
        backup = original.with_name(original.stem + '.backup-' + ident + '.duckdb')
        stage = original.with_name('.' + original.stem + '.incoming-' + ident + '.duckdb')
        # Never begin a swap until both complete durable copies are verified.
        durable_copy(original, backup, original_hash)
        durable_copy(candidate, stage, candidate_hash)
        if sha256(original) != original_hash or sha256(candidate) != candidate_hash:
            raise PromotionError('database changed while preparing promotion')
        record = {
            'version': 1, 'original': str(original), 'candidate': str(candidate),
            'backup': str(backup), 'stage': str(stage),
            'original_sha256': original_hash, 'candidate_sha256': candidate_hash,
            'phase': 'PREPARED', 'id': ident,
        }
        write_journal(original, record, create=True)
        try:
            record['lifecycle_epoch'] = rotate_epoch(original)
            record['phase'] = 'EPOCH_ROTATED'
            write_journal(original, record)
            os.replace(stage, original)
            fsync_directory(original.parent)
            record['phase'] = 'SWAPPED_UNCERTIFIED'
            write_journal(original, record)
            if sha256(original) != candidate_hash:
                raise PromotionError('promoted warehouse SHA drift')
            if post_validator is None:
                validate_promoted(original, expected_events=expected_events,
                                  expected_cycles=expected_cycles, contracts=contracts)
            else:
                post_validator(original)
            record['phase'] = 'ACTIVE_UNCERTIFIED'
            write_journal(original, record)
            return record
        except Exception:
            # Rollback itself can fail (e.g. EIO), leaving journal+backup for
            # a separate explicit recovery attempt. Do not hide that error.
            _rollback_locked(original, record)
            raise


def rollback(original: Path, *, timeout: float) -> dict:
    record = read_journal(original)
    if record['phase'] == 'ROLLED_BACK':
        if sha256(original) != record['original_sha256']:
            raise PromotionError('rolled-back target hash drifted')
        return record
    with locked_pair(original, Path(record['candidate']), timeout):
        # Explicitly refuse a successful database that has subsequently changed.
        assert_no_wal(original)
        assert_no_wal(Path(record['candidate']))
        return _rollback_locked(original, record)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('preflight', 'promote', 'status', 'rollback'))
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--candidate', type=Path)
    ap.add_argument('--original-sha256')
    ap.add_argument('--candidate-sha256')
    ap.add_argument('--expected-events', type=int, default=198)
    ap.add_argument('--expected-cycles', type=int, default=5)
    ap.add_argument('--contracts', type=Path, default=Path('contracts/election_cycles.json'))
    ap.add_argument('--lock-timeout', type=float, default=30.0)
    ap.add_argument('--ack-quiescent', action='store_true')
    ap.add_argument('--confirm-promote', action='store_true')
    ap.add_argument('--confirm-rollback', action='store_true')
    args = ap.parse_args(argv)
    original = args.original.expanduser().absolute()
    try:
        if args.action == 'status':
            result = read_journal(original)
        elif args.action == 'rollback':
            if not (args.ack_quiescent and args.confirm_rollback):
                raise PromotionError('rollback requires --ack-quiescent and --confirm-rollback')
            result = rollback(original, timeout=args.lock_timeout)
        else:
            if not args.candidate or not args.original_sha256 or not args.candidate_sha256:
                raise PromotionError('candidate and both explicit SHA-256 values are required')
            candidate = args.candidate.expanduser().absolute()
            if args.action == 'preflight':
                result = preflight(original, candidate,
                                   args.original_sha256, args.candidate_sha256)
            else:
                if not (args.ack_quiescent and args.confirm_promote):
                    raise PromotionError('promote requires --ack-quiescent and --confirm-promote')
                result = promote(original, candidate, args.original_sha256,
                                 args.candidate_sha256, timeout=args.lock_timeout,
                                 expected_events=args.expected_events,
                                 expected_cycles=args.expected_cycles,
                                 contracts=args.contracts.expanduser().absolute())
        print(json.dumps({'result': 'PASS', **result}, indent=2))
        return 0
    except (PromotionError, ValueError, OSError, TimeoutError, ImportError) as exc:
        print(f'PROMOTION DENIED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
