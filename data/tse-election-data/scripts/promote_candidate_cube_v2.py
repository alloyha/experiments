#!/usr/bin/env python3
"""Versioned, append-only, recoverable activation of a previously certified OFFLINE cube clone.

Operational contract: all writers/readers must coordinate on stable POSIX lock inodes;
external quiescence is an operator requirement. No existing v1 journal is modified.
The initial 'prepare' action NEVER changes the live warehouse or lifecycle epoch.
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
import shutil
import stat
import sys
import time
import uuid
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import stage_candidate_cube as staging
from scripts import audit_staged_candidate_cube_contracts as offline
from scripts import preflight_candidate_cube_promotion_v2 as gate
from scripts.warehouse_lifecycle import lifecycle_epoch, rotate_epoch

SCOPE = 'candidate_cube_v2_append_only_recoverable_activation_not_freshness'
STATES = ('ADMITTED', 'PREPARED', 'EPOCH_ROTATION_INTENT', 'EPOCH_ROTATED',
          'SWAP_INTENT', 'SWAPPED_UNCERTIFIED', 'ACTIVE_UNCERTIFIED',
          'ROLLBACK_INTENT', 'ROLLED_BACK')
TERMINAL = {'ACTIVE_UNCERTIFIED', 'ROLLED_BACK'}


class PromotionV2Error(RuntimeError):
    pass


def sha(path: Path) -> str:
    evidence.regular_file(path)
    return evidence.digest(path)


def no_wal(db: Path) -> None:
    staging.assert_no_wal(db)


def state_dir(main: Path) -> Path:
    return main.with_name(main.name + gate.V2_STATE_SUFFIX)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive_existing_locks(paths: list[Path], timeout: float):
    """Do not create/replace any lock inode, and acquire warehouse-before-partition."""
    if timeout < 0:
        raise PromotionV2Error('invalid negative lock timeout')
    deadline = time.monotonic() + timeout
    with ExitStack() as held:
        for path in paths:
            evidence.regular_file(path)
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
            fd = os.open(path, flags)
            stream = held.enter_context(os.fdopen(fd, 'rb'))
            if not stat.S_ISREG(os.fstat(fd).st_mode) or not os.path.samestat(os.fstat(fd), path.stat()):
                raise PromotionV2Error('lock inode changed or nonregular: ' + str(path))
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise PromotionV2Error('busy cooperative writer/reader: ' + str(path)) from exc
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        yield


def lock_paths(main: Path, preview: Path, root: Path) -> list[Path]:
    # Clone has no .build.lock (the stage process does not create one); it is
    # pinned by SHA and the mandatory external quiescence acknowledged by operator.
    return staging.locks_for(main, preview, root)


def read_json(path: Path) -> dict:
    return gate.existing_json(path)


def validate_source_evidence(approval: dict, *, main: Path, preview: Path, clone: Path,
                             root: Path, allow_v2: bool) -> dict:
    """No nested flocks: caller holds ALL exclusive cooperating locks."""
    if (approval.get('result') != 'PASS' or approval.get('scope') != gate.SCOPE
        or approval.get('main') != str(main) or approval.get('preview') != str(preview)
        or approval.get('clone') != str(clone) or approval.get('source_root') != str(root)
        or approval.get('cycle_count') != 5):
        raise PromotionV2Error('V2 preflight proof does not approve this exact source identity')
    for db, field in ((main, 'main_sha256'), (preview, 'preview_sha256'),
                      (clone, 'clone_sha256')):
        no_wal(db)
        if sha(db) != approval[field]:
            raise PromotionV2Error(f'warehouse SHA differs from V2 approval: {db}')
    inputs = (
        ('manifest', 'manifest_sha256'), ('stage_report', 'stage_report_sha256'),
        ('offline_contract_report', 'offline_contract_report_sha256'),
    )
    for name, digest_name in inputs:
        p = Path(approval[name])
        if sha(p) != approval[digest_name]:
            raise PromotionV2Error(f'evidence SHA differs from V2 approval: {name}')
    manifest = read_json(Path(approval['manifest']))
    staging.check_identity(main, preview, root, manifest)
    if (approval.get('total_rows') != manifest['total_rows']
        or approval.get('total_nominal_votes') != manifest['total_nominal_votes']):
        raise PromotionV2Error('V2 approval totals differ from manifest')
    staging.verify_sources(main, preview, root, manifest)
    stage = read_json(Path(approval['stage_report']))
    offline_record = read_json(Path(approval['offline_contract_report']))
    parsed_manifest, parsed_stage = offline.validate_metadata(
        main=main, preview=preview, clone=clone, root=root,
        manifest_path=Path(approval['manifest']),
        manifest_hash=approval['manifest_sha256'],
        stage_path=Path(approval['stage_report']),
        clone_hash=approval['clone_sha256'], report=main.parent / '.v2-dryrun-nonexistent.json',
    )
    if parsed_manifest != manifest or parsed_stage != stage:
        raise PromotionV2Error('stage and manifest disagree')
    hashes = gate.validate_offline_report(
        record=offline_record, main=main, preview=preview, clone=clone, root=root,
        manifest_path=Path(approval['manifest']),
        manifest_hash=approval['manifest_sha256'],
        stage_path=Path(approval['stage_report']),
        stage_hash=approval['stage_report_sha256'], manifest=manifest)
    if hashes != approval.get('contract_sql_sha256'):
        raise PromotionV2Error('SQL contract evidence changed')
    if not allow_v2:
        legacy = gate.legacy_anchor(main, approval['main_sha256'])
        if legacy != approval.get('legacy_anchor'):
            raise PromotionV2Error('historical V1 lineage differs from preflight approval')
    return manifest


def write_event(folder: Path, ledger: list[tuple[Path, dict]], phase: str,
                *, anchor: dict | None = None) -> list[tuple[Path, dict]]:
    """Atomic create-only per transition; orphan .tmp files are never authoritative."""
    if phase not in STATES:
        raise PromotionV2Error('unrecognized journal phase')
    ordinal = len(ledger)
    name = f'{ordinal:04d}-{phase.lower().replace("_", "-")}.json'
    target = folder / name
    record = {'version': 2, 'sequence': ordinal, 'phase': phase,
              'previous_sha256': sha(ledger[-1][0]) if ledger else None,
              'written_utc': datetime.now(timezone.utc).isoformat()}
    if ordinal == 0:
        if phase != 'ADMITTED' or anchor is None:
            raise PromotionV2Error('first event must be ADMITTED with a full anchor')
        record['anchor'] = anchor
    elif anchor is not None:
        raise PromotionV2Error('only first event carries an anchor')
    temp = folder / f'.{name}.{uuid.uuid4().hex}.tmp'
    try:
        with temp.open('x', encoding='utf8') as out:
            json.dump(record, out, sort_keys=True, indent=2)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(temp, target, follow_symlinks=False)
        fsync_dir(folder)
    finally:
        temp.unlink(missing_ok=True)
    return ledger + [(target, record)]


def read_ledger(folder: Path) -> list[tuple[Path, dict]]:
    if folder.is_symlink() or not folder.is_dir():
        raise PromotionV2Error('missing/unsafe V2 state directory')
    files = sorted(folder.glob('*.json'))
    if not files:
        raise PromotionV2Error('V2 directory exists but has no authoritative event')
    ledger = []
    for i, f in enumerate(files):
        data = read_json(f)
        phase = data.get('phase')
        if (data.get('version') != 2 or data.get('sequence') != i or phase not in STATES
            or f.name != f'{i:04d}-{phase.lower().replace("_", "-")}.json'
            or data.get('previous_sha256') != (sha(ledger[-1][0]) if ledger else None)):
            raise PromotionV2Error('V2 append-only ledger corrupt/has gap or broken hash link')
        if i == 0 and (phase != 'ADMITTED' or not isinstance(data.get('anchor'), dict)):
            raise PromotionV2Error('V2 admission anchor missing')
        if i > 0 and 'anchor' in data:
            raise PromotionV2Error('only admission event may define immutable anchor')
        if i:
            prev = ledger[-1][1]['phase']
            if (phase == 'ROLLBACK_INTENT' and prev != 'ROLLED_BACK' and prev != 'ROLLBACK_INTENT'):
                pass
            elif (phase == 'ROLLED_BACK' and prev == 'ROLLBACK_INTENT'):
                pass
            elif (prev, phase) not in zip(STATES[:6], STATES[1:7]):
                raise PromotionV2Error(f'illegal V2 transition {prev} -> {phase}')
        ledger.append((f, data))
    return ledger


def durable_copy(source: Path, destination: Path, approved: str) -> None:
    evidence.regular_file(source)
    if destination.exists() or destination.is_symlink():
        raise PromotionV2Error('immutable durable target already exists: ' + str(destination))
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if sha(destination) != approved:
        raise PromotionV2Error('durable copy integrity failed')
    fsync_dir(destination.parent)


def checked_identity(main: Path, anchor: dict) -> str:
    if main.is_symlink() or not main.is_file():
        raise PromotionV2Error('unsafe/missing live Gold')
    current = sha(main)
    if current not in (anchor['main_sha256'], anchor['clone_sha256']):
        raise PromotionV2Error('live Gold bytes unknown; manual forensic recovery required')
    return current


def immutable_legacy(anchor: dict) -> None:
    legacy = anchor['legacy_anchor']
    if (sha(Path(legacy['journal'])) != legacy['journal_sha256']
        or sha(Path(legacy['legacy_backup'])) != legacy['legacy_backup_sha256']):
        raise PromotionV2Error('V1 history or backup mutated; refuse activation/recovery')


def prepare(*, main: Path, approval_path: Path, approval_sha: str,
            timeout: float) -> dict:
    approval_path = approval_path.absolute()
    main = main.absolute()
    if not gate.full_sha(approval_sha) or sha(approval_path) != approval_sha:
        raise PromotionV2Error('explicit V2 approval SHA mismatch')
    approval = read_json(approval_path)
    preview, clone, root = [Path(approval[key]) for key in ('preview', 'clone', 'source_root')]
    if main != Path(approval.get('main', '')):
        raise PromotionV2Error('V2 approval belongs to another live Gold')
    folder = state_dir(main)
    if folder.exists() or folder.is_symlink():
        raise PromotionV2Error('existing V2 state; use status/recovery, never start over')
    with exclusive_existing_locks(lock_paths(main, preview, root), timeout):
        if folder.exists() or folder.is_symlink():
            raise PromotionV2Error('V2 state appeared while acquiring writer locks')
        manifest = validate_source_evidence(approval, main=main, preview=preview,
                                            clone=clone, root=root, allow_v2=False)
        if approval['main_sha256'] == approval['clone_sha256']:
            raise PromotionV2Error('V2 clone must differ from currently active main')
        required = main.stat().st_size + clone.stat().st_size + 512 * 1024 * 1024
        if shutil.disk_usage(main.parent).free < required:
            raise PromotionV2Error('insufficient disk for immutable baseline+incoming')
        txid = uuid.uuid4().hex
        backup = main.with_name(main.stem + '.backup-v2-' + txid + '.duckdb')
        incoming = main.with_name('.' + main.stem + '.incoming-v2-' + txid + '.duckdb')
        anchor = {
            'id': txid, 'main': str(main), 'preview': str(preview), 'clone': str(clone),
            'root': str(root), 'backup': str(backup), 'incoming': str(incoming),
            'main_sha256': approval['main_sha256'],
            'clone_sha256': approval['clone_sha256'],
            'approval_path': str(approval_path), 'approval_sha256': approval_sha,
            'legacy_anchor': approval['legacy_anchor'],
            'epoch_before': lifecycle_epoch(main),
            'manifest_sha256': approval['manifest_sha256'],
            'expected_rows': manifest['total_rows'],
            'expected_votes': manifest['total_nominal_votes'],
        }
        # Seal first admission event INSIDE an unpublished work directory.
        # SIGKILL before rename cannot expose an empty authoritative state.
        work = folder.with_name('.' + folder.name + '.' + txid + '.building')
        work.mkdir(mode=0o700)
        fsync_dir(work.parent)
        write_event(work, [], 'ADMITTED', anchor=anchor)
        if folder.exists() or folder.is_symlink():
            raise PromotionV2Error('V2 state appeared during admission sealing')
        os.rename(work, folder)
        fsync_dir(folder.parent)
        ledger = read_ledger(folder)
        durable_copy(main, backup, anchor['main_sha256'])
        durable_copy(clone, incoming, anchor['clone_sha256'])
        if sha(main) != anchor['main_sha256'] or sha(clone) != anchor['clone_sha256']:
            raise PromotionV2Error('source changed while preparing copies')
        ledger = write_event(folder, ledger, 'PREPARED')
        return {'result': 'PASS', 'phase': 'PREPARED', 'scope': SCOPE,
                'state_dir': str(folder), 'backup': str(backup),
                'incoming': str(incoming), 'main_unchanged': True,
                'lifecycle_epoch_unchanged': lifecycle_epoch(main) == anchor['epoch_before']}


def _load_and_check(main: Path):
    ledger = read_ledger(state_dir(main))
    anchor = ledger[0][1]['anchor']
    if anchor.get('main') != str(main):
        raise PromotionV2Error('state belongs to another warehouse')
    for key in ('backup', 'incoming', 'clone'):
        value = Path(anchor[key])
        if value.is_symlink() or value.parent != main.parent or value == main or value.suffix != '.duckdb':
            raise PromotionV2Error(f'unsafe ledger {key} path')
    immutable_legacy(anchor)
    return ledger, anchor


def status(main: Path, timeout: float = 30.0) -> dict:
    main = main.absolute()
    # Even status must not race cooperative writer; journal is never modified.
    folder = state_dir(main)
    if not folder.is_dir():
        raise PromotionV2Error('no V2 state directory; run prepare after approved preflight')
    ledger, anchor = _load_and_check(main)
    preview, root = Path(anchor['preview']), Path(anchor['root'])
    with evidence.noncreating_shared_locks(lock_paths(main, preview, root), timeout):
        ledger, anchor = _load_and_check(main)
        phase = ledger[-1][1]['phase']
        current = checked_identity(main, anchor)
        epoch = lifecycle_epoch(main)
        expected = anchor['clone_sha256'] if phase == 'ACTIVE_UNCERTIFIED' else anchor['main_sha256']
        consistent = current == expected if phase in TERMINAL else None
        mutation_started = any(event[1]['phase'] == 'EPOCH_ROTATION_INTENT' for event in ledger)
        epoch_consistent = (epoch != anchor['epoch_before']) if mutation_started else (epoch == anchor['epoch_before'])
        wal = Path(str(main) + '.wal')
        wal_present = wal.exists() or wal.is_symlink()
        return {'result': 'PASS', 'phase': phase, 'scope': SCOPE,
                'state_dir': str(folder), 'events': len(ledger),
                'main_sha256': current, 'epoch': epoch,
                'phase_matches_live_bytes': consistent,
                'lifecycle_epoch_consistent': epoch_consistent,
                'wal_present': wal_present,
                'recovery_required': phase not in TERMINAL or not consistent or not epoch_consistent or wal_present,
                'v1_journal_preserved': True,
                'limitation': 'not a freshness certificate; cooperative locks only'}


def default_smoke(path: Path, anchor: dict) -> None:
    """Live-path read-only smoke, not a cube/graph freshness certificate."""
    import duckdb
    with duckdb.connect(str(path), read_only=True) as con:
        records = con.execute('SELECT cycle_id, count(*), sum(nominal_valid_votes) '
                              'FROM main.cube_candidate_performance_municipality '
                              'GROUP BY 1').fetchall()
        if len(records) != 5 or sum(r[1] for r in records) != anchor['expected_rows'] \
           or sum(int(r[2]) for r in records) != anchor['expected_votes']:
            raise PromotionV2Error('live candidate cube inventory failed')
        if con.execute('SELECT count(*) FROM main.dim_election').fetchone()[0] != 198:
            raise PromotionV2Error('historical event registry altered')
        if con.execute('SELECT count(*) FROM main.dim_election_cycle').fetchone()[0] != 5:
            raise PromotionV2Error('historical cycle registry altered')


def _rollback_locked(main: Path, ledger: list, anchor: dict,
                     hook: Callable[[str], None]) -> dict:
    phase = ledger[-1][1]['phase']
    if phase == 'ROLLED_BACK':
        if checked_identity(main, anchor) != anchor['main_sha256']:
            raise PromotionV2Error('rolled-back Gold bytes drifted')
        return {'result': 'PASS', 'phase': 'ROLLED_BACK', 'already_done': True}
    if phase == 'ACTIVE_UNCERTIFIED' or phase in STATES[:6] or phase == 'ROLLBACK_INTENT':
        current = checked_identity(main, anchor)
    else:
        raise PromotionV2Error('invalid rollback phase')
    mutation_started = any(item[1]['phase'] == 'EPOCH_ROTATION_INTENT' for item in ledger)
    if not mutation_started and current != anchor['main_sha256']:
        raise PromotionV2Error('unexpected active bytes before V2 mutation intent')
    if phase != 'ROLLBACK_INTENT':
        ledger = write_event(state_dir(main), ledger, 'ROLLBACK_INTENT')
    hook('after_rollback_intent')
    # ADMITTED / PREPARED never rotated the epoch, because activation first
    # durably records EPOCH_ROTATION_INTENT. Thus cancelling preparation can
    # preserve existing certificates and must not touch the live DuckDB.
    needs_epoch = mutation_started
    if needs_epoch:
        rotate_epoch(main)
        hook('after_rollback_epoch')
    if current == anchor['clone_sha256']:
        backup = Path(anchor['backup'])
        if sha(backup) != anchor['main_sha256']:
            raise PromotionV2Error('baseline backup corrupted; cannot restore')
        restore = main.with_name('.' + main.stem + '.restore-v2-' + uuid.uuid4().hex + '.duckdb')
        durable_copy(backup, restore, anchor['main_sha256'])
        os.replace(restore, main)
        fsync_dir(main.parent)
    if sha(main) != anchor['main_sha256']:
        raise PromotionV2Error('rollback restoration failed: main not baseline')
    hook('after_rollback_restore')
    ledger = write_event(state_dir(main), ledger, 'ROLLED_BACK')
    return {'result': 'PASS', 'phase': 'ROLLED_BACK', 'main_sha256': anchor['main_sha256'],
            'certificates_invalidated': True}


def recover(*, main: Path, timeout: float,
            hook: Callable[[str], None] = lambda _: None) -> dict:
    main = main.absolute()
    ledger, anchor = _load_and_check(main)
    with exclusive_existing_locks(lock_paths(main, Path(anchor['preview']), Path(anchor['root'])), timeout):
        ledger, anchor = _load_and_check(main)
        for db in (main, Path(anchor['preview'])):
            no_wal(db)
        admitted_only = ledger[-1][1]['phase'] == 'ADMITTED'
        backup = Path(anchor['backup'])
        if not admitted_only and sha(backup) != anchor['main_sha256']:
            raise PromotionV2Error('cannot recover without exact V2 baseline backup')
        # After admission, an unfinished copy may have been left behind. Such
        # bytes are untrusted but harmless if no activation intent exists.
        # Never overwrite or unlink them automatically.
        if admitted_only and checked_identity(main, anchor) != anchor['main_sha256']:
            raise PromotionV2Error('main drift during interrupted admission')
        return _rollback_locked(main, ledger, anchor, hook)


def activate(*, main: Path, timeout: float,
             validator: Callable[[Path, dict], None] = default_smoke,
             hook: Callable[[str], None] = lambda _: None) -> dict:
    main = main.absolute()
    ledger, anchor = _load_and_check(main)
    with exclusive_existing_locks(lock_paths(main, Path(anchor['preview']), Path(anchor['root'])), timeout):
        ledger, anchor = _load_and_check(main)
        if ledger[-1][1]['phase'] != 'PREPARED':
            raise PromotionV2Error('activation requires PREPARED; use status/recovery')
        if lifecycle_epoch(main) != anchor['epoch_before']:
            raise PromotionV2Error('lifecycle drift since V2 prepare')
        if checked_identity(main, anchor) != anchor['main_sha256']:
            raise PromotionV2Error('live baseline bytes changed after prepare')
        approval_path = Path(anchor['approval_path'])
        if sha(approval_path) != anchor['approval_sha256']:
            raise PromotionV2Error('approved preflight changed')
        approved = read_json(approval_path)
        validate_source_evidence(approved, main=main,
                                 preview=Path(anchor['preview']),
                                 clone=Path(anchor['clone']), root=Path(anchor['root']),
                                 allow_v2=True)
        for key, expected in (('backup', anchor['main_sha256']),
                              ('incoming', anchor['clone_sha256'])):
            if sha(Path(anchor[key])) != expected:
                raise PromotionV2Error('durable promotion copy drift: ' + key)
        if shutil.disk_usage(main.parent).free < 512 * 1024 * 1024:
            raise PromotionV2Error('insufficient rescue headroom')
        # All possible process-kill boundaries from here are discoverable from
        # the PREPARED/INTENT records, both pinned SHA values and epoch token.
        try:
            ledger = write_event(state_dir(main), ledger, 'EPOCH_ROTATION_INTENT')
            hook('after_epoch_rotation_intent')
            new_epoch = rotate_epoch(main)
            hook('after_epoch_rotation')
            ledger = write_event(state_dir(main), ledger, 'EPOCH_ROTATED')
            hook('after_epoch_rotated_event')
            ledger = write_event(state_dir(main), ledger, 'SWAP_INTENT')
            hook('after_swap_intent')
            os.replace(anchor['incoming'], main)
            fsync_dir(main.parent)
            hook('after_swap')
            ledger = write_event(state_dir(main), ledger, 'SWAPPED_UNCERTIFIED')
            if sha(main) != anchor['clone_sha256']:
                raise PromotionV2Error('swapped Gold differs from approved clone')
            hook('before_live_smoke')
            validator(main, anchor)
            hook('after_live_smoke')
            ledger = write_event(state_dir(main), ledger, 'ACTIVE_UNCERTIFIED')
            return {'result': 'PASS', 'phase': 'ACTIVE_UNCERTIFIED',
                    'scope': SCOPE, 'main_sha256': anchor['clone_sha256'],
                    'epoch': new_epoch, 'certified': False}
        except Exception:
            # Failure of rollback is deliberately visible; never claim a usable
            # main when restoration/recording is uncertain. SIGKILL needs explicit recover.
            current = read_ledger(state_dir(main))
            _rollback_locked(main, current, anchor, lambda _: None)
            raise


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'status', 'activate', 'recover'))
    p.add_argument('--main-db', type=Path, required=True)
    p.add_argument('--preflight-report', type=Path)
    p.add_argument('--preflight-sha256')
    p.add_argument('--lock-timeout', type=float, default=300.0)
    p.add_argument('--ack-quiescent', action='store_true')
    p.add_argument('--confirm-prepare', action='store_true')
    p.add_argument('--confirm-activate', action='store_true')
    p.add_argument('--confirm-recover', action='store_true')
    a = p.parse_args(argv)
    try:
        if a.action == 'status':
            report = status(a.main_db, timeout=a.lock_timeout)
        else:
            if not a.ack_quiescent:
                raise PromotionV2Error('all mutations require --ack-quiescent')
            if a.action == 'prepare':
                if not a.confirm_prepare or not a.preflight_report or not a.preflight_sha256:
                    raise PromotionV2Error('prepare requires approval report, SHA and confirmation')
                report = prepare(main=a.main_db, approval_path=a.preflight_report,
                                 approval_sha=a.preflight_sha256, timeout=a.lock_timeout)
            elif a.action == 'activate':
                if not a.confirm_activate:
                    raise PromotionV2Error('activation requires --confirm-activate')
                report = activate(main=a.main_db, timeout=a.lock_timeout)
            else:
                if not a.confirm_recover:
                    raise PromotionV2Error('recovery requires --confirm-recover')
                report = recover(main=a.main_db, timeout=a.lock_timeout)
        print(json.dumps(report, indent=2))
        return 0
    except (PromotionV2Error, gate.PreflightError, evidence.EvidenceError,
            staging.StageError, offline.OfflineAuditError,
            OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print(f'V2 PROMOTION DENIED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
