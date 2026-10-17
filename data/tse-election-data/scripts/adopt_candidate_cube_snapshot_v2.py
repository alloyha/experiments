#!/usr/bin/env python3
"""Adopt an activated V2 snapshot into NEW candidate-cube freshness certificates.

No dbt run, Parquet republish, or database mutation. Reuse the existing
candidate_cube_freshness certificate format and live input/output fingerprints.
Only the previously approved clone's *identical bytes* may be adopted.

"check" is safe in PREPARED and ACTIVE_UNCERTIFIED; "adopt" requires an
active V2 ledger and external quiescence, then publishes a complete new
certificate directory atomically (Linux renameat2 RENAME_NOREPLACE).
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import preflight_candidate_cube_promotion_v2 as gate
from scripts import promote_candidate_cube_v2 as v2
from scripts import stage_candidate_cube as staging
from scripts.warehouse_lifecycle import lifecycle_epoch

SCOPE = 'candidate_cube_v2_snapshot_adoption_no_dbt_run'


class AdoptionError(RuntimeError):
    pass


def _read(path: Path) -> dict:
    return gate.existing_json(path)


def _check_under_locks(main: Path, *, require_active: bool) -> tuple[dict, dict, dict]:
    """Caller holds stable warehouse-before-partition locks throughout."""
    ledger, anchor = v2._load_and_check(main)
    phase = ledger[-1][1]['phase']
    if phase not in ({'ACTIVE_UNCERTIFIED'} if require_active else {'PREPARED', 'ACTIVE_UNCERTIFIED'}):
        raise AdoptionError(f'V2 ledger phase is not eligible for snapshot adoption: {phase}')
    approval_path = Path(anchor['approval_path'])
    if v2.sha(approval_path) != anchor['approval_sha256']:
        raise AdoptionError('V2 approval changed since prepare')
    approval = _read(approval_path)
    preview, clone, root = (Path(anchor[key]) for key in ('preview', 'clone', 'root'))
    if (approval.get('result') != 'PASS' or approval.get('scope') != gate.SCOPE
        or approval.get('main') != str(main)
        or approval.get('preview') != str(preview)
        or approval.get('clone') != str(clone)
        or approval.get('source_root') != str(root)
        or approval.get('legacy_anchor') != anchor['legacy_anchor']
        or approval.get('main_sha256') != anchor['main_sha256']
        or approval.get('clone_sha256') != anchor['clone_sha256']):
        raise AdoptionError('immutable approval/ledger anchor conflict')
    for db in (main, preview, clone):
        v2.no_wal(db)
    expected_main = anchor['clone_sha256'] if phase == 'ACTIVE_UNCERTIFIED' else anchor['main_sha256']
    if v2.sha(main) != expected_main:
        raise AdoptionError('live main SHA does not correspond to V2 phase')
    if v2.sha(preview) != approval['preview_sha256'] or v2.sha(clone) != anchor['clone_sha256']:
        raise AdoptionError('preview or clone hash drift')
    if v2.sha(Path(anchor['backup'])) != anchor['main_sha256']:
        raise AdoptionError('V2 baseline backup drift')
    epoch = lifecycle_epoch(main)
    if phase == 'PREPARED':
        if epoch != anchor['epoch_before']:
            raise AdoptionError('PREPARED epoch drift')
    elif epoch == anchor['epoch_before']:
        raise AdoptionError('activated warehouse has not rotated its lifecycle epoch')
    if approval.get('manifest_sha256') != anchor['manifest_sha256']:
        raise AdoptionError('manifest identity disagrees with V2 admission')
    sources = (('manifest', 'manifest_sha256'), ('stage_report', 'stage_report_sha256'),
               ('offline_contract_report', 'offline_contract_report_sha256'))
    for name, digest_field in sources:
        if v2.sha(Path(approval[name])) != approval[digest_field]:
            raise AdoptionError(f'upstream proof mutated: {name}')
    manifest = _read(Path(approval['manifest']))
    stage = _read(Path(approval['stage_report']))
    offline = _read(Path(approval['offline_contract_report']))
    if (manifest.get('result') != 'PASS' or set(manifest.get('cycles', {})) != set(evidence.CYCLES)
        or manifest.get('main_sha256') != anchor['main_sha256']
        or manifest.get('preview_sha256') != approval['preview_sha256']
        or manifest.get('total_rows') != anchor['expected_rows']
        or manifest.get('total_nominal_votes') != anchor['expected_votes']
        or stage.get('result') != 'PASS' or stage.get('output_sha256') != anchor['clone_sha256']
        or stage.get('manifest_sha256') != anchor['manifest_sha256']):
        raise AdoptionError('incomplete or inconsistent source manifest/stage')
    # Authenticate the same exact dbt SQL implementation that passed offline.
    hashes = gate.validate_offline_report(
        record=offline, main=main, preview=preview, clone=clone, root=root,
        manifest_path=Path(approval['manifest']), manifest_hash=approval['manifest_sha256'],
        stage_path=Path(approval['stage_report']), stage_hash=approval['stage_report_sha256'],
        manifest=manifest,
    )
    if hashes != approval.get('contract_sql_sha256'):
        raise AdoptionError('offline dbt SQL source has changed')
    for cycle in evidence.CYCLES:
        sig = evidence.publication_signature(evidence.partition_for(root, cycle))
        if manifest['cycles'][cycle]['source_signature'] != sig:
            raise AdoptionError(f'published Parquet drift: {cycle}')
    return anchor, approval, manifest


def _linux_rename_noreplace(source: Path, destination: Path) -> None:
    """Linux-only, create-only atomic directory publication (no overwrite)."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, 'renameat2', None)
    if renameat2 is None:
        raise AdoptionError('Linux renameat2 RENAME_NOREPLACE required')
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(destination))


def readiness(main: Path, timeout: float = 60.0) -> dict:
    main = main.expanduser().absolute()
    ledger, anchor = v2._load_and_check(main)
    with evidence.noncreating_shared_locks(
        v2.lock_paths(main, Path(anchor['preview']), Path(anchor['root'])), timeout):
        anchor, approval, manifest = _check_under_locks(main, require_active=False)
        phase = v2.read_ledger(v2.state_dir(main))[-1][1]['phase']
        return {'result': 'PASS', 'scope': SCOPE, 'phase': phase,
                'ready_for_adoption': phase == 'ACTIVE_UNCERTIFIED',
                'main_sha256': v2.sha(main), 'clone_sha256': anchor['clone_sha256'],
                'cycles': len(evidence.CYCLES), 'total_rows': manifest['total_rows'],
                'lifecycle_epoch': lifecycle_epoch(main),
                'limitation': 'read-only readiness, not a freshness certificate'}


def adopt(*, main: Path, cert_dir: Path, timeout: float,
          ack_quiescent: bool, confirm_adopt: bool) -> dict:
    if not (ack_quiescent and confirm_adopt):
        raise AdoptionError('adopt requires --ack-quiescent and --confirm-adopt')
    main = main.expanduser().absolute()
    cert_dir = cert_dir.expanduser().absolute()
    if (cert_dir.exists() or cert_dir.is_symlink() or cert_dir.parent.is_symlink()
        or not cert_dir.parent.is_dir()):
        raise AdoptionError('new certificate destination must not exist and parent must be real')
    ledger, anchor = v2._load_and_check(main)
    with v2.exclusive_existing_locks(
        v2.lock_paths(main, Path(anchor['preview']), Path(anchor['root'])), timeout):
        anchor, approval, manifest = _check_under_locks(main, require_active=True)
        epoch_before = lifecycle_epoch(main)
        from scripts import candidate_cube_freshness as freshness
        # No local DuckDB mutation. Issue only into a private, unpublished
        # directory and publish that directory after ALL five have passed.
        staging_dir = cert_dir.with_name('.' + cert_dir.name + '.' + uuid.uuid4().hex + '.building')
        staging_dir.mkdir(mode=0o700)
        try:
            certificates = {}
            for cycle in evidence.CYCLES:
                inputs = freshness.input_signature(main, Path(anchor['root']), cycle)
                outputs = freshness.output_signature(main, cycle)
                expected = manifest['cycles'][cycle]
                if (inputs['source'] != expected['source_signature']
                    or outputs['cube']['rows'] != expected['rows']
                    or outputs['cube']['nominal_votes'] != expected['nominal_votes']
                    or len(outputs['cube']['columns']) != 29):
                    raise AdoptionError(f'freshness input/output does not match offline contract: {cycle}')
                freshness.issue_certificate(staging_dir, main, Path(anchor['root']),
                                            cycle, inputs, outputs)
                certificates[cycle] = evidence.digest(freshness.certificate_path(staging_dir, cycle))
            # Refuse to publish if anything changed during expensive fingerprints.
            _check_under_locks(main, require_active=True)
            if lifecycle_epoch(main) != epoch_before:
                raise AdoptionError('warehouse epoch changed while certifying snapshot')
            report = {
                'result': 'PASS', 'scope': SCOPE,
                'main': str(main), 'main_sha256': anchor['clone_sha256'],
                'clone': anchor['clone'], 'epoch': epoch_before,
                'v2_state_dir': str(v2.state_dir(main)),
                'v2_terminal_event_sha256': evidence.digest(v2.read_ledger(v2.state_dir(main))[-1][0]),
                'approval_sha256': anchor['approval_sha256'],
                'manifest_sha256': anchor['manifest_sha256'],
                'offline_contract_report_sha256': approval['offline_contract_report_sha256'],
                'cycles': certificates, 'total_rows': manifest['total_rows'],
                'certificates_complete': True,
                'issued_at_utc': datetime.now(timezone.utc).isoformat(),
                'limitations': ('one complete set of cube-specific freshness certificates; '
                                'not graph recertification; cooperative locks and operator quiescence'),
            }
            evidence.write_create_only(staging_dir / 'adoption.json', report)
            v2.fsync_dir(staging_dir)
            _linux_rename_noreplace(staging_dir, cert_dir)
            v2.fsync_dir(cert_dir.parent)
            return {'result': 'PASS', 'scope': SCOPE, 'certificates_dir': str(cert_dir),
                    'cycle_count': 5, 'total_rows': manifest['total_rows'],
                    'main_sha256': anchor['clone_sha256'], 'epoch': epoch_before,
                    'graph_freshness_certified': False}
        finally:
            # Intentionally retain orphan staging directories on errors or SIGKILL
            # for forensic review; they are NOT active certificates.
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'adopt'))
    parser.add_argument('--main-db', type=Path, required=True)
    parser.add_argument('--attestations-dir', type=Path)
    parser.add_argument('--lock-timeout', type=float, default=300.0)
    parser.add_argument('--ack-quiescent', action='store_true')
    parser.add_argument('--confirm-adopt', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.action == 'check':
            result = readiness(args.main_db, timeout=args.lock_timeout)
        else:
            if args.attestations_dir is None:
                raise AdoptionError('--attestations-dir is required for adopt')
            result = adopt(main=args.main_db, cert_dir=args.attestations_dir,
                           timeout=args.lock_timeout, ack_quiescent=args.ack_quiescent,
                           confirm_adopt=args.confirm_adopt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (AdoptionError, v2.PromotionV2Error, gate.PreflightError,
            evidence.EvidenceError, OSError, ValueError, KeyError,
            TypeError, ImportError, RuntimeError) as exc:
        print(f'CUBE V2 ADOPTION DENIED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
