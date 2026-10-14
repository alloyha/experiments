#!/usr/bin/env python3
"""Bounded, verified, restartable TSE extracted-cache eviction.

No writes to the canonical raw lake without --apply, exact reviewed plan,
verified plan authorization (or scratch canary), and --ack-quiescent. The ack means *all* dbt, ingest,
rehydration, and external readers/writers have been stopped; POSIX flock here
coordinates this tool's invocations only, NOT those independent consumers.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
import uuid
import stat
from pathlib import PurePosixPath
from pathlib import Path

# Existing PR42 verifier is the single source of source/manifest/index truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import certify_cache_roundtrip as base

Error = base.CertificationError
INDEX = 'current_objects.jsonl'
STATE = 'resource_state.json'
MANIFEST = 'ingest_manifest.jsonl'
JOURNAL = 'cache_eviction_journal.json'
LOCK = '.cache_eviction.lock'


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


def read_bytes(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise Error(f'missing regular file: {path}')
    return path.read_bytes()


def verified_path(root: Path, relative: str) -> Path:
    path = base.require_path(root, relative, kind='eviction')
    current = root
    for segment in Path(relative).parts:
        current = current / segment
        if current.is_symlink():
            raise Error(f'symlink is not a safe eviction target: {relative}')
    if not path.is_file() or not path.is_relative_to(root / 'raw'):
        raise Error(f'outside raw or missing: {relative}')
    return path


def validate_zip_matches(root: Path, item: dict) -> list[dict]:
    """Prove each extracted file is byte-identical to its selected ZIP member.

    This is streaming and verifies the ZIP member CRC through ZipExtFile. It
    rejects flattened-name collisions and any unmatched extracted output.
    """
    members = item['row']['selected_members']
    outputs = item['outputs']
    if len(members) != len(outputs) or len(set(Path(x).name for x in members)) != len(members):
        raise Error('ambiguous ZIP member to extracted-file mapping')
    result = []
    with zipfile.ZipFile(item['source']) as archive:
        for name, rel in zip(members, outputs):
            name_path = PurePosixPath(name)
            if (name_path.is_absolute() or any(part in ('.', '..') for part in name.split('/'))
                    or name not in archive.namelist() or name_path.name != Path(rel).name):
                raise Error(f'unsafe/unmatched selected archive member: {name}')
            info = archive.getinfo(name)
            if (info.is_dir() or info.flag_bits & 0x1
                    or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK):
                raise Error('directory/encrypted archive member is not supported')
            local = verified_path(root, rel)
            if info.file_size != local.stat().st_size:
                raise Error(f'ZIP member size differs from extracted file: {rel}')
            disk_hash = hashlib.sha256()
            archive_hash = hashlib.sha256()
            size = 0
            with local.open('rb') as disk, archive.open(name) as stream:
                while True:
                    chunk = stream.read(1 << 20)
                    if not chunk:
                        break
                    archive_hash.update(chunk)
                    size += len(chunk)
                for chunk in iter(lambda: disk.read(1 << 20), b''):
                    disk_hash.update(chunk)
            if size != local.stat().st_size or size != info.file_size or disk_hash.digest() != archive_hash.digest():
                raise Error(f'archive payload differs from extracted file: {rel}')
            result.append({'relative': rel, 'bytes': size, 'sha256': disk_hash.hexdigest()})
    return result


def metadata_hashes(root: Path) -> dict[str, str]:
    meta = root / '_metadata'
    return {name: hash_bytes(read_bytes(meta / name)) for name in (STATE, INDEX, MANIFEST)}


def build_plan(root: Path, domains: set[str], years: set[int], max_resources: int,
               max_bytes: int, max_source_bytes: int) -> dict:
    if not domains or any(not d or '/' in d or '\\' in d for d in domains):
        raise Error('at least one explicit valid domain is required')
    if max_resources < 1 or max_bytes < 1 or max_source_bytes < 1:
        raise Error('all safety limits must be positive')
    if not (root / 'raw').is_dir():
        raise Error('not a raw lake root')
    if (root / '_metadata' / JOURNAL).exists():
        raise Error('pending eviction journal; use --recover first')
    state = base.read_json(root / '_metadata' / STATE)
    candidates = []
    for key, row in state.items():
        if not isinstance(row, dict) or row.get('domain') not in domains:
            continue
        try:
            year = int(row.get('year', -1))
        except (ValueError, TypeError):
            continue
        if years and year not in years:
            continue
        if not row.get('selection_complete') or row.get('materialization_state', 'present') != 'present':
            continue
        source_rel = row.get('source_object')
        if not isinstance(source_rel, str) or not source_rel.endswith('.zip'):
            continue
        outputs = row.get('extracted_objects')
        if not isinstance(outputs, list) or not outputs:
            continue
        # Metadata-only ordering first; avoid hashing every multi-GiB source.
        try:
            sizes = [verified_path(root, rel).stat().st_size for rel in outputs]
            total = sum(sizes)
        except (OSError, Error):
            continue
        if total <= max_bytes:
            candidates.append((total, str(key), year, str(row['domain']), str(row['resource_id'])))
    selected = []
    used = 0
    inspected = 0
    for estimate, key, year, domain, rid in sorted(candidates):
        if len(selected) >= max_resources or used + estimate > max_bytes:
            continue
        inspected += 1
        print(f'[verify {inspected}/{len(candidates)}] {domain}/{year} '
              f'resource={rid} csv_bytes={estimate}', file=sys.stderr, flush=True)
        # Existing verifier independently validates ZIP SHA, manifest, selected
        # objects and EXACT physical index, including legacy states.
        item = base.validate_resource(root, rid, year, domain,
            max_source_bytes, max_bytes, allow_legacy_materialization=True)
        if item['key'] != key:
            raise Error('selected state key changed during planning')
        files = validate_zip_matches(root, item)
        if sum(f['bytes'] for f in files) != estimate:
            raise Error('physical file sizes changed during planning')
        selected.append({'key': key, 'resource_id': rid, 'year': year,
                         'domain': domain, 'source': item['source_rel'],
                         'source_sha256': item['row']['source_sha256'],
                         'outputs': files})
        used += estimate
        print(f'[verified {len(selected)}] {domain}/{year} '
              f'cumulative_bytes={used}', file=sys.stderr, flush=True)
    if not selected:
        raise Error('no fully verified recoverable resources meet the limits')
    return {'schema_version': 1, 'root': str(root), 'metadata_sha256': metadata_hashes(root),
            'resources': selected, 'resource_count': len(selected), 'reclaimable_bytes': used,
            'safety': 'maintenance-only: stop independent readers and writers before apply'}


def validate_plan(root: Path, plan: dict) -> None:
    if plan.get('schema_version') != 1 or plan.get('root') != str(root):
        raise Error('plan root/schema mismatch')
    if metadata_hashes(root) != plan.get('metadata_sha256'):
        raise Error('stale plan: authoritative metadata changed')
    resources = plan.get('resources')
    if (not isinstance(resources, list) or not resources
            or plan.get('resource_count') != len(resources)):
        raise Error('plan resource count mismatch or empty plan')
    expected_bytes = sum(sum(f['bytes'] for f in r['outputs']) for r in resources)
    if expected_bytes != plan.get('reclaimable_bytes'):
        raise Error('plan reclaimable byte count mismatch')
    if len({r['key'] for r in resources}) != len(resources):
        raise Error('duplicate resource key in plan')
    for resource in resources:
        item = base.validate_resource(root, resource['resource_id'], resource['year'],
            resource['domain'], 1 << 50, 1 << 50, allow_legacy_materialization=True)
        if item['key'] != resource['key'] or item['row']['source_sha256'] != resource['source_sha256']:
            raise Error('source version changed after plan')
        if validate_zip_matches(root, item) != resource['outputs']:
            raise Error('extracted payload changed after plan')


def atomic_bytes(path: Path, payload: bytes) -> None:
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temp.open('xb') as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)
    sync_dir(path.parent)


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def exclusive(root: Path):
    # Advisory lock is outside canonical root so plan/canary never modify it.
    token = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    path = Path(tempfile.gettempdir()) / f'tse-cache-eviction-{token}.lock'
    if path.is_symlink():
        raise Error('lock file cannot be symlink')
    with path.open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Error('Another cache eviction operation is holding the lock; '
                        'do not start a second invocation') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_journal(root: Path, plan: dict) -> dict:
    metadata = root / '_metadata'
    state = base.read_json(metadata / STATE)
    index = base.read_index(metadata / INDEX)
    keys = {r['key'] for r in plan['resources']}
    paths = {f['relative'] for r in plan['resources'] for f in r['outputs']}
    for key in keys:
        if key not in state:
            raise Error(f'missing state key: {key}')
    for key in keys:
        state[key] = {**state[key], 'materialization_state': 'evicted'}
    filtered = [line for line in index if line.get('object') not in paths]
    if len(index) - len(filtered) != len(paths):
        raise Error('current index does not contain exactly one row per planned output')
    return {'schema_version': 1, 'plan_sha256': hash_bytes(canonical(plan)),
            'sources': [{'relative':r['source'], 'sha256':r['source_sha256']} for r in plan['resources']],
            'pre': {k: plan['metadata_sha256'][k] for k in (STATE, INDEX)},
            'post': {STATE: canonical(state).decode(),
                     INDEX: ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in filtered)},
            'outputs': [f for r in plan['resources'] for f in r['outputs']]}


def recover_pending(root: Path) -> int:
    """Roll forward a durable journal; never blindly rewrite unrelated changes."""
    path = root / '_metadata' / JOURNAL
    if not path.exists():
        return 0
    j = base.read_json(path)
    if j.get('schema_version') != 1 or not j.get('outputs') or not j.get('post'):
        raise Error('invalid pending journal')
    # Validate all physical targets and recovery sources *before* mutating
    # either authoritative control-plane file. Interrupted runs can then
    # safely roll forward without deleting bytes from an untrusted version.
    for source in j.get('sources', []):
        target = verified_path(root, source['relative'])
        if base.sha256(target) != source['sha256']:
            raise Error('recovery source changed while journal pending')
    for item in j['outputs']:
        target = base.require_path(root, item['relative'], kind='journal output')
        if not target.is_relative_to(root / 'raw') or target.is_symlink():
            raise Error('invalid recovery deletion path')
        if target.exists() and (verified_path(root, item['relative']).stat().st_size != item['bytes']
                                 or base.sha256(target) != item['sha256']):
            raise Error('target changed while journal pending: ' + item['relative'])
    for name in (STATE, INDEX):
        dest = root / '_metadata' / name
        observed = hash_bytes(read_bytes(dest))
        intended = j['post'][name].encode()
        if observed not in (j['pre'][name], hash_bytes(intended)):
            raise Error(f'{name} modified by another writer; cannot recover automatically')
        if observed != hash_bytes(intended):
            atomic_bytes(dest, intended)
    count = 0
    for file in j['outputs']:
        relative = file['relative']
        # A path missing after an interrupted unlink is expected; symlink is not.
        target = base.require_path(root, relative, kind='journal output')
        if not target.is_relative_to(root / 'raw') or target.is_symlink():
            raise Error('invalid recovery deletion path')
        if target.exists():
            if verified_path(root, relative).stat().st_size != file['bytes'] or base.sha256(target) != file['sha256']:
                raise Error('target changed while journal pending: ' + relative)
            target.unlink()
            count += 1
            sync_dir(target.parent)
    path.unlink()
    sync_dir(path.parent)
    return count


def canary(root: Path, plan: dict) -> dict:
    """Isolated official prune + ensure; verify original extracted SHA and index."""
    import certify_cache_roundtrip as base
    with tempfile.TemporaryDirectory(prefix='tse-eviction-canary-') as scratch:
        lake = Path(scratch) / 'tse'
        meta = lake / '_metadata'
        meta.mkdir(parents=True)
        state = base.read_json(root / '_metadata' / STATE)
        index = base.read_index(root / '_metadata' / INDEX)
        chosen = {r['key'] for r in plan['resources']}
        subset_state = {key: state[key] for key in chosen}
        paths = {f['relative'] for r in plan['resources'] for f in r['outputs']}
        subset_index = [row for row in index if row.get('object') in paths]
        (meta / STATE).write_bytes(canonical(subset_state))
        (meta / INDEX).write_text(''.join(json.dumps(r)+'\n' for r in subset_index))
        (meta / 'pending_resources.json').write_text('{}\n')
        shutil.copy2(root / '_metadata' / MANIFEST, meta / MANIFEST)
        for r in plan['resources']:
            base.copy_into(root, lake, r['source'])
            for output in r['outputs']:
                base.copy_into(root, lake, output['relative'])
        scriptdir = Path(__file__).resolve().parent
        domains = sorted({r['domain'] for r in plan['resources']})
        years = sorted({r['year'] for r in plan['resources']})
        # Original pruning intentionally only touches disposable lake.
        prune = [sys.executable, str(scriptdir / 'prune_extracted_raw.py'),
                 '--root', str(lake / 'raw'), '--apply']
        for domain in domains:
            prune += ['--domain', domain]
        run = subprocess.run(prune, text=True, capture_output=True, timeout=180)
        if run.returncode:
            raise Error(f'canary prune failed: {run.stderr[-800:]}')
        guard = base.offline_policy(Path(scratch) / 'policy')
        cmd = [sys.executable, '-c', base.OFFLINE_BOOTSTRAP,
               str(scriptdir / 'ensure_materialized.py'), '--root', str(lake),
               '--extractor', 'python']
        for domain in domains:
            cmd += ['--domain', domain]
        for year in years:
            cmd += ['--year', str(year)]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(guard) + os.pathsep + env.get('PYTHONPATH', '')
        run = subprocess.run(cmd, text=True, capture_output=True, timeout=300, env=env)
        if run.returncode:
            raise Error(f'canary offline recovery failed: {run.stderr[-800:]}')
        rebuilt = base.read_index(meta / INDEX)
        observed_paths = {r.get('object') for r in rebuilt}
        if observed_paths != paths or len(rebuilt) != len(paths):
            raise Error('canary index not exactly rebuilt')
        recovered_state = base.read_json(meta / STATE)
        for r in plan['resources']:
            if recovered_state[r['key']].get('materialization_state') != 'present':
                raise Error('canary state not recovered')
            for f in r['outputs']:
                obj = verified_path(lake, f['relative'])
                if base.sha256(obj) != f['sha256'] or obj.stat().st_size != f['bytes']:
                    raise Error('recovered canary bytes differ: ' + f['relative'])
    return {'result': 'PASS', 'schema_version': 1,
            'plan_sha256': hash_bytes(canonical(plan)),
            'root': str(root),
            'certified_scopes': sorted([{'year': y, 'domain': d} for y, d in
                {(r['year'], r['domain']) for r in plan['resources']}], key=lambda x: (x['year'],x['domain'])),
            'tool_sha256': base.sha256(Path(__file__).resolve()),
            'recovered_resources': len(plan['resources']),
            'recovered_files': len(paths), 'scratch_deleted': True,
            'canonical_read_only': True}


def outside_root(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved.is_relative_to(root):
        raise Error('report/plan paths must not be inside canonical data root')
    return resolved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    for mode in ('plan', 'canary', 'apply', 'apply-verified', 'recover'):
        group.add_argument('--'+mode, action='store_true')
    ap.add_argument('--domain', action='append')
    ap.add_argument('--year', action='append', type=int)
    ap.add_argument('--max-resources', type=int, default=1)
    ap.add_argument('--max-bytes', type=int, default=64 * 1024**2)
    ap.add_argument('--max-source-bytes', type=int, default=4 * 1024**3)
    ap.add_argument('--plan-file', type=Path)
    ap.add_argument('--canary-report', type=Path)
    ap.add_argument('--expect-plan-sha256', type=str,
                    help='Required for --apply-verified: exact SHA-256 printed by --plan')
    ap.add_argument('--output-json', type=Path)
    ap.add_argument('--ack-quiescent', action='store_true',
                    help='I stopped ALL independent raw readers, writers, dbt and ingest jobs')
    args = ap.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        meta = root / '_metadata'
        if not meta.is_dir():
            raise Error('missing _metadata')
        if args.plan:
            if not args.output_json:
                raise Error('--plan requires --output-json')
            output = outside_root(args.output_json, root)
            with exclusive(root):
                report = build_plan(root, set(args.domain or []), set(args.year or []),
                    args.max_resources, args.max_bytes, args.max_source_bytes)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(canonical(report))
            print(json.dumps({'result': 'PLAN', 'resources':report['resource_count'],
                              'reclaimable_bytes':report['reclaimable_bytes'],
                              'plan_sha256':hash_bytes(canonical(report)), 'file':str(output)}))
            return 0
        if args.canary:
            if not args.plan_file or not args.output_json:
                raise Error('--canary requires --plan-file and --output-json')
            plan = json.loads(read_bytes(outside_root(args.plan_file,root)))
            report_path = outside_root(args.output_json,root)
            with exclusive(root):
                validate_plan(root,plan)
                report = canary(root,plan)
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_bytes(canonical(report))
            print(json.dumps(report))
            return 0
        if not args.ack_quiescent:
            raise Error('--apply/--apply-verified/--recover require --ack-quiescent (all consumers stopped)')
        with exclusive(root):
            if args.recover:
                removed = recover_pending(root)
                print(json.dumps({'result':'RECOVERED','removed_files':removed}))
                return 0
            if not args.plan_file:
                raise Error('--apply/--apply-verified requires --plan-file')
            plan = json.loads(read_bytes(outside_root(args.plan_file,root)))
            if args.apply_verified:
                if args.canary_report:
                    raise Error('--apply-verified does not accept --canary-report')
                expected = args.expect_plan_sha256
                if (not isinstance(expected, str) or len(expected) != 64
                        or any(c not in '0123456789abcdef' for c in expected)
                        or hash_bytes(canonical(plan)) != expected):
                    raise Error('--apply-verified requires matching --expect-plan-sha256 from reviewed plan')
            else:
                if args.expect_plan_sha256 is not None:
                    raise Error('--expect-plan-sha256 is only valid with --apply-verified')
                if not args.canary_report:
                    raise Error('--apply requires --canary-report')
                report = json.loads(read_bytes(outside_root(args.canary_report,root)))
                covered = {(r['year'], r['domain']) for r in report.get('certified_scopes', [])}
                needed = {(r['year'], r['domain']) for r in plan['resources']}
                if (report.get('result') != 'PASS' or report.get('root') != str(root)
                        or not report.get('scratch_deleted') or not report.get('canonical_read_only')
                        or report.get('tool_sha256') != base.sha256(Path(__file__).resolve())
                        or not needed or not needed.issubset(covered)):
                    raise Error('missing/mismatched successful scoped canary evidence')
            validate_plan(root,plan)
            journal = build_journal(root,plan)
            target = meta / JOURNAL
            if target.exists():
                raise Error('pending journal; recover first')
            atomic_bytes(target,canonical(journal))
            deleted = recover_pending(root)
            print(json.dumps({'result':'EVICTED', 'resources':plan['resource_count'],
                              'removed_files':deleted,'expected_bytes':plan['reclaimable_bytes'],
                              'authorization':'verified-only' if args.apply_verified else 'canary'}))
            return 0
    except (Error,OSError,ValueError,KeyError,zipfile.BadZipFile,subprocess.TimeoutExpired) as exc:
        print(f'cache eviction DENIED: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
