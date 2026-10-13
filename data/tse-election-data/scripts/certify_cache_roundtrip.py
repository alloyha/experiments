#!/usr/bin/env python3
"""Certify an active TSE source -> prune -> offline rehydrate round trip.

Only the selected resource is copied into a fresh disposable temporary lake.
Original data is strictly read-only. No network connection is allowed from the
child Python processes. Fail closed rather than claiming recoverability from
source adjacency or from a hash alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath


class CertificationError(RuntimeError):
    """An invalid precondition or failed certification gate."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(root: Path, relative: object, *, kind: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise CertificationError(f'{kind}: invalid relative path: {relative!r}')
    pure = PurePosixPath(relative)
    if (pure.is_absolute() or any(part in {'.', '..'} for part in relative.split('/'))
            or any(part == '' for part in relative.split('/'))):
        raise CertificationError(f'{kind}: unsafe relative path: {relative!r}')
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
        raise CertificationError(f'{kind}: symlink or escaped root: {relative!r}')
    return path


def read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise CertificationError(f'missing authoritative file: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise CertificationError(f'expected JSON object: {path}')
    return value


def read_index(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise CertificationError(f'missing current object index: {path}')
    result = []
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CertificationError(f'invalid current-index row {n}')
            result.append(row)
    return result


def fingerprints(paths: list[Path]) -> dict[str, str]:
    return {str(p): sha256(p) for p in paths if p.is_file()}


def copy_into(root: Path, destination: Path, relative: str) -> None:
    source = require_path(root, relative, kind='copy source')
    if not source.is_file() or source.is_symlink():
        raise CertificationError(f'missing input for disposable copy: {relative}')
    target = require_path(destination, relative, kind='copy destination')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256(target) != sha256(source):
        raise CertificationError(f'copy checksum mismatch: {relative}')


def validate_resource(root: Path, resource_id: str, year: int, domain: str,
                      max_source: int, max_extracted: int) -> dict:
    metadata = root / '_metadata'
    state = read_json(metadata / 'resource_state.json')
    index = read_index(metadata / 'current_objects.jsonl')
    matches = [(key, row) for key, row in state.items()
               if isinstance(row, dict) and row.get('resource_id') == resource_id
               and int(row.get('year', -1)) == year and row.get('domain') == domain]
    if len(matches) != 1:
        raise CertificationError(f'expected exactly one {year}/{domain}/{resource_id} active row, got {len(matches)}')
    key, row = matches[0]
    manifest = metadata / 'ingest_manifest.jsonl'
    if not manifest.is_file() or manifest.is_symlink():
        raise CertificationError('append-only ingest manifest is required for certification')
    with manifest.open(encoding='utf-8') as stream:
        published = any(
            isinstance(entry, dict) and entry.get('resource_id') == resource_id
            and entry.get('source_sha256') == row.get('source_sha256')
            and entry.get('source_object') == row.get('source_object')
            for line in stream if line.strip() for entry in (json.loads(line),)
        )
    if not published:
        raise CertificationError('active immutable source is absent from append-only manifest')
    if not row.get('selection_complete') or row.get('materialization_state') != 'present':
        raise CertificationError('selected active resource is not completely materialized')
    source_rel = row.get('source_object')
    source = require_path(root, source_rel, kind='source')
    if source.parent.name != 'source' or not source.name.lower().endswith('.zip'):
        raise CertificationError('only an immutable ZIP under source/ can be certified')
    version = source.parent.parent
    required_partitions = {'raw', f'year={year}', f'domain={domain}'}
    if not version.is_relative_to(root / 'raw') or not required_partitions.issubset(set(version.relative_to(root).parts)):
        raise CertificationError('immutable source path does not match selected year/domain')
    expected_sha = row.get('source_sha256')
    if (not isinstance(expected_sha, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha)
            or version.name != f'sha256={expected_sha}'):
        raise CertificationError('source digest metadata does not match immutable version path')
    if not source.is_file() or source.is_symlink() or not zipfile.is_zipfile(source):
        raise CertificationError('missing or invalid ZIP source')
    if source.stat().st_size > max_source:
        raise CertificationError('source exceeds safety limit')
    if int(row.get('source_size_bytes', -1)) != source.stat().st_size or sha256(source) != expected_sha:
        raise CertificationError('immutable source size/SHA-256 mismatch')
    outputs = row.get('extracted_objects')
    selected = row.get('selected_members')
    if (not isinstance(outputs, list) or not outputs or not isinstance(selected, list)
            or len(selected) != len(outputs) or len(set(outputs)) != len(outputs)):
        raise CertificationError('incomplete or ambiguous selected members/outputs')
    actual_bytes = 0
    files = []
    for rel in outputs:
        p = require_path(root, rel, kind='extracted')
        if not p.is_relative_to(version / 'extracted') or not p.is_file() or p.is_symlink():
            raise CertificationError(f'extracted object missing or outside version: {rel}')
        actual_bytes += p.stat().st_size
        if actual_bytes > max_extracted:
            raise CertificationError('extracted data exceeds safety limit')
        files.append(p)
    if not all(name.split('/')[-1] in {p.name for p in files} for name in selected):
        raise CertificationError('selected archive members do not map to extracted outputs')
    # The actual active physical index must be exactly the set of expected outputs.
    matching = [r for r in index if r.get('source_object') == source_rel]
    if (len(matching) != len(outputs)
            or {r.get('object') for r in matching} != set(outputs)):
        raise CertificationError('current index does not exactly match the selected version')
    if any(r.get('object') in outputs and r.get('source_object') != source_rel for r in index):
        raise CertificationError('another source claims a selected physical object')
    return {'key': key, 'row': row, 'index': matching, 'source_rel': source_rel,
            'source': source, 'outputs': outputs, 'files': files,
            'source_bytes': source.stat().st_size, 'extracted_bytes': actual_bytes}


def csv_fingerprint(path: Path) -> dict:
    """Streaming, order-sensitive normalized row digest using the TSE CSV dialect.

    This supplements byte identity; it is NOT equivalent to running the dbt DAG.
    """
    h = hashlib.sha256()
    rows = 0
    with path.open('r', encoding='latin-1', newline='') as stream:
        reader = csv.reader(stream, delimiter=';', quotechar='"')
        header = next(reader, None)
        if not header:
            raise CertificationError(f'empty or headerless CSV: {path}')
        for item in reader:
            if len(item) != len(header):
                raise CertificationError(f'irregular CSV column count: {path} at row {rows+2}')
            h.update(json.dumps(item, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            h.update(b'\n')
            rows += 1
    return {'rows': rows, 'columns': header, 'ordered_row_sha256': h.hexdigest()}


def duckdb_fingerprint(path: Path) -> dict:
    """Use the same reader dialect as check_raw_contracts.py."""
    try:
        import duckdb
    except ImportError as exc:
        raise CertificationError('duckdb is required: use the project .venv/bin/python') from exc
    literal = "'" + str(path).replace("'", "''") + "'"
    relation = (f'read_csv({literal}, delim=\';\', quote=\'"\', header=true, '
                'all_varchar=true, encoding=\'latin-1\', sample_size=20480)')
    with duckdb.connect(':memory:') as connection:
        cols = [r[0] for r in connection.execute(f'DESCRIBE SELECT * FROM {relation}').fetchall()]
        if not cols:
            raise CertificationError(f'no columns in {path}')
        expressions = ', '.join('"' + c.replace('"', '""') + '"' for c in cols)
        row = connection.execute(
            f'SELECT count(*), sum(hash({expressions})::HUGEINT) FROM {relation}'
        ).fetchone()
    return {'rows': row[0], 'columns': cols, 'sum_row_hash': str(row[1])}


def snapshot(root: Path, outputs: list[str], *, reader: str) -> dict:
    fn = duckdb_fingerprint if reader == 'duckdb' else csv_fingerprint
    return {relative: {'sha256': sha256(require_path(root, relative, kind='snapshot')),
                       'bytes': require_path(root, relative, kind='snapshot').stat().st_size,
                       'reader': fn(require_path(root, relative, kind='snapshot'))}
            for relative in outputs}


def offline_policy(directory: Path) -> Path:
    directory.mkdir(parents=True)
    # Applied at Python interpreter startup inside both child processes.
    (directory / 'sitecustomize.py').write_text(
        'import socket\n'
        'def _deny_network(*args, **kwargs):\n'
        '    raise RuntimeError("PR42: network disabled during cache round trip")\n'
        'socket.socket.connect = _deny_network\n'
        'socket.socket.connect_ex = _deny_network\n'
        'socket.create_connection = _deny_network\n', encoding='utf-8')
    return directory


# Explicit bootstrap is required: other sitecustomize modules may already be
# imported by a managed Python environment before PYTHONPATH is consulted.
OFFLINE_BOOTSTRAP = (
    'import runpy, socket, sys\n'
    'def _deny(*args, **kwargs):\n'
    '    raise RuntimeError("PR42: network disabled during cache round trip")\n'
    'socket.socket.connect = _deny\n'
    'socket.socket.connect_ex = _deny\n'
    'socket.create_connection = _deny\n'
    'sys.argv = sys.argv[1:]\n'
    'runpy.run_path(sys.argv[0], run_name="__main__")\n'
)


def run_command(command: list[str], *, policy: Path, project: Path) -> str:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(policy) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    res = subprocess.run([command[0], '-c', OFFLINE_BOOTSTRAP, *command[1:]],
                         cwd=project, env=env, capture_output=True,
                         text=True, check=False, timeout=300)
    if res.returncode != 0:
        raise CertificationError(f'command failed ({res.returncode}): {command[-1]}\n'
                                 f'stdout={res.stdout[-2000:]}\nstderr={res.stderr[-2000:]}')
    return res.stdout


def run_certification(raw_root: Path, project: Path, resource_id: str,
                      year: int, domain: str, *, reader: str = 'duckdb',
                      max_source: int = 64 * 1024**2,
                      max_extracted: int = 512 * 1024**2) -> dict:
    root = raw_root.resolve(strict=True)
    project = project.resolve(strict=True)
    if not root.joinpath('raw').is_dir():
        raise CertificationError(f'not a TSE raw lake root: {root}')
    for name in ('prune_extracted_raw.py', 'ensure_materialized.py',
                 'check_raw_contracts.py', 'report_local_storage.py'):
        if not (project / 'scripts' / name).is_file():
            raise CertificationError(f'missing project script: {name}')
    chosen = validate_resource(root, resource_id, year, domain, max_source, max_extracted)
    protected = [root / '_metadata' / n for n in
                 ('resource_state.json', 'current_objects.jsonl',
                  'ingest_manifest.jsonl', 'pending_resources.json')]
    protected += [chosen['source'], *chosen['files']]
    baseline = fingerprints(protected)
    before = snapshot(root, chosen['outputs'], reader=reader)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='tse-pr42-roundtrip-') as temporary:
        work = Path(temporary).resolve()
        if work.is_relative_to(root) or root.is_relative_to(work):
            raise CertificationError('scratch must be separate from production data')
        disposable = work / 'tse'
        (disposable / '_metadata').mkdir(parents=True)
        copy_into(root, disposable, chosen['source_rel'])
        for relative in chosen['outputs']:
            copy_into(root, disposable, relative)
        # Only one resource lives in the disposable control plane; an archive
        # traversal or broad domain prune cannot touch the canonical lake.
        (disposable / '_metadata' / 'resource_state.json').write_text(
            json.dumps({chosen['key']: chosen['row']}, ensure_ascii=False, indent=2), encoding='utf-8')
        (disposable / '_metadata' / 'current_objects.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in chosen['index']), encoding='utf-8')
        (disposable / '_metadata' / 'pending_resources.json').write_text('{}\n', encoding='utf-8')
        manifest = root / '_metadata' / 'ingest_manifest.jsonl'
        if manifest.is_file():
            shutil.copy2(manifest, disposable / '_metadata' / 'ingest_manifest.jsonl')
        original_manifest = sha256(disposable / '_metadata' / 'ingest_manifest.jsonl') if manifest.is_file() else None
        policy = offline_policy(work / 'policy')
        python = sys.executable
        prune_log = run_command(
            [python, str(project / 'scripts' / 'prune_extracted_raw.py'),
             '--root', str(disposable / 'raw'), '--domain', domain, '--apply'],
            policy=policy, project=project)
        remaining_index = read_index(disposable / '_metadata' / 'current_objects.jsonl')
        state_after_prune = read_json(disposable / '_metadata' / 'resource_state.json')
        if (remaining_index or state_after_prune[chosen['key']].get('materialization_state') != 'evicted'
                or any((disposable / rel).exists() for rel in chosen['outputs'])
                or sha256(disposable / chosen['source_rel']) != chosen['row']['source_sha256']):
            raise CertificationError('prune failed index-first/eviction/source-preservation gate')
        audit_command = [python, str(project / 'scripts' / 'report_local_storage.py'),
                         '--raw-root', str(disposable), '--duckdb',
                         str(work / 'nonexistent-warehouse.duckdb'),
                         '--candidate-fact-root', str(work / 'nonexistent-facts'),
                         '--verify-hashes', '--strict', '--json']
        after_prune_audit = json.loads(run_command(audit_command, policy=policy, project=project))
        if (after_prune_audit.get('issues') or
                after_prune_audit.get('evicted_missing_objects') != len(chosen['outputs']) or
                after_prune_audit.get('unexpected_missing_objects') != 0 or
                after_prune_audit.get('current_index_rows') != 0 or
                after_prune_audit.get('source_hashes_verified') != 1):
            raise CertificationError('strict storage audit failed during intentional eviction')
        rehydrate_log = run_command(
            [python, str(project / 'scripts' / 'ensure_materialized.py'),
             '--root', str(disposable), '--year', str(year),
             '--election-type', str(chosen['row']['election_type']),
             '--domain', domain, '--extractor', 'python'],
            policy=policy, project=project)
        if 'rehydrated=1' not in rehydrate_log or 'errors=0' not in rehydrate_log:
            raise CertificationError('rehydration did not report exactly one recovered resource')
        state_after = read_json(disposable / '_metadata' / 'resource_state.json')
        restored_index = read_index(disposable / '_metadata' / 'current_objects.jsonl')
        if (state_after[chosen['key']].get('materialization_state') != 'present'
                or state_after[chosen['key']].get('source_sha256') != chosen['row']['source_sha256']
                or state_after[chosen['key']].get('source_object') != chosen['source_rel']
                or {(v.get('source_object'), v.get('object')) for v in restored_index}
                != {(v.get('source_object'), v.get('object')) for v in chosen['index']}
                or len(restored_index) != len(chosen['index'])):
            raise CertificationError('recovery failed active-state/index-equivalence gate')
        after = snapshot(disposable, chosen['outputs'], reader=reader)
        if before != after:
            raise CertificationError('materialized payload or downstream reader output changed')
        if original_manifest is not None and sha256(disposable / '_metadata' / 'ingest_manifest.jsonl') != original_manifest:
            raise CertificationError('recovery mutated the append-only manifest')
        if sha256(disposable / chosen['source_rel']) != chosen['row']['source_sha256']:
            raise CertificationError('recovery modified immutable source')
        final_audit = json.loads(run_command(audit_command, policy=policy, project=project))
        if (final_audit.get('issues') or final_audit.get('evicted_missing_objects') != 0
                or final_audit.get('unexpected_missing_objects') != 0
                or final_audit.get('current_index_rows') != len(chosen['outputs'])
                or final_audit.get('source_hashes_verified') != 1):
            raise CertificationError('strict final storage audit did not certify restored state')
        # Actual raw-contracts, using the disposable physical index. These
        # contracts do not require a production DuckDB warehouse.
        contract_log = run_command(
            [python, str(project / 'scripts' / 'check_raw_contracts.py'),
             '--root', str(disposable)], policy=policy, project=project)
        if 'current object references .... PASS' not in contract_log:
            raise CertificationError('raw physical contracts not passed')
        elapsed = round(time.monotonic() - started, 3)
    if fingerprints(protected) != baseline:
        raise CertificationError('canonical source or metadata changed during certification')
    return {'schema_version': 1, 'result': 'PASS', 'read_only_canonical': True,
            'offline_enforced': True, 'disposable_copy_deleted': True,
            'resource_id': resource_id, 'year': year, 'domain': domain,
            'source_sha256': chosen['row']['source_sha256'],
            'source_bytes': chosen['source_bytes'], 'extracted_bytes': chosen['extracted_bytes'],
            'objects': len(chosen['outputs']), 'reader': reader,
            'payload_and_reader_equivalent': True, 'index_rebuilt': True,
            'manifest_unchanged': True, 'strict_storage_audit_passed': True,
            'elapsed_seconds': elapsed,
            'prune': prune_log.strip().splitlines()[-3:],
            'rehydrate': rehydrate_log.strip().splitlines()[-1:],
            'contract': contract_log.strip().splitlines()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True,
                   help='Canonical authoritative data/tse root (READ ONLY)')
    p.add_argument('--resource-id', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--domain', required=True)
    p.add_argument('--project-root', type=Path,
                   default=Path(__file__).resolve().parents[1])
    p.add_argument('--reader', choices=('duckdb', 'csv'), default='duckdb')
    p.add_argument('--max-source-mib', type=int, default=64)
    p.add_argument('--max-extracted-mib', type=int, default=512)
    p.add_argument('--output-json', type=Path)
    a = p.parse_args()
    try:
        if a.max_source_mib < 1 or a.max_extracted_mib < 1:
            raise CertificationError('size limits must be positive')
        root = a.raw_root.resolve(strict=True)
        if a.output_json and a.output_json.resolve(strict=False).is_relative_to(root):
            raise CertificationError('output-json must be outside canonical raw root')
        result = run_certification(root, a.project_root, a.resource_id, a.year, a.domain,
                                   reader=a.reader,
                                   max_source=a.max_source_mib * 1024**2,
                                   max_extracted=a.max_extracted_mib * 1024**2)
        content = json.dumps(result, ensure_ascii=False, indent=2)
        if a.output_json:
            a.output_json.parent.mkdir(parents=True, exist_ok=True)
            a.output_json.write_text(content + '\n', encoding='utf-8')
        print(content)
        return 0
    except (OSError, ValueError, CertificationError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        print(f'cache roundtrip certification FAILED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
