#!/usr/bin/env python3
"""PR43: real dbt downstream equivalence around disposable offline raw cache recovery.

Canonical TSE data is READ ONLY. All prune/rehydrate and dbt builds run in a
private scratch directory. The only supported downstream dependency slice is
candidate_assets -> bronze_candidate_assets -> silver_candidate_assets.
"""
from __future__ import annotations

import argparse
import datetime as dt
import decimal
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import certify_cache_roundtrip as base

CertificationError = base.CertificationError
MODEL = 'silver_candidate_assets'
RELATIONS = ('bronze_candidate_assets', 'silver_candidate_assets')
DOMAIN = 'candidate_assets'


def eligible_rows(root: Path, year: int, max_source: int, max_extracted: int) -> list[dict]:
    """List candidates from state for reporting; validation is always delegated to PR42."""
    state = base.read_json(root / '_metadata' / 'resource_state.json')
    results = []
    for row in state.values():
        if (not isinstance(row, dict) or row.get('domain') != DOMAIN
                or str(row.get('year')) != str(year) or not row.get('selection_complete')
                or (row.get('materialization_state') != 'present'
                    and 'materialization_state' in row)):
            continue
        src = base.require_path(root, row.get('source_object'), kind='candidate source')
        if not src.is_file() or src.suffix.lower() != '.zip':
            continue
        source_bytes = src.stat().st_size
        outputs = row.get('extracted_objects') or []
        if not outputs or source_bytes > max_source:
            continue
        size = 0
        complete = True
        for rel in outputs:
            path = base.require_path(root, rel, kind='candidate output')
            if not path.is_file():
                complete = False
                break
            size += path.stat().st_size
        if complete and size <= max_extracted:
            results.append({'resource_id': row['resource_id'], 'source_bytes': source_bytes,
                            'extracted_bytes': size, 'resource_name': row.get('resource_name')})
    return sorted(results, key=lambda r: (r['extracted_bytes'], r['resource_id']))


def inspect_candidates(root: Path, year: int, max_source: int,
                       max_extracted: int) -> dict:
    """Metadata-only inventory. Never hash, extract, download or alter source files.

    Include other years to show whether selecting a different election cycle is
    a safer alternative to weakening source/extraction budgets.
    """
    state = base.read_json(root / '_metadata' / 'resource_state.json')
    candidates = []
    for key, row in sorted(state.items()):
        if not isinstance(row, dict) or row.get('domain') != DOMAIN:
            continue
        reasons = []
        year_value = row.get('year')
        if str(year_value) != str(year):
            reasons.append('different_year')
        if not row.get('selection_complete'):
            reasons.append('selection_incomplete')
        if (row.get('materialization_state') != 'present'
                and 'materialization_state' in row):
            reasons.append('not_materialized_present')
        source_name = row.get('source_object')
        source_bytes = None
        suffix = None
        try:
            source = base.require_path(root, source_name, kind='candidate source')
            suffix = source.suffix.lower()
            if not source.is_file() or source.is_symlink():
                reasons.append('source_missing')
            else:
                source_bytes = source.stat().st_size
                if source_bytes > max_source:
                    reasons.append('source_over_limit')
            if suffix != '.zip':
                reasons.append('not_zip_source')
        except (OSError, CertificationError, ValueError):
            reasons.append('invalid_source_path')
        outputs = row.get('extracted_objects')
        extracted_bytes = None
        if not isinstance(outputs, list) or not outputs:
            reasons.append('no_extracted_outputs')
        else:
            extracted_bytes = 0
            for value in outputs:
                try:
                    output = base.require_path(root, value, kind='candidate output')
                    if not output.is_file() or output.is_symlink():
                        reasons.append('extracted_output_missing')
                        extracted_bytes = None
                        break
                    extracted_bytes += output.stat().st_size
                except (OSError, CertificationError, ValueError):
                    reasons.append('invalid_output_path')
                    extracted_bytes = None
                    break
            if extracted_bytes is not None and extracted_bytes > max_extracted:
                reasons.append('extracted_over_limit')
        candidates.append({
            'resource_id': row.get('resource_id'),
            'resource_name': row.get('resource_name'),
            'year': year_value,
            'materialization_state': row.get('materialization_state'),
            'legacy_materialization_state_missing': 'materialization_state' not in row,
            'source_format': suffix,
            'source_bytes': source_bytes,
            'extracted_bytes': extracted_bytes,
            'extracted_count': len(outputs) if isinstance(outputs, list) else 0,
            'eligible_in_requested_year': not reasons,
            'exclusion_reasons': sorted(set(reasons)),
        })
    return {
        'schema_version': 1,
        'mode': 'read_only_candidate_inventory',
        'requested_year': year,
        'domain': DOMAIN,
        'limits_bytes': {'source': max_source, 'extracted': max_extracted},
        'scanned_candidate_resources': len(candidates),
        'eligible_in_requested_year': sum(row['eligible_in_requested_year'] for row in candidates),
        'other_year_zip_resources': sorted({int(str(row['year'])) for row in candidates
             if str(row['year']).isdigit() and str(row['year']) != str(year)
             and row['source_format'] == '.zip'
             and row['source_bytes'] is not None and row['source_bytes'] <= max_source
             and row['extracted_bytes'] is not None and row['extracted_bytes'] <= max_extracted
             and (row['materialization_state'] == 'present'
                  or row['legacy_materialization_state_missing'])}),
        'resources': sorted(candidates, key=lambda row: (str(row['year']), str(row['resource_id']))),
    }


def choose_resource(root: Path, year: int, resource_id: str | None,
                    max_source: int, max_extracted: int) -> str:
    if resource_id:
        return resource_id
    candidates = eligible_rows(root, year, max_source, max_extracted)
    if not candidates:
        raise CertificationError('no candidate_assets ZIP fits size limits; run --inspect-candidates to see all years and exclusion reasons before choosing a resource or raising limits')
    # Explicit, auditable smallest resource selection; report exact ID in JSON.
    return candidates[0]['resource_id']


def normalize_value(value: object) -> object:
    """Stable typed representation including NULL, decimals, dates and float specials."""
    if value is None:
        return ['null']
    if isinstance(value, bool):
        return ['bool', value]
    if isinstance(value, int):
        return ['int', str(value)]
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, decimal.Decimal):
        return ['decimal', str(value)]
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, bytes):
        return ['binary', value.hex()]
    if isinstance(value, str):
        return ['str', value]
    raise CertificationError(f'unsupported dbt relation value type: {type(value).__name__}')


def multiset_signature(connection: object, relation: str, *, max_rows: int = 500_000) -> dict:
    if relation not in RELATIONS:
        raise CertificationError(f'unapproved dbt relation: {relation}')
    if max_rows < 1:
        raise CertificationError('max-rows must be positive')
    cursor = connection.execute(f'SELECT * FROM "main"."{relation}"')
    schema = [(col[0], str(col[1])) for col in cursor.description]
    if not schema:
        raise CertificationError(f'empty schema: {relation}')
    counts: Counter[str] = Counter()
    row_count = 0
    while True:
        rows = cursor.fetchmany(4096)
        if not rows:
            break
        row_count += len(rows)
        if row_count > max_rows:
            raise CertificationError(f'{relation}: row safety limit exceeded ({max_rows})')
        for row in rows:
            if len(row) != len(schema):
                raise CertificationError(f'{relation}: bad row width')
            payload = json.dumps([normalize_value(v) for v in row],
                                 ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            counts[hashlib.sha256(payload).hexdigest()] += 1
    if row_count == 0:
        raise CertificationError(f'{relation}: vacuous zero-row result is not certification')
    digest = hashlib.sha256()
    for item, count in sorted(counts.items()):
        digest.update(f'{item}:{count}\n'.encode('ascii'))
    return {'schema': schema, 'row_count': row_count,
            'distinct_row_signatures': len(counts), 'multiset_sha256': digest.hexdigest()}


def capture_relations(duckdb_file: Path, *, max_rows: int) -> dict:
    try:
        import duckdb
    except ImportError as exc:
        raise CertificationError('install project dependencies; duckdb is required') from exc
    result = {}
    with duckdb.connect(str(duckdb_file), read_only=True) as conn:
        for relation in RELATIONS:
            result[relation] = multiset_signature(conn, relation, max_rows=max_rows)
    return result


def isolated_dbt_project(original: Path, temporary: Path) -> Path:
    source = original / 'tse_dbt'
    if not (source / 'dbt_project.yml').is_file():
        raise CertificationError(f'could not find dbt project: {source}')
    if not (source / 'dbt_packages').is_dir():
        raise CertificationError('missing dbt_packages; run make dbt-deps in your worktree first')
    target = temporary / 'dbt_project'
    ignore = shutil.ignore_patterns('target', 'logs', '.user.yml', '*.duckdb', '.duckdb_tmp')
    shutil.copytree(source, target, symlinks=False, ignore=ignore)
    return target


def write_profile(directory: Path, database: Path, tmpdir: Path) -> Path:
    """No httpfs extension, no production DuckDB and no references to user profiles."""
    p = directory / 'profiles'
    p.mkdir(parents=True)
    for path in (database, tmpdir):
        if '\n' in str(path) or "'" in str(path):
            raise CertificationError('invalid scratch pathname')
    (p / 'profiles.yml').write_text(
        'tse_analytics:\n'
        '  target: pr43\n'
        '  outputs:\n'
        '    pr43:\n'
        '      type: duckdb\n'
        f'      path: {json.dumps(str(database))}\n'
        '      threads: 1\n'
        '      settings:\n'
        '        memory_limit: "1GB"\n'
        '        threads: 1\n'
        f'        temp_directory: {json.dumps(str(tmpdir))}\n', encoding='utf-8')
    return p


def run_dbt(project: Path, dbt_dir: Path, disposable: Path, dbt_bin: Path,
            policy: Path, work: Path, phase: str, year: int, election_type: str, max_rows: int) -> tuple[dict, str]:
    database = work / f'{phase}.duckdb'
    profiles = write_profile(work / phase, database, work / phase / 'spill')
    (work / phase / 'spill').mkdir(parents=True, exist_ok=True)
    if election_type not in ('general', 'municipal'):
        raise CertificationError(f'unsupported election type: {election_type}')
    variables = {'tse_raw_root': str(disposable), 'election_years': [year],
                 'election_types': [election_type],
                 'incremental_years': [year],
                 'incremental_election_types': [election_type],
                 'tse_csv_encoding': 'latin-1'}
    command = [sys.executable, str(dbt_bin), 'run',
               '--project-dir', str(dbt_dir), '--profiles-dir', str(profiles),
               '--select', '+silver_candidate_assets', '--vars', json.dumps(variables),
               '--target-path', str(work / phase / 'target'),
               '--log-path', str(work / phase / 'logs'), '--no-partial-parse']
    # Uses PR42's Python socket guard; no packages or extensions are fetched.
    output = base.run_command(command, policy=policy, project=project)
    result = capture_relations(database, max_rows=max_rows)
    return result, output


def certify(raw_root: Path, project: Path, *, year: int, resource_id: str | None,
            max_source: int, max_extracted: int, max_rows: int = 500_000) -> dict:
    if year not in (2018, 2020, 2022, 2024, 2026):
        raise CertificationError(f'unsupported cycle: {year}')
    if max_rows < 1:
        raise CertificationError("max_rows must be positive")
    root = raw_root.resolve(strict=True)
    project = project.resolve(strict=True)
    if root.is_relative_to(project) or project.is_relative_to(root):
        raise CertificationError('canonical raw root and code checkout must be separate')
    chosen_id = choose_resource(root, year, resource_id, max_source, max_extracted)
    chosen = base.validate_resource(root, chosen_id, year, DOMAIN,
                                    max_source, max_extracted,
                                    allow_legacy_materialization=True)
    election_type = chosen['row'].get('election_type')
    expected_type = 'general' if year % 4 == 2 else 'municipal'
    if election_type != expected_type:
        raise CertificationError(f'invalid cycle type: {year}/{election_type}, expected {expected_type}')
    dbt_bin = project / '.venv' / 'bin' / 'dbt'
    if not dbt_bin.is_file():
        raise CertificationError('missing dbt CLI: run make setup and make dbt-deps first')
    protected = [root / '_metadata' / name for name in
                 ('resource_state.json', 'current_objects.jsonl',
                  'ingest_manifest.jsonl', 'pending_resources.json')]
    protected.extend([chosen['source'], *chosen['files']])
    baseline = base.fingerprints(protected)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='tse-pr43-dbt-') as temp:
        work = Path(temp).resolve()
        if work.is_relative_to(root) or root.is_relative_to(work):
            raise CertificationError('scratch cannot overlap canonical raw storage')
        disposable = work / 'tse'
        (disposable / '_metadata').mkdir(parents=True)
        base.copy_into(root, disposable, chosen['source_rel'])
        for rel in chosen['outputs']:
            base.copy_into(root, disposable, rel)
        (disposable / '_metadata/resource_state.json').write_text(
            json.dumps({chosen['key']: chosen['row']}), encoding='utf-8')
        (disposable / '_metadata/current_objects.jsonl').write_text(
            ''.join(json.dumps(v) + '\n' for v in chosen['index']), encoding='utf-8')
        (disposable / '_metadata/pending_resources.json').write_text('{}\n', encoding='utf-8')
        shutil.copy2(root / '_metadata/ingest_manifest.jsonl',
                     disposable / '_metadata/ingest_manifest.jsonl')
        manifest_sha = base.sha256(disposable / '_metadata/ingest_manifest.jsonl')
        policy = base.offline_policy(work / 'policy')
        isolated_project = isolated_dbt_project(project, work)
        original_source_sha = base.sha256(disposable / chosen['source_rel'])
        output_sha = {rel: base.sha256(disposable / rel) for rel in chosen['outputs']}
        first, first_log = run_dbt(project, isolated_project, disposable, dbt_bin, policy, work, 'before', year, election_type, max_rows)
        prune = base.run_command(
            [sys.executable, str(project / 'scripts/prune_extracted_raw.py'),
             '--root', str(disposable / 'raw'), '--domain', DOMAIN, '--apply'],
            policy=policy, project=project)
        evicted = base.read_json(disposable / '_metadata/resource_state.json')
        if (evicted[chosen['key']].get('materialization_state') != 'evicted'
                or base.read_index(disposable / '_metadata/current_objects.jsonl')
                or any((disposable / rel).exists() for rel in chosen['outputs'])):
            raise CertificationError('raw prune failed the durable state/index gate')
        rehydrate = base.run_command(
            [sys.executable, str(project / 'scripts/ensure_materialized.py'),
             '--root', str(disposable), '--year', str(year),
             '--election-type', chosen['row']['election_type'],
             '--domain', DOMAIN, '--extractor', 'python'],
            policy=policy, project=project)
        if 'rehydrated=1' not in rehydrate or 'errors=0' not in rehydrate:
            raise CertificationError('raw rehydration did not succeed')
        restored = base.read_json(disposable / '_metadata/resource_state.json')
        repaired_index = base.read_index(disposable / '_metadata/current_objects.jsonl')
        if (restored[chosen['key']].get('materialization_state') != 'present'
                or len(repaired_index) != len(chosen['index'])
                or {(x.get('source_object'), x.get('object')) for x in repaired_index}
                != {(x.get('source_object'), x.get('object')) for x in chosen['index']}):
            raise CertificationError('recovery did not restore index/state exactly')
        if base.sha256(disposable / chosen['source_rel']) != original_source_sha:
            raise CertificationError('immutable source changed')
        if any(base.sha256(disposable / rel) != sha for rel, sha in output_sha.items()):
            raise CertificationError('recovered cache differs at the byte level')
        second, second_log = run_dbt(project, isolated_project, disposable, dbt_bin, policy, work, 'after', year, election_type, max_rows)
        if first != second:
            different = [name for name in RELATIONS if first.get(name) != second.get(name)]
            raise CertificationError('dbt downstream relation divergence: ' + ', '.join(different))
        audit = json.loads(base.run_command(
            [sys.executable, str(project / 'scripts/report_local_storage.py'),
             '--raw-root', str(disposable), '--duckdb', str(work / 'after.duckdb'),
             '--candidate-fact-root', str(work / 'unallocated-facts'), '--verify-hashes',
             '--strict', '--json'], policy=policy, project=project))
        if (audit.get('issues') or audit['source_hashes_verified'] != 1
                or audit['current_index_rows'] != len(chosen['outputs'])):
            raise CertificationError('strict storage audit failed after downstream execution')
        if base.sha256(disposable / '_metadata/ingest_manifest.jsonl') != manifest_sha:
            raise CertificationError('append-only manifest modified')
        # This is not a comprehensive OS-level network sandbox: only Python
        # child socket calls are intercepted; profile intentionally has no httpfs.
        duration = round(time.monotonic() - started, 3)
    if base.fingerprints(protected) != baseline:
        raise CertificationError('canonical metadata or payload changed during execution')
    return {'schema_version': 1, 'result': 'PASS', 'scope': 'candidate_assets -> bronze_candidate_assets -> silver_candidate_assets',
            'year': year, 'domain': DOMAIN, 'resource_id': chosen_id,
            'legacy_materialization_state_missing': chosen['legacy_materialization_state_missing'],
            'source_sha256': chosen['row']['source_sha256'],
            'source_bytes': chosen['source_bytes'], 'extracted_bytes': chosen['extracted_bytes'],
            'dbt_relations': first, 'dbt_relation_equivalent': True,
            'raw_payload_equivalent': True, 'index_rebuilt': True,
            'manifest_unchanged': True, 'canonical_data_read_only': True,
            'scratch_deleted': True, 'python_network_guarded': True,
            'strict_storage_audit_passed': True, 'elapsed_seconds': duration,
            'prune_summary': prune.strip().splitlines()[-3:],
            'rehydration_summary': rehydrate.strip().splitlines()[-1:],
            'dbt_before_tail': first_log.strip().splitlines()[-3:],
            'dbt_after_tail': second_log.strip().splitlines()[-3:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--resource-id', help='optional; chooses smallest eligible 2026 candidate_assets ZIP if omitted')
    parser.add_argument('--max-source-mib', type=int, default=256)
    parser.add_argument('--max-extracted-mib', type=int, default=512)
    parser.add_argument('--max-rows', type=int, default=500_000)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--inspect-candidates', action='store_true',
                        help='Read-only candidate_assets inventory with selection exclusion reasons; no dbt execution')
    args = parser.parse_args()
    try:
        if min(args.max_source_mib, args.max_extracted_mib, args.max_rows) < 1:
            raise CertificationError('all size/row limits must be positive')
        root = args.raw_root.resolve(strict=True)
        if args.output_json and args.output_json.resolve(strict=False).is_relative_to(root):
            raise CertificationError('JSON report cannot be written into canonical data root')
        if args.inspect_candidates:
            result = inspect_candidates(root, args.year,
                                        args.max_source_mib * 1024**2,
                                        args.max_extracted_mib * 1024**2)
        else:
            result = certify(root, args.project_root, year=args.year,
                             resource_id=args.resource_id,
                             max_source=args.max_source_mib * 1024**2,
                             max_extracted=args.max_extracted_mib * 1024**2,
                             max_rows=args.max_rows)
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(serialized, encoding='utf-8')
        print(serialized, end='')
        return 0
    except (OSError, ValueError, CertificationError, subprocess.TimeoutExpired) as exc:
        print(f'PR43 downstream certification FAILED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
