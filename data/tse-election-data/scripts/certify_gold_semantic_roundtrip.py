#!/usr/bin/env python3
"""PR44: disposable recovery vs uninterrupted incremental Gold/Semantic certification.

Requires a complete canonical 2026 `candidate` and `candidate_assets` pair. All
writes (including partition probes and dbt databases) happen in TemporaryDirectory.
This is a scoped Gold/Semantic experiment, NOT approval for global cache eviction.
"""
from __future__ import annotations

import argparse
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
import certify_dbt_cache_roundtrip as prior

Error = base.CertificationError
YEAR_TYPES = {2018: 'general', 2020: 'municipal', 2022: 'general',
              2024: 'municipal', 2026: 'general'}
DOMAINS = ('candidate', 'candidate_assets')
GOLD = 'dim_candidate'
SEMANTIC = 'candidate_summary'
PERSISTED_SEMANTIC = 'pr44_candidate_summary_persisted'
PROBE_SELECTED = '__PR44_SELECTED_PARTITION_PROBE__'
PROBE_OUTSIDE = '__PR44_UNTOUCHED_PARTITION_PROBE__'


def inspect_sources(root: Path, year: int, source_limit: int,
                    extracted_limit: int) -> dict:
    """Metadata-only selection inventory; no SHA reads, no ZIP extraction."""
    state = base.read_json(root / '_metadata/resource_state.json')
    result = []
    for row in state.values():
        if not isinstance(row, dict) or row.get('domain') not in DOMAINS:
            continue
        reasons = []
        if str(row.get('year')) != str(year):
            reasons.append('different_year')
        if not row.get('selection_complete'):
            reasons.append('selection_incomplete')
        if 'materialization_state' in row and row.get('materialization_state') != 'present':
            reasons.append('not_materialized_present')
        size = None
        extracted = None
        try:
            src = base.require_path(root,row.get('source_object'),kind='source')
            if not src.is_file() or src.is_symlink():
                reasons.append('source_missing')
            else:
                size = src.stat().st_size
                if size > source_limit:
                    reasons.append('source_over_limit')
                if src.suffix.lower() != '.zip':
                    reasons.append('not_zip')
            outputs = row.get('extracted_objects') or []
            if not outputs:
                reasons.append('no_extracted_objects')
            else:
                paths = [base.require_path(root,v,kind='output') for v in outputs]
                if not all(v.is_file() and not v.is_symlink() for v in paths):
                    reasons.append('extracted_missing')
                else:
                    extracted = sum(v.stat().st_size for v in paths)
                    if extracted > extracted_limit:
                        reasons.append('extracted_over_limit')
        except (Error, OSError, ValueError):
            reasons.append('invalid_or_unsafe_path')
        result.append({'domain':row.get('domain'),'year':row.get('year'),
                       'resource_id':row.get('resource_id'),'resource_name':row.get('resource_name'),
                       'source_bytes':size,'extracted_bytes':extracted,
                       'legacy_missing_state':'materialization_state' not in row,
                       'metadata_eligible':not reasons,'reasons':sorted(set(reasons))})
    return {'schema_version':1,'mode':'read_only_inventory',
            'requested_year':year,
            'resources':sorted(result,key=lambda x:(str(x['year']),x['domain'],str(x['resource_id']))),
            'warning':'metadata-only; manifest and SHA validation happen in the actual gate'}


def select_pair(root: Path, year: int, asset_id: str | None,
                candidate_id: str | None, source_limit: int,
                extracted_limit: int) -> dict:
    if year not in YEAR_TYPES:
        raise Error('unsupported year')
    chosen = {}
    for domain, requested in zip(DOMAINS, (candidate_id, asset_id)):
        rid = prior.choose_resource(root, year, requested, source_limit, extracted_limit) if domain == 'candidate_assets' else None
        if domain == 'candidate':
            state = base.read_json(root / '_metadata/resource_state.json')
            options = []
            for row in state.values():
                if (not isinstance(row, dict) or row.get('domain') != 'candidate'
                        or str(row.get('year')) != str(year)):
                    continue
                if requested is not None and row.get('resource_id') != requested:
                    continue
                if not row.get('selection_complete') or ('materialization_state' in row
                    and row['materialization_state'] != 'present'):
                    continue
                try:
                    source = base.require_path(root, row.get('source_object'), kind='candidate source')
                    outputs = [base.require_path(root, rel, kind='candidate output')
                               for rel in row.get('extracted_objects') or []]
                except Error:
                    continue
                if (source.is_file() and source.suffix.lower() == '.zip'
                        and outputs and all(p.is_file() for p in outputs)
                        and source.stat().st_size <= source_limit
                        and sum(p.stat().st_size for p in outputs) <= extracted_limit):
                    options.append((sum(p.stat().st_size for p in outputs),row['resource_id']))
            if not options:
                raise Error('no candidate resource fits limits (requires complete ZIP + extracted files); inspect candidate state or supply --candidate-resource-id')
            rid = min(options)[1]
        item = base.validate_resource(root, rid, year, domain, source_limit,
                                      extracted_limit, allow_legacy_materialization=True)
        if item['row'].get('election_type') != YEAR_TYPES[year]:
            raise Error(f'{domain} election type mismatch')
        chosen[domain] = item
    if chosen['candidate']['key'] == chosen['candidate_assets']['key']:
        raise Error('candidate and candidate_assets cannot share a state key')
    return chosen


def quote_ident(value: str) -> str:
    if not value or '\x00' in value:
        raise Error('invalid SQL identifier')
    return '"' + value.replace('"', '""') + '"'


def relation_signature(conn: object, relation: str, year: int, kind: str,
                       *, max_rows: int) -> dict:
    if (relation, kind) not in ((GOLD, 'gold'), (PERSISTED_SEMANTIC, 'semantic')):
        raise Error('unapproved persisted relation')
    if max_rows <= 0:
        raise Error('max_rows must be positive')
    query = (f'SELECT * FROM main.{quote_ident(relation)} '
             'WHERE election_year = ? AND election_type = ?')
    cursor = conn.execute(query, [year, YEAR_TYPES[year]])
    schema = [[col[0], str(col[1])] for col in cursor.description]
    if not schema:
        raise Error('no columns in selected relation')
    counts: Counter[str] = Counter()
    total = 0
    for_chunk = 4096
    while True:
        rows = cursor.fetchmany(for_chunk)
        if not rows:
            break
        total += len(rows)
        if total > max_rows:
            raise Error(f'{relation} exceeded row safety limit')
        for values in rows:
            encoded = json.dumps([prior.normalize_value(v) for v in values],
                                 ensure_ascii=False, separators=(',', ':')).encode()
            counts[hashlib.sha256(encoded).hexdigest()] += 1
    if total == 0:
        raise Error(f'{relation} returned no rows for {year}')
    digest = hashlib.sha256()
    for key, count in sorted(counts.items()):
        digest.update(f'{key}:{count}\n'.encode('ascii'))
    return {'schema': schema, 'row_count': total,
            'distinct_signatures': len(counts), 'multiset_sha256': digest.hexdigest()}


def assert_gold_table_and_semantic_view(conn: object) -> None:
    for name, expected in ((GOLD, 'BASE TABLE'), (SEMANTIC, 'VIEW')):
        row = conn.execute('SELECT table_type FROM information_schema.tables '
                           'WHERE table_schema = ? AND table_name = ?',
                           ['main', name]).fetchone()
        if row != (expected,):
            raise Error(f'{name} must be a {expected} (got {row})')


def save_semantic_snapshot(conn: object) -> None:
    conn.execute(f'DROP TABLE IF EXISTS main.{quote_ident(PERSISTED_SEMANTIC)}')
    conn.execute(f'CREATE TABLE main.{quote_ident(PERSISTED_SEMANTIC)} AS '
                 f'SELECT * FROM main.{quote_ident(SEMANTIC)}')
    row = conn.execute('SELECT table_type FROM information_schema.tables '
                       'WHERE table_schema = ? AND table_name = ?',
                       ['main', PERSISTED_SEMANTIC]).fetchone()
    if row != ('BASE TABLE',):
        raise Error('semantic snapshot is not persisted')


def outside_probe_sha(conn: object, year: int) -> str:
    outside_year = 2018 if year != 2018 else 2020
    cursor = conn.execute(f'SELECT * FROM main.{quote_ident(GOLD)} '
                          'WHERE election_year = ? AND election_type = ? AND candidate_id = ?',
                          [outside_year, YEAR_TYPES[outside_year], PROBE_OUTSIDE])
    row = cursor.fetchone()
    if row is None or cursor.fetchone() is not None:
        raise Error('outside partition probe missing or duplicated')
    encoded=json.dumps([prior.normalize_value(v) for v in row],
                       ensure_ascii=False,separators=(',',':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def seed_partition_probes(conn: object, year: int) -> str:
    """Poison the active election partition; preserve an unrelated partition.

    Both probes are inserted into *scratch* copies of the same baseline database.
    The selected probe must disappear via partition_replace_pre_hook. The outside
    probe must survive byte-for-byte through the incremental update.
    """
    assert_gold_table_and_semantic_view(conn)
    cursor = conn.execute(f'SELECT * FROM main.{quote_ident(GOLD)} '
                          'WHERE election_year = ? AND election_type = ? LIMIT 1',
                          [year, YEAR_TYPES[year]])
    row = cursor.fetchone()
    if row is None:
        raise Error('Gold selected partition is empty; cannot create probes')
    columns = [x[0] for x in cursor.description]
    for required in ('election_year', 'election_type', 'candidate_id', 'election_id'):
        if required not in columns:
            raise Error(f'{required} not in Gold schema')
    source = dict(zip(columns, row))
    cols = ', '.join(quote_ident(col) for col in columns)
    placeholders = ', '.join('?' for _ in columns)
    sql = f'INSERT INTO main.{quote_ident(GOLD)} ({cols}) VALUES ({placeholders})'
    for marker, probe_year in ((PROBE_SELECTED, year), (PROBE_OUTSIDE, 2018 if year != 2018 else 2020)):
        probe_type = YEAR_TYPES[probe_year]
        probe = {**source, 'candidate_id': marker,
                 'election_year': probe_year, 'election_type': probe_type,
                 'cycle_id': f'{probe_year}:{probe_type}',
                 'election_id': f'{probe_year}:{probe_type}:{source["election_code"]}'}
        if 'candidate_name' in probe:
            probe['candidate_name'] = marker
        exists = conn.execute(f'SELECT count(*) FROM main.{quote_ident(GOLD)} '
                              'WHERE election_year = ? AND election_type = ? AND candidate_id = ?',
                              [probe_year, probe_type, marker]).fetchone()[0]
        if exists:
            raise Error('probe key already present: '+marker)
        conn.execute(sql, [probe[name] for name in columns])
    verify_probes(conn, year, expected_selected=1, expected_outside=1)
    return outside_probe_sha(conn,year)


def verify_probes(conn: object, year: int, *, expected_selected: int,
                  expected_outside: int) -> None:
    outside_year = 2018 if year != 2018 else 2020
    for probe, y, expected in ((PROBE_SELECTED, year, expected_selected),
                               (PROBE_OUTSIDE, outside_year, expected_outside)):
        result = conn.execute(f'SELECT count(*) FROM main.{quote_ident(GOLD)} '
                              'WHERE election_year = ? AND election_type = ? AND candidate_id = ?',
                              [y, YEAR_TYPES[y], probe]).fetchone()[0]
        if result != expected:
            raise Error(f'partition probe {probe}: expected {expected}, got {result}')


def capture_db(path: Path, year: int, max_rows: int) -> dict:
    import duckdb
    with duckdb.connect(str(path)) as conn:
        assert_gold_table_and_semantic_view(conn)
        save_semantic_snapshot(conn)
        gold = relation_signature(conn, GOLD, year, 'gold', max_rows=max_rows)
        semantic = relation_signature(conn, PERSISTED_SEMANTIC, year, 'semantic', max_rows=max_rows)
    return {'gold': gold, 'semantic': semantic}


def verify_db(path: Path, year: int, max_rows: int, expected_outside_sha: str) -> dict:
    import duckdb
    with duckdb.connect(str(path), read_only=True) as conn:
        assert_gold_table_and_semantic_view(conn)
        verify_probes(conn, year, expected_selected=0, expected_outside=1)
        if outside_probe_sha(conn,year)!=expected_outside_sha:
            raise Error('unselected partition row mutated')
        gold = relation_signature(conn, GOLD, year, 'gold', max_rows=max_rows)
        semantic = relation_signature(conn, PERSISTED_SEMANTIC, year, 'semantic', max_rows=max_rows)
    return {'gold': gold, 'semantic': semantic,
            'selected_probe_removed': True, 'outside_probe_preserved': True}


def isolated_run(project: Path, dbt: Path, raw: Path, database: Path,
                 phase: Path, year: int, policy: Path) -> str:
    """Run Gold+Semantic dependency DAG; always use scoped scratch-only paths."""
    phase.mkdir(parents=True, exist_ok=True)
    profiles = prior.write_profile(phase, database, phase / 'spill')
    (phase / 'spill').mkdir(parents=True, exist_ok=True)
    vars_json = json.dumps({'tse_raw_root':str(raw), 'election_years':[year],
                            'election_types':[YEAR_TYPES[year]],
                            'incremental_years':[year],
                            'incremental_election_types':[YEAR_TYPES[year]],
                            'tse_csv_encoding':'latin-1'})
    command = [sys.executable, str(dbt), 'run',
               '--project-dir', str(project), '--profiles-dir', str(profiles),
               '--select', '+candidate_summary', '--vars', vars_json,
               '--target-path', str(phase/'target'),
               '--log-path', str(phase/'logs'), '--no-partial-parse']
    return base.run_command(command, policy=policy, project=dbt.parents[2])


def certify(root: Path, checkout: Path, year: int,
            candidate_id: str | None, asset_id: str | None,
            max_source: int, max_extracted: int, max_rows: int,
            min_free_gib: int) -> dict:
    root = root.resolve(strict=True)
    checkout = checkout.resolve(strict=True)
    if root.is_relative_to(checkout) or checkout.is_relative_to(root):
        raise Error('canonical raw root overlaps project checkout')
    if not (root/'raw').is_dir():
        raise Error('not a TSE raw root')
    if max_rows <= 0:
        raise Error('max_rows must be positive')
    selected = select_pair(root, year, asset_id, candidate_id,
                           max_source, max_extracted)
    originals = [root/'_metadata'/x for x in ('resource_state.json',
                 'current_objects.jsonl', 'ingest_manifest.jsonl',
                 'pending_resources.json')]
    for item in selected.values():
        originals += [item['source'], *item['files']]
    original_sha = base.fingerprints(originals)
    if len(original_sha) != len([x for x in originals if x.is_file()]):
        raise Error('unreadable canonical control-plane or resource files')
    dbt = checkout/'.venv/bin/dbt'
    if not dbt.is_file():
        raise Error('dbt binary missing; run make dbt-deps')
    if shutil.disk_usage(tempfile.gettempdir()).free < min_free_gib*1024**3:
        raise Error(f'not enough free temporary disk space (requires {min_free_gib} GiB free)')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='tse-pr44-gold-semantic-') as tmp:
        temp = Path(tmp).resolve()
        if temp.is_relative_to(root) or root.is_relative_to(temp):
            raise Error('scratch root overlaps canonical raw data')
        lake = temp/'tse'
        meta = lake/'_metadata'
        meta.mkdir(parents=True)
        subset_state = {}
        subset_index = []
        expected_objects = set()
        for domain, item in selected.items():
            subset_state[item['key']] = item['row']
            subset_index.extend(item['index'])
            base.copy_into(root, lake, item['source_rel'])
            for rel in item['outputs']:
                base.copy_into(root, lake, rel)
                expected_objects.add((item['source_rel'], rel))
        (meta/'resource_state.json').write_text(json.dumps(subset_state,ensure_ascii=False))
        (meta/'current_objects.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n'
                                                      for row in subset_index))
        (meta/'pending_resources.json').write_text('{}\n')
        shutil.copy2(root/'_metadata/ingest_manifest.jsonl', meta/'ingest_manifest.jsonl')
        manifest_sha = base.sha256(meta/'ingest_manifest.jsonl')
        policy = base.offline_policy(temp/'policy')
        isolated_project = prior.isolated_dbt_project(checkout, temp)
        baseline_db = temp/'baseline.duckdb'
        before_log = isolated_run(isolated_project,dbt,lake,baseline_db,
                                  temp/'initial',year,policy)
        baseline_signatures = capture_db(baseline_db,year,max_rows)
        import duckdb
        with duckdb.connect(str(baseline_db),read_only=True) as conn:
            joined_assets = conn.execute(
                f'SELECT count(*) FROM main.{quote_ident(GOLD)} '
                'WHERE election_year = ? AND election_type = ? '
                'AND declared_assets_count > 0',
                [year,YEAR_TYPES[year]],
            ).fetchone()[0]
        if joined_assets <= 0:
            raise Error('Gold slice has no enriched candidates; assets dependency is unproven')
        # Both recovery and uninterrupted control start from the exact same
        # committed DuckDB state, copied only after all connections close.
        control_db = temp/'control.duckdb'
        recovered_db = temp/'recovered.duckdb'
        shutil.copy2(baseline_db, control_db)
        shutil.copy2(baseline_db, recovered_db)
        probe_hashes={}
        for path in (control_db,recovered_db):
            with duckdb.connect(str(path)) as conn:
                probe_hashes[path] = seed_partition_probes(conn,year)
        if probe_hashes[control_db] != probe_hashes[recovered_db]:
            raise Error('control/recovery probes differ before updates')
        control_log = isolated_run(isolated_project,dbt,lake,control_db,
                                   temp/'control',year,policy)
        control = capture_db(control_db,year,max_rows)
        verify_db(control_db,year,max_rows,probe_hashes[control_db])
        # Prune/recover only candidate_assets in disposable tree. Candidate
        # source stays active to feed Bronze->Gold during both arms.
        eviction = base.run_command([sys.executable,str(checkout/'scripts/prune_extracted_raw.py'),
                                     '--root',str(lake/'raw'),'--domain','candidate_assets','--apply'],
                                    policy=policy,project=checkout)
        state = base.read_json(meta/'resource_state.json')
        remaining = base.read_index(meta/'current_objects.jsonl')
        cand = selected['candidate']
        assets = selected['candidate_assets']
        if (state[assets['key']].get('materialization_state') != 'evicted'
            or state[cand['key']].get('materialization_state') == 'evicted'
            or {(x.get('source_object'),x.get('object')) for x in remaining}
               != {(x.get('source_object'),x.get('object')) for x in cand['index']}
            or any((lake/rel).exists() for rel in assets['outputs'])):
            raise Error('domain-scoped eviction modified unrelated candidate resource or index')
        rehydration = base.run_command([sys.executable,str(checkout/'scripts/ensure_materialized.py'),
                                       '--root',str(lake),'--year',str(year),
                                       '--election-type',YEAR_TYPES[year],
                                       '--domain','candidate_assets','--extractor','python'],
                                      policy=policy,project=checkout)
        if 'rehydrated=1' not in rehydration or 'errors=0' not in rehydration:
            raise Error('asset resource did not rehydrate offline')
        rest = base.read_json(meta/'resource_state.json')
        restored_pairs = [(x.get('source_object'),x.get('object'))
                          for x in base.read_index(meta/'current_objects.jsonl')]
        if (rest[cand['key']] != cand['row']
            or rest[assets['key']].get('materialization_state') != 'present'
            or len(restored_pairs) != len(expected_objects)
            or set(restored_pairs) != expected_objects):
            raise Error('post-recovery state/index mismatch')
        for item in selected.values():
            for src in [item['source'],*item['files']]:
                rel=src.relative_to(root)
                if base.sha256(lake/rel) != original_sha[str(src)]:
                    raise Error(f'payload diverged: {rel}')
        recovered_log = isolated_run(isolated_project,dbt,lake,recovered_db,
                                     temp/'recovered',year,policy)
        recovered = capture_db(recovered_db,year,max_rows)
        verify_db(recovered_db,year,max_rows,probe_hashes[recovered_db])
        if baseline_signatures != control or control != recovered:
            raise Error('persisted Gold/Semantic diverged across baseline/control/recovery')
        audit = json.loads(base.run_command([sys.executable,str(checkout/'scripts/report_local_storage.py'),
                          '--raw-root',str(lake),'--duckdb',str(recovered_db),
                          '--candidate-fact-root',str(temp/'no-physical-facts'),
                          '--verify-hashes','--strict','--json'],policy=policy,project=checkout))
        if (audit.get('issues') or audit.get('current_index_rows') != len(expected_objects)
            or audit.get('source_hashes_verified') != 2):
            raise Error('strict two-resource storage audit failed')
        base.run_command([sys.executable,str(checkout/'scripts/check_raw_contracts.py'),
                          '--root',str(lake)],policy=policy,project=checkout)
        if base.sha256(meta/'ingest_manifest.jsonl') != manifest_sha:
            raise Error('manifest mutated')
        elapsed = round(time.monotonic()-started,3)
    if base.fingerprints(originals) != original_sha:
        raise Error('canonical control plane or source bytes mutated')
    return {'schema_version':1,'result':'PASS',
            'scope':'candidate + candidate_assets -> dim_candidate (incremental) -> candidate_summary (persisted snapshot)',
            'year':year,'election_type':YEAR_TYPES[year],
            'candidate_resource_id':selected['candidate']['row']['resource_id'],
            'assets_resource_id':selected['candidate_assets']['row']['resource_id'],
            'legacy':{domain:bool(item['legacy_materialization_state_missing'])
                      for domain,item in selected.items()},
            'asset_enriched_candidate_rows':joined_assets,
            'gold':baseline_signatures['gold'],
            'semantic':baseline_signatures['semantic'],
            'control_and_recovered_signatures_equal':True,
            'selected_partition_replaced':True,
            'unselected_partition_preserved':True,
            'persisted_duckdb_gold_and_semantic_equal':True,
            'offline_python_guard':True,'canonical_read_only':True,
            'scratch_destroyed':True,'strict_storage_audit_passed':True,
            'elapsed_seconds':elapsed,
            'dbt_model_runs':{'baseline':before_log.strip().splitlines()[-2:],
                              'control':control_log.strip().splitlines()[-2:],
                              'recovered':recovered_log.strip().splitlines()[-2:]},
            'prune':eviction.strip().splitlines()[-3:],
            'rehydrate':rehydration.strip().splitlines()[-1:]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--checkout',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--year',type=int,default=2026)
    p.add_argument('--candidate-resource-id')
    p.add_argument('--assets-resource-id')
    p.add_argument('--max-source-mib',type=int,default=128)
    p.add_argument('--max-extracted-mib',type=int,default=512)
    p.add_argument('--max-rows',type=int,default=500000)
    p.add_argument('--min-free-gib',type=int,default=3)
    p.add_argument('--output-json',type=Path)
    p.add_argument('--inspect-sources',action='store_true',help='read-only metadata preflight')
    args = p.parse_args()
    try:
        if min(args.max_source_mib,args.max_extracted_mib,args.max_rows,args.min_free_gib)<1:
            raise Error('limits must all be positive')
        root=args.raw_root.resolve(strict=True)
        if args.output_json and args.output_json.resolve(strict=False).is_relative_to(root):
            raise Error('report cannot be inside canonical raw root')
        if args.inspect_sources:
            result=inspect_sources(root,args.year,args.max_source_mib*1024**2,
                                   args.max_extracted_mib*1024**2)
        else:
            result=certify(root,args.checkout,args.year,args.candidate_resource_id,
                           args.assets_resource_id,args.max_source_mib*1024**2,
                           args.max_extracted_mib*1024**2,args.max_rows,args.min_free_gib)
        output=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
        if args.output_json:
            args.output_json.parent.mkdir(parents=True,exist_ok=True)
            args.output_json.write_text(output,encoding='utf-8')
        print(output,end='')
        return 0
    except (Error,OSError,ValueError,subprocess.TimeoutExpired) as e:
        print(f'PR44 Gold/Semantic certification FAILED: {e}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
