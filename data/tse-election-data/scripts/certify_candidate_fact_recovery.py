#!/usr/bin/env python3
"""PR45: real candidate fact Parquet publisher crash/recovery + dbt consumer gate.

All data is SYNTHETIC and all writes are confined to TemporaryDirectory.
Never reads or writes the canonical TSE lake/warehouse/Parquet root.
Requires Linux/WSL POSIX SIGKILL, flock, symlinks and atomic rename.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import certify_cache_roundtrip as base
import certify_dbt_cache_roundtrip as prior
import write_candidate_fact_partition as publisher
from write_candidate_fact_partition import current_generation, publish_partition

Error = base.CertificationError
YEAR, CYCLE = 2026, 'general'
OTHER_YEAR, OTHER_CYCLE = 2024, 'municipal'
FACT = 'fact_candidate_votes'
RANKING = 'candidate_ranking'
SELECTED_ROWS = (('candidate-A', 11), ('candidate-B', 23))
UPDATED_ROWS = (('candidate-A', 50), ('candidate-B', 17))
SOURCE_COLUMNS = ('election_year', 'election_type', 'election_scope', 'election_id',
                  'election_code', 'round_number', 'uf', 'office_code',
                  'candidate_id', 'nominal_valid_votes')


def require_scratch(root: Path, paths: list[Path]) -> None:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise Error('scratch root missing')
    for p in paths:
        resolved = p.resolve(strict=False)
        if not resolved.is_relative_to(root) or resolved == root:
            raise Error('attempt to use path outside temporary scratch root')


def partition_dir(root: Path, year: int, election_type: str) -> Path:
    if (year, election_type) not in ((YEAR, CYCLE), (OTHER_YEAR, OTHER_CYCLE)):
        raise Error('unapproved partition')
    return root / f'election_type={election_type}' / f'year={year}'


def publish_table(conn: object, *, year: int, cycle: str, records: tuple) -> None:
    """Replace physical source staging table with a deterministic fixture."""
    if (year, cycle) not in ((YEAR, CYCLE), (OTHER_YEAR, OTHER_CYCLE)):
        raise Error('unapproved source partition')
    if not records or len({rid for rid, _ in records}) != len(records):
        raise Error('empty or duplicate candidate fixture')
    conn.execute('''CREATE OR REPLACE TABLE main.int_candidate_fact_partition AS
        SELECT CAST(NULL AS INTEGER) AS election_year,
               CAST(NULL AS VARCHAR) AS election_type,
               CAST(NULL AS VARCHAR) AS election_scope,
               CAST(NULL AS VARCHAR) AS election_id,
               CAST(NULL AS VARCHAR) AS election_code,
               CAST(NULL AS INTEGER) AS round_number,
               CAST(NULL AS VARCHAR) AS uf,
               CAST(NULL AS VARCHAR) AS office_code,
               CAST(NULL AS VARCHAR) AS candidate_id,
               CAST(NULL AS BIGINT) AS nominal_valid_votes WHERE false''')
    rows = [(year, cycle, 'national' if cycle == CYCLE else 'municipal',
             f'{year}:{cycle}', '100', 1, 'GO', '6', rid, votes)
            for rid, votes in records]
    conn.executemany('INSERT INTO main.int_candidate_fact_partition VALUES (?,?,?,?,?,?,?,?,?,?)', rows)


def seed_candidate_dimension(conn: object) -> None:
    """Synthetic persisted Gold dimension; same join columns as real model."""
    conn.execute('''CREATE TABLE main.dim_candidate (
        election_year INTEGER, election_type VARCHAR, election_code VARCHAR,
        candidate_id VARCHAR, electoral_unit VARCHAR, uf VARCHAR,
        office VARCHAR, office_scope VARCHAR, candidate_number VARCHAR,
        candidate_name VARCHAR, ballot_name VARCHAR, party_number VARCHAR,
        party VARCHAR, party_name VARCHAR)''')
    data = [(YEAR, CYCLE, '100', rid, 'GO', 'GO', 'Deputado Federal',
             'state', str(i+10), f'Candidate {i}', f'Candidate {i}',
             str(30+i), f'P{i}', f'Party {i}')
            for i, (rid, _) in enumerate(SELECTED_ROWS)]
    conn.executemany('INSERT INTO main.dim_candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)


def sha256(path: Path) -> str:
    return base.sha256(path)


def active_snapshot(facts: Path, year: int, cycle: str) -> dict:
    """Read the *real* public symlink and its immutable generation through DuckDB."""
    import duckdb
    partition = partition_dir(facts, year, cycle)
    name = current_generation(partition)
    if name in ('none', 'legacy') or not re.fullmatch(r'generation-[0-9a-f]{32}\.parquet', name):
        raise Error(f'no valid immutable generation: {name}')
    active = partition / 'data.parquet'
    if not active.is_symlink() or os.readlink(active) != name:
        raise Error('public pointer does not select immutable generation')
    immutable = partition / name
    if not immutable.is_file() or immutable.is_symlink():
        raise Error('immutable parquet target missing')
    with duckdb.connect(':memory:') as conn:
        query = "SELECT * FROM read_parquet(?, hive_partitioning=false)"
        relation = conn.execute(query, [str(active)])
        schema = [[col[0], str(col[1])] for col in relation.description]
        rows = relation.fetchall()
    if not rows:
        raise Error('vacuous physical partition')
    if [x[0] for x in schema] != list(SOURCE_COLUMNS):
        raise Error('unexpected physical fact schema')
    digest = rows_signature(rows, schema)
    return {'generation': name, 'sha256': sha256(immutable),
            'row_count': len(rows), 'schema': schema, 'multiset_sha256': digest}


def rows_signature(rows: list[tuple], schema: list[list[str]]) -> str:
    """Order independent, duplicate sensitive, schema aware."""
    from collections import Counter
    if not schema or not rows:
        raise Error('vacuous signature')
    counts = Counter()
    for row in rows:
        if len(row) != len(schema):
            raise Error('invalid row width')
        blob = json.dumps([prior.normalize_value(v) for v in row],
                          separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        counts[hashlib.sha256(blob).hexdigest()] += 1
    h = hashlib.sha256()
    h.update(json.dumps(schema, separators=(',', ':')).encode('utf-8'))
    h.update(b'\n')
    for digest, count in sorted(counts.items()):
        h.update(f'{digest}:{count}\n'.encode('ascii'))
    return h.hexdigest()


def run_consumer(project: Path, dbt_bin: Path, database: Path,
                 facts: Path, scratch: Path, label: str, policy: Path) -> str:
    """Actually execute repository's fact_candidate_votes and candidate_ranking dbt models."""
    phase = scratch / label
    phase.mkdir()
    (phase / 'spill').mkdir()
    profiles = prior.write_profile(phase, database, phase / 'spill')
    variables = {'candidate_fact_root': str(facts),
                 'tse_raw_root': str(scratch / 'unreferenced_raw'),
                 'election_years': [YEAR], 'election_types': [CYCLE],
                 'incremental_years': [YEAR], 'incremental_election_types': [CYCLE]}
    command = [sys.executable, str(dbt_bin), 'run',
               '--project-dir', str(project), '--profiles-dir', str(profiles),
               '--select', FACT, RANKING, '--vars', json.dumps(variables),
               '--target-path', str(phase/'target'),
               '--log-path', str(phase/'logs'), '--no-partial-parse']
    return base.run_command(command, policy=policy, project=dbt_bin.parent.parent.parent)


def persisted_ranking(database: Path, *, max_rows: int = 10_000) -> dict:
    import duckdb
    with duckdb.connect(str(database), read_only=True) as conn:
        row = conn.execute("SELECT table_type FROM information_schema.tables "
                           "WHERE table_schema='main' AND table_name='candidate_ranking'").fetchone()
        if row != ('BASE TABLE',):
            raise Error('candidate_ranking was not persisted by real dbt model')
        cursor = conn.execute('SELECT * FROM main.candidate_ranking WHERE election_year=? AND election_type=?',
                              [YEAR, CYCLE])
        schema = [[c[0], str(c[1])] for c in cursor.description]
        rows = cursor.fetchmany(max_rows+1)
        if not rows or len(rows) > max_rows or cursor.fetchone() is not None:
            raise Error('empty or oversized persisted semantic relation')
    return {'schema': schema, 'row_count': len(rows),
            'multiset_sha256': rows_signature(rows, schema)}


def kill_before_pointer_swap(db: Path, facts: Path, expected: str,
                             project_root: Path, *, timeout: int = 90) -> dict:
    """SIGKILL worker after immutable generation fsync, before public pointer swap."""
    code = r'''
import os, signal, sys
from pathlib import Path
from scripts import write_candidate_fact_partition as publisher
warehouse, facts, expected = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
original = publisher.os.replace
public = facts / 'election_type=general/year=2026/data.parquet'
def intercepted(src, dest):
    if Path(dest) == public and Path(src).name.startswith('.data-parquet-'):
        os.kill(os.getpid(), signal.SIGKILL)
        raise AssertionError('SIGKILL did not terminate publisher')
    return original(src, dest)
publisher.os.replace = intercepted
publisher.publish_partition(warehouse, facts, 2026, 'general', expected_generation=expected)
raise AssertionError('failure injection did not trigger')
'''
    result = subprocess.run([sys.executable, '-c', code, str(db), str(facts), expected],
                            cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=timeout, check=False)
    if result.returncode != -signal.SIGKILL:
        raise Error(f'expected actual SIGKILL, got {result.returncode}: {result.stderr[-1000:]}')
    return {'signal': 'SIGKILL', 'returncode': result.returncode,
            'boundary': 'after immutable fsync, before symlink swap'}


def certify(project_root: Path, *, max_rows: int = 10_000, min_free_gib: int = 2) -> dict:
    if os.name != 'posix' or not hasattr(signal, 'SIGKILL'):
        raise Error('requires Linux/WSL POSIX SIGKILL')
    if max_rows <= 0 or min_free_gib < 0:
        raise Error('invalid resource limits')
    project_root = project_root.resolve(strict=True)
    if shutil.disk_usage(tempfile.gettempdir()).free < min_free_gib*1024**3:
        raise Error(f'insufficient scratch space; need {min_free_gib} GiB free')
    dbt_bin = project_root / '.venv/bin/dbt'
    if not dbt_bin.is_file():
        raise Error('missing dbt; run make dbt-deps')
    import duckdb
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='tse-pr45-physical-') as path:
        root = Path(path).resolve()
        physical = root / 'baseline_facts'
        database = root / 'baseline.duckdb'
        require_scratch(root, [physical, database])
        with duckdb.connect(str(database)) as conn:
            seed_candidate_dimension(conn)
            publish_table(conn, year=OTHER_YEAR, cycle=OTHER_CYCLE,
                          records=(('unrelated-2024', 333),))
        publisher.publish_partition(database, physical, OTHER_YEAR, OTHER_CYCLE)
        with duckdb.connect(str(database)) as conn:
            publish_table(conn, year=YEAR, cycle=CYCLE, records=SELECTED_ROWS)
        publisher.publish_partition(database, physical, YEAR, CYCLE)
        before = active_snapshot(physical, YEAR, CYCLE)
        if before['row_count'] != len(SELECTED_ROWS):
            raise Error('bad initial selected partition')
        other = active_snapshot(physical, OTHER_YEAR, OTHER_CYCLE)
        isolated_project = prior.isolated_dbt_project(project_root, root)
        policy = base.offline_policy(root / 'policy')
        baseline_log = run_consumer(isolated_project, dbt_bin, database, physical, root,
                                    'initial_consumer', policy)
        old_ranking = persisted_ranking(database, max_rows=max_rows)
        control_db, recovered_db = root/'control.duckdb', root/'recovered.duckdb'
        control_facts, recovered_facts = root/'control_facts', root/'recovered_facts'
        shutil.copy2(database, control_db); shutil.copy2(database, recovered_db)
        shutil.copytree(physical, control_facts, symlinks=True)
        shutil.copytree(physical, recovered_facts, symlinks=True)
        for db in (control_db, recovered_db):
            with duckdb.connect(str(db)) as conn:
                publish_table(conn, year=YEAR, cycle=CYCLE, records=UPDATED_ROWS)
        control_old = current_generation(partition_dir(control_facts,YEAR,CYCLE))
        recovered_old = current_generation(partition_dir(recovered_facts,YEAR,CYCLE))
        if control_old != recovered_old:
            raise Error('different initial generation pointers')
        publisher.publish_partition(control_db, control_facts, YEAR, CYCLE,
                                    expected_generation=control_old)
        control_physical = active_snapshot(control_facts, YEAR, CYCLE)
        control_log = run_consumer(isolated_project, dbt_bin, control_db, control_facts, root,
                                   'control_consumer', policy)
        control_ranking = persisted_ranking(control_db, max_rows=max_rows)
        interrupted = kill_before_pointer_swap(recovered_db, recovered_facts, recovered_old,
                                               project_root)
        after_kill = active_snapshot(recovered_facts, YEAR, CYCLE)
        if after_kill != before:
            # Ignore generation equality across copied root; content identity still necessary.
            raise Error('SIGKILL exposed an incomplete or premature publication')
        old_generation_path = partition_dir(recovered_facts,YEAR,CYCLE) / recovered_old
        old_generation_sha = sha256(old_generation_path)
        publisher.publish_partition(recovered_db, recovered_facts, YEAR, CYCLE,
                                    expected_generation=recovered_old)
        recovered_physical = active_snapshot(recovered_facts, YEAR, CYCLE)
        if old_generation_path.is_file() is False or sha256(old_generation_path) != old_generation_sha:
            raise Error('old immutable generation lost after recovery')
        staging = list(partition_dir(recovered_facts,YEAR,CYCLE).glob('.generation-*.tmp'))
        staging += list(partition_dir(recovered_facts,YEAR,CYCLE).glob('.data-parquet-*.link'))
        if staging:
            raise Error('abandoned unpublished staging files remain')
        if recovered_physical['multiset_sha256'] != control_physical['multiset_sha256'] or recovered_physical['schema'] != control_physical['schema']:
            raise Error('control/recovery Parquet row multiset diverged')
        if recovered_physical['multiset_sha256'] == before['multiset_sha256']:
            raise Error('published update did not change physical content')
        for facts in (control_facts, recovered_facts):
            if active_snapshot(facts, OTHER_YEAR, OTHER_CYCLE) != other:
                raise Error('unrelated 2024 partition modified')
        try:
            publisher.publish_partition(recovered_db,recovered_facts,YEAR,CYCLE,
                                        expected_generation=recovered_old)
        except RuntimeError as exc:
            if 'generation changed' not in str(exc):
                raise
        else:
            raise Error('stale generation not fenced after recovery')
        recovered_log = run_consumer(isolated_project, dbt_bin, recovered_db, recovered_facts,
                                     root, 'recovered_consumer', policy)
        recovered_ranking = persisted_ranking(recovered_db, max_rows=max_rows)
        if recovered_ranking != control_ranking:
            raise Error('persisted candidate_ranking differs control vs recovered')
        if old_ranking == recovered_ranking:
            raise Error('semantic consumer did not observe published update')
        result = {'schema_version':1,'result':'PASS',
                  'scope':'synthetic real Parquet publisher + real dbt fact_candidate_votes/candidate_ranking',
                  'year':YEAR,'election_type':CYCLE,
                  'source_fixture':'synthetic, no TSE canonical source accessed',
                  'physical_before':before,'physical_control':control_physical,
                  'physical_recovered':recovered_physical,
                  'persisted_ranking_before':old_ranking,
                  'persisted_ranking_control':control_ranking,
                  'persisted_ranking_recovered':recovered_ranking,
                  'kill':interrupted,
                  'control_recovery_parquet_equivalent':True,
                  'control_recovery_semantic_equivalent':True,
                  'published_update_detected':True,
                  'old_generation_retained':True,
                  'stale_generation_fenced':True,
                  'unrelated_partition_preserved':True,
                  'no_canonical_lake_access':True,
                  'scratch_destroyed':True,
                  'python_network_guarded':True,
                  'elapsed_seconds':round(time.monotonic()-started,3),
                  'dbt_model_runs':{name:log.strip().splitlines()[-2:]
                    for name,log in [('initial',baseline_log),('control',control_log),('recovered',recovered_log)]}}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--max-rows',type=int,default=10_000)
    parser.add_argument('--min-free-gib',type=int,default=2)
    parser.add_argument('--output-json',type=Path)
    args = parser.parse_args()
    try:
        result=certify(args.project_root,max_rows=args.max_rows,min_free_gib=args.min_free_gib)
        output=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
        if args.output_json:
            args.output_json.parent.mkdir(parents=True,exist_ok=True)
            args.output_json.write_text(output,encoding='utf-8')
        print(output,end='')
        return 0
    except (Error,OSError,ValueError,subprocess.TimeoutExpired,RuntimeError) as exc:
        print(f'PR45 physical certification FAILED: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
