"""PR38: isolate dbt run artifacts and DuckDB spill for independent cycles.

All warehouse and Parquet data stays inside TemporaryDirectory; no canonical TSE writes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from scripts import build_candidate_fact_partition as job
from scripts import write_candidate_fact_partition as publisher


FAKE_DBT = r'''#!{python}
import json, os, pathlib, sys, time
import duckdb
args = sys.argv

def arg(key):
    return pathlib.Path(args[args.index(key) + 1])

if '--target-path' not in args or '--log-path' not in args:
    raise SystemExit('missing isolated dbt artifact flags')
target, logs = arg('--target-path'), arg('--log-path')
if 'TSE_DUCKDB_TEMP_DIRECTORY' in os.environ:
    raise SystemExit('obsolete spill override passed to dbt')
spills = pathlib.Path(os.environ['TSE_DUCKDB_PATH'] + '.tmp')
for p in [target, logs, spills]:
    if not p.is_absolute():
        raise SystemExit('non-absolute per-run path')
    p.mkdir(parents=True, exist_ok=True)
vars_ = json.loads(args[args.index('--vars') + 1])
payload = {{'year': vars_['incremental_years'][0],
           'type': vars_['incremental_election_types'][0],
           'db': os.environ['TSE_DUCKDB_PATH'],
           'target': str(target), 'logs': str(logs), 'spills': str(spills)}}
(target/'manifest.json').write_text(json.dumps(payload))
(logs/'dbt.log').write_text(json.dumps(payload))
(spills/'spill.marker').write_text('isolated')
if os.environ.get('START_FILE'):
    pathlib.Path(os.environ['START_FILE']).write_text('ready')
if os.environ.get('RELEASE_FILE'):
    path = pathlib.Path(os.environ['RELEASE_FILE'])
    deadline = time.monotonic() + 12
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    if not path.exists():
        raise SystemExit('release timeout')
if os.environ.get('FAIL_DBT'):
    raise SystemExit(18)
with duckdb.connect(os.environ['TSE_DUCKDB_PATH']) as con:
    con.execute('create or replace table main.int_candidate_fact_partition '
                '(election_year integer, election_type varchar, election_id integer, '
                'round_number integer, uf varchar, municipality_code varchar, '
                'office_code varchar, is_transit_vote boolean, candidate_id varchar, '
                'zone integer, nominal_votes integer)')
    con.execute("insert into main.int_candidate_fact_partition values "
                "(?, ?, 1, 1, 'GO', '5208707', '13', false, 'C1', 10, ?)",
                [payload['year'], payload['type'], int(os.getenv('VOTES', '17'))])
'''


class DbtRunIsolationTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix='pr38-isolation-')
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.db = self.root / 'warehouse.duckdb'
        self.facts = self.root / 'facts'
        self.project = self.root / 'tse_dbt'
        self.project.mkdir()
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.fake_dbt = self.root / 'fake-dbt'
        self.fake_dbt.write_text(FAKE_DBT.format(python=sys.executable))
        self.fake_dbt.chmod(0o755)

    def config(self, *, db=None, facts=None, year=2026, cycle='general', **extras):
        return dict(db_path=db or self.db, fact_root=facts or self.facts,
                    raw_root=self.raw, project_dir=self.project,
                    profiles_dir=self.project, dbt_executable=self.fake_dbt,
                    year=year, election_type=cycle, **extras)

    def published_votes(self, facts, year, cycle):
        pointer = facts / f'election_type={cycle}' / f'year={year}' / 'data.parquet'
        with duckdb.connect() as con:
            return con.execute('select nominal_votes from read_parquet(?)',
                               [str(pointer)]).fetchone()[0]

    def manifests(self, db):
        root = db.with_name(f'.{db.name}.dbt-runs')
        return sorted(root.glob('*/target/manifest.json'))

    def test_each_execution_has_unique_artifacts_and_database_scoped_spill(self):
        job.build_and_publish(**self.config())
        job.build_and_publish(**self.config())
        files = self.manifests(self.db)
        self.assertEqual(len(files), 2)
        payloads = [json.loads(f.read_text()) for f in files]
        self.assertEqual(len({p['target'] for p in payloads}), 2)
        for payload in payloads:
            self.assertTrue((Path(payload['logs']) / 'dbt.log').is_file())
            self.assertTrue((Path(payload['spills']) / 'spill.marker').is_file())
            self.assertEqual(payload['spills'], str(self.db) + '.tmp')
            self.assertEqual(Path(payload['target']).parent, Path(payload['logs']).parent)
            self.assertEqual(payload['year'], 2026)
        self.assertFalse((self.project / 'target').exists())
        self.assertFalse((self.project / 'logs').exists())
        self.assertEqual(self.published_votes(self.facts, 2026, 'general'), 17)

    def test_failure_keeps_artifact_evidence_and_previous_publication(self):
        job.build_and_publish(**self.config())
        partition = self.facts / 'election_type=general' / 'year=2026'
        before = publisher.current_generation(partition)
        with mock.patch.dict(os.environ, {'FAIL_DBT': '1'}):
            with self.assertRaises(subprocess.CalledProcessError):
                job.build_and_publish(**self.config())
        self.assertEqual(publisher.current_generation(partition), before)
        self.assertEqual(len(self.manifests(self.db)), 2)
        self.assertEqual(self.published_votes(self.facts, 2026, 'general'), 17)

    def test_explicit_artifact_root_is_respected(self):
        custom = self.root / 'custom-artifacts'
        job.build_and_publish(**self.config(artifacts_root=custom))
        files = sorted(custom.glob('*/target/manifest.json'))
        self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text())
        self.assertEqual(Path(payload['target']).parents[1], custom)
        self.assertEqual(self.manifests(self.db), [])

    def test_independent_warehouses_and_electoral_cycles_overlap_without_artifact_collision(self):
        db2 = self.root / 'other.duckdb'
        facts2 = self.root / 'other-facts'
        start_a, start_b = self.root / 'started-a', self.root / 'started-b'
        release = self.root / 'release'
        processes = []
        def launch(db, facts, year, cycle, started):
            args = [sys.executable, str(Path(job.__file__).resolve()),
                    '--db', str(db), '--root', str(facts), '--raw-root', str(self.raw),
                    '--project-dir', str(self.project), '--profiles-dir', str(self.project),
                    '--dbt-executable', str(self.fake_dbt), '--year', str(year),
                    '--election-type', cycle]
            env = {**os.environ, 'START_FILE': str(started), 'RELEASE_FILE': str(release)}
            p = subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
            processes.append(p)
            return p
        try:
            a = launch(self.db, self.facts, 2026, 'general', start_a)
            b = launch(db2, facts2, 2024, 'municipal', start_b)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and not (start_a.exists() and start_b.exists()):
                self.assertIsNone(a.poll(), 'first build exited early')
                self.assertIsNone(b.poll(), 'second build exited early')
                time.sleep(.02)
            self.assertTrue(start_a.exists() and start_b.exists(),
                            'both independent jobs must reach dbt concurrently')
        finally:
            release.touch()
            for p in processes:
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(timeout=5)
        self.assertTrue(all(p.returncode == 0 for p in processes))
        self.assertEqual(self.published_votes(self.facts, 2026, 'general'), 17)
        self.assertEqual(self.published_votes(facts2, 2024, 'municipal'), 17)
        [first] = self.manifests(self.db)
        [second] = self.manifests(db2)
        self.assertNotEqual(first.parent, second.parent)
        self.assertNotEqual(
            json.loads(first.read_text())['spills'],
            json.loads(second.read_text())['spills'])

    def test_makefile_exposes_isolated_artifact_root(self):
        makefile = (Path(__file__).resolve().parents[1] / 'Makefile').read_text()
        recipe = makefile.split('candidate-fact-partition: dbt-deps', 1)[1].split('\nunit:', 1)[0]
        self.assertIn('--artifacts-root', recipe)
        self.assertIn('DBT_ARTIFACTS_ROOT_ABS', recipe)


if __name__ == '__main__':
    unittest.main()
