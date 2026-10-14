"""PR44 -- contract tests for scratch-only persisted incremental Gold/Semantic gate."""
import decimal
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import certify_gold_semantic_roundtrip as c


class Cursor:
    def __init__(self, values, cols):
        self.values = values
        self.description = [(name, datatype) for name,datatype in cols]
        self.pos = 0

    def fetchmany(self, n):
        batch=self.values[self.pos:self.pos+n]
        self.pos+=len(batch)
        return batch


class Connection:
    def __init__(self, rows, columns):
        self.rows,self.columns=rows,columns
        self.queries=[]

    def execute(self, sql, params=None):
        self.queries.append((sql,params))
        return Cursor(self.rows,self.columns)


class GoldSemanticContracts(unittest.TestCase):
    def test_order_independent_duplicate_sensitive(self):
        rows=[('1',decimal.Decimal('10.00')),('2',decimal.Decimal('0.00')),
              ('1',decimal.Decimal('10.00'))]
        cols=[('candidate_id','VARCHAR'),('amount','DECIMAL(18,2)')]
        a=c.relation_signature(Connection(rows,cols),'dim_candidate',2026,'gold',max_rows=20)
        b=c.relation_signature(Connection(rows[::-1],cols),'dim_candidate',2026,'gold',max_rows=20)
        self.assertEqual(a,b)
        self.assertEqual(a['row_count'],3)
        self.assertEqual(a['distinct_signatures'],2)
        d=c.relation_signature(Connection(rows[:-1],cols),'dim_candidate',2026,'gold',max_rows=20)
        self.assertNotEqual(d['multiset_sha256'],a['multiset_sha256'])

    def test_value_and_schema_change_fail(self):
        cols=[('candidate_id','VARCHAR')]
        a=c.relation_signature(Connection([('one',)],cols),'dim_candidate',2026,'gold',max_rows=2)
        b=c.relation_signature(Connection([('two',)],cols),'dim_candidate',2026,'gold',max_rows=2)
        x=c.relation_signature(Connection([('one',)],[('candidate_id','INTEGER')]),'dim_candidate',2026,'gold',max_rows=2)
        self.assertNotEqual(a,b)
        self.assertNotEqual(a,x)

    def test_scope_has_year_and_election_type(self):
        fake=Connection([(100,)], [('amount','INTEGER')])
        c.relation_signature(fake,'dim_candidate',2026,'gold',max_rows=10)
        sql,parameters=fake.queries[0]
        self.assertIn('WHERE election_year = ? AND election_type = ?',sql)
        self.assertEqual(parameters,[2026,'general'])

    def test_semantic_persisted_name_only(self):
        fake=Connection([('x',)],[('value','VARCHAR')])
        c.relation_signature(fake,c.PERSISTED_SEMANTIC,2026,'semantic',max_rows=5)
        self.assertIn('pr44_candidate_summary_persisted',fake.queries[0][0])
        for name,kind in [('candidate_summary','semantic'),('dim_party','gold'),
                          ('dim_candidate','semantic')]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(c.Error,'unapproved'):
                    c.relation_signature(fake,name,2026,kind,max_rows=5)

    def test_empty_and_row_limit_are_hard_failures(self):
        with self.assertRaisesRegex(c.Error,'no rows'):
            c.relation_signature(Connection([],[('value','INTEGER')]),'dim_candidate',2026,'gold',max_rows=10)
        with self.assertRaisesRegex(c.Error,'row safety limit'):
            c.relation_signature(Connection([(1,),(2,)],[('value','INTEGER')]),'dim_candidate',2026,'gold',max_rows=1)

    def test_reject_unknown_year_and_unsafe_identifier(self):
        with self.assertRaisesRegex(c.Error,'unsupported year'):
            c.select_pair(Path('/tmp/unused'),2019,None,None,100,100)
        with self.assertRaisesRegex(c.Error,'invalid SQL'):
            c.quote_ident('\x00')

    def test_run_dbt_scoped_to_gold_semantic_with_disposable_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            checkout=root/'checkout';(checkout/'.venv/bin').mkdir(parents=True)
            dbt=checkout/'.venv/bin/dbt'
            scratch=root/'scratch';scratch.mkdir()
            with mock.patch.object(c.base,'run_command',return_value='success') as run:
                c.isolated_run(scratch/'dbt_project',dbt,scratch/'tse',scratch/'before.duckdb',
                               scratch/'before',2026,scratch/'policy')
            command=run.call_args.args[0]
            self.assertIn('+candidate_summary',command)
            self.assertIn('--no-partial-parse',command)
            self.assertNotIn('dbt deps',command)
            self.assertIn('run',command)
            vars_=json.loads(command[command.index('--vars')+1])
            self.assertEqual(vars_['election_years'],[2026])
            self.assertEqual(vars_['incremental_years'],[2026])
            self.assertEqual(vars_['tse_raw_root'],str(scratch/'tse'))
            profile_path=Path(command[command.index('--profiles-dir')+1])/'profiles.yml'
            profile=profile_path.read_text()
            self.assertIn(str(scratch/'before.duckdb'),profile)
            self.assertNotIn('httpfs',profile)

    def test_choose_two_domains_fails_if_one_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);meta=root/'_metadata';meta.mkdir()
            (meta/'resource_state.json').write_text('{}')
            with self.assertRaisesRegex(c.Error,'no candidate resource'):
                c.select_pair(root,2026,'assets',None,100,100)

    def test_consent_rejects_canonical_root_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(c.Error,'overlaps'):
                c.certify(root,root,2026,None,None,10,10,100,3)

    def test_partition_probe_constants_differ(self):
        self.assertNotEqual(c.PROBE_SELECTED,c.PROBE_OUTSIDE)
        self.assertNotEqual(c.YEAR_TYPES[2026],c.YEAR_TYPES[2024])

    def test_real_duckdb_probe_lifecycle_when_installed(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB is installed by make setup in WSL')
        with duckdb.connect(':memory:') as db:
            db.execute('CREATE TABLE dim_candidate AS SELECT 2026 AS election_year, '
                       "'general' AS election_type, '2026:general' AS cycle_id, '2026:general:1' AS election_id, '1' AS election_code, "
                       "'abc' AS candidate_id, 'Ana' AS candidate_name")
            db.execute('CREATE VIEW candidate_summary AS SELECT election_year, election_type, '
                       'count(*) AS candidates FROM dim_candidate GROUP BY 1,2')
            c.seed_partition_probes(db,2026)
            c.verify_probes(db,2026,expected_selected=1,expected_outside=1)
            db.execute("DELETE FROM dim_candidate WHERE election_year = 2026 AND candidate_id = ?",[c.PROBE_SELECTED])
            c.verify_probes(db,2026,expected_selected=0,expected_outside=1)
            c.save_semantic_snapshot(db)
            x=c.relation_signature(db,c.PERSISTED_SEMANTIC,2026,'semantic',max_rows=100)
            self.assertEqual(x['row_count'],1)


if __name__=='__main__':
    unittest.main()
