"""PR45: falsifiable contracts for physical Parquet -> persisted semantic recovery.

Entire test suite uses synthetic content. No TSE canonical lake is accessed.
The optional real end-to-end test belongs to the explicit runner invocation.
"""
from __future__ import annotations

import json
import importlib.util
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import certify_candidate_fact_recovery as c


class SyntheticConnection:
    def __init__(self):
        self.calls = []
    def execute(self, sql, args=None):
        self.calls.append((sql,args))
        return self
    def executemany(self, sql, data):
        self.calls.append((sql,data))
        return self


class PhysicalRecoveryContracts(unittest.TestCase):
    def test_schema_signature_order_independent_and_duplicate_sensitive(self):
        schema = [['candidate_id','VARCHAR'],['valid_votes','BIGINT']]
        left = [('a',10),('a',10),('b',20)]
        self.assertEqual(c.rows_signature(left,schema), c.rows_signature(left[::-1],schema))
        self.assertNotEqual(c.rows_signature(left,schema), c.rows_signature(left[:-1],schema))

    def test_value_change_detected(self):
        schema = [['id','VARCHAR'],['votes','BIGINT']]
        self.assertNotEqual(c.rows_signature([('a',5)],schema),
                            c.rows_signature([('a',6)],schema))

    def test_schema_change_detected(self):
        self.assertNotEqual(c.rows_signature([('1',)],[['id','VARCHAR']]),
                            c.rows_signature([('1',)],[['id','INTEGER']]))

    def test_empty_and_ragged_relation_denied(self):
        for values,columns in [([], [['id','VARCHAR']]),([('a',)],[]),
                               ([('a',1)],[['id','VARCHAR']])]:
            with self.subTest(values=values,columns=columns):
                with self.assertRaises(c.Error):
                    c.rows_signature(values,columns)

    def test_scratch_rejects_outside_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c.require_scratch(root,[root/'yes',root/'nested'/'yes'])
            with self.assertRaisesRegex(c.Error,'outside'):
                c.require_scratch(root,[root.parent/'outside'])
            with self.assertRaisesRegex(c.Error,'outside'):
                c.require_scratch(root,[root])

    def test_scratch_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
            root=Path(d); (root/'link').symlink_to(Path(out),target_is_directory=True)
            with self.assertRaisesRegex(c.Error,'outside'):
                c.require_scratch(root,[root/'link'/'db.duckdb'])

    def test_partition_scope_is_explicit(self):
        p=Path('/tmp/fixture')
        self.assertEqual(c.partition_dir(p,2026,'general'),p/'election_type=general/year=2026')
        self.assertEqual(c.partition_dir(p,2024,'municipal'),p/'election_type=municipal/year=2024')
        with self.assertRaisesRegex(c.Error,'unapproved'):
            c.partition_dir(p,2026,'municipal')

    def test_fixture_requires_unique_nonempty_rows(self):
        con=SyntheticConnection()
        with self.assertRaisesRegex(c.Error,'empty or duplicate'):
            c.publish_table(con,year=2026,cycle='general',records=())
        with self.assertRaisesRegex(c.Error,'empty or duplicate'):
            c.publish_table(con,year=2026,cycle='general',records=(('a',1),('a',2)))

    def test_fixture_year_partition_and_columns(self):
        con=SyntheticConnection()
        c.publish_table(con,year=2026,cycle='general',records=(('x',17),))
        self.assertEqual(len(con.calls),2)
        inserted=con.calls[1][1]
        self.assertEqual(len(inserted),1)
        self.assertEqual(len(inserted[0]),len(c.SOURCE_COLUMNS))
        self.assertEqual((inserted[0][0],inserted[0][1]),(2026,'general'))
        with self.assertRaisesRegex(c.Error,'unapproved'):
            c.publish_table(con,year=2020,cycle='municipal',records=(('x',1),))

    def test_dimension_contains_join_keys_for_ranking(self):
        con=SyntheticConnection();c.seed_candidate_dimension(con)
        create=con.calls[0][0]
        for column in ('candidate_id','election_code','electoral_unit','office_scope','party_name'):
            self.assertIn(column,create)
        self.assertEqual(len(con.calls[1][1]),2)

    def test_consumer_dbt_selects_only_real_fact_and_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);project=root/'dbt_project';project.mkdir()
            bin=root/'.venv/bin/dbt';bin.parent.mkdir(parents=True)
            phases=root/'phase_root';phases.mkdir()
            with mock.patch.object(c.prior,'write_profile',return_value=phases), \
                 mock.patch.object(c.base,'run_command',return_value='ok') as called:
                c.run_consumer(project,bin,root/'db.duckdb',root/'facts',phases,'phase',root/'policy')
            argv=called.call_args.args[0]
            self.assertEqual(argv[argv.index('--select')+1:argv.index('--vars')],[c.FACT,c.RANKING])
            variables=json.loads(argv[argv.index('--vars')+1])
            self.assertEqual(variables['candidate_fact_root'],str(root/'facts'))
            self.assertEqual(variables['election_years'],[2026])
            self.assertEqual(variables['incremental_election_types'],['general'])
            self.assertIn('--no-partial-parse',argv)
            self.assertNotIn('+candidate_ranking',argv)

    def test_kill_boundary_is_actual_sigkill_not_exception(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            fake=mock.Mock(returncode=-signal.SIGKILL,stderr='')
            with mock.patch.object(c.subprocess,'run',return_value=fake) as run:
                result=c.kill_before_pointer_swap(root/'db',root/'facts','generation-old.parquet',root)
            self.assertEqual(result['signal'],'SIGKILL')
            args=run.call_args.args[0]
            self.assertIn('signal.SIGKILL',args[2])
            self.assertIn('expected_generation=expected',args[2])
            fake.returncode=1
            with mock.patch.object(c.subprocess,'run',return_value=fake):
                with self.assertRaisesRegex(c.Error,'expected actual SIGKILL'):
                    c.kill_before_pointer_swap(root/'db',root/'facts','none',root)

    def test_runner_fail_closed_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for max_rows,min_free in [(0,0),(20,-1)]:
                with self.subTest(max_rows=max_rows,min_free=min_free):
                    with self.assertRaisesRegex(c.Error,'invalid resource limits'):
                        c.certify(root,max_rows=max_rows,min_free_gib=min_free)

    @unittest.skipUnless(importlib.util.find_spec('duckdb') is not None,
                         'needs real DuckDB (provided by project WSL environment)')
    def test_real_publisher_produces_readable_immutable_partition(self):
        import duckdb
        from write_candidate_fact_partition import publish_partition
        with tempfile.TemporaryDirectory(prefix='pr45-real-publisher-') as d:
            root = Path(d)
            database, facts = root/'db.duckdb', root/'facts'
            with duckdb.connect(str(database)) as conn:
                c.publish_table(conn,year=2026,cycle='general',records=c.SELECTED_ROWS)
            publish_partition(database,facts,2026,'general')
            snap=c.active_snapshot(facts,2026,'general')
            self.assertEqual(snap['row_count'],len(c.SELECTED_ROWS))
            self.assertEqual(len(snap['schema']),len(c.SOURCE_COLUMNS))
            self.assertEqual(len(snap['sha256']),64)
            self.assertTrue((facts/'election_type=general/year=2026/data.parquet').is_symlink())

    def test_direct_script_entrypoint_import_does_not_require_scripts_package(self):
        # `python scripts/certify_candidate_fact_recovery.py` adds scripts/, not
        # its parent, to sys.path. The old `from scripts import ...` failed
        # in certify(), although test discovery imported the module successfully.
        class EnteredScratch(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dbt = root / '.venv/bin/dbt'
            dbt.parent.mkdir(parents=True)
            dbt.write_text('placeholder')
            import types
            with mock.patch.dict(sys.modules, {
                'scripts': None,
                'duckdb': types.ModuleType('duckdb'),
            }), mock.patch.object(c.tempfile, 'TemporaryDirectory',
                                side_effect=EnteredScratch('publisher imported')):
                with self.assertRaisesRegex(EnteredScratch, 'publisher imported'):
                    c.certify(root, max_rows=10, min_free_gib=0)

    def test_no_canonical_root_argument_exposed(self):
        import inspect
        self.assertNotIn('raw_root',inspect.signature(c.certify).parameters)
        self.assertNotIn('warehouse_path',inspect.signature(c.certify).parameters)


if __name__=='__main__':
    unittest.main()
