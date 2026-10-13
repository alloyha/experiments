"""PR43: deterministic order-insensitive dbt relation signatures, safety and scoped selection."""
from __future__ import annotations

import datetime
import decimal
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import certify_dbt_cache_roundtrip as c


class FakeCursor:
    def __init__(self, rows, types):
        self.rows = list(rows)
        self.description = [(n, t) for n, t in types]
        self.pos = 0

    def fetchmany(self, n):
        value = self.rows[self.pos:self.pos + n]
        self.pos += len(value)
        return value


class FakeConnection:
    def __init__(self, rows, types):
        self.rows = rows
        self.types = types
        self.calls = []

    def execute(self, query):
        self.calls.append(query)
        return FakeCursor(self.rows, self.types)


class DbtCacheRoundtripTests(unittest.TestCase):
    def test_order_independent_signature_with_duplicates(self):
        schema = [('candidate_id', 'VARCHAR'), ('assets', 'DECIMAL(18,2)')]
        rows = [('A', decimal.Decimal('10.00')), ('A', decimal.Decimal('10.00')),
                ('B', decimal.Decimal('1.00'))]
        left = c.multiset_signature(FakeConnection(rows, schema), 'silver_candidate_assets')
        right = c.multiset_signature(FakeConnection(rows[::-1], schema), 'silver_candidate_assets')
        self.assertEqual(left, right)
        self.assertEqual(left['row_count'], 3)
        self.assertEqual(left['distinct_row_signatures'], 2)

    def test_changed_value_fails_relation_equivalence(self):
        cols = [('id', 'VARCHAR'), ('amount', 'INTEGER')]
        a = c.multiset_signature(FakeConnection([('1', 10)], cols), 'silver_candidate_assets')
        b = c.multiset_signature(FakeConnection([('1', 11)], cols), 'silver_candidate_assets')
        self.assertNotEqual(a, b)

    def test_changed_schema_fails_relation_equivalence(self):
        first = c.multiset_signature(FakeConnection([('1',)], [('id', 'VARCHAR')]), 'bronze_candidate_assets')
        second = c.multiset_signature(FakeConnection([('1',)], [('id', 'INTEGER')]), 'bronze_candidate_assets')
        self.assertNotEqual(first, second)

    def test_empty_relation_is_not_pass(self):
        with self.assertRaisesRegex(c.CertificationError, 'vacuous'):
            c.multiset_signature(FakeConnection([], [('id', 'VARCHAR')]), 'silver_candidate_assets')

    def test_row_limit_is_fail_closed(self):
        with self.assertRaisesRegex(c.CertificationError, 'limit exceeded'):
            c.multiset_signature(FakeConnection([('1',), ('2',)], [('id', 'VARCHAR')]),
                                 'silver_candidate_assets', max_rows=1)

    def test_unapproved_relation_rejected(self):
        with self.assertRaisesRegex(c.CertificationError, 'unapproved'):
            c.multiset_signature(FakeConnection([], []), 'financial_production')

    def test_float_and_null_and_date_canonicalization(self):
        self.assertEqual(c.normalize_value(None), ['null'])
        self.assertEqual(c.normalize_value(float('-0.0')), ['float', '-0x0.0p+0'])
        self.assertEqual(c.normalize_value(datetime.date(2026, 10, 13)), ['date', '2026-10-13'])
        self.assertEqual(c.normalize_value(True), ['bool', True])
        self.assertEqual(c.normalize_value(b'\x00'), ['binary', '00'])

    def test_unsupported_type_rejected(self):
        with self.assertRaisesRegex(c.CertificationError, 'unsupported'):
            c.normalize_value(object())

    def test_isolated_profile_only_scratch_paths(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            profile = c.write_profile(tmp, tmp / 'db.duckdb', tmp / 'spill')
            txt = (profile / 'profiles.yml').read_text()
            self.assertIn(str(tmp / 'db.duckdb'), txt)
            self.assertNotIn('httpfs', txt)
            self.assertNotIn('prod', txt)

    def test_dbt_project_copy_without_target_or_logs(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            src = tmp / 'source' / 'tse_dbt'
            (src / 'dbt_packages').mkdir(parents=True)
            (src / 'target').mkdir()
            (src / 'logs').mkdir()
            (src / 'dbt_project.yml').write_text('name: test\n')
            (src / 'dbt_packages' / 'ok.txt').write_text('x')
            dest = c.isolated_dbt_project(tmp / 'source', tmp / 'scratch')
            self.assertTrue((dest / 'dbt_packages' / 'ok.txt').is_file())
            self.assertFalse((dest / 'target').exists())
            self.assertFalse((dest / 'logs').exists())

    def test_dbt_project_requires_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            src = tmp / 'tse_dbt'
            src.mkdir()
            (src / 'dbt_project.yml').write_text('name: test\n')
            with self.assertRaisesRegex(c.CertificationError, 'dbt_packages'):
                c.isolated_dbt_project(tmp, tmp / 'scratch')

    def test_scoped_resource_selection_smallest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m = root / '_metadata'
            m.mkdir()
            def make(rowid, n):
                rel = f'raw/election_type=general/year=2026/domain=candidate_assets/resource={rowid}/source/a.zip'
                output = f'raw/election_type=general/year=2026/domain=candidate_assets/resource={rowid}/extracted/a.csv'
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / output).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b'fake-zip')
                (root / output).write_bytes(b'a' * n)
                return {'year':2026,'domain':'candidate_assets','resource_id':rowid,
                        'selection_complete':True,'materialization_state':'present',
                        'source_object':rel,'extracted_objects':[output]}
            rows = {'r1':make('big', 20), 'r2':make('small', 3),
                    'notactive':{'domain':'candidate_history', 'year':2026}}
            (m / 'resource_state.json').write_text(json.dumps(rows))
            self.assertEqual(c.choose_resource(root, 2026, None, 100, 100), 'small')
            self.assertEqual(c.choose_resource(root, 2026, 'big', 100, 100), 'big')
            with self.assertRaisesRegex(c.CertificationError, 'no candidate_assets'):
                c.choose_resource(root, 2026, None, 1, 1)

    def test_candidate_inventory_explains_year_and_format_without_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meta = root / '_metadata'
            meta.mkdir()
            src = root / 'raw/election_type=general/year=2026/domain=candidate_assets/source/a.csv'
            src.parent.mkdir(parents=True)
            src.write_bytes(b'hello')
            out = root / 'raw/election_type=general/year=2026/domain=candidate_assets/extracted/a.csv'
            out.parent.mkdir(parents=True)
            out.write_bytes(b'content')
            old = root / 'raw/election_type=general/year=2022/domain=candidate_assets/source/a.zip'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'zip')
            old_out = root / 'raw/election_type=general/year=2022/domain=candidate_assets/extracted/a.csv'
            old_out.parent.mkdir(parents=True)
            old_out.write_bytes(b'abc')
            state = {
                'new': {'year': 2026, 'domain': 'candidate_assets', 'resource_id': 'new',
                        'selection_complete': True, 'materialization_state':'present',
                        'source_object':src.relative_to(root).as_posix(),
                        'extracted_objects':[out.relative_to(root).as_posix()]},
                'old': {'year': 2022, 'domain': 'candidate_assets', 'resource_id': 'old',
                        'selection_complete': True, 'materialization_state':'present',
                        'source_object':old.relative_to(root).as_posix(),
                        'extracted_objects':[old_out.relative_to(root).as_posix()]},
            }
            (meta / 'resource_state.json').write_text(json.dumps(state))
            before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = c.inspect_candidates(root, 2026, 256*1024**2, 512*1024**2)
            self.assertEqual(result['scanned_candidate_resources'],2)
            self.assertEqual(result['eligible_in_requested_year'],0)
            self.assertEqual(result['other_year_zip_resources'], [2022])
            reasons = {r['resource_id']:r['exclusion_reasons'] for r in result['resources']}
            self.assertIn('not_zip_source', reasons['new'])
            self.assertIn('different_year', reasons['old'])
            self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_legacy_missing_flag_is_preflight_candidate_not_state_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meta = root / '_metadata'
            meta.mkdir()
            source = root / 'raw/source/legacy.zip'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'legacy archive fixture')
            output = root / 'raw/extracted/asset.csv'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'id;value\n1;42\n')
            state = {'row': {
                'year': 2026, 'domain': c.DOMAIN, 'resource_id': 'legacy',
                'selection_complete': True, 'source_object': source.relative_to(root).as_posix(),
                'extracted_objects': [output.relative_to(root).as_posix()],
            }}
            state_path = meta / 'resource_state.json'
            state_path.write_text(json.dumps(state))
            baseline = state_path.read_bytes()
            self.assertEqual(c.choose_resource(root, 2026, None, 1024, 1024), 'legacy')
            inventory = c.inspect_candidates(root, 2026, 1024, 1024)
            self.assertEqual(inventory['eligible_in_requested_year'], 1)
            self.assertTrue(inventory['resources'][0]['legacy_materialization_state_missing'])
            self.assertEqual(state_path.read_bytes(), baseline)
            state['row']['materialization_state'] = None
            state_path.write_text(json.dumps(state))
            self.assertEqual(c.inspect_candidates(root, 2026, 1024, 1024)
                             ['resources'][0]['exclusion_reasons'], ['not_materialized_present'])
            self.assertEqual(c.eligible_rows(root, 2026, 1024, 1024), [])

    def test_candidate_inventory_diagnoses_size_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / '_metadata').mkdir()
            src = root / 'raw/source/large.zip'
            src.parent.mkdir(parents=True)
            src.write_bytes(b'AAAA')
            out = root / 'raw/extracted/a.csv'
            out.parent.mkdir(parents=True)
            out.write_bytes(b'BBBBBB')
            (root / '_metadata/resource_state.json').write_text(json.dumps({
                'row': {'year': 2026, 'domain':'candidate_assets', 'resource_id':'a',
                        'selection_complete':True, 'materialization_state':'present',
                        'source_object':src.relative_to(root).as_posix(),
                        'extracted_objects':[out.relative_to(root).as_posix()]}
            }))
            result=c.inspect_candidates(root,2026,3,5)
            self.assertEqual(result['eligible_in_requested_year'],0)
            self.assertEqual(set(result['resources'][0]['exclusion_reasons']),
                             {'source_over_limit','extracted_over_limit'})

    def test_scoped_selector_accepts_numeric_year_string(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / '_metadata').mkdir()
            src = root / 'raw/source/a.zip'
            src.parent.mkdir(parents=True)
            src.write_bytes(b'archive')
            output = root / 'raw/extracted/a.csv'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'rows')
            (root / '_metadata/resource_state.json').write_text(json.dumps({
                'row': {'year': '2026', 'domain':'candidate_assets',
                        'resource_id':'string-year', 'selection_complete':True,
                        'materialization_state':'present',
                        'source_object':src.relative_to(root).as_posix(),
                        'extracted_objects':[output.relative_to(root).as_posix()]}
            }))
            self.assertEqual(c.choose_resource(root,2026,None,100,100),'string-year')

    def test_run_dbt_arguments_only_disposable(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / 'proj').mkdir()
            (tmp / 'dbt').mkdir()
            (tmp / 'tse').mkdir()
            (tmp / 'policy').mkdir()
            (tmp / 'dbt-bin').write_text('# test')
            fake_results = {'bronze_candidate_assets': {'row_count':1},
                            'silver_candidate_assets': {'row_count':1}}
            with mock.patch.object(c.base, 'run_command', return_value='dbt success') as run, \
                 mock.patch.object(c, 'capture_relations', return_value=fake_results) as capture:
                result, _ = c.run_dbt(tmp / 'proj', tmp / 'dbt', tmp / 'tse',
                                      tmp / 'dbt-bin', tmp / 'policy', tmp,
                                      'before', 2026, 'general', 321)
            self.assertEqual(result, fake_results)
            command = run.call_args.args[0]
            self.assertIn('+silver_candidate_assets', command)
            self.assertIn('--no-partial-parse', command)
            vars_str = command[command.index('--vars') + 1]
            variables = json.loads(vars_str)
            self.assertEqual(variables['tse_raw_root'], str(tmp / 'tse'))
            self.assertEqual(capture.call_args.kwargs['max_rows'], 321)
            self.assertEqual(capture.call_args.args[0], tmp / 'before.duckdb')


if __name__ == '__main__':
    unittest.main()
