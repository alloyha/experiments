"""Coverage for the offline, non-destructive conformed-Gold migration."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'scripts/migrate_conformed_gold.py'
spec = importlib.util.spec_from_file_location('migrate_conformed_gold', MOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class OfflineGoldMigrationTests(unittest.TestCase):
    def test_generated_identity_is_not_a_cycle(self):
        self.assertIn("cast(election_code as varchar)", m.EVENT)
        self.assertNotEqual(m.IDENTITY, m.EVENT)
        self.assertEqual(len(m.PHYSICAL), 8)

    def test_parquet_path_escapes_quotes(self):
        self.assertEqual(m.quoted_path(Path("/tmp/father's/test")), "'/tmp/father''s/test'")

    def test_singular_sql_uses_real_dbt_contracts(self):
        for name in m.SINGULAR:
            with self.subTest(name=name):
                text = m.dbt_sql(ROOT, name)
                self.assertNotIn('{{', text)
                self.assertIn('main.dim_election', text)

    def test_ack_is_required(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest('DuckDB installed in project .venv')
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = directory / 'source.duckdb'
            source.write_bytes(b'placeholder')
            with self.assertRaisesRegex(m.MigrationError, 'ack-quiescent'):
                m.run(source, directory / 'candidate.duckdb', directory, ROOT, ack_quiescent=False)
            self.assertEqual(source.read_bytes(), b'placeholder')
            self.assertFalse((directory / 'candidate.duckdb').exists())

    def test_absent_parquet_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(m.MigrationError, 'partitions missing'):
                m.validate_candidate_files(Path(td))

    def test_full_synthetic_offline_copy(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB installed in project .venv')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'tse_analytics.duckdb'
            candidate = folder / 'tse_analytics.conformed-candidate.duckdb'
            fact_root = folder / 'fact_candidate_votes'
            partition = fact_root / 'election_type=general' / 'year=2022'
            partition.mkdir(parents=True)
            parquet = partition / 'data.parquet'
            with duckdb.connect(str(source)) as db:
                db.execute(f"""copy (select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general:522'::varchar election_id, 14::integer nominal_valid_votes)
                    to {m.quoted_path(parquet)} (format parquet)""")
                db.execute("""create table dim_election as select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general:522'::varchar election_id""")
                db.execute("""create table dim_candidate as select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general'::varchar election_id, 'candidateA'::varchar candidate_id,
                    100::integer declared_assets_value""")
                db.execute("""create table dim_geography as select 2022::integer election_year,
                    'general'::varchar election_type, '12345'::varchar municipality_code""")
                db.execute("""create table dim_party as select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general:522'::varchar election_id, 45::integer party_number""")
                db.execute("""create table fact_electorate_municipality as select 2022::integer election_year,
                    'general'::varchar election_type, 50::integer electorate""")
                db.execute("""create table fact_tally_munzona as select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general:522'::varchar election_id,
                    50::integer eligible_voters, 36::integer turnout,
                    14::integer abstentions, 14::integer nominal_valid_votes""")
                db.execute("""create table fact_turnout as select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general:522'::varchar election_id,
                    50::integer eligible_voters, 36::integer turnout,
                    14::integer abstentions""")
                db.execute("""create table fact_party_votes as select 2022::integer election_year,
                    'general'::varchar election_type, '522'::varchar election_code,
                    '2022:general:522'::varchar election_id, 14::integer nominal_valid_votes,
                    14::integer party_valid_votes""")
                db.execute(f'''create view fact_candidate_votes as
                    select * from read_parquet({m.quoted_path(parquet)})''')
                db.execute("create table candidate_ranking as select 14::integer nominal_valid_votes")
                db.execute("create table election_results as select 14::integer top_nominal_valid_votes")
            original = m.sha256_file(source)
            report = m.run(source, candidate, fact_root, ROOT, ack_quiescent=True)
            self.assertEqual(report['result'], 'PASS')
            self.assertEqual(report['cycles'], 1)
            self.assertEqual(report['dbt_assertions'], dict.fromkeys(m.SINGULAR, 0))
            self.assertEqual(m.sha256_file(source), original)
            with duckdb.connect(str(candidate), read_only=True) as db:
                self.assertEqual(db.execute('select cycle_id from dim_election_cycle').fetchone()[0], '2022:general')
                self.assertEqual(db.execute('select election_id from dim_candidate').fetchone()[0], '2022:general:522')
                self.assertEqual(db.execute('select cycle_id, nominal_valid_votes from fact_candidate_votes').fetchone(),
                                 ('2022:general', 14))
            with duckdb.connect(str(source), read_only=True) as db:
                self.assertNotIn('cycle_id', m.columns(db, 'dim_candidate'))
                self.assertEqual(db.execute('select election_id from dim_candidate').fetchone()[0], '2022:general')



class ObservedElectionCoverageTests(unittest.TestCase):
    """TDD: never invent election events to make a failing membership test green."""

    def setUp(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB installed in project .venv')
        self.db = duckdb.connect(':memory:')
        self.db.execute("""create table dim_election (
            election_id varchar, cycle_id varchar, election_year integer,
            election_type varchar, election_code varchar,
            election_scope varchar, cycle_label varchar)""")
        self.db.execute("""insert into dim_election values
            ('2022:general:522', '2022:general', 2022, 'general', '522',
            'federal_state', '2022 general 522')""")
        for relation in m.EVENT_EVIDENCE:
            self.db.execute(f"""create table {relation} (
                election_year integer, election_type varchar,
                election_code varchar, election_scope varchar)""")

    def tearDown(self):
        if hasattr(self, 'db'):
            self.db.close()

    def observe(self, relation, *, code='523', scope='federal_state'):
        self.db.execute(f'insert into {relation} values (2022, ?, ?, ?)',
                        ['general', code, scope])

    def test_missing_real_event_requires_explicit_authorization(self):
        self.observe('fact_tally_munzona')
        with self.assertRaisesRegex(m.MigrationError, '--reconcile-observed-events'):
            m.reconcile_observed_events(self.db, authorize=False)
        self.assertEqual(self.db.execute('select count(*) from dim_election').fetchone()[0], 1)

    def test_witnessed_event_is_enriched_with_source_provenance(self):
        self.observe('fact_tally_munzona')
        self.observe('fact_party_votes')
        repairs = m.reconcile_observed_events(self.db, authorize=True)
        self.assertEqual(repairs, [{
            'election_id': '2022:general:523',
            'election_scope': 'federal_state',
            'sources': ['fact_party_votes', 'fact_tally_munzona'],
        }])
        self.assertEqual(self.db.execute('select count(*) from dim_election').fetchone()[0], 2)
        self.assertEqual(self.db.execute("""select cycle_label from dim_election
                                where election_id='2022:general:523'""").fetchone()[0],
                         '2022 general 523')

    def test_conflicting_source_scope_fails_without_insert(self):
        self.observe('fact_tally_munzona', scope='federal_state')
        self.observe('fact_party_votes', scope='municipal')
        with self.assertRaisesRegex(m.MigrationError, 'unknown/contradictory scope'):
            m.reconcile_observed_events(self.db, authorize=True)
        self.assertEqual(self.db.execute('select count(*) from dim_election').fetchone()[0], 1)

    def test_missing_scope_fails_without_insert(self):
        self.observe('fact_tally_munzona', scope=None)
        with self.assertRaisesRegex(m.MigrationError, 'unknown/contradictory scope'):
            m.reconcile_observed_events(self.db, authorize=True)
        self.assertEqual(self.db.execute('select count(*) from dim_election').fetchone()[0], 1)

    def test_wrong_scope_for_election_type_fails(self):
        self.observe('fact_tally_munzona', scope='municipal')
        with self.assertRaisesRegex(m.MigrationError, 'contradictory election_scope'):
            m.reconcile_observed_events(self.db, authorize=True)
        self.assertEqual(self.db.execute('select count(*) from dim_election').fetchone()[0], 1)

    def test_no_missing_events_is_idempotent(self):
        self.observe('fact_tally_munzona', code='522')
        self.assertEqual(m.reconcile_observed_events(self.db, authorize=False), [])
        self.assertEqual(self.db.execute('select count(*) from dim_election').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
