"""PR52: candidate/party cubes conserve multiset votes and contest identity."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'tse_dbt/models/semantic'
SQL_TESTS = ROOT / 'tse_dbt/tests'
CANDIDATE = 'cube_candidate_performance_municipality'
PARTY = 'cube_party_performance_municipality'
CONSERVATION = {
    CANDIDATE: 'assert_cube_candidate_vote_conservation.sql',
    PARTY: 'assert_cube_party_vote_conservation.sql',
}
RATES = {
    CANDIDATE: 'assert_cube_candidate_share_and_rank.sql',
    PARTY: 'assert_cube_party_share_and_rank.sql',
}
REF = re.compile(r"\{\{\s*ref\('([a-z_]+)'\)\s*\}\}")
# A config() may contain a nested quoted Jinja expression, e.g.
# pre_hook="{{ partition_replace_pre_hook() }}". A non-greedy generic regex
# stops at its inner `) }}` and leaves invalid Jinja in the SQL fixture.
# Match the single-line form, or the multiline header through its *own*
# standalone closing `) }}` line; this is a lightweight test renderer only.
CONFIG = re.compile(
    r"\A\s*\{\{\s*config\((?:[^\n]*\)\s*\}\}|[\s\S]*?^\)\s*\}\})",
    re.MULTILINE,
)
PARTITION_FILTER = re.compile(
    r"\{\{\s*incremental_partition_predicate\(\s*'election_year'\s*,\s*'election_type'\s*\)\s*\}\}"
)
# Fixture tables have election_year but no virtual Hive `year` field.
# Resolve the physical partition predicate against their matching logical
# year so synthetic tests cover the same selected-cycle behavior.
HIVE_PARTITION_FILTER = re.compile(
    r"\{\{\s*incremental_partition_predicate\(\s*'year'\s*,\s*'election_type'\s*\)\s*\}\}"
)


def render(path: Path, *, year=2024, election_type='municipal') -> str:
    # Lightweight rendering of the supported fixture-only Jinja macros.
    # A real dbt compile is still required in the worktree.
    text = CONFIG.sub('', path.read_text(encoding='utf8'), count=1)
    text = REF.sub(lambda m: m.group(1), text)
    text = PARTITION_FILTER.sub(
        f"election_year in ({int(year)}) and election_type in ('{election_type}')",
        text,
    )
    text = HIVE_PARTITION_FILTER.sub(
        f"election_year in ({int(year)}) and election_type in ('{election_type}')",
        text,
    )
    return text.strip()


class PerformanceStaticContracts(unittest.TestCase):
    def test_fixture_renderer_strips_config_and_scopes_without_dbt(self):
        # Regression for the quoted pre_hook Jinja inside multiline config().
        candidate = render(MODELS / f'{CANDIDATE}.sql', year=2026,
                           election_type='general')
        party = render(MODELS / f'{PARTY}.sql')
        for sql in (candidate, party):
            self.assertNotIn('{{', sql)
            self.assertNotIn('}}', sql)
            self.assertNotIn('pre_hook=', sql)
            self.assertIn('with ', sql.lower())
        self.assertIn("election_year in (2026) and election_type in ('general')",
                      candidate)
        self.assertNotIn("election_year in (2024) and election_type in ('municipal')",
                         candidate)

    def test_candidate_is_partition_replace_incremental_not_full_table(self):
        candidate_sql = (MODELS / f'{CANDIDATE}.sql').read_text()
        party_sql = (MODELS / f'{PARTY}.sql').read_text()
        self.assertIn("materialized='incremental'", candidate_sql)
        self.assertIn("partition_replace_pre_hook()", candidate_sql)
        self.assertIn("incremental_partition_predicate('election_year', 'election_type')", candidate_sql)
        self.assertIn("incremental_partition_predicate('year', 'election_type')", candidate_sql)
        source_fact = (ROOT / 'tse_dbt/models/gold/facts/fact_candidate_votes.sql')
        if source_fact.exists():
            fact_sql = source_fact.read_text()
            self.assertIn('hive_partitioning = true', fact_sql)
            self.assertIn('year=*/data.parquet', fact_sql)
        self.assertIn("'election_id', 'round_number'", candidate_sql)
        self.assertIn("config(materialized='view')", party_sql)
        for filename in (CONSERVATION[CANDIDATE], RATES[CANDIDATE]):
            self.assertIn("incremental_partition_predicate('election_year', 'election_type')",
                          (SQL_TESTS / filename).read_text())
        # Two sides of the conservation test MUST use the same filter.
        self.assertEqual((SQL_TESTS / CONSERVATION[CANDIDATE]).read_text().count(
            "incremental_partition_predicate('election_year', 'election_type')"), 2)
        self.assertEqual((SQL_TESTS / CONSERVATION[CANDIDATE]).read_text().count(
            "incremental_partition_predicate('year', 'election_type')"), 1)

    def test_only_conformed_gold_inputs(self):
        for model, expected in (
            (CANDIDATE, {'fact_candidate_votes', 'dim_candidate', 'dim_geography'}),
            (PARTY, {'fact_party_votes', 'dim_party', 'dim_geography'}),
        ):
            with self.subTest(model=model):
                refs = set(REF.findall((MODELS / f'{model}.sql').read_text()))
                self.assertEqual(refs, expected)
                self.assertNotIn('fact_electorate_municipality', refs)
                self.assertNotIn('fact_turnout', refs)

    def test_multiset_conservation_is_not_except_distinct(self):
        for filename in CONSERVATION.values():
            with self.subTest(test=filename):
                sql = (SQL_TESTS / filename).read_text().lower()
                self.assertEqual(len(re.findall(r'(?m)^\s*except\s+all\b', sql)), 2)

    def test_candidate_and_party_remain_separate_domains(self):
        candidate = (MODELS / f'{CANDIDATE}.sql').read_text()
        party = (MODELS / f'{PARTY}.sql').read_text()
        self.assertNotIn("ref('fact_party_votes')", candidate)
        self.assertNotIn("ref('fact_candidate_votes')", party)
        self.assertIn('total_legend_valid_votes', party)
        self.assertNotIn('elected_status', candidate)
        self.assertIn('municipal_candidate_rank', candidate)
        self.assertIn('municipal_party_rank', party)


class PerformanceDuckDBTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB is provided by the project virtual environment')
        self.db = duckdb.connect(':memory:')
        self.db.execute('''create table fact_candidate_votes (
            cycle_id varchar, election_id varchar, election_year integer,
            election_type varchar, election_scope varchar, election_code varchar,
            round_number integer, uf varchar, municipality_code varchar,
            zone integer, office_code varchar, is_transit_vote boolean,
            candidate_id varchar, nominal_valid_votes bigint)''')
        self.db.execute('''create table fact_party_votes (
            cycle_id varchar, election_id varchar, election_year integer,
            election_type varchar, election_scope varchar, election_code varchar,
            round_number integer, uf varchar, municipality_code varchar,
            zone integer, office_code varchar, is_transit_vote boolean,
            party_number varchar, nominal_valid_votes bigint,
            legend_valid_votes bigint, total_legend_valid_votes bigint,
            party_valid_votes bigint)''')
        self.db.execute('''create table dim_candidate (
            election_id varchar, candidate_id varchar, candidate_number varchar,
            candidate_name varchar, ballot_name varchar, party_number varchar,
            party varchar, party_name varchar, office varchar,
            office_scope varchar)''')
        self.db.execute('''create table dim_party (
            election_id varchar, party_number varchar, party varchar,
            party_name varchar, party_id varchar)''')
        self.db.execute('''create table dim_geography (
            cycle_id varchar, uf varchar, municipality_code varchar,
            municipality varchar)''')
        self.db.execute("insert into dim_geography values ('2024:municipal','GO','5208707','Goiânia')")
        self.db.execute('''insert into dim_candidate values
            ('2024:municipal:620','C1','45','Candidate 1','C1','45','P1','Party One','Prefeito','municipal'),
            ('2024:municipal:620','C2','18','Candidate 2','C2','18','P2','Party Two','Prefeito','municipal')''')
        self.db.execute('''insert into dim_party values
            ('2024:municipal:620','45','P1','Party One','P1ID'),
            ('2024:municipal:620','18','P2','Party Two','P2ID')''')
        for candidate, zone, transit, votes in [
            ('C1', 1, False, 40), ('C1', 2, False, 60),
            ('C2', 1, False, 25), ('C1', 1, True, 3),
        ]:
            self.db.execute('''insert into fact_candidate_votes values
                ('2024:municipal','2024:municipal:620',2024,'municipal',
                 'municipal','620',2,'GO','5208707',?, '11',?,?,?)''',
                [zone, transit, candidate, votes])
        for party, zone, nominal, legend, total_legend in [
            ('45', 1, 40, 2, 3), ('45', 2, 60, 5, 6),
            ('18', 1, 20, 0, 0),
        ]:
            self.db.execute('''insert into fact_party_votes values
                ('2024:municipal','2024:municipal:620',2024,'municipal',
                 'municipal','620',2,'GO','5208707',?, '11',false,?,?,?,?,?)''',
                [zone, party, nominal, legend, total_legend, nominal + total_legend])
        for model in (CANDIDATE, PARTY):
            self.db.execute(f'create view {model} as ' + render(MODELS / f'{model}.sql'))

    def tearDown(self):
        if hasattr(self, 'db'):
            self.db.close()

    def violations(self, filename):
        return self.db.execute(render(SQL_TESTS / filename)).fetchall()

    def assert_clean(self):
        for sql in (*CONSERVATION.values(), *RATES.values()):
            with self.subTest(contract=sql):
                self.assertEqual(self.violations(sql), [])

    def test_exact_rollup_and_eligible_scope(self):
        self.assert_clean()
        rows = self.db.execute(f'''select candidate_id, nominal_valid_votes,
            observed_candidate_nominal_valid_votes, municipal_candidate_rank,
            candidate_share_of_observed_nominal_votes
            from {CANDIDATE}
            where is_transit_vote = false
            order by municipal_candidate_rank''').fetchall()
        self.assertEqual([(r[0], r[1], r[2], r[3]) for r in rows],
                         [('C1', 100, 125, 1), ('C2', 25, 125, 2)])
        self.assertAlmostEqual(rows[0][4], .8)
        self.assertEqual(self.db.execute(f'''select party_number, nominal_valid_votes,
            total_legend_valid_votes, party_valid_votes
            from {PARTY} order by municipal_party_rank''').fetchall(),
            [('45', 100, 9, 109), ('18', 20, 0, 20)])
        self.assertEqual(self.db.execute(f'''select nominal_valid_votes
            from {CANDIDATE} where is_transit_vote=true''').fetchone()[0], 3)

    def test_missing_metadata_preserves_votes(self):
        self.db.execute("delete from dim_candidate where candidate_id='C2'")
        self.db.execute("delete from dim_party where party_number='18'")
        self.db.execute("delete from dim_geography")
        self.assert_clean()
        self.assertEqual(self.db.execute(f'''select candidate_matched,
            geography_matched from {CANDIDATE}
            where candidate_id='C2' ''').fetchone(), (False, False))
        self.assertEqual(self.db.execute(f'''select party_matched,
            geography_matched from {PARTY}
            where party_number='18' ''').fetchone(), (False, False))

    def test_duplicate_geography_is_detected_for_both_facts(self):
        self.db.execute("insert into dim_geography values ('2024:municipal','GO','5208707','Goiânia')")
        for sql in CONSERVATION.values():
            with self.subTest(contract=sql):
                self.assertTrue(self.violations(sql))

    def test_duplicate_descriptive_members_are_detected(self):
        self.db.execute('insert into dim_candidate select * from dim_candidate where candidate_id=\'C1\'')
        self.assertTrue(self.violations(CONSERVATION[CANDIDATE]))
        self.db.execute('insert into dim_party select * from dim_party where party_number=\'45\'')
        self.assertTrue(self.violations(CONSERVATION[PARTY]))

    def test_identical_duplicate_output_rows_are_detected(self):
        for model in (CANDIDATE, PARTY):
            with self.subTest(model=model):
                self.db.execute('create table duplicated_output as select * from ' + model)
                self.db.execute('insert into duplicated_output select * from ' + model + ' limit 1')
                self.db.execute(f'drop view {model}')
                self.db.execute(f'alter table duplicated_output rename to {model}')
                self.assertTrue(self.violations(CONSERVATION[model]))
                self.db.execute(f'drop table {model}')
                self.db.execute(f'create view {model} as ' + render(MODELS / f'{model}.sql'))

    def test_mutated_amount_is_detected(self):
        self.db.execute(f'create table changed_output as select * from {PARTY}')
        self.db.execute("update changed_output set party_valid_votes=party_valid_votes+1 where party_number='45'")
        self.db.execute(f'drop view {PARTY}')
        self.db.execute(f'alter table changed_output rename to {PARTY}')
        self.assertTrue(self.violations(CONSERVATION[PARTY]))
        self.assertTrue(self.violations(RATES[PARTY]))

    def test_scoped_cycle_ignores_unselected_source_rows(self):
        # A tiny regression for the first-ever run, when is_incremental()
        # would be false: the model must nevertheless honor the selected cycle.
        self.db.execute("""insert into fact_candidate_votes values
            ('2026:general','2026:general:700',2026,'general',
             'federal_state','700',1,'GO','5208707',1,'1',false,'C2026',123)""")
        self.db.execute(f'drop view {CANDIDATE}')
        self.db.execute(f'create view {CANDIDATE} as ' + render(
            MODELS / f'{CANDIDATE}.sql', year=2026, election_type='general'))
        ids = self.db.execute(f'select candidate_id from {CANDIDATE}').fetchall()
        self.assertEqual(ids, [('C2026',)])
        self.assertEqual(self.db.execute(render(
            SQL_TESTS / CONSERVATION[CANDIDATE],
            year=2026, election_type='general')).fetchall(), [])

    def test_zero_observed_total_has_null_share(self):
        self.db.execute('''insert into fact_candidate_votes values
            ('2024:municipal','2024:municipal:620',2024,'municipal',
             'municipal','620',2,'GO','5208707',1,'13',false,'CZ',0)''')
        self.db.execute('''insert into fact_party_votes values
            ('2024:municipal','2024:municipal:620',2024,'municipal',
             'municipal','620',2,'GO','5208707',1,'13',false,'90',0,0,0,0)''')
        self.assert_clean()
        self.assertIsNone(self.db.execute(f'''select candidate_share_of_observed_nominal_votes
            from {CANDIDATE} where office_code='13' ''').fetchone()[0])
        self.assertIsNone(self.db.execute(f'''select party_share_of_observed_valid_votes
            from {PARTY} where office_code='13' ''').fetchone()[0])


if __name__ == '__main__':
    unittest.main()
