"""PR1 regression: Kimball cycle/event keys and executable dbt test semantics."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConformedElectionGrainTests(unittest.TestCase):
    def test_candidate_event_id_includes_election_code(self):
        sql = (ROOT / 'tse_dbt/models/gold/dimensions/dim_candidate.sql').read_text()
        self.assertIn("c.election_type || ':' || c.election_code as election_id", sql)
        self.assertIn("c.election_type as cycle_id", sql)

    def test_cycle_dimension_is_unique_grain(self):
        sql = (ROOT / 'tse_dbt/models/gold/dimensions/dim_election_cycle.sql').read_text()
        self.assertIn('select distinct', sql.lower())
        self.assertIn("ref('dim_election')", sql)
        self.assertNotIn('election_code as', sql)

    def test_cycle_key_present_in_gold_models(self):
        for name in ('dimensions/dim_election.sql', 'dimensions/dim_candidate.sql',
                     'dimensions/dim_geography.sql', 'dimensions/dim_party.sql',
                     'facts/fact_electorate_municipality.sql', 'facts/fact_turnout.sql',
                     'facts/fact_tally_munzona.sql', 'facts/fact_party_votes.sql',
                     'facts/fact_candidate_votes.sql'):
            with self.subTest(model=name):
                self.assertIn('cycle_id', (ROOT / 'tse_dbt/models/gold' / name).read_text())

    def test_membership_assertion_has_only_conformed_refs(self):
        sql = (ROOT / 'tse_dbt/tests/assert_conformed_dimension_membership.sql').read_text()
        refs = set(re.findall(r"ref\('([^']+)'\)", sql))
        self.assertEqual(refs, {
            'dim_election_cycle', 'dim_election', 'dim_candidate', 'dim_party',
            'fact_tally_munzona', 'fact_turnout', 'fact_party_votes',
            'fact_candidate_votes', 'fact_electorate_municipality', 'dim_geography',
        })
        self.assertIn('select distinct election_id, cycle_id', sql)

    def test_event_and_cycle_are_not_equivalent(self):
        def cycle(y, typ):
            return f'{y}:{typ}'
        def event(y, typ, code):
            return f'{cycle(y, typ)}:{code}'
        self.assertEqual(event(2018, 'general', 297), '2018:general:297')
        self.assertNotEqual(event(2018, 'general', 297), event(2018, 'general', 339))
        self.assertEqual(cycle(2018, 'general'), cycle(2018, 'general'))
        self.assertNotEqual(cycle(2018, 'general'), event(2018, 'general', 297))

    def test_dbt_identity_assertion_with_synthetic_warehouse(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB is provided by the project .venv')
        sql = (ROOT / 'tse_dbt/tests/assert_conformed_election_identity.sql').read_text()
        names = re.findall(r"\{\{\s*ref\('([^']+)'\)\s*\}\}", sql)
        events = {'dim_election', 'dim_candidate', 'dim_party', 'fact_tally_munzona',
                  'fact_turnout', 'fact_party_votes', 'fact_candidate_votes'}
        cycles = {'dim_election_cycle', 'dim_geography', 'fact_electorate_municipality'}
        self.assertEqual(set(names), events | cycles)
        for name in set(names):
            sql = sql.replace("{{ ref('" + name + "') }}", name)
        conn = duckdb.connect(':memory:')
        for name in events:
            conn.execute(f"""create table {name} as select 2018 as election_year,
                'general' as election_type, '297'::varchar as election_code,
                '2018:general'::varchar as cycle_id,
                '2018:general:297'::varchar as election_id""")
        for name in cycles:
            conn.execute(f"""create table {name} as select 2018 as election_year,
                'general' as election_type, '2018:general'::varchar as cycle_id""")
        self.assertEqual(conn.execute(sql).fetchall(), [])
        conn.execute("update dim_candidate set election_id = '2018:general'")
        self.assertEqual(conn.execute(sql).fetchone()[0], 'dim_candidate')
        conn.execute("update dim_candidate set election_id = '2018:general:297'")
        conn.execute("update fact_electorate_municipality set cycle_id = '2018:general:297'")
        self.assertEqual(conn.execute(sql).fetchone()[0], 'fact_electorate_municipality')


if __name__ == '__main__':
    unittest.main()
