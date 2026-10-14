"""PR50: event-grain dimension is derived only from published, conformed Gold.

The fixture exercises the real DuckDB SQL when the project .venv has DuckDB.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'tse_dbt/models/gold/dimensions/dim_election.sql'
SCOPE_TEST = ROOT / 'tse_dbt/tests/assert_dim_election_gold_source_coverage.sql'
PARENTS = ('dim_candidate', 'dim_party', 'fact_tally_munzona', 'fact_party_votes')
MISSING_SCOPE = object()
REF_RE = re.compile(r"\{\{\s*ref\('([a-z_]+)'\)\s*\}\}")
CODES_26 = {
    (2018, 'general', '431'),
    *((2020, 'municipal', x) for x in ('455','456','457','458','459','481','507',
                                     '520','593','596','608','622','625')),
    *((2024, 'municipal', x) for x in ('6223','6231','6232','6239','6267',
                                     '6282','6283','6284','6285','6286','6290','6293')),
}


def render(path: Path) -> str:
    # This model has one config statement followed by ordinary ref() calls.
    text = path.read_text(encoding='utf-8')
    text = re.sub(r"\{\{\s*config\([\s\S]*?\)\s*\}\}", '', text, count=1)
    return REF_RE.sub(lambda m: m.group(1), text).strip()


class ElectionEventCoverageContractTests(unittest.TestCase):
    def test_only_persisted_gold_sources(self):
        sql = MODEL.read_text(encoding='utf-8')
        self.assertEqual(set(REF_RE.findall(sql)), set(PARENTS))
        for forbidden in ('bronze_candidates', 'bronze_party_votes_raw',
                          'silver_tally_munzona', 'fact_candidate_votes'):
            self.assertNotIn(f"ref('{forbidden}')", sql)
        self.assertNotIn('max(election_scope)', sql.lower())
        self.assertIn('error(', sql)

    def test_upstream_gold_has_no_reverse_dependency(self):
        # These are the current direct parent definitions; the DAG tool also
        # checks transitive dependencies throughout the repository.
        locations = {
            'dim_candidate': 'dimensions/dim_candidate.sql',
            'dim_party': 'dimensions/dim_party.sql',
            'fact_tally_munzona': 'facts/fact_tally_munzona.sql',
            'fact_party_votes': 'facts/fact_party_votes.sql',
        }
        for name, rel in locations.items():
            with self.subTest(parent=name):
                path = ROOT / 'tse_dbt/models/gold' / rel
                if not path.exists():
                    self.skipTest('Run against the full repository checkout')
                text = path.read_text()
                self.assertNotIn("ref('dim_election')", text)
                self.assertNotIn("ref('dim_election_cycle')", text)

    def test_source_contract_uses_identical_gold_refs(self):
        self.assertEqual(set(REF_RE.findall(SCOPE_TEST.read_text())),
                         set(PARENTS) | {'dim_election'})

    def test_regression_codes_are_distinct(self):
        # Specifically the 26 witnessed orphan event keys from the real
        # October 2026 Gold copy, never hardcoded in the production model.
        self.assertEqual(len(CODES_26), 26)


class ElectionEventGoldSqlTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB is available in the project .venv')
        self.conn = duckdb.connect(':memory:')
        for p in PARENTS:
            self.conn.execute(f'''create table {p} (
                election_year integer, election_type varchar,
                election_code varchar, election_scope varchar)''')

    def tearDown(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def observe(self, model, year, typ, code, scope=MISSING_SCOPE):
        scope = scope if scope is not MISSING_SCOPE else (
            'general' == typ and 'federal_state' or 'municipal'
        )
        self.conn.execute(f'insert into {model} values (?, ?, ?, ?)',
                          (year, typ, code, scope))

    def rebuild(self):
        self.conn.execute('create or replace table dim_election as ' + render(MODEL))

    def test_historical_candidate_only_26_and_other_sources(self):
        for year, typ, code in CODES_26:
            self.observe('dim_candidate', year, typ, code)
        self.observe('dim_party', 2022, 'general', '545')
        self.observe('fact_tally_munzona', 2018, 'general', '339')
        self.observe('fact_party_votes', 2024, 'municipal', '619')
        self.rebuild()
        expected = set(CODES_26) | {
            (2022, 'general', '545'), (2018, 'general', '339'),
            (2024, 'municipal', '619')
        }
        self.assertEqual(set(self.conn.execute('select election_year, election_type, election_code from dim_election').fetchall()), expected)
        self.assertEqual(self.conn.execute(render(SCOPE_TEST)).fetchall(), [])
        self.assertEqual(self.conn.execute("select evidence_sources from dim_election where election_id='2020:municipal:481'").fetchone()[0], 'dim_candidate')
        self.rebuild()  # repeatable, no accumulating duplicate IDs
        self.assertEqual(self.conn.execute('select count(*) from dim_election').fetchone()[0], len(expected))

    def test_incremental_refresh_of_one_cycle_keeps_other_cycles(self):
        self.observe('dim_candidate', 2020, 'municipal', '481')
        self.observe('dim_candidate', 2026, 'general', '700')
        self.rebuild()
        self.conn.execute("delete from dim_candidate where election_year=2026")
        self.observe('dim_candidate', 2026, 'general', '701')
        self.rebuild()
        found = set(self.conn.execute('select election_id from dim_election').fetchall())
        self.assertEqual(found, {('2020:municipal:481',), ('2026:general:701',)})

    def test_partial_parent_replacement_and_full_refresh_are_fresh(self):
        self.observe('dim_candidate', 2024, 'municipal', '619')
        self.observe('dim_party', 2024, 'municipal', '619')
        self.rebuild()
        self.conn.execute("delete from dim_candidate where election_code='619'")
        self.rebuild()
        self.assertEqual(self.conn.execute('select count(*) from dim_election').fetchone()[0], 1)
        self.conn.execute("delete from dim_party where election_code='619'")
        self.rebuild()
        self.assertEqual(self.conn.execute('select count(*) from dim_election').fetchone()[0], 0)

    def test_scope_conflict_is_a_hard_error(self):
        self.observe('dim_candidate', 2024, 'municipal', '619')
        self.observe('fact_party_votes', 2024, 'municipal', '619', 'federal_state')
        with self.assertRaisesRegex(Exception, 'conflicting event scopes'):
            self.rebuild()

    def test_null_scope_is_a_hard_error(self):
        self.observe('dim_candidate', 2024, 'municipal', '619', scope=None)
        with self.assertRaisesRegex(Exception, 'unknown scope'):
            self.rebuild()

    def test_null_key_and_incompatible_scope_fail(self):
        self.observe('dim_candidate', 2024, 'municipal', None)
        with self.assertRaisesRegex(Exception, 'null/empty key'):
            self.rebuild()
        self.conn.execute('delete from dim_candidate')
        self.observe('dim_candidate', 2024, 'municipal', '619', 'federal_state')
        with self.assertRaisesRegex(Exception, 'scope contradicts'):
            self.rebuild()


if __name__ == '__main__':
    unittest.main()
