"""PR51: independently grained electorate and turnout analytical marts.

Python stdlib tests; SQL is executed against an in-memory DuckDB when installed.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'tse_dbt/models/semantic'
TESTS = ROOT / 'tse_dbt/tests'
REF = re.compile(r"\{\{\s*ref\('([a-z_]+)'\)\s*\}\}")
CONFIG = re.compile(r"^\s*\{\{\s*config\([\s\S]*?\)\s*\}\}", re.M)


def render(path: Path) -> str:
    source = CONFIG.sub('', path.read_text(encoding='utf-8'), count=1)
    if '{{' in source and not REF.search(source):
        raise AssertionError(f'unexpected dbt expression in {path}')
    result = REF.sub(lambda m: m.group(1), source).strip()
    if '{{' in result:
        raise AssertionError(f'unrendered dbt expression in {path}')
    return result


class AnalyticalMartStaticTests(unittest.TestCase):
    def test_model_dependencies_are_gold_only(self):
        self.assertEqual(set(REF.findall((MODELS / 'obt_electorate_municipality.sql').read_text())),
                         {'fact_electorate_municipality', 'dim_geography'})
        self.assertEqual(set(REF.findall((MODELS / 'cube_participation_municipality.sql').read_text())),
                         {'fact_turnout', 'dim_geography'})

    def test_marts_keep_separate_grains(self):
        electorate = (MODELS / 'obt_electorate_municipality.sql').read_text()
        participation = (MODELS / 'cube_participation_municipality.sql').read_text()
        self.assertNotIn("ref('fact_turnout')", electorate)
        self.assertNotIn("ref('fact_electorate_municipality')", participation)
        for key in ('election_id', 'round_number', 'office_code', 'is_transit_vote'):
            self.assertIn(key, participation)
        self.assertIn('lag(electorate)', electorate)
        self.assertIn('sum(turnout)', participation)
        self.assertIn('sum(eligible_voters)', participation)

    def test_conservation_respects_duplicate_multiplicity(self):
        # EXCEPT DISTINCT hides identical duplicate rows introduced by a join.
        for filename in ('assert_obt_electorate_conservation.sql',
                         'assert_cube_participation_conservation.sql'):
            with self.subTest(sql=filename):
                sql = (TESTS / filename).read_text(encoding='utf-8').lower()
                self.assertEqual(sql.count('except all'), 2)

    def test_conservation_contracts_are_declared(self):
        for name in ('assert_obt_electorate_conservation.sql',
                     'assert_cube_participation_conservation.sql',
                     'assert_cube_participation_rates.sql'):
            self.assertTrue((TESTS / name).is_file())


class AnalyticalMartDuckDBTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB provided by project .venv')
        self.db = duckdb.connect(':memory:')
        self.db.execute('''create table fact_electorate_municipality (
            cycle_id varchar, election_year integer, election_type varchar,
            election_scope varchar, uf varchar, municipality_code varchar,
            municipality varchar, electorate integer)''')
        self.db.execute('''create table dim_geography (
            cycle_id varchar, uf varchar, municipality_code varchar,
            municipality varchar)''')
        self.db.execute('''create table fact_turnout (
            cycle_id varchar, election_id varchar, election_year integer,
            election_type varchar, election_scope varchar, election_code varchar,
            round_number integer, uf varchar, municipality_code varchar,
            office_code varchar, office_scope varchar, is_transit_vote boolean,
            zone integer, eligible_voters integer, turnout integer,
            abstentions integer, uncounted_voters integer,
            voters_uninstalled_sections integer, generated_at timestamp)''')
        self.db.executemany('insert into fact_electorate_municipality values (?, ?, ?, ?, ?, ?, ?, ?)', [
            ('2020:municipal', 2020, 'municipal', 'municipal', 'GO', '00001', 'Original A', 100),
            ('2024:municipal', 2024, 'municipal', 'municipal', 'GO', '00001', 'Original A', 120),
            ('2024:municipal', 2024, 'municipal', 'municipal', 'GO', '00002', 'Original B', 30),
        ])
        self.db.executemany('insert into dim_geography values (?, ?, ?, ?)', [
            ('2020:municipal', 'GO', '00001', 'Conformed A'),
            ('2024:municipal', 'GO', '00001', 'Conformed A'),
        ])
        def turnout(code, round_, office, transit, zone, eligible, attended, abst, municipality='00001'):
            return ('2024:municipal', f'2024:municipal:{code}', 2024, 'municipal',
                    'municipal', code, round_, 'GO', municipality, office,
                    'municipal', transit, zone, eligible, attended, abst,
                    eligible - attended - abst, 0, '2024-10-06 20:00:00')
        self.db.executemany('insert into fact_turnout values (' + ','.join('?' for _ in range(19)) + ')', [
            turnout('619', 1, '11', False, 1, 100, 70, 30),
            turnout('619', 1, '11', False, 2, 50, 40, 10),
            turnout('619', 1, '13', False, 1, 100, 70, 30),
            turnout('620', 2, '11', False, 1, 100, 60, 40),
            turnout('619', 1, '11', True, 1, 5, 4, 1),
            turnout('619', 1, '11', False, 1, 0, 0, 0, municipality='00002'),
        ])
        self.db.execute('create view obt_electorate_municipality as ' +
                        render(MODELS / 'obt_electorate_municipality.sql'))
        self.db.execute('create view cube_participation_municipality as ' +
                        render(MODELS / 'cube_participation_municipality.sql'))

    def tearDown(self):
        if hasattr(self, 'db'):
            self.db.close()

    def assert_singular_passes(self, file: str):
        rows = self.db.execute(render(TESTS / file)).fetchall()
        self.assertEqual(rows, [], f'{file} violations: {rows[:5]}')

    def test_electorate_snapshots_and_previous_cycle(self):
        rows = self.db.execute('''select cycle_id, electorate, previous_cycle_id,
               previous_electorate, electorate_delta, electorate_growth_rate,
               municipality, geography_matched
               from obt_electorate_municipality where municipality_code='00001'
               order by election_year''').fetchall()
        self.assertEqual(rows[0], ('2020:municipal', 100, None, None, None, None,
                                   'Conformed A', True))
        self.assertEqual(rows[1][:5], ('2024:municipal', 120, '2020:municipal', 100, 20))
        self.assertAlmostEqual(rows[1][5], 0.2)
        other = self.db.execute("""select municipality, geography_matched,
            previous_cycle_id from obt_electorate_municipality
            where municipality_code='00002'""").fetchone()
        self.assertEqual(other, ('Original B', False, None))
        self.assert_singular_passes('assert_obt_electorate_conservation.sql')

    def test_participation_distinguishes_offices_rounds_and_transit(self):
        rows = self.db.execute('''select election_id, round_number, office_code,
           is_transit_vote, eligible_voters, turnout, abstentions, zone_rows,
           turnout_rate from cube_participation_municipality
           where municipality_code='00001' order by election_id, office_code, is_transit_vote''').fetchall()
        self.assertEqual(len(rows), 4)
        primary = next(r for r in rows if r[0].endswith('619') and r[2]=='11' and not r[3])
        self.assertEqual(primary[4:8], (150, 110, 40, 2))
        self.assertAlmostEqual(primary[8], 110 / 150)
        # Another office reports a repeated voter population, kept separate.
        other_office = next(r for r in rows if r[2] == '13')
        self.assertEqual(other_office[4], 100)
        second_round = next(r for r in rows if r[0].endswith('620'))
        self.assertEqual(second_round[5], 60)
        transit = next(r for r in rows if r[3])
        self.assertEqual(transit[4:7], (5, 4, 1))
        self.assert_singular_passes('assert_cube_participation_conservation.sql')
        self.assert_singular_passes('assert_cube_participation_rates.sql')

    def test_zero_denominator_is_not_zero_percent(self):
        row = self.db.execute("""select turnout_rate, abstention_rate,
                geography_matched from cube_participation_municipality
                where municipality_code='00002'""").fetchone()
        self.assertEqual(row, (None, None, False))

    def test_duplicate_geography_fails_conservation(self):
        self.db.execute("""insert into dim_geography values
                        ('2024:municipal', 'GO', '00001', 'Duplicate A')""")
        self.assertTrue(self.db.execute(
            render(TESTS / 'assert_obt_electorate_conservation.sql')).fetchall())
        self.assertTrue(self.db.execute(
            render(TESTS / 'assert_cube_participation_conservation.sql')).fetchall())

    def test_identical_duplicate_cube_row_fails_conservation(self):
        base = render(MODELS / 'cube_participation_municipality.sql')
        self.db.execute('''create or replace view cube_participation_municipality as
            select * from (''' + base + ''')
            union all
            select * from (''' + base + ''')
            where election_id = '2024:municipal:620' and office_code = '11'
              and is_transit_vote = false''')
        self.assertTrue(self.db.execute(
            render(TESTS / 'assert_cube_participation_conservation.sql')).fetchall())

    def test_identical_duplicate_obt_row_fails_conservation(self):
        base = render(MODELS / 'obt_electorate_municipality.sql')
        self.db.execute('''create or replace view obt_electorate_municipality as
            select * from (''' + base + ''')
            union all
            select * from (''' + base + ''')
            where cycle_id = '2024:municipal' and municipality_code = '00001' ''')
        self.assertTrue(self.db.execute(
            render(TESTS / 'assert_obt_electorate_conservation.sql')).fetchall())

    def test_modified_turnout_is_detected(self):
        self.db.execute("""create or replace view cube_participation_municipality as
            select * replace (turnout + 1 as turnout)
            from (""" + render(MODELS / 'cube_participation_municipality.sql') + ')')
        self.assertTrue(self.db.execute(
            render(TESTS / 'assert_cube_participation_conservation.sql')).fetchall())
        self.assertTrue(self.db.execute(render(TESTS / 'assert_cube_participation_rates.sql')).fetchall())


if __name__ == '__main__':
    unittest.main()
