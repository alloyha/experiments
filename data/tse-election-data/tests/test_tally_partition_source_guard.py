"""Regression tests for empty or missing selected tally upstream cycles."""
import unittest
from pathlib import Path
import duckdb


class TallyGuardTests(unittest.TestCase):
    def setUp(self):
        self.db = duckdb.connect(':memory:')
        self.db.execute('create table upstream(year integer, kind varchar)')
        self.db.execute('create table target(year integer, kind varchar, marker integer)')

    def tearDown(self):
        self.db.close()

    def guard_and_replace(self, years):
        predicate = f"year in ({','.join(map(str, years))}) and kind in ('general','municipal')"
        target_predicate = predicate.replace('year', 't.year').replace('kind', 't.kind')
        source_predicate = predicate.replace('year', 's.year').replace('kind', 's.kind')
        check = f"""
        select case when
            not exists(select 1 from upstream where {predicate})
            or exists(
                select 1 from (select distinct t.year, t.kind from target t
                               where {target_predicate}) existing
                where not exists(select 1 from upstream s where {source_predicate}
                   and s.year=existing.year and s.kind=existing.kind)
            )
        then error('C3 unsafe tally partition replacement') else 1 end
        """
        self.db.execute(check).fetchone()
        self.db.execute(f'delete from target where {predicate}')
        self.db.execute(f'insert into target select year,kind,42 from upstream where {predicate}')

    def test_empty_upstream_preserves_target(self):
        self.db.execute("insert into target values (2018,'general',7)")
        with self.assertRaisesRegex(Exception, 'unsafe tally'):
            self.guard_and_replace([2018])
        self.assertEqual(self.db.execute('select marker from target').fetchall(), [(7,)])

    def test_one_missing_selected_cycle_preserves_all(self):
        self.db.execute("insert into target values (2018,'general',7),(2020,'municipal',8)")
        self.db.execute("insert into upstream values (2018,'general')")
        with self.assertRaisesRegex(Exception, 'unsafe tally'):
            self.guard_and_replace([2018,2020])
        self.assertEqual(self.db.execute('select count(*) from target').fetchone()[0],2)

    def test_complete_selected_cycles_allow_replace(self):
        self.db.execute("insert into target values (2018,'general',7),(2020,'municipal',8)")
        self.db.execute("insert into upstream values (2018,'general'),(2020,'municipal')")
        self.guard_and_replace([2018,2020])
        self.assertEqual(self.db.execute('select sum(marker) from target').fetchone()[0],84)

    def test_empty_new_target_rejected_when_upstream_empty(self):
        with self.assertRaisesRegex(Exception, 'unsafe tally'):
            self.guard_and_replace([2022])

    def test_macro_wired_before_delete(self):
        root=Path(__file__).resolve().parents[0]
        # Run with the patched project root as argument? Default uses cwd.
        project=Path.cwd()
        if (project/'data/tse-election-data/tse_dbt').exists():
            project=project/'data/tse-election-data'
        for rel in ('models/silver/tally/silver_tally_munzona.sql',
                    'models/gold/facts/fact_tally_munzona.sql'):
            text=(project/'tse_dbt'/rel).read_text()
            self.assertLess(text.index('assert_tally_upstream_partitions'),
                            text.index('partition_replace_pre_hook'))

if __name__ == '__main__':
    unittest.main()
