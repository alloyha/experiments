"""Zone-count ablation: exact SQL arms, semantic substitution guards, fail-closed."""
from __future__ import annotations

import unittest
import json
import io
from contextlib import redirect_stdout
from unittest.mock import patch

from research import benchmark_graph_zone_count_ablation as ablation
from research import benchmark_graph_candidate_conservation as bench

try:
    import duckdb
except ImportError:
    duckdb = None


class AblationContractTests(unittest.TestCase):
    def test_sql_arm_differs_only_in_distinct_counter(self):
        baseline = ablation.build_query('with_distinct', 'AC')
        cheaper = ablation.build_query('no_distinct', 'AC')
        self.assertIn('COUNT(DISTINCT zone)', baseline)
        self.assertNotIn('COUNT(DISTINCT zone)', cheaper)
        for clause in ('COUNT(zone)', 'COUNT(*)', 'SUM(nominal_valid_votes)',
                       'GROUP BY ' + ', '.join(ablation.KEYS), "WHERE uf = 'AC'"):
            self.assertIn(clause, baseline)
            self.assertIn(clause, cheaper)
        self.assertIn('SUM(zone_count)', baseline)
        self.assertNotIn('SUM(zone_count)', cheaper)

    def test_unknown_arm_or_uf_fails(self):
        with self.assertRaises(bench.BenchmarkError):
            ablation.build_query('hash_approx', None)
        for bad in ('ac', "A'C", 'AC;DROP', 'ABC'):
            with self.subTest(bad=bad), self.assertRaises(bench.BenchmarkError):
                ablation.build_query('no_distinct', bad)

    def test_duplicate_and_null_prevent_count_star_substitution(self):
        result = ablation.unpack_row('with_distinct', (2, 5, 17, 4, 2))
        self.assertEqual(result['null_zone_rows'], 1)
        self.assertEqual(result['duplicate_nonnull_zone_rows'], 2)
        self.assertFalse(result['count_star_exact_substitute_for_distinct'])
        self.assertFalse(result['count_zone_exact_substitute_for_distinct'])

    def test_unique_nonnull_group_zone_proves_exact_count_star(self):
        result = ablation.unpack_row('with_distinct', (2, 5, 17, 5, 5))
        self.assertTrue(result['count_star_exact_substitute_for_distinct'])
        self.assertTrue(result['count_zone_exact_substitute_for_distinct'])

    def test_inconsistent_inventory_fails_closed(self):
        with self.assertRaises(bench.BenchmarkError):
            ablation.unpack_row('with_distinct', (2, 5, 17, 4, 6))
        with self.assertRaises(bench.BenchmarkError):
            ablation.unpack_row('no_distinct', (0, 0, 0, 0))

    def test_shared_invariants_mismatch_fails_closed(self):
        items = [
            {'variant': 'no_distinct', 'values': ablation.unpack_row('no_distinct', (2, 5, 17, 4)), 'elapsed_seconds': 1},
            {'variant': 'with_distinct', 'values': ablation.unpack_row('with_distinct', (2, 6, 17, 4, 4)), 'elapsed_seconds': 2},
        ]
        with self.assertRaises(bench.BenchmarkError):
            ablation.prove_shared_invariants(items)
        items[1]['values'] = ablation.unpack_row('with_distinct', (2, 5, 17, 4, 4))
        data = ablation.prove_shared_invariants(items)
        self.assertEqual(data['speedup_without_distinct'], 2)
        self.assertFalse(data['distinct_zone_semantics']['count_star_exact_substitute_for_distinct'])

    def test_cli_success_after_written_summary_returns_zero_and_json(self):
        # Regression: execute() had written summary.json, but main() then
        # failed with NameError because json was not imported.
        expected = {
            'result': 'PASS',
            'shared_comparison': {'shared_aggregates_equal': True},
        }
        stdout = io.StringIO()
        with patch.object(ablation, 'execute', return_value=expected) as execute:
            with redirect_stdout(stdout):
                result = ablation.main([
                    '--db', '/unused/gold.duckdb',
                    '--source-root', '/unused/fact_candidate_votes',
                    '--expected-main-sha256', 'a' * 64,
                    '--cycle', '2018:general',
                    '--output-dir', '/unused/zone-output',
                ])
        self.assertEqual(result, 0)
        execute.assert_called_once()
        data = json.loads(stdout.getvalue())
        self.assertEqual(data['result'], 'PASS')
        self.assertEqual(data['summary'], '/unused/zone-output/summary.json')
        self.assertEqual(data['comparison'], expected['shared_comparison'])

    @unittest.skipUnless(duckdb is not None, 'real DuckDB Python required')
    def test_real_duckdb_nulls_duplicates_and_unique_groups(self):
        con = duckdb.connect(':memory:')
        try:
            con.execute('''CREATE TABLE fact_candidate_votes AS SELECT * FROM (
               VALUES
                 ('2020:municipal', 7, 1, 'AC', 10, '01', false, 11, 1, 5),
                 ('2020:municipal', 7, 1, 'AC', 10, '01', false, 11, 1, 7),
                 ('2020:municipal', 7, 1, 'AC', 10, '01', false, 11, NULL, 3),
                 ('2020:municipal', 7, 1, 'AC', 10, '01', false, 12, 2, 9),
                 ('2020:municipal', 7, 1, 'GO', 20, '01', false, 13, 4, 2)
             ) v(cycle_id,election_id,round_number,uf,municipality_code,office_code,
                  is_transit_vote,candidate_id,zone,nominal_valid_votes)''')
            def execute(arm, uf):
                sql = ablation.build_query(arm, uf)
                return ablation.unpack_row(arm, con.execute(sql).fetchone())
            a = execute('with_distinct', 'AC')
            b = execute('no_distinct', 'AC')
            self.assertEqual((a['groups'], a['source_rows'], a['votes']), (2, 4, 24))
            self.assertEqual(a['distinct_zone_total'], 2)
            self.assertEqual(a['null_zone_rows'], 1)
            self.assertEqual(a['duplicate_nonnull_zone_rows'], 1)
            self.assertEqual(a['source_rows'], b['source_rows'])
            self.assertFalse(a['count_star_exact_substitute_for_distinct'])
            # A region containing one unambiguous zone is safe.
            c = execute('with_distinct', 'GO')
            self.assertTrue(c['count_star_exact_substitute_for_distinct'])
        finally:
            con.close()


if __name__ == '__main__':
    unittest.main()
