"""Exact multiset equivalence and fail-closed benchmark contract tests."""
import collections
import json
from threading import Event
from pathlib import Path
import tempfile
import time
import unittest

from scripts import benchmark_graph_candidate_conservation as bench

try:
    import duckdb
except ImportError:
    duckdb = None


SAMPLE = '''-- Original checked-in SQL CTE fixture
WITH expected AS (SELECT {cols} FROM src_expected),
actual AS (SELECT {cols} FROM src_actual)
{tail}'''.format(cols=', '.join(bench.COLUMNS), tail=bench.TAIL)


class SqlRewriteTests(unittest.TestCase):
    def test_unchanged_contract_tail_passes(self):
        self.assertIn('UNION ALL', bench.original_query(SAMPLE).upper())
        signed = bench.signed_query(SAMPLE)
        self.assertIn('SUM(ABS(multiplicity_delta))', signed)
        self.assertIn('GROUP BY ' + ', '.join(bench.COLUMNS), signed)
        self.assertIn('FROM expected', signed)
        self.assertIn('FROM actual', signed)

    def test_unknown_contract_rejected_without_guessing(self):
        for sql in (
            SAMPLE.replace('except all', 'except'),
            SAMPLE.replace('missing_or_changed', 'different'),
            SAMPLE.replace('from actual)', 'from another_table)'),
            SAMPLE.replace("(select 'missing_or_changed'", "(select 'xyz'"),
        ):
            with self.subTest(sql=sql[-100:]):
                with self.assertRaises(bench.BenchmarkError):
                    bench.signed_query(sql)

    def test_only_two_variants_are_accepted(self):
        with self.assertRaises(bench.BenchmarkError):
            bench.query_for(SAMPLE, 'approx_hash')

    def test_checks_missing_ctes(self):
        with self.assertRaises(bench.BenchmarkError):
            bench.checked_ctes('SELECT 1\n' + bench.TAIL)

    def test_write_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            bench.durable_json(path, {'status': 'ok'})
            with self.assertRaises(FileExistsError):
                bench.durable_json(path, {'status': 'overwritten'})

    def test_speedup_pairs_separate_cycles(self):
        rows = [
            {'cycle': '2018:general', 'repeat': 1, 'variant': 'except_all', 'result': 'PASS', 'elapsed_seconds': 100},
            {'cycle': '2018:general', 'repeat': 1, 'variant': 'signed', 'result': 'PASS', 'elapsed_seconds': 25},
            {'cycle': '2020:municipal', 'repeat': 1, 'variant': 'except_all', 'result': 'PASS', 'elapsed_seconds': 10},
            {'cycle': '2020:municipal', 'repeat': 1, 'variant': 'signed', 'result': 'PASS', 'elapsed_seconds': 5},
        ]
        compared = bench.speedups(rows)
        self.assertEqual(compared['2018:general']['pairs'][0]['speedup_except_over_signed'], 4)
        self.assertEqual(compared['2020:municipal']['pairs'][0]['speedup_except_over_signed'], 2)

    def test_paired_variants_alternate_order(self):
        self.assertEqual(bench.ordered_variants(1, ('except_all', 'signed')),
                         ('except_all', 'signed'))
        self.assertEqual(bench.ordered_variants(2, ('except_all', 'signed')),
                         ('signed', 'except_all'))

    def test_single_arm_does_not_execute_unrequested_expensive_variant(self):
        for repeat in (1, 2, 3):
            self.assertEqual(bench.ordered_variants(repeat, ('signed',)), ('signed',))
            self.assertEqual(bench.ordered_variants(repeat, ('except_all',)), ('except_all',))

    def test_invalid_or_duplicate_arm_fails_closed(self):
        for variants in ((), ('signed', 'signed'), ('except_all', 'except_all'),
                         ('hash_only',), ('signed', 'invalid')):
            with self.subTest(variants=variants):
                with self.assertRaises(bench.BenchmarkError):
                    bench.ordered_variants(1, variants)

    def test_unpaired_result_never_reports_speedup(self):
        rows = [{'cycle': '2018:general', 'repeat': 1, 'variant': 'signed',
                 'result': 'PASS', 'elapsed_seconds': 11.0}]
        comparison = bench.speedups(rows)['2018:general']
        self.assertEqual(comparison['pairs'], [])
        self.assertIsNone(comparison['median_except_all_s'])
        self.assertIsNone(comparison['median_signed_s'])

    def test_wal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'main.duckdb'
            db.write_bytes(b'synthetic')
            bench.assert_wal_free(db)
            Path(str(db) + '.wal').write_bytes(b'uncommitted')
            with self.assertRaises(bench.BenchmarkError):
                bench.assert_wal_free(db)


class GuardedCountStarTests(unittest.TestCase):
    BASE = '''WITH expected_rollup AS (
        SELECT election_id, candidate_id, count(distinct zone) as zones,
               sum(nominal_valid_votes) as votes FROM physical_fact
        GROUP BY election_id, candidate_id
    ), expected AS (SELECT source_id, target_id, cycle_id, election_id,
        round_number, office_code, is_transit_vote, nominal_valid_votes,
        zone_rows, zone_count FROM baseline),
    actual AS (SELECT source_id, target_id, cycle_id, election_id,
        round_number, office_code, is_transit_vote, nominal_valid_votes,
        zone_rows, zone_count FROM graph_edges)
    ''' + bench.TAIL

    MODEL = '''with candidate_vote_rollup as (
        select election_id, candidate_id, count(distinct zone) as zone_count
        from fact_candidate_votes group by election_id, candidate_id
    ), party_vote_rollup as (
        select election_id, party_number, count(distinct zone) as zone_count
        from fact_party_votes group by election_id, party_number
    ) select * from candidate_vote_rollup'''

    def test_rewrite_modifies_only_one_expected_count(self):
        result = bench.rewrite_expected_zone_count(self.BASE)
        self.assertIn('count(*) as zones', result)
        self.assertNotIn('count(distinct zone) as zones', result)
        self.assertIn('sum(nominal_valid_votes)', result)
        self.assertEqual(bench.query_for(self.BASE, 'signed_countstar'),
                         bench.signed_query(result))

    def test_rewrite_graph_preserves_party_and_independent_sources(self):
        result = bench.rewrite_graph_zone_count(self.MODEL)
        self.assertIn('candidate_vote_rollup as (', result)
        self.assertIn('count(*) as zone_count', result)
        self.assertEqual(result.count('count(distinct zone) as zone_count'), 1)
        self.assertIn('from fact_candidate_votes', result)
        self.assertIn('from fact_party_votes', result)

    def test_guard_rewrite_rejects_ambiguous_contract_change(self):
        bad = [
            self.BASE.replace('count(distinct zone) as zones', 'count(*) as zones'),
            self.BASE.replace('count(distinct zone) as zones', 'sum(zone) as zones'),
            self.BASE.replace('count(distinct zone) as zones',
                              'count(distinct zone) as zones, count(distinct zone) as zones2'),
        ]
        for sql in bad:
            with self.subTest(sql=sql[:90]):
                with self.assertRaises(bench.BenchmarkError):
                    bench.rewrite_expected_zone_count(sql)
        with self.assertRaises(bench.BenchmarkError):
            bench.rewrite_graph_zone_count(self.MODEL.replace('party_vote_rollup as (', 'not_party AS ('))

    def test_alternating_paired_signed_and_guarded(self):
        with self.assertRaises(bench.BenchmarkError):
            bench.ordered_variants(1, ('except_all', 'signed_countstar'))
        pair = ('signed', 'signed_countstar')
        self.assertEqual(bench.ordered_variants(1, pair), pair)
        self.assertEqual(bench.ordered_variants(2, pair), pair[::-1])
        records = [
            {'result': 'PASS', 'cycle': '2018:general', 'variant': 'signed',
             'repeat': 1, 'elapsed_seconds': 350.0},
            {'result': 'PASS', 'cycle': '2018:general', 'variant': 'signed_countstar',
             'repeat': 1, 'elapsed_seconds': 99.0},
        ]
        result = bench.speedups(records)['2018:general']['signed_countstar_pairs'][0]
        self.assertAlmostEqual(result['speedup_signed_over_countstar'], 3.5354)
        self.assertEqual(result['saved_seconds'], 251.0)

    def test_exact_independent_physical_guard_sql(self):
        sql = bench.zone_uniqueness_sql()
        self.assertIn('candidate_id, zone', sql)
        self.assertIn('zone IS NULL', sql)
        self.assertIn('zone IS NOT NULL THEN n-1', sql)
        self.assertNotIn('approx', sql.lower())

    @unittest.skipUnless(duckdb is not None, 'DuckDB required for physical uniqueness guard')
    def test_real_duckdb_duplicate_and_null_groups_fail_guard(self):
        sql = bench.zone_uniqueness_sql()
        # A narrow fixture retains all nine (group, zone) keys of the physical source.
        with duckdb.connect(':memory:') as con:
            con.execute('''CREATE TABLE fact_candidate_votes AS SELECT
                '2018:general'::VARCHAR AS cycle_id, 1::BIGINT AS election_id,
                1::BIGINT AS round_number, 'AC'::VARCHAR AS uf,
                '01'::VARCHAR AS municipality_code, '1'::VARCHAR AS office_code,
                false AS is_transit_vote, 33::BIGINT AS candidate_id,
                10::BIGINT AS zone WHERE FALSE''')
            con.execute("INSERT INTO fact_candidate_votes VALUES ('2018:general',1,1,'AC','01','1',false,33,10)")
            con.execute("INSERT INTO fact_candidate_votes VALUES ('2018:general',1,1,'AC','01','1',false,33,11)")
            result = con.execute(sql).fetchone()
            self.assertEqual(tuple(map(int, result)), (0, 0, 2, 2))
            con.execute("INSERT INTO fact_candidate_votes VALUES ('2018:general',1,1,'AC','01','1',false,33,11)")
            self.assertEqual(tuple(map(int, con.execute(sql).fetchone()))[:2], (0, 1))
            con.execute("INSERT INTO fact_candidate_votes VALUES ('2018:general',1,1,'AC','01','1',false,33,NULL)")
            self.assertEqual(tuple(map(int, con.execute(sql).fetchone()))[:2], (1, 1))

class QueryTimeoutTests(unittest.TestCase):
    class ShortQuery:
        def __init__(self):
            self.interrupted = False

        def interrupt(self):
            self.interrupted = True

        def execute(self, sql):
            self.sql = sql
            return self

        def fetchone(self):
            self.reads = getattr(self, 'reads', 0) + 1
            return (42,) if self.reads == 1 else None

    class BlockingQuery:
        def __init__(self):
            self.interrupted_event = Event()

        def interrupt(self):
            self.interrupted_event.set()

        def execute(self, sql):
            self.sql = sql
            return self

        def fetchone(self):
            if not self.interrupted_event.wait(timeout=3):
                raise AssertionError('watchdog did not call interrupt')
            raise RuntimeError('Interrupted!')

    def test_supported_query_completes_without_interrupt(self):
        conn = self.ShortQuery()
        self.assertEqual(
            bench.fetchone_with_timeout(conn, 'SELECT 42', max_seconds=1, label='fast'),
            (42,))
        self.assertEqual(conn.sql, 'SELECT 42')
        self.assertEqual(conn.reads, 2)  # First row, then EOF to finalize profiler.
        self.assertFalse(conn.interrupted)

    def test_rejects_unexpected_extra_scalar_rows(self):
        class ExtraRows(self.ShortQuery):
            def fetchone(self):
                return (42,)
        with self.assertRaisesRegex(bench.BenchmarkError, 'exactly one scalar row'):
            bench.fetchone_with_timeout(ExtraRows(), 'SELECT too_many', max_seconds=1, label='bad')

    def test_rejects_missing_scalar_row(self):
        class NoRows(self.ShortQuery):
            def fetchone(self):
                return None
        with self.assertRaisesRegex(bench.BenchmarkError, 'returned no rows'):
            bench.fetchone_with_timeout(NoRows(), 'SELECT empty', max_seconds=1, label='bad')

    def test_cooperative_watchdog_interrupts_long_query(self):
        conn = self.BlockingQuery()
        started = time.monotonic()
        with self.assertRaisesRegex(bench.BenchmarkError, 'query exceeded 1s'):
            bench.fetchone_with_timeout(conn, 'SELECT expensive', max_seconds=1, label='slow')
        self.assertTrue(conn.interrupted_event.is_set())
        self.assertLess(time.monotonic() - started, 3)

    def test_missing_interrupt_rejected_before_query(self):
        class Unsupported:
            def execute(self, sql):
                raise AssertionError('must not execute without watchdog')

        with self.assertRaisesRegex(bench.BenchmarkError, 'lacks interrupt'):
            bench.fetchone_with_timeout(Unsupported(), 'SELECT 1', max_seconds=1, label='bad')

    def test_invalid_timeout_rejected(self):
        with self.assertRaisesRegex(bench.BenchmarkError, 'positive'):
            bench.fetchone_with_timeout(self.ShortQuery(), 'SELECT 1', max_seconds=0, label='bad')

    def test_genuine_query_error_propagates(self):
        class BrokenQuery(self.ShortQuery):
            def execute(self, sql):
                raise ValueError('original SQL error')

        with self.assertRaisesRegex(ValueError, 'original SQL error'):
            bench.fetchone_with_timeout(BrokenQuery(), 'SELECT broken', max_seconds=1, label='bad')


class DuckDBProfilerConfigurationTests(unittest.TestCase):
    def test_json_format_is_enabled_before_json_file_path(self):
        class StrictMockDuckDBConnection:
            def __init__(self):
                self.calls = []
                self.json_enabled = False

            def execute(self, sql):
                self.calls.append(sql)
                if sql == "SET enable_profiling = 'json'":
                    self.json_enabled = True
                elif sql.startswith('SET profiling_output = '):
                    if not self.json_enabled:
                        raise ValueError('Profiler file type must match query_tree')
                return self

        with tempfile.TemporaryDirectory() as tmp:
            con = StrictMockDuckDBConnection()
            name = Path(tmp) / "profile's-results.json"
            bench.configure_json_profiler(con, name)
            self.assertEqual(con.calls, [
                "SET enable_profiling = 'json'",
                "SET profiling_output = '" + str(name).replace("'", "''") + "'",
            ])

    def test_single_row_reader_triggers_eof_dependent_profile(self):
        # Regression: reading the scalar row alone is NOT the end of a DuckDB
        # result. This fake backend emits the JSON file only at EOF.
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / 'deferred.json'
            class DeferredProfile:
                def __init__(self):
                    self.reads = 0
                def interrupt(self):
                    pass
                def execute(self, sql):
                    return self
                def fetchone(self):
                    self.reads += 1
                    if self.reads == 1:
                        return (5,)
                    profile.write_text('{"latency": 0, "children": []}')
                    return None
            con = DeferredProfile()
            self.assertEqual(bench.fetchone_with_timeout(
                con, 'SELECT 5', max_seconds=2, label='eof'), (5,))
            self.assertEqual(con.reads, 2)
            self.assertTrue(profile.is_file())

    def test_non_json_output_is_rejected_before_touching_connection(self):
        class NoCalls:
            def execute(self, sql):
                raise AssertionError('must validate output extension first')
        with self.assertRaisesRegex(bench.BenchmarkError, 'requires a .json'):
            bench.configure_json_profiler(NoCalls(), Path('/tmp/profile.txt'))

    @unittest.skipUnless(duckdb is not None, 'DuckDB required for profiler integration')
    def test_real_duckdb_can_write_parseable_json_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / 'cardinality.profile.json'
            con = duckdb.connect(':memory:')
            try:
                bench.configure_json_profiler(con, profile)
                row = bench.fetchone_with_timeout(
                    con, 'SELECT sum(i) FROM range(10000) t(i)',
                    max_seconds=10, label='profiler-smoke')
                self.assertEqual(row, (49995000,))
            finally:
                con.close()
            self.assertTrue(profile.is_file())
            parsed = json.loads(profile.read_text(encoding='utf-8'))
            self.assertIsInstance(parsed, dict)
            self.assertTrue('children' in parsed or 'latency' in parsed)


@unittest.skipUnless(duckdb is not None, 'DuckDB required for exact EXCEPT ALL equivalence test')
class DuckDBMultisetTests(unittest.TestCase):
    def test_signed_equals_except_all_for_nulls_multiplicities_and_changes(self):
        with duckdb.connect(':memory:') as con:
            columns = (
                'source_id VARCHAR, target_id VARCHAR, cycle_id VARCHAR, '
                'election_id VARCHAR, round_number INTEGER, office_code VARCHAR, '
                'is_transit_vote BOOLEAN, nominal_valid_votes HUGEINT, '
                'zone_rows BIGINT, zone_count BIGINT'
            )
            con.execute('CREATE TABLE src_expected (' + columns + ')')
            con.execute('CREATE TABLE src_actual (' + columns + ')')
            a = ('["contest","42"]', '["candidate","17"]', '2018:general',
                 '42', 1, '1', None, 100, 2, 2)
            b = ('["contest","42"]', '["candidate","19"]', '2018:general',
                 '42', None, '2', False, 77, 3, 1)
            c = ('["contest","44"]', '["candidate","21"]', '2018:general',
                 '44', 2, '1', True, 17, 1, 1)
            pairs = [
                ([a, a, b], [a, a, b]),  # exact duplicate conservation
                ([a, a], [a]),          # one missing duplicate
                ([a], [a, a, a]),       # two unexpected duplicates
                ([a, b], [a, c]),       # changed identity + nullable fields
                ([a, b], [a, a, c]),    # both kinds at once
                ([], []),               # empty set still conserves
                ([b, b], [b]),          # NULL grouping equals NULL
            ]
            for expected, actual in pairs:
                with self.subTest(expected=len(expected), actual=len(actual)):
                    con.execute('DELETE FROM src_expected')
                    con.execute('DELETE FROM src_actual')
                    if expected:
                        con.executemany('INSERT INTO src_expected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', expected)
                    if actual:
                        con.executemany('INSERT INTO src_actual VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', actual)
                    oracle = sum((collections.Counter(expected) - collections.Counter(actual)).values())
                    oracle += sum((collections.Counter(actual) - collections.Counter(expected)).values())
                    except_count = con.execute(bench.original_query(SAMPLE)).fetchone()[0]
                    signed_count = con.execute(bench.signed_query(SAMPLE)).fetchone()[0]
                    self.assertEqual(int(except_count), oracle)
                    self.assertEqual(int(signed_count), oracle)


if __name__ == '__main__':
    unittest.main()
