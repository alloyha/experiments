"""Historical five-cycle guard falsifiers; all fixtures are disposable.

No canonical TSE or existing attestation files are accessed. Requires DuckDB
only for the real Parquet tests; baseline input guards use stdlib only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import audit_graph_candidate_zone_history as history


class HistoryInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='graph-history-input-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db_dir = self.root / 'warehouse'
        self.db_dir.mkdir()
        self.db = self.db_dir / 'main.duckdb'
        self.db.write_bytes(b'synthetic-warehouse-identity')
        self.facts = self.db_dir / 'fact_candidate_votes'
        self.facts.mkdir()
        self.out = self.root / 'new-report.json'
        self.sha = hashlib.sha256(self.db.read_bytes()).hexdigest()

    def test_output_must_not_be_in_warehouse(self):
        with self.assertRaisesRegex(history.HistoryGuardError, 'outside'):
            history.expected_input(self.db, self.facts, self.db_dir / 'proof.json',
                                   expected_sha256=self.sha)

    def test_refuse_existing_report_and_invalid_sha(self):
        self.out.write_text('historical')
        with self.assertRaisesRegex(history.HistoryGuardError, 'Output must be absent'):
            history.expected_input(self.db, self.facts, self.out, expected_sha256=self.sha)
        self.out.unlink()
        with self.assertRaisesRegex(history.HistoryGuardError, 'sha256'):
            history.expected_input(self.db, self.facts, self.out, expected_sha256='incorrect')

    def test_reject_wal_without_creating_result(self):
        wal = Path(str(self.db) + '.wal')
        wal.write_bytes(b'uncommitted')
        with self.assertRaisesRegex(history.HistoryGuardError, 'WAL'):
            history.audit(db=self.db, root=self.facts, out=self.out,
                          expected_sha256=self.sha)
        self.assertFalse(self.out.exists())


class IndependentAggregationTests(unittest.TestCase):
    class Connection:
        def __init__(self, record):
            self.record = record

        def execute(self, query, params):
            if query != history.GUARD_SQL or len(params) != 1:
                raise AssertionError('expected one pinned physical SQL query')
            return self

        def fetchone(self):
            return self.record

    def test_count_distinct_tuple_matches_unique_inventory(self):
        result = history.check_exact_uniqueness(
            self.Connection((2, 0, 2, 2)), Path('/synthetic.parquet'), expected_rows=2)
        self.assertEqual(result['distinct_group_zone_keys'], 2)
        self.assertEqual(result['null_zone_rows'], 0)
        self.assertEqual(result['duplicate_nonnull_zone_rows'], 0)

    def test_duplicate_fails_closed(self):
        with self.assertRaisesRegex(history.HistoryGuardError,
                                    'duplicate_nonnull_zone_rows=1'):
            history.check_exact_uniqueness(
                self.Connection((2, 0, 2, 1)), Path('/synthetic.parquet'), expected_rows=2)

    def test_null_zone_fails_closed(self):
        with self.assertRaisesRegex(history.HistoryGuardError, 'null_zone_rows=1'):
            history.check_exact_uniqueness(
                self.Connection((2, 1, 1, 1)), Path('/synthetic.parquet'), expected_rows=2)

    def test_inventory_mismatch_fails_closed(self):
        with self.assertRaisesRegex(history.HistoryGuardError, 'inventory mismatch'):
            history.check_exact_uniqueness(
                self.Connection((2, 0, 2, 2)), Path('/synthetic.parquet'), expected_rows=3)



@unittest.skipUnless(importlib.util.find_spec('duckdb') is not None,
                     'Real DuckDB needed for exact physical Parquet falsifiers')
class RealHistoryTests(unittest.TestCase):
    def setUp(self):
        import duckdb
        self.duckdb = duckdb
        temp = tempfile.TemporaryDirectory(prefix='graph-history-real-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        warehouse = self.root / 'warehouse'
        warehouse.mkdir()
        self.db = warehouse / 'test.duckdb'
        with duckdb.connect(str(self.db)) as connection:
            connection.execute('CREATE TABLE test_marker AS SELECT 1 AS n')
        self.facts = warehouse / 'fact_candidate_votes'
        self.facts.mkdir()
        self.out = self.root / 'report.json'
        self.sha = history.locks.digest(self.db)
        self.warehouse_lock = history.locks.lock_path(self.db)
        self.warehouse_lock.touch()
        for cycle in history.CYCLES:
            self.make_cycle(cycle)

    def make_cycle(self, cycle: str, *, null_zone: bool = False,
                   duplicate: bool = False, wrong_cycle: bool = False,
                   missing_zone: bool = False) -> None:
        year, kind = cycle.split(':')
        part = history.locks.partition_for(self.facts, cycle)
        part.mkdir(parents=True, exist_ok=True)
        (part / '.publish.lock').touch()
        immutable = part / 'generation-fixture.parquet'
        public = part / 'data.parquet'
        if public.is_symlink():
            public.unlink()
        # Every physical file is created from scratch in the temporary directory.
        if immutable.exists():
            immutable.unlink()
        logical_year = int(year) - 2 if wrong_cycle else int(year)
        zone_expr = 'NULL::INTEGER' if null_zone else '10::INTEGER'
        zone_projection = '' if missing_zone else f', {zone_expr} AS "zone"'
        duplicate_union = ' UNION ALL SELECT * FROM data' if duplicate else ''
        with self.duckdb.connect(':memory:') as con:
            sql = f"""COPY (WITH data AS (
                SELECT {logical_year}::INTEGER AS election_year,
                       '{kind}'::VARCHAR AS election_type,
                       1::INTEGER AS election_id, 1::INTEGER AS round_number,
                       'GO'::VARCHAR AS uf, '5208707'::VARCHAR AS municipality_code,
                       '13'::VARCHAR AS office_code, false AS is_transit_vote,
                       'C1'::VARCHAR AS candidate_id{zone_projection},
                       17::INTEGER AS nominal_valid_votes
             ) SELECT * FROM data{duplicate_union})
             TO '{immutable}' (FORMAT PARQUET)"""
            con.execute(sql)
        public.symlink_to(immutable.name)

    def run_audit(self, **params):
        return history.audit(db=self.db, root=self.facts,
                             out=self.out, expected_sha256=self.sha, **params)

    def test_all_five_cycles_pass_and_report_is_create_only(self):
        report = self.run_audit()
        self.assertEqual(report['result'], 'PASS')
        self.assertTrue(report['all_five_cycles_verified'])
        self.assertEqual([item['cycle'] for item in report['cycles']], list(history.CYCLES))
        self.assertTrue(all(item['source_rows'] == 1 for item in report['cycles']))
        self.assertTrue(all(item['duplicate_nonnull_zone_rows'] == 0 for item in report['cycles']))
        self.assertEqual(json.loads(self.out.read_text()), report)
        self.assertFalse(list(self.facts.rglob('*.zone-uniqueness.json')))
        with self.assertRaisesRegex(history.HistoryGuardError, 'Output must be absent'):
            self.run_audit()

    def test_duplicate_is_rejected_without_a_report(self):
        self.make_cycle('2024:municipal', duplicate=True)
        with self.assertRaisesRegex(history.HistoryGuardError, 'duplicate_nonnull_zone_rows=1'):
            self.run_audit()
        self.assertFalse(self.out.exists())

    def test_null_zone_is_rejected_without_a_report(self):
        self.make_cycle('2022:general', null_zone=True)
        with self.assertRaisesRegex(history.HistoryGuardError, 'null_zone_rows=1'):
            self.run_audit()
        self.assertFalse(self.out.exists())

    def test_missing_zone_column_fails_closed(self):
        self.make_cycle('2018:general', missing_zone=True)
        with self.assertRaisesRegex(history.HistoryGuardError, 'missing mandatory columns'):
            self.run_audit()
        self.assertFalse(self.out.exists())

    def test_out_of_cycle_parquet_fails_closed(self):
        self.make_cycle('2020:municipal', wrong_cycle=True)
        with self.assertRaisesRegex(history.HistoryGuardError, 'out_of_cycle=1'):
            self.run_audit()
        self.assertFalse(self.out.exists())

    def test_missing_partition_lock_never_fabricated(self):
        lock = history.locks.partition_for(self.facts, '2024:municipal') / '.publish.lock'
        lock.unlink()
        with self.assertRaises(Exception):
            self.run_audit()
        self.assertFalse(lock.exists())
        self.assertFalse(self.out.exists())

    def test_wrong_warehouse_digest_fails(self):
        with self.assertRaisesRegex(history.HistoryGuardError, 'Warehouse SHA differs'):
            history.audit(db=self.db, root=self.facts, out=self.out,
                          expected_sha256='0' * 64)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
