"""CCC B2 guarded migration safety falsifiers; SQLite tests are not a real DuckDB smoke."""
from __future__ import annotations
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.audit_party_id_materialization import summarize_rows
from scripts.migrate_party_id_canonical import (
    MigrationRejected, create_durable_backup, ensure_distinct_canonical_ids,
    evaluate_before, unchanged_except_party_id, parse_expected, capture_rows,
    verify_after,
)

EXPECTED={'2018:general': 1, '2020:municipal': 1}

class MigrationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.con=sqlite3.connect(':memory:')
        self.addCleanup(self.con.close)
        self.con.execute('''CREATE TABLE dim_party (
            election_year INTEGER, election_type TEXT, election_code TEXT,
            party_number TEXT, cycle_id TEXT, election_id TEXT, party_id TEXT,
            party_name TEXT)''')
        for y,t,e,p in [(2018,'general','85','12'),(2020,'municipal','620','45')]:
            event=f'{y}:{t}:{e}'
            self.con.execute('INSERT INTO dim_party VALUES (?,?,?,?,?,?,?,?)',
                (y,t,e,p,f'{y}:{t}',event,f'{event}:{e}:{p}',f'Party {p}'))

    def rows(self):
        return summarize_rows(self.con, 'main')

    def test_plan_accepts_only_exact_legacy_inventory(self):
        self.assertEqual(evaluate_before(self.rows(), EXPECTED, 2)['status'], 'READY')

    def test_wrong_counts_or_missing_cycles_fails_closed(self):
        for expected, legacy in [(EXPECTED, 1), ({'2018:general':1}, 2),
                                 ({'2018:general':2,'2020:municipal':1}, 2)]:
            with self.subTest(expected=expected,legacy=legacy), self.assertRaises(MigrationRejected):
                evaluate_before(self.rows(), expected, legacy)

    def test_mixed_ids_fail(self):
        self.con.execute("UPDATE dim_party SET party_id='2020:municipal:620:45' WHERE election_year=2020")
        with self.assertRaisesRegex(MigrationRejected, 'mixed or unexpected'):
            evaluate_before(self.rows(), EXPECTED, 2)

    def test_corrupted_cycle_and_duplicate_grain_fail(self):
        self.con.execute("UPDATE dim_party SET cycle_id='wrong' WHERE election_year=2020")
        with self.assertRaisesRegex(MigrationRejected, 'structural defect'):
            evaluate_before(self.rows(), EXPECTED, 2)
        self.con.execute("UPDATE dim_party SET cycle_id='2020:municipal' WHERE election_year=2020")
        self.con.execute('INSERT INTO dim_party SELECT * FROM dim_party WHERE election_year=2018')
        with self.assertRaises(MigrationRejected):
            evaluate_before(self.rows(), EXPECTED, 3)

    def test_exact_sql_update_preserves_other_fields(self):
        before=capture_rows(self.con,'main')
        self.con.commit()
        self.con.execute('BEGIN')
        self.con.execute("""UPDATE main.dim_party SET party_id = CAST(election_id AS VARCHAR)
            || ':' || CAST(party_number AS VARCHAR)
            WHERE party_id = CAST(election_id AS VARCHAR) || ':' ||
            CAST(election_code AS VARCHAR) || ':' || CAST(party_number AS VARCHAR)""")
        after=capture_rows(self.con,'main')
        unchanged_except_party_id(before,after)
        verify_after(self.rows(), EXPECTED)
        self.con.execute('ROLLBACK')
        self.assertEqual(evaluate_before(self.rows(), EXPECTED, 2)['status'], 'READY')

    def test_non_id_change_is_rejected(self):
        before=capture_rows(self.con,'main')
        self.con.execute("UPDATE dim_party SET party_name='CORRUPTED'")
        with self.assertRaisesRegex(MigrationRejected, 'Non-party_id'):
            unchanged_except_party_id(before, capture_rows(self.con,'main'))

    def test_collision_rejected(self):
        # The fixed 2018/2020 IDs are distinct, but an unexpected colliding
        # derived identifier must refuse migration even if old IDs differ.
        self.con.execute("UPDATE dim_party SET election_id='2018:general:85', party_number='12' WHERE election_year=2020")
        with self.assertRaisesRegex(MigrationRejected, 'collide'):
            ensure_distinct_canonical_ids(self.con, 'main', 2)

    def test_strict_cli_inventory_parse(self):
        self.assertEqual(parse_expected(['2018:general=88','2020:municipal=395']),
                         {'2018:general':88,'2020:municipal':395})
        for x in ([], ['x=1'], ['2020:municipal=0'],
                  ['2020:municipal=1','2020:municipal=1']):
            with self.subTest(x=x), self.assertRaises(MigrationRejected):
                parse_expected(x)

    def test_backup_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'tiny.duckdb'; source.write_bytes(b'DB1234')
            dest=Path(tmp)/'tiny.bak'
            create_durable_backup(source,dest)
            self.assertEqual(dest.read_bytes(), source.read_bytes())
            with self.assertRaises(MigrationRejected):
                create_durable_backup(source,dest)

if __name__=='__main__': unittest.main()
