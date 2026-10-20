"""CCC B2: execute the complete guarded party-ID migration on real, disposable DuckDB.

No canonical TSE warehouse or external files are opened by these tests.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.audit_party_id_materialization import audit_database
from scripts.migrate_party_id_canonical import MigrationRejected, migration
from scripts.warehouse_lifecycle import lifecycle_epoch


CYCLES = {
    "2018:general": 88,
    "2020:municipal": 395,
    "2022:general": 58,
    "2024:municipal": 134,
    "2026:general": 42,
}
TOTAL = sum(CYCLES.values())


class RealDuckDBPartyIdentityMigrationTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="ccc-b2-duckdb-e2e-")
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        self.db = root / "fixture.duckdb"
        self.backup = root / "fixture.before-migration.duckdb"
        with duckdb.connect(str(self.db)) as con:
            con.execute("""
                CREATE TABLE main.dim_party (
                    election_year INTEGER NOT NULL,
                    election_type VARCHAR NOT NULL,
                    election_code VARCHAR NOT NULL,
                    party_number VARCHAR NOT NULL,
                    cycle_id VARCHAR NOT NULL,
                    election_id VARCHAR NOT NULL,
                    party_id VARCHAR NOT NULL,
                    party_name VARCHAR NOT NULL
                )
            """)
            records = []
            for cycle, count in CYCLES.items():
                year, election_type = cycle.split(":", 1)
                for n in range(count):
                    code = f"{int(year)}{n:04d}"
                    number = "45"
                    event = f"{cycle}:{code}"
                    records.append((int(year), election_type, code, number,
                                    cycle, event, f"{event}:{code}:{number}",
                                    f"Party snapshot {cycle} record {n}"))
            con.executemany("INSERT INTO main.dim_party VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
        self.assertEqual(TOTAL, 717)

    def test_full_duckdb_backup_transaction_and_post_audit(self):
        with duckdb.connect(str(self.db), read_only=True) as con:
            original_other_columns = con.execute("""
                SELECT * EXCLUDE (party_id) FROM main.dim_party
                ORDER BY election_year, election_type, election_code, party_number
            """).fetchall()
        previous_epoch = lifecycle_epoch(self.db)
        plan = migration(self.db, "main", CYCLES, TOTAL, apply=False)
        self.assertEqual(plan["status"], "READY")
        self.assertFalse(self.backup.exists())
        self.assertEqual(lifecycle_epoch(self.db), previous_epoch)

        result = migration(self.db, "main", CYCLES, TOTAL,
                           apply=True, backup=self.backup)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["updated_rows"], TOTAL)
        self.assertTrue(self.backup.is_file())
        self.assertNotEqual(lifecycle_epoch(self.db), previous_epoch)

        after = audit_database(self.db, "main", list(CYCLES))
        self.assertEqual(after["status"], "PASS")
        self.assertEqual(after["total_rows"], TOTAL)
        self.assertEqual({r["cycle"]: r["rows"] for r in after["cycles"]}, CYCLES)
        for cycle in after["cycles"]:
            self.assertEqual(cycle["noncanonical_party_ids"], 0)
            self.assertEqual(cycle["legacy_party_ids"], 0)
            self.assertEqual(cycle["duplicate_party_id_rows"], 0)
            self.assertEqual(cycle["duplicate_grain_rows"], 0)

        with duckdb.connect(str(self.db), read_only=True) as con:
            new_other_columns = con.execute("""
                SELECT * EXCLUDE (party_id) FROM main.dim_party
                ORDER BY election_year, election_type, election_code, party_number
            """).fetchall()
        self.assertEqual(new_other_columns, original_other_columns)

        before = audit_database(self.backup, "main", list(CYCLES))
        self.assertEqual(before["status"], "FAIL")
        self.assertEqual(before["total_rows"], TOTAL)
        self.assertEqual(sum(r["legacy_party_ids"] for r in before["cycles"]), TOTAL)

        # A second migration is forbidden: no silent partial/idempotent overwrite.
        with self.assertRaises(MigrationRejected):
            migration(self.db, "main", CYCLES, TOTAL, apply=False)

    def test_corrupted_identity_refuses_write_or_backup(self):
        with duckdb.connect(str(self.db)) as con:
            con.execute("""
                UPDATE main.dim_party SET party_id = 'unexpected'
                WHERE election_year = 2018 AND election_code = '20180000'
            """)
        previous_epoch = lifecycle_epoch(self.db)
        with self.assertRaises(MigrationRejected):
            migration(self.db, "main", CYCLES, TOTAL,
                      apply=True, backup=self.backup)
        self.assertFalse(self.backup.exists())
        self.assertEqual(lifecycle_epoch(self.db), previous_epoch)


if __name__ == "__main__":
    unittest.main()
