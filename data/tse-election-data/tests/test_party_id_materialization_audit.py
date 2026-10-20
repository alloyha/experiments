"""CCC B2 audit regressions using executable SQL on in-memory SQLite.

The real target is DuckDB; SQL here deliberately uses the portable subset.
A separate smoke run against the WSL DuckDB is required before promotion.
"""
from __future__ import annotations

import sqlite3
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from scripts.audit_party_id_materialization import (
    assess, audit_database, relation, summarize_rows,
)


class PartyIdentityMaterializationAuditTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("""CREATE TABLE dim_party (
            election_year INTEGER, election_type TEXT, election_code TEXT,
            party_number TEXT, cycle_id TEXT, election_id TEXT, party_id TEXT
        )""")

    def row(self, year=2024, kind="municipal", event="620", party="45", *,
            legacy=False, value=None, cycle=None):
        event_id = f"{year}:{kind}:{event}"
        canonical = f"{event_id}:{party}"
        old = f"{event_id}:{event}:{party}"
        self.db.execute("INSERT INTO dim_party VALUES(?,?,?,?,?,?,?)", (
            year, kind, event, party, cycle if cycle is not None else f"{year}:{kind}",
            event_id, old if legacy else canonical if value is None else value,
        ))

    def test_canonical_multiple_cycles_pass(self):
        self.row()
        self.row(year=2026, kind="general", event="700")
        report = assess(summarize_rows(self.db, "main"),
                        ["2024:municipal", "2026:general"])
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["total_rows"], 2)

    def test_legacy_identity_requires_migration(self):
        self.row(legacy=True)
        result = summarize_rows(self.db, "main")[0]
        self.assertEqual(result["legacy_party_ids"], 1)
        self.assertEqual(result["noncanonical_party_ids"], 1)
        self.assertEqual(result["status"], "DRIFT")
        self.assertEqual(assess([result], ["2024:municipal"])["status"], "FAIL")

    def test_duplicate_grain_and_party_id_fail(self):
        self.row()
        self.row()
        result = summarize_rows(self.db, "main")[0]
        self.assertEqual(result["duplicate_grain_rows"], 1)
        self.assertEqual(result["duplicate_party_id_rows"], 1)
        self.assertEqual(result["status"], "DRIFT")

    def test_missing_expected_cycle_and_empty_fail_closed(self):
        self.assertEqual(assess([], ["2018:general"])["status"], "EMPTY")
        self.row()
        report = assess(summarize_rows(self.db, "main"),
                        ["2018:general", "2024:municipal"])
        self.assertEqual(report["missing_expected_cycles"], ["2018:general"])
        self.assertEqual(report["status"], "FAIL")

    def test_corrupt_cycle_is_rejected(self):
        self.row(cycle="2026:general")
        result = summarize_rows(self.db, "main")[0]
        self.assertEqual(result["invalid_cycle_ids"], 1)
        self.assertEqual(result["status"], "DRIFT")

    def test_missing_key_fails(self):
        self.db.execute("INSERT INTO dim_party VALUES(?,?,?,?,?,?,?)",
                        (2024, "municipal", "620", None, "2024:municipal",
                         "2024:municipal:620", None))
        record = summarize_rows(self.db, "main")[0]
        self.assertGreaterEqual(record["null_key_rows"], 1)
        self.assertEqual(record["status"], "DRIFT")

    def test_schema_name_cannot_inject_sql(self):
        with self.assertRaises(ValueError):
            relation('main"; DROP TABLE dim_party; --')

    def test_database_missing_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nonexistent.duckdb"
            with self.assertRaises(FileNotFoundError):
                audit_database(target, "main", [])
            self.assertFalse(target.exists())

    def test_existing_warehouse_opens_read_only_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing.duckdb"
            target.write_bytes(b"fixture-not-a-real-duckdb")
            conn = MagicMock()
            conn.__enter__.return_value = conn
            fake_duckdb = types.SimpleNamespace(connect=Mock(return_value=conn))
            with patch.dict(sys.modules, {"duckdb": fake_duckdb}), \
                 patch("scripts.audit_party_id_materialization.inspect_catalog"), \
                 patch("scripts.audit_party_id_materialization.summarize_rows", return_value=[]):
                result = audit_database(target, "main", ["2024:municipal"])
            fake_duckdb.connect.assert_called_once_with(str(target.resolve()), read_only=True)
            self.assertEqual(result["status"], "EMPTY")
            self.assertEqual(result["missing_expected_cycles"], ["2024:municipal"])

    def test_cli_does_not_mask_missing_cycles(self):
        self.row()
        self.assertEqual(assess(summarize_rows(self.db, "main"), ["2018:general"])["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
