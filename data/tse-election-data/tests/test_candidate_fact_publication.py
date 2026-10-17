"""TDD: verify the dbt physical publishing boundary, in an isolated DuckDB."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from scripts import write_candidate_fact_partition as publisher


class CandidateFactPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pr35-physical-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.database = self.root / "warehouse.duckdb"
        self.output = self.root / "fact_candidate_votes"
        self.target = self.output / "election_type=general/year=2026/data.parquet"

    def table(self, name: str, values: list[tuple[int, str, int]]) -> None:
        with duckdb.connect(str(self.database)) as con:
            con.execute(f"""
                create or replace table main.{name} (
                    election_year integer, election_type varchar,
                    election_id integer, round_number integer, uf varchar,
                    municipality_code varchar, office_code varchar,
                    is_transit_vote boolean, candidate_id varchar,
                    zone integer, nominal_votes integer
                )
            """)
            if values:
                con.executemany(
                    f"insert into main.{name} values "
                    "(?, ?, 1, 1, 'GO', '5208707', '13', false, 'C1', 10, ?)",
                    values,
                )

    def existing_target(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect() as con:
            con.execute("create table previous_partition as select 2026 as election_year, 'general' as election_type, 999 as nominal_votes")
            con.execute(f"copy previous_partition to '{self.target}' (format parquet)")
        return hashlib.sha256(self.target.read_bytes()).hexdigest()

    def publish(self):
        argv = [
            "write_candidate_fact_partition.py",
            "--db", str(self.database),
            "--root", str(self.output),
            "--year", "2026",
            "--election-type", "general",
        ]
        with mock.patch("sys.argv", argv):
            return publisher.main()

    def test_reads_enabled_physical_model_not_stale_legacy_relation(self):
        self.table("int_candidate_fact_partition", [(2026, "general", 17)])
        # A stale relation must never determine the external fact partition.
        self.table("int_candidate_votes", [(2026, "general", 999)])
        self.publish()
        with duckdb.connect() as con:
            result = con.execute(
                "select nominal_votes from read_parquet(?)", [str(self.target)]
            ).fetchall()
        self.assertEqual(result, [(17,)])

    def test_missing_physical_model_does_not_fall_back_to_legacy_table(self):
        self.table("int_candidate_votes", [(2026, "general", 999)])
        with self.assertRaisesRegex(RuntimeError, "int_candidate_fact_partition"):
            self.publish()
        self.assertFalse(self.target.exists())

    def test_scope_mismatch_preserves_existing_partition(self):
        before = self.existing_target()
        self.table("int_candidate_fact_partition", [(2024, "municipal", 31)])
        with self.assertRaisesRegex(RuntimeError, "not scoped"):
            self.publish()
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), before)

    def test_empty_source_preserves_existing_partition(self):
        before = self.existing_target()
        self.table("int_candidate_fact_partition", [])
        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.publish()
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), before)

    def test_failed_staging_validation_preserves_existing_partition(self):
        before = self.existing_target()
        self.table("int_candidate_fact_partition", [(2026, "general", 21)])
        with mock.patch.object(
            publisher,
            "validate_staged_parquet",
            side_effect=RuntimeError("injected staged verification failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected staged verification failure"):
                self.publish()
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), before)
        self.assertFalse(list(self.target.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
