"""GG-009: direct physical publisher must participate in the warehouse gate.

Each DuckDB and Parquet path is confined to TemporaryDirectory. No TSE scans.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from scripts import build_candidate_fact_partition as builder
from scripts import write_candidate_fact_partition as publisher
from scripts.graph_guarded_read import graph_read_locks


class WarehouseFencedPublisherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="gg009-publisher-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / "warehouse.duckdb"
        self.fact_root = self.root / "facts"
        self.part = self.fact_root / "election_type=municipal/year=2024"
        self.part.mkdir(parents=True)
        with duckdb.connect(str(self.db)) as con:
            con.execute("""CREATE TABLE int_candidate_fact_partition AS
                SELECT 2024 AS election_year, 'municipal' AS election_type,
                       31 AS nominal_votes""")

    def publish(self, **kwargs):
        return publisher.publish_partition(self.db, self.fact_root, 2024,
                                           "municipal", **kwargs)

    def test_standalone_publisher_cannot_bypass_graph_reader(self):
        with graph_read_locks(self.db, self.fact_root, "2024:municipal", timeout=0):
            with self.assertRaisesRegex(TimeoutError, "warehouse lock"):
                self.publish(warehouse_lock_timeout=0)
        self.assertFalse((self.part / "data.parquet").exists())
        self.publish(warehouse_lock_timeout=0)
        self.assertTrue((self.part / "data.parquet").is_symlink())

    def test_existing_coordinator_lock_is_borrowed_without_deadlock(self):
        # If the publisher tried to reacquire this flock, timeout=0 would fail.
        with builder.warehouse_write_lock(self.db, timeout=0) as lock:
            self.publish(borrowed_warehouse_lock_fd=lock.fileno(),
                         warehouse_lock_timeout=0, lock_timeout=0)
        target = self.part / "data.parquet"
        self.assertTrue(target.is_symlink())
        with duckdb.connect(":memory:") as con:
            row = con.execute("SELECT nominal_votes FROM read_parquet(?)",
                              [str(target)]).fetchone()
        self.assertEqual(row, (31,))

    def test_borrowed_lock_must_match_warehouse_inode(self):
        with self.db.open("rb") as wrong:
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                self.publish(borrowed_warehouse_lock_fd=wrong.fileno())
        self.assertFalse((self.part / "data.parquet").exists())

    def test_invalid_borrowed_lock_descriptor_is_distinct_from_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid borrowed warehouse lock descriptor"):
            self.publish(borrowed_warehouse_lock_fd=-1)
        self.assertFalse(self.db.with_name(self.db.name + ".build.lock").exists())
        self.assertFalse((self.part / "data.parquet").exists())

    def test_wrong_descriptor_is_rejected_even_if_lock_path_exists(self):
        with builder.warehouse_write_lock(self.db, timeout=0):
            with self.db.open("rb") as wrong:
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    self.publish(borrowed_warehouse_lock_fd=wrong.fileno())
        self.assertFalse((self.part / "data.parquet").exists())

    def test_direct_publisher_holds_warehouse_lock_during_export(self):
        original = publisher.validate_staged_parquet

        def probe(conn, staged, expected):
            # A graph reader must fail before it can enter while publication is
            # holding warehouse AND partition locks. The DB is real and on disk.
            with self.assertRaisesRegex(TimeoutError, "graph read lock"):
                with graph_read_locks(self.db, self.fact_root,
                                      "2024:municipal", timeout=0):
                    self.fail("reader entered while publisher exporting")
            return original(conn, staged, expected)

        with patch.object(publisher, "validate_staged_parquet", side_effect=probe):
            self.publish(warehouse_lock_timeout=0)

    def test_independent_warehouses_can_publish_concurrently(self):
        other = self.root / "other.duckdb"
        other_root = self.root / "other-facts"
        with duckdb.connect(str(other)) as con:
            con.execute("""CREATE TABLE int_candidate_fact_partition AS
                SELECT 2024 AS election_year, 'municipal' AS election_type,
                       19 AS nominal_votes""")
        with graph_read_locks(self.db, self.fact_root, "2024:municipal", timeout=0):
            path = publisher.publish_partition(other, other_root, 2024,
                                               "municipal", warehouse_lock_timeout=0)
        self.assertTrue(path.is_symlink())


if __name__ == "__main__":
    unittest.main()
