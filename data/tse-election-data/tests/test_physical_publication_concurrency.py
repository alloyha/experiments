"""PR36 falsification gate: partition locks, immutable generations and recovery.

All test data is confined to TemporaryDirectory. Requires POSIX flock + DuckDB.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from scripts import write_candidate_fact_partition as publisher


class PhysicalPublicationConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pr36-publish-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "warehouse.duckdb"
        self.out = self.root / "facts"
        self.partition = self.out / "election_type=general" / "year=2026"
        self.target = self.partition / "data.parquet"
        self.set_votes(10)

    def set_votes(self, value):
        with duckdb.connect(str(self.db)) as con:
            con.execute("""
                create or replace table main.int_candidate_fact_partition
                (election_year integer, election_type varchar, nominal_votes integer)
            """)
            con.execute("insert into main.int_candidate_fact_partition values (2026, 'general', ?)", [value])

    def publish(self, **kwargs):
        return publisher.publish_partition(self.db, self.out, 2026, "general", **kwargs)

    def read_vote(self, path=None):
        with duckdb.connect() as con:
            return con.execute(
                "select nominal_votes from read_parquet(?, hive_partitioning=false)",
                [str(path or self.target)],
            ).fetchone()[0]

    def test_publication_uses_atomic_symlink_to_immutable_generation(self):
        self.publish()
        self.assertTrue(self.target.is_symlink())
        generation = os.readlink(self.target)
        self.assertTrue(generation.startswith("generation-"), generation)
        self.assertTrue((self.partition / generation).is_file())
        self.assertEqual(self.read_vote(), 10)
        self.assertEqual(self.read_vote(self.partition / generation), 10)

    def test_old_generation_survives_next_publication(self):
        self.publish()
        old_generation = self.partition / os.readlink(self.target)
        self.set_votes(20)
        self.publish()
        self.assertEqual(self.read_vote(), 20)
        self.assertEqual(self.read_vote(old_generation), 10)
        self.assertNotEqual(old_generation.name, os.readlink(self.target))

    def test_gold_glob_resolves_atomic_generation_pointer(self):
        self.publish()
        pattern = self.out / "election_type=*/year=*/data.parquet"
        with duckdb.connect() as con:
            rows = con.execute(
                "select nominal_votes from read_parquet(?, hive_partitioning=true)",
                [str(pattern)],
            ).fetchall()
        self.assertEqual(rows, [(10,)])

    def test_killed_writer_releases_partition_lock(self):
        self.partition.mkdir(parents=True, exist_ok=True)
        lock_path = self.partition / ".publish.lock"
        script = (
            "import fcntl,sys; f=open(sys.argv[1], 'a+b'); "
            "fcntl.flock(f, fcntl.LOCK_EX); print('LOCKED', flush=True); "
            "sys.stdin.read()"
        )
        worker = subprocess.Popen(
            [sys.executable, "-c", script, str(lock_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(worker.stdout.readline().strip(), "LOCKED")
            with self.assertRaises(TimeoutError):
                self.publish(lock_timeout=0)
        finally:
            worker.kill()
            worker.communicate(timeout=5)
        self.publish(lock_timeout=1)
        self.assertEqual(self.read_vote(), 10)

    def test_explicit_stale_generation_is_fenced_out(self):
        baseline = publisher.current_generation(self.partition)
        self.publish()
        current = os.readlink(self.target)
        self.set_votes(30)
        with self.assertRaisesRegex(RuntimeError, "generation changed"):
            self.publish(expected_generation=baseline)
        self.assertEqual(os.readlink(self.target), current)
        self.assertEqual(self.read_vote(), 10)

    def test_lock_timeout_rejects_simultaneous_publisher(self):
        self.partition.mkdir(parents=True, exist_ok=True)
        lock = self.partition / ".publish.lock"
        with lock.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                with self.assertRaisesRegex(TimeoutError, "partition lock"):
                    self.publish(lock_timeout=0)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        self.assertFalse(self.target.exists())
        self.publish(lock_timeout=1)
        self.assertEqual(self.read_vote(), 10)

    def test_crash_before_pointer_swap_preserves_old_generation(self):
        self.publish()
        original = os.readlink(self.target)
        self.set_votes(20)
        replace = os.replace

        def fail_publish(source, destination):
            if Path(destination) == self.target:
                raise OSError("injected pointer swap failure")
            return replace(source, destination)

        with mock.patch.object(publisher.os, "replace", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "pointer swap failure"):
                self.publish()
        self.assertEqual(os.readlink(self.target), original)
        self.assertEqual(self.read_vote(), 10)
        self.assertEqual(list(self.partition.glob(".generation-*.tmp")), [])
        self.assertEqual(list(self.partition.glob(".data-parquet-*.link")), [])

    def test_next_writer_recovers_abandoned_unpublished_temporaries(self):
        self.partition.mkdir(parents=True, exist_ok=True)
        (self.partition / ".generation-abandoned.tmp").write_bytes(b"broken")
        (self.partition / ".data-parquet-abandoned.link").symlink_to("missing.parquet")
        self.publish()
        self.assertEqual(list(self.partition.glob(".generation-*.tmp")), [])
        self.assertEqual(list(self.partition.glob(".data-parquet-*.link")), [])
        self.assertEqual(self.read_vote(), 10)

    def test_makefile_fences_snapshot_built_before_publication(self):
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        recipe = makefile.split("candidate-fact-partition: dbt-deps", 1)[1].split("\nunit:", 1)[0]
        self.assertIn("--expected-generation", recipe)
        self.assertIn("--show-generation", recipe)
        self.assertLess(recipe.index("--show-generation"), recipe.index("$(DBT) run"))

    def test_legacy_regular_file_is_retained_during_migration(self):
        self.partition.mkdir(parents=True, exist_ok=True)
        with duckdb.connect() as con:
            con.execute("create table legacy as select 2026 as election_year, 'general' as election_type, 999 as nominal_votes")
            con.execute(f"copy legacy to '{self.target}' (format parquet)")
        old_sha = hashlib.sha256(self.target.read_bytes()).hexdigest()
        self.publish()
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(self.read_vote(), 10)
        legacy_copies = list(self.partition.glob("generation-legacy-*.parquet"))
        self.assertEqual(len(legacy_copies), 1)
        self.assertEqual(hashlib.sha256(legacy_copies[0].read_bytes()).hexdigest(), old_sha)
        self.assertEqual(self.read_vote(legacy_copies[0]), 999)


if __name__ == "__main__":
    unittest.main()
