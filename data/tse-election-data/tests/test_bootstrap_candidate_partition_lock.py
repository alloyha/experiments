"""Isolated real-filesystem tests for one-time historical lock initialization."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import bootstrap_candidate_partition_lock as mod


class CandidatePartitionLockBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.warehouse = Path(self.tmp.name) / "warehouse"
        self.warehouse.mkdir()
        self.main = self.warehouse / "tse_analytics.duckdb"
        self.preview = self.warehouse / "tse_event_registry_preview.duckdb"
        self.main.write_bytes(b"main-unchanged")
        self.preview.write_bytes(b"preview-unchanged")
        self.oh = hashlib.sha256(self.main.read_bytes()).hexdigest()
        self.ph = hashlib.sha256(self.preview.read_bytes()).hexdigest()
        for db in (self.main, self.preview):
            (self.warehouse / (db.name + ".build.lock")).touch()
        self.root = self.warehouse / "fact_candidate_votes"
        self.part = self.root / "election_type=general" / "year=2018"
        self.part.mkdir(parents=True)
        self.parquet = self.part / "data.parquet"
        self.parquet.write_bytes(b"legacy-parquet-fixed")
        self.lock = self.part / ".publish.lock"

    def run_bootstrap(self, mainhash=None, previewhash=None):
        return mod.bootstrap(self.main, self.preview, self.root, "2018:general",
                             mainhash if mainhash is not None else self.oh,
                             previewhash if previewhash is not None else self.ph,
                             timeout=2)

    def test_new_lock_creation_preserves_payload_and_warehouse_epochs(self):
        before_db = (self.main.read_bytes(), self.preview.read_bytes())
        before_parquet = self.parquet.read_bytes()
        out = self.run_bootstrap()
        self.assertTrue(out["created"])
        self.assertEqual(out["scope"], "cooperative_lock_sidecar_bootstrap_only")
        self.assertTrue(self.lock.is_file())
        self.assertEqual(self.main.read_bytes(), before_db[0])
        self.assertEqual(self.preview.read_bytes(), before_db[1])
        self.assertEqual(self.parquet.read_bytes(), before_parquet)
        self.assertFalse(Path(str(self.main) + ".lifecycle").exists())

    def test_idempotent_existing_lock_keeps_inode(self):
        first = self.run_bootstrap()
        stat_before = self.lock.stat()
        second = self.run_bootstrap()
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(self.lock.stat().st_ino, stat_before.st_ino)

    def test_hash_mismatch_refuses_creation(self):
        with self.assertRaisesRegex(mod.BootstrapError, "SHA-256"):
            self.run_bootstrap("a" * 64)
        self.assertFalse(self.lock.exists())

    def test_missing_warehouse_lock_refuses_creation(self):
        (self.warehouse / (self.main.name + ".build.lock")).unlink()
        with self.assertRaisesRegex(mod.BootstrapError, "Missing or unsafe"):
            self.run_bootstrap()
        self.assertFalse(self.lock.exists())

    def test_symlink_partition_lock_refuses_creation(self):
        destination = self.part / "target.txt"
        destination.write_text("important")
        self.lock.symlink_to(destination)
        with self.assertRaisesRegex(mod.BootstrapError, "symlink"):
            self.run_bootstrap()
        self.assertEqual(destination.read_text(), "important")

    def test_wal_refuses_creation(self):
        wal = Path(str(self.preview) + ".wal")
        wal.write_bytes(b"pending")
        with self.assertRaisesRegex(RuntimeError, "WAL"):
            self.run_bootstrap()
        self.assertFalse(self.lock.exists())

    def test_cli_reports_wal_without_traceback(self):
        Path(str(self.preview) + ".wal").write_bytes(b"pending")
        argv = ["--main-db", str(self.main), "--preview-db", str(self.preview),
                "--source-root", str(self.root), "--cycle", "2018:general",
                "--main-sha256", self.oh, "--preview-sha256", self.ph,
                "--ack-quiescent", "--confirm-bootstrap"]
        with mock.patch("sys.stderr"):
            self.assertEqual(mod.main(argv), 2)
        self.assertFalse(self.lock.exists())

    def test_explicit_ack_required_on_cli(self):
        with mock.patch("sys.stderr"):
            status = mod.main(["--main-db", str(self.main), "--preview-db", str(self.preview),
                               "--source-root", str(self.root), "--cycle", "2018:general",
                               "--main-sha256", self.oh, "--preview-sha256", self.ph])
        self.assertEqual(status, 2)
        self.assertFalse(self.lock.exists())

    def test_invalid_cycle_fails_closed(self):
        with self.assertRaisesRegex(mod.BootstrapError, "YYYY"):
            mod.bootstrap(self.main, self.preview, self.root,
                                     "2018:general;DROP", self.oh, self.ph, timeout=2)
        self.assertFalse(self.lock.exists())

    @unittest.skipUnless(os.name == "posix", "requires POSIX flock")
    def test_existing_exclusive_lock_conflict_refuses_without_replacing(self):
        import fcntl
        self.lock.touch()
        previous_inode = self.lock.stat().st_ino
        fd = os.open(self.lock, os.O_RDWR)
        try:
            # Different open file descriptions conflict even in one process.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(mod.BootstrapError, "Timed out acquiring"):
                self.run_bootstrap_with_timeout(.05)
            self.assertEqual(self.lock.stat().st_ino, previous_inode)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def run_bootstrap_with_timeout(self, timeout):
        return mod.bootstrap(self.main, self.preview, self.root, "2018:general",
                             self.oh, self.ph, timeout=timeout)

if __name__ == "__main__":
    unittest.main()
