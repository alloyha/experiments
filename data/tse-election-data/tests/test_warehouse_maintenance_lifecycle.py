"""GG-009 maintenance: cross-process fencing and no stale proof after recreation."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import warehouse_lifecycle as maintenance
from scripts import graph_freshness as graph
from scripts import candidate_cube_freshness as cube


class WarehouseMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gg009-clean-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "warehouse.duckdb"
        self.db.write_bytes(b"same DuckDB replacement bytes")
        self.facts = self.root / "facts"
        self.facts.mkdir()
        self.certs = self.root / "graph_certs"

    def test_shared_reader_blocks_deletion_before_epoch_rotation(self):
        old = maintenance.lifecycle_epoch(self.db)
        lock = self.db.with_name(self.db.name + ".build.lock")
        lock.touch()
        with lock.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            try:
                # Separate process required: cross-process flock semantics.
                result = subprocess.run(
                    [sys.executable, str(Path(maintenance.__file__).resolve()),
                     "warehouse", "--db", str(self.db), "--lock-timeout", "0"],
                    capture_output=True, text=True, timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("warehouse lock", result.stderr)
                self.assertTrue(self.db.is_file())
                self.assertEqual(maintenance.lifecycle_epoch(self.db), old)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        maintenance.clean_warehouse(self.db, timeout=0)
        self.assertFalse(self.db.exists())
        self.assertTrue(lock.is_file(), "stable lock inode must survive cleanup")
        self.assertNotEqual(maintenance.lifecycle_epoch(self.db), old)

    def test_identical_recreated_db_cannot_reuse_graph_certificate(self):
        original = self.db.read_bytes()
        proof = {"unchanged": True}
        graph.issue_certificate(self.certs, self.db, self.facts, "2024:municipal", proof)
        graph.audit_certificate(self.certs, self.db, self.facts,
                                "2024:municipal", proof)
        maintenance.clean_warehouse(self.db)
        self.db.write_bytes(original)
        with self.assertRaisesRegex(graph.GraphFreshnessError, "different warehouse"):
            graph.audit_certificate(self.certs, self.db, self.facts,
                                    "2024:municipal", proof)

    def test_identical_recreated_db_cannot_reuse_cube_certificate(self):
        original = self.db.read_bytes()
        inputs, outputs = {"same": "input"}, {"same": "output"}
        cube.issue_certificate(self.certs, self.db, self.facts, "2024:municipal",
                               inputs, outputs)
        with patch.object(cube, "input_signature", return_value=inputs), \
             patch.object(cube, "output_signature", return_value=outputs):
            self.assertEqual(cube.audit_freshness(self.db, self.facts, self.certs,
                                                  {"2024:municipal"})["fresh_cycles"],
                             ["2024:municipal"])
            maintenance.clean_warehouse(self.db)
            self.db.write_bytes(original)
            with self.assertRaisesRegex(cube.FreshnessError, "lifecycle"):
                cube.audit_freshness(self.db, self.facts, self.certs,
                                     {"2024:municipal"})

    def test_sigkill_during_cleanup_cannot_leave_prior_authority_valid(self):
        # Simulate interruption after the durable epoch rotation but before unlink.
        before = maintenance.lifecycle_epoch(self.db)
        script = '''import os,sys
from pathlib import Path
from scripts.warehouse_lifecycle import maintenance_lock,rotate_epoch
with maintenance_lock(Path(sys.argv[1]),timeout=0):
 rotate_epoch(Path(sys.argv[1]))
 os.kill(os.getpid(),9)
'''
        run = subprocess.run([sys.executable, "-c", script, str(self.db)],
                             cwd=Path(maintenance.__file__).resolve().parents[1],
                             capture_output=True, timeout=5)
        self.assertEqual(run.returncode, -9)
        self.assertTrue(self.db.exists())
        self.assertNotEqual(maintenance.lifecycle_epoch(self.db), before)
        # The SIGKILL releases the lock; next cleanup may proceed.
        maintenance.clean_warehouse(self.db, timeout=0)

    def test_clean_all_rejects_missing_confirmation_and_outside_paths(self):
        project = self.root / "project"
        project.mkdir()
        db = project / "warehouse.duckdb"
        db.write_bytes(b"fixture")
        raw = project / "raw"
        facts = project / "facts"
        artifact = project / "runs"
        dbt = project / "dbt"
        venv = project / ".venv"
        for path in (raw, facts, artifact, dbt, venv):
            path.mkdir()
        base = ["all", "--db", str(db), "--project-root", str(project),
                "--raw-root", str(raw), "--fact-root", str(facts),
                "--artifacts-root", str(artifact), "--dbt-dir", str(dbt),
                "--venv", str(venv)]
        self.assertEqual(maintenance.main(base), 1)
        self.assertTrue(db.exists())
        self.assertEqual(maintenance.main(base + ["--confirm", "--raw-root",
                                                  str(self.root / "outside")]), 1)
        self.assertTrue(db.exists())
        self.assertEqual(maintenance.main(base + ["--confirm"]), 0)
        self.assertFalse(db.exists())
        self.assertFalse(raw.exists())
        self.assertFalse(facts.exists())
        self.assertFalse(venv.exists())
        self.assertTrue(Path(str(db) + ".build.lock").exists())
        self.assertEqual(len(maintenance.lifecycle_epoch(db)), 32)

    def test_clean_all_rejects_parent_of_lifecycle_sidecars(self):
        project = self.root / "project"
        warehouse_dir = project / "warehouse"
        warehouse_dir.mkdir(parents=True)
        db = warehouse_dir / "canary.duckdb"
        db.write_bytes(b"fixture")
        marker = maintenance.epoch_path(db)
        with self.assertRaisesRegex(ValueError, "remove warehouse lock/epoch"):
            maintenance.clean_warehouse(db, extra_paths=(warehouse_dir,))
        self.assertTrue(db.exists())
        self.assertFalse(marker.exists())

    def test_epoch_file_is_not_symlink_or_malformed(self):
        marker = maintenance.epoch_path(self.db)
        marker.symlink_to(self.db)
        with self.assertRaisesRegex(ValueError, "symlink"):
            maintenance.lifecycle_epoch(self.db)
        with self.assertRaisesRegex(ValueError, "symlink"):
            maintenance.clean_warehouse(self.db)
        self.assertTrue(self.db.exists())
        marker.unlink()
        marker.write_text("not-an-epoch")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            maintenance.lifecycle_epoch(self.db)


if __name__ == "__main__":
    unittest.main()
