"""GG-009: cube certifier must share the warehouse lock with graph readers."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import certify_candidate_cube as cube
from scripts.graph_guarded_read import graph_read_locks


class CubeWriterFencingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="gg009-cube-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / "canary.duckdb"
        self.db.write_bytes(b"fixture -- only mocks read this database")
        self.source = self.root / "facts"
        self.part = self.source / "election_type=municipal/year=2024"
        self.part.mkdir(parents=True)
        (self.part / "data.parquet").write_bytes(b"fixture")
        self.certs = self.root / "certificates"
        self.args = SimpleNamespace(
            year=2024, election_type="municipal", attestations_dir=self.certs,
            dbt="dbt", memory_limit="512MB", run_timeout=5,
            test_timeout=5, lock_timeout=0.0,
        )
        self.lock = self.db.with_name(self.db.name + ".build.lock")

    def exclusive_probe(self):
        script = (
            "import fcntl,sys\n"
            "with open(sys.argv[1],'a+b') as f:\n"
            " try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            " except BlockingIOError: sys.exit(17)\n"
        )
        return subprocess.run([sys.executable, "-c", script, str(self.lock)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=5).returncode

    def test_reader_blocks_certification_before_any_revocation_or_scan(self):
        self.certs.mkdir()
        active = self.certs / "2024-municipal.json"
        active.write_text("old certificate", encoding="utf8")
        with graph_read_locks(self.db, self.source, "2024:municipal", timeout=0):
            with patch.object(cube, "revoke_certificate") as revoke, \
                 patch.object(cube, "snapshot_other_cycles") as scan, \
                 patch.object(cube, "dbt_call") as dbt:
                with self.assertRaisesRegex(TimeoutError, "warehouse lock"):
                    cube.certify(self.args, self.db, self.source)
                revoke.assert_not_called()
                scan.assert_not_called()
                dbt.assert_not_called()
        self.assertEqual(active.read_text(encoding="utf8"), "old certificate")

    def test_lock_spans_snapshots_revoke_dbt_and_issue(self):
        attempts = []

        def must_be_fenced(*args, **kwargs):
            attempts.append(self.exclusive_probe())
            self.assertEqual(attempts[-1], 17, "exclusive lock was not held")
            return None

        def fake_dbt(*args, lock_fd, **kwargs):
            os.fstat(lock_fd)  # descriptor must remain open
            must_be_fenced()
            return 0.01

        with patch.object(cube, "snapshot_other_cycles", side_effect=lambda *a: (must_be_fenced(), [])[1]), \
             patch.object(cube, "input_signature", side_effect=lambda *a: (must_be_fenced(), {"input": 1})[1]), \
             patch.object(cube, "output_signature", return_value={"output": 2}), \
             patch.object(cube, "materialized_cycles", return_value={"2024:municipal": (1, 10)}), \
             patch.object(cube, "revoke_certificate", side_effect=must_be_fenced), \
             patch.object(cube, "issue_certificate", side_effect=must_be_fenced), \
             patch.object(cube, "dbt_call", side_effect=fake_dbt):
            result = cube.certify(self.args, self.db, self.source)
        self.assertEqual(result["cycle"], "2024:municipal")
        self.assertGreaterEqual(len(attempts), 6)
        self.assertEqual(self.exclusive_probe(), 0, "certifier did not release lock")

    def test_db_subprocess_must_inherit_the_exclusive_descriptor(self):
        with patch.object(cube.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            cube.dbt_call("dbt", "run", self.db, "512MB", "{}", 5, lock_fd=21)
        self.assertEqual(run.call_args.kwargs["pass_fds"], (21,))
        self.assertTrue(run.call_args.kwargs["start_new_session"])
        with self.assertRaises(TypeError):
            cube.dbt_call("dbt", "run", self.db, "512MB", "{}", 5)


if __name__ == "__main__":
    unittest.main()
