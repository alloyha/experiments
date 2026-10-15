"""GG-009 writer gateway regression: real POSIX locks, fake isolated dbt child."""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from scripts import run_coordinated_dbt as gateway
from scripts.build_candidate_fact_partition import warehouse_write_lock


FAKE_DBT = '''#!{python}
import json, os, pathlib, sys, time
args = sys.argv[1:]
if os.environ.get('CAPTURE'):
    pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({{'args': args, 'warehouse': os.getenv('TSE_DUCKDB_PATH')}}))
if os.environ.get('START_FILE'):
    pathlib.Path(os.environ['START_FILE']).write_text(str(os.getpid()))
if os.environ.get('HOLD_UNTIL'):
    path = pathlib.Path(os.environ['HOLD_UNTIL'])
    for _ in range(400):
        if path.exists(): break
        time.sleep(.025)
    else: sys.exit(19)
if '--target-path' in args:
    target = pathlib.Path(args[args.index('--target-path') + 1])
    target.mkdir(parents=True, exist_ok=True)
    (target/'run_results.json').write_text(json.dumps({{'run': args[0]}}))
sys.exit(int(os.getenv('DBT_EXIT', '0')))
'''


class CoordinatedDbtGatewayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="gg009-writer-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / "warehouse.duckdb"
        self.db.write_bytes(b"fake-db")
        self.other = self.root / "other.duckdb"
        self.other.write_bytes(b"fake-db")
        self.artifacts = self.root / "artifacts"
        self.dbt = self.root / "fake-dbt"
        self.dbt.write_text(FAKE_DBT.format(python=sys.executable))
        self.dbt.chmod(0o755)
        self.gateway = Path(gateway.__file__).resolve()

    def args(self, db=None, command="build"):
        return [sys.executable, str(self.gateway), "--db", str(db or self.db),
                "--dbt-executable", str(self.dbt),
                "--artifacts-root", str(self.artifacts), "--lock-timeout", "0", "--", command,
                "--project-dir", str(self.root)]

    def await_file(self, path: Path):
        for _ in range(200):
            if path.exists(): return
            time.sleep(.02)
        self.fail(f"Child did not start: {path}")

    def test_direct_script_entrypoint_imports_project_package_from_elsewhere(self):
        """Regression: direct invocation must not depend on cwd or PYTHONPATH."""
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        run = subprocess.run(self.args(), cwd=self.root, env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue((self.artifacts / "latest_run_results.json").is_file())

    def test_build_inherits_lock_and_isolates_artifacts(self):
        capture = self.root / "args.json"
        with mock.patch.dict(os.environ, {"CAPTURE": str(capture)}):
            self.assertEqual(gateway.main(self.args()[2:]), 0)
        import json
        data = json.loads(capture.read_text())
        self.assertEqual(data["warehouse"], str(self.db.resolve()))
        self.assertIn("--target-path", data["args"])
        self.assertIn("--log-path", data["args"])
        self.assertEqual(json.loads((self.artifacts / "latest_run_results.json").read_text()),
                         {"run": "build"})

    def test_writer_waits_for_existing_graph_reader(self):
        lock = self.db.with_name(self.db.name + ".build.lock")
        with lock.open("a+b") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            worker = subprocess.run(self.args(), capture_output=True, text=True, timeout=10)
            self.assertEqual(worker.returncode, 1, worker.stderr)
            self.assertIn("warehouse lock", worker.stderr)
            fcntl.flock(fh, fcntl.LOCK_UN)
        self.assertEqual(subprocess.run(self.args(), timeout=10).returncode, 0)

    def test_sigkill_parent_does_not_release_live_dbt_child_lock(self):
        start = self.root / "started"
        release = self.root / "release"
        env = {**os.environ, "START_FILE": str(start), "HOLD_UNTIL": str(release)}
        parent = subprocess.Popen(self.args(), env=env, start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        child = None
        try:
            self.await_file(start)
            child = int(start.read_text())
            parent.kill()
            parent.wait(timeout=5)
            with self.assertRaises(TimeoutError):
                with warehouse_write_lock(self.db, timeout=0): pass
        finally:
            release.touch()
            if parent.poll() is None: parent.kill()
            parent.wait(timeout=5)
            for _ in range(150):
                try:
                    with warehouse_write_lock(self.db, timeout=0): break
                except TimeoutError:
                    time.sleep(.03)
            else:
                if child: os.kill(child, signal.SIGKILL)
                self.fail("dbt child left orphaned warehouse lock")
        self.assertEqual(subprocess.run(self.args(), timeout=10).returncode, 0)

    def test_two_warehouses_do_not_share_writer_lock(self):
        with warehouse_write_lock(self.db, timeout=0):
            self.assertEqual(subprocess.run(self.args(db=self.other), timeout=10).returncode, 0)
            blocked = subprocess.run(self.args(), capture_output=True, text=True, timeout=10)
            self.assertEqual(blocked.returncode, 1)

    def test_compile_and_deps_are_non_writers(self):
        # Should not acquire the warehouse lock or alter target path.
        capture = self.root / "compile.json"
        with warehouse_write_lock(self.db, timeout=0), \
             mock.patch.dict(os.environ, {"CAPTURE": str(capture)}):
            self.assertEqual(gateway.main(self.args(command="compile")[2:]), 0)
        import json
        self.assertNotIn("--target-path", json.loads(capture.read_text())["args"])

    def test_nonzero_dbt_propagates_and_does_not_leave_stale_run_results(self):
        with mock.patch.dict(os.environ, {"DBT_EXIT": "7"}):
            self.assertEqual(gateway.main(self.args()[2:]), 7)
        self.assertFalse((self.artifacts / "latest_run_results.json").exists())
        self.assertEqual(len(list(self.artifacts.glob("*/target/run_results.json"))), 1)
        # Invalid CLI must be rejected before invoking dbt.
        self.assertEqual(gateway.main(self.args()[2:]+["--target-path", "unsafe"]), 1)

    def test_makefile_db_is_single_gateway_for_all_targets(self):
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        self.assertIn("scripts/run_coordinated_dbt.py", makefile)
        self.assertIn("--dbt-executable", makefile)
        for target in ("dbt-bronze", "dbt-silver", "dbt-gold", "semantic", "dbt-build",
                       "dbt-full-refresh", "test-only", "pr-gate"):
            self.assertIn(f"{target}:", makefile)
            recipe = makefile.split(f"{target}:", 1)[1].split("\n", 1)[1]
            self.assertIn("$(DBT)", recipe.split("\n#", 1)[0].split("\n\n", 1)[0])


if __name__ == "__main__":
    unittest.main()
