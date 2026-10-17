"""PR37: real subprocess / POSIX flock tests on isolated DuckDB databases."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from scripts import build_candidate_fact_partition as job
from scripts import write_candidate_fact_partition as publisher


FAKE_DBT = '''#!{python}
import json, os, pathlib, sys, time
import duckdb
args = sys.argv
if '--vars' not in args or '+int_candidate_fact_partition' not in args:
    raise SystemExit('incorrect dbt invocation')
variables = json.loads(args[args.index('--vars')+1])
if os.getenv('ARGS_CAPTURE'):
    pathlib.Path(os.environ['ARGS_CAPTURE']).write_text(json.dumps(variables))
if os.getenv('START_FILE'):
    pathlib.Path(os.environ['START_FILE']).write_text(str(os.getpid()))
if os.getenv('HOLD_UNTIL'):
    target = pathlib.Path(os.environ['HOLD_UNTIL'])
    deadline = time.monotonic() + 15
    while not target.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    if not target.exists():
        raise SystemExit('fake dbt hold timeout')
if os.getenv('FAIL_BUILD'):
    raise SystemExit(42)
with duckdb.connect(os.environ['TSE_DUCKDB_PATH']) as con:
    con.execute('create or replace table main.int_candidate_fact_partition(election_year integer, election_type varchar, election_id integer, round_number integer, uf varchar, municipality_code varchar, office_code varchar, is_transit_vote boolean, candidate_id varchar, zone integer, nominal_votes integer)')
    con.execute("insert into main.int_candidate_fact_partition values (?, ?, 1, 1, 'GO', '5208707', '13', false, 'C1', 10, ?)",
                [variables['incremental_years'][0], variables['incremental_election_types'][0], int(os.getenv('VOTES', '17'))])
'''


class WarehouseBuildCoordinationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pr37-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "warehouse.duckdb"
        self.other_db = self.root / "other.duckdb"
        self.facts = self.root / "facts"
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.project = self.root / "tse_dbt"
        self.project.mkdir()
        self.bin = self.root / "fake-dbt"
        self.bin.write_text(FAKE_DBT.format(python=sys.executable))
        self.bin.chmod(0o755)
        self.partition = self.facts / "election_type=general/year=2026"
        self.pointer = self.partition / "data.parquet"

    def config(self, db=None, fact_root=None):
        return {
            "db_path": db or self.db,
            "fact_root": fact_root or self.facts,
            "raw_root": self.raw,
            "project_dir": self.project,
            "profiles_dir": self.project,
            "dbt_executable": self.bin,
            "year": 2026,
            "election_type": "general",
            "warehouse_lock_timeout": 0.0,
        }

    def build(self, **overrides):
        cfg = self.config()
        cfg.update(overrides)
        return job.build_and_publish(**cfg)

    def launch(self, db=None, facts=None, *, env=None, timeout=5):
        cfg = self.config(db=db, fact_root=facts)
        args = [
            sys.executable, str(Path(job.__file__).resolve()),
            "--db", str(cfg["db_path"]),
            "--root", str(cfg["fact_root"]),
            "--raw-root", str(cfg["raw_root"]),
            "--project-dir", str(cfg["project_dir"]),
            "--profiles-dir", str(cfg["profiles_dir"]),
            "--dbt-executable", str(cfg["dbt_executable"]),
            "--year", "2026", "--election-type", "general",
            "--warehouse-lock-timeout", str(timeout),
        ]
        return subprocess.Popen(args, env={**os.environ, **(env or {})},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True, text=True)

    def await_file(self, path, timeout=7):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(.02)
        self.fail(f"timeout awaiting {path}")

    def published_votes(self, pointer=None):
        with duckdb.connect() as con:
            return con.execute("select nominal_votes from read_parquet(?)",
                               [str(pointer or self.pointer)]).fetchone()[0]

    def test_build_publishes_expected_scope_and_variables(self):
        capture = self.root / "vars.json"
        with mock.patch.dict(os.environ, {"ARGS_CAPTURE": str(capture), "VOTES": "31"}):
            self.build()
        self.assertEqual(self.published_votes(), 31)
        self.assertTrue(self.pointer.is_symlink())
        variables = json.loads(capture.read_text())
        self.assertEqual(variables["incremental_years"], [2026])
        self.assertEqual(variables["incremental_election_types"], ["general"])
        self.assertTrue(variables["build_candidate_fact_partition"])

    def test_failed_ctas_does_not_advance_publication(self):
        self.build()
        generation = publisher.current_generation(self.partition)
        with mock.patch.dict(os.environ, {"FAIL_BUILD": "1"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.build()
        self.assertEqual(publisher.current_generation(self.partition), generation)
        self.assertEqual(self.published_votes(), 17)
        # The lock must also be available after a failed child.
        with mock.patch.dict(os.environ, {"VOTES": "21"}):
            self.build()
        self.assertEqual(self.published_votes(), 21)

    def test_same_warehouse_rejects_second_ctas_while_first_is_active(self):
        started = self.root / "started"
        release = self.root / "release"
        worker = self.launch(env={"START_FILE": str(started), "HOLD_UNTIL": str(release)})
        try:
            self.await_file(started)
            with self.assertRaisesRegex(TimeoutError, "warehouse lock"):
                self.build()
            self.assertFalse(self.pointer.exists())
            release.touch()
            out, err = worker.communicate(timeout=10)
            self.assertEqual(worker.returncode, 0, err)
            self.assertEqual(self.published_votes(), 17)
        finally:
            release.touch()
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=5)

    def test_different_warehouses_are_independent(self):
        started = self.root / "started"
        release = self.root / "release"
        worker = self.launch(env={"START_FILE": str(started), "HOLD_UNTIL": str(release)})
        try:
            self.await_file(started)
            second_facts = self.root / "other-facts"
            with mock.patch.dict(os.environ, {"VOTES": "23"}):
                self.build(db_path=self.other_db, fact_root=second_facts)
            pointer = second_facts / "election_type=general/year=2026/data.parquet"
            self.assertEqual(self.published_votes(pointer), 23)
            self.assertFalse(self.pointer.exists())
        finally:
            release.touch()
            if worker.poll() is None:
                worker.communicate(timeout=10)
            else:
                worker.communicate(timeout=5)

    def test_sigkill_parent_cannot_release_child_warehouse_lock_early(self):
        started = self.root / "started"
        release = self.root / "release"
        parent = self.launch(env={"START_FILE": str(started), "HOLD_UNTIL": str(release)})
        child_pid = None
        try:
            self.await_file(started)
            child_pid = int(started.read_text())
            parent.kill()
            parent.communicate(timeout=5)
            with self.assertRaisesRegex(TimeoutError, "warehouse lock"):
                self.build()
            self.assertFalse(self.pointer.exists())
        finally:
            release.touch()
            if child_pid is not None:
                # The orphaned fake dbt exits once it sees the release marker.
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    try:
                        with job.warehouse_write_lock(self.db, timeout=0):
                            break
                    except TimeoutError:
                        time.sleep(.05)
                else:
                    os.kill(child_pid, signal.SIGKILL)
                    self.fail("orphan child did not release warehouse lock")
            if parent.poll() is None:
                parent.kill()
            parent.communicate(timeout=5)
        self.build()
        self.assertEqual(self.published_votes(), 17)

    def test_publisher_failure_releases_warehouse_lock_without_pointer_change(self):
        self.build()
        generation = publisher.current_generation(self.partition)
        with mock.patch.object(job.publisher, "publish_partition", side_effect=RuntimeError("injected publish failure")):
            with self.assertRaisesRegex(RuntimeError, "injected publish failure"):
                self.build()
        self.assertEqual(publisher.current_generation(self.partition), generation)
        self.build()
        self.assertNotEqual(publisher.current_generation(self.partition), generation)

    def test_lock_timeout_and_invalid_arguments(self):
        with self.assertRaises(ValueError):
            with job.warehouse_write_lock(self.db, timeout=-1):
                pass
        with self.assertRaises(ValueError):
            self.build(year=0)
        with self.assertRaises(FileNotFoundError):
            self.build(dbt_executable=self.root / "not-found")

    def test_warehouse_lock_identity_is_canonical_path(self):
        alias = self.root / "subdirectory" / ".." / self.db.name
        with job.warehouse_write_lock(self.db, timeout=0):
            with self.assertRaises(TimeoutError):
                with job.warehouse_write_lock(alias, timeout=0):
                    pass

    def test_makefile_delegates_to_coordinator(self):
        text = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        recipe = text.split("candidate-fact-partition: dbt-deps", 1)[1].split("\nunit:", 1)[0]
        self.assertIn("scripts/build_candidate_fact_partition.py", recipe)
        self.assertIn("--dbt-executable", recipe)
        self.assertNotIn("$(DBT) run", recipe)


if __name__ == "__main__":
    unittest.main()
