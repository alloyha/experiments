"""GG-009 end-to-end: real dbt mutates an isolated DuckDB Gold table.

The certifier/reader must share the gateway's preexisting POSIX warehouse lock.
This test never opens canonical electoral warehouse or raw data. It does not
promote GG-009 because uncoordinated DuckDB callers can still bypass the lock.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts import graph_guarded_read as guarded
from scripts import graph_freshness as freshness
from scripts.certify_gold_graph_cycle import SCOPED_SOURCES, quote_sql
from test_gold_graph_invariants import GoldGraphTests
from scripts import run_coordinated_dbt as gateway

CYCLE = "2024:municipal"

# The launcher waits *after* the gateway has acquired the warehouse flock and
# *before* exec-ing actual dbt. This is a deterministic concurrency boundary.
# Using exec keeps the same process and inherited flock fd alive in real dbt.
HOLD_THEN_EXEC_DBT = '''#!{python}
import os, pathlib, sys, time
start = pathlib.Path(os.environ["GG009_START"])
release = pathlib.Path(os.environ["GG009_RELEASE"])
start.write_text(str(os.getpid()))  # PID is also the dbt child process-group ID
deadline = time.monotonic() + 20
while not release.exists() and time.monotonic() < deadline:
    time.sleep(.02)
if not release.exists():
    raise SystemExit("fixture writer release timed out")
real_dbt = os.environ["GG009_REAL_DBT"]
os.execv(real_dbt, [real_dbt, *sys.argv[1:]])
'''


class RealDbtGraphHandoffTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("DuckDB is required for the real dbt integration test")
        self.dbt = Path(sys.executable).with_name("dbt")
        if not self.dbt.is_file() or not os.access(self.dbt, os.X_OK):
            self.skipTest("Project dbt executable is not installed")
        self.tmp = tempfile.TemporaryDirectory(prefix="gg009-dbt-handoff-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "warehouse.duckdb"
        self.source = self.root / "facts"
        self.partition = self.source / "election_type=municipal/year=2024"
        self.partition.mkdir(parents=True)
        self.certs = self.root / "certs"
        self.cycle = CYCLE
        self.project = self.root / "minimal-dbt"
        models = self.project / "models"
        models.mkdir(parents=True)
        (self.project / "dbt_project.yml").write_text(
            "name: gg009_fixture\nversion: '1.0'\nconfig-version: 2\n"
            "profile: gg009_fixture\nmodel-paths: ['models']\n",
            encoding="utf-8",
        )
        (self.project / "profiles.yml").write_text(
            "gg009_fixture:\n  target: dev\n  outputs:\n    dev:\n"
            "      type: duckdb\n      path: \"{{ env_var('TSE_DUCKDB_PATH') }}\"\n"
            "      schema: main\n      threads: 1\n",
            encoding="utf-8",
        )
        (models / "dim_party.sql").write_text(
            "{{ config(materialized='table') }}\n"
            "select cycle_id, election_id, party_number, "
            "party_name || '_dbt_changed' as party_name\n"
            "from main.source_dim_party\n",
            encoding="utf-8",
        )
        self.launcher = self.root / "hold-then-dbt"
        self.launcher.write_text(HOLD_THEN_EXEC_DBT.format(python=sys.executable),
                                 encoding="utf-8")
        self.launcher.chmod(0o700)

    def _seed_and_certify(self):
        """Use the real PR54 graph fixture, detached onto one scratch DuckDB."""
        fixture = GoldGraphTests("test_graph_projection_conserves_relationships")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.db.execute(f"ATTACH {quote_sql(str(self.db))} AS evidence")
        try:
            for table in SCOPED_SOURCES:
                fixture.db.execute(
                    f"CREATE TABLE evidence.{table} AS SELECT * FROM main.{table}"
                )
            fixture.db.execute(
                "CREATE TABLE evidence.source_dim_party AS "
                "SELECT * FROM main.dim_party"
            )
        finally:
            fixture.db.execute("DETACH evidence")
        publication = self.partition / "data.parquet"
        fixture.db.execute(
            "COPY (SELECT * EXCLUDE (cycle_id) FROM fact_candidate_votes) "
            f"TO {quote_sql(str(publication))} (FORMAT PARQUET)"
        )
        command = [
            sys.executable, str(freshness.ROOT / "scripts/graph_freshness.py"),
            "--db", str(self.db), "--source-root", str(self.source),
            "--attestations-dir", str(self.certs),
            "--cycle", self.cycle, "--memory-limit", "512MB", "certify",
        ]
        done = subprocess.run(command, cwd=freshness.ROOT, capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def _guarded_read(self):
        return guarded.read_graph(
            db=self.db, source_root=self.source,
            attestations_dir=self.certs, cycle=self.cycle,
            relation="edges", kind="VOTED_FOR_CANDIDATE",
            limit=3, memory_limit="512MB", lock_timeout=0,
        )

    def _await_file(self, path, process, timeout=10):
        deadline = time.monotonic() + timeout
        while not path.is_file() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(path.is_file(), "Writer failed before acquiring lock")

    @staticmethod
    def _stop_writer(writer: subprocess.Popen, started: Path) -> None:
        """Reap both isolated process groups on failure/timeout.

        Gateway and dbt use distinct sessions. Killing only the gateway may
        leave dbt holding the inherited flock FD and stdout open.
        """
        if started.is_file():
            try:
                child_pid = int(started.read_text(encoding="utf-8").strip())
                # Inherited marker belongs to the exec-preserved child session.
                # Refuse to signal the test runner's own process group.
                if (child_pid > 1 and child_pid != os.getpgrp()
                        and os.getpgid(child_pid) == child_pid):
                    os.killpg(child_pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError):
                pass  # Child already terminated, or marker was incomplete.
        if writer.poll() is None:
            writer.kill()
        writer.wait(timeout=5)

    def test_guarded_reader_handoff_real_dbt_commit_stales_certificate(self):
        """Reader -> exclusive writer -> stale proof denial, all on disk."""
        import duckdb
        self._seed_and_certify()
        original = self._guarded_read()
        self.assertEqual(original["status"], "cooperatively_guarded_not_production_certified")
        self.assertTrue(original["rows"])

        started = self.root / "writer-started"
        release = self.root / "release-dbt"
        command = [
            sys.executable, str(Path(gateway.__file__).resolve()),
            "--db", str(self.db), "--dbt-executable", str(self.launcher),
            "--artifacts-root", str(self.root / "dbt-runs"),
            "--lock-timeout", "12", "--", "run",
            "--project-dir", str(self.project),
            "--profiles-dir", str(self.project),
            "--select", "dim_party",
        ]
        env = {**os.environ, "GG009_START": str(started),
               "GG009_RELEASE": str(release), "GG009_REAL_DBT": str(self.dbt)}
        env.pop("VIRTUAL_ENV", None)
        env.pop("DBT_PROFILES_DIR", None)
        env.pop("DBT_PROJECT_DIR", None)
        writer = None
        writer_log = self.root / "writer.log"
        try:
            # The writer must not reach its child while this reader owns SH locks.
            with guarded.graph_read_locks(self.db, self.source, self.cycle, timeout=0):
                # File-backed output cannot hold communicate() hostage if
                # the gateway dies before its separate dbt session exits.
                with writer_log.open("w", encoding="utf-8") as log:
                    writer = subprocess.Popen(
                        command, cwd=str(self.root), env=env,
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                time.sleep(.3)
                self.assertIsNone(writer.poll(), "Writer failed while reader held the lock")
                self.assertFalse(started.exists(), "Writer bypassed shared reader lock")

            # Writer now takes EX lock, and holds it during the real dbt child.
            self._await_file(started, writer)
            with self.assertRaisesRegex(TimeoutError, "graph read lock"):
                self._guarded_read()
            release.touch()
            try:
                writer.wait(timeout=90)
            except subprocess.TimeoutExpired:
                output = writer_log.read_text(encoding="utf-8", errors="replace")
                self.fail(f"Real dbt writer timed out after release; output:\n{output[-8000:]}")
            output = writer_log.read_text(encoding="utf-8", errors="replace")
            self.assertEqual(writer.returncode, 0, output)
        finally:
            release.touch()
            if writer is not None and writer.poll() is None:
                self._stop_writer(writer, started)

        with duckdb.connect(str(self.db), read_only=True) as connection:
            changed = connection.execute(
                "SELECT count(*) FROM dim_party "
                "WHERE cycle_id = ? AND ends_with(party_name, '_dbt_changed')",
                [self.cycle],
            ).fetchone()[0]
        self.assertGreater(changed, 0, "Real dbt did not mutate the Gold dimension")

        # The old certificate cannot authorize rows from the new Gold state.
        with self.assertRaisesRegex(Exception, "freshness failure"):
            self._guarded_read()


if __name__ == "__main__":
    unittest.main()
