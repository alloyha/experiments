"""Offline fail-closed stage tests; optional real DuckDB fixture when installed."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from scripts import stage_candidate_cube as mod
from scripts import consolidate_candidate_cube_transfer as evidence


class StageFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main = self.root / "tse_analytics.duckdb"
        self.preview = self.root / "tse_event_registry_preview.duckdb"
        self.fact = self.root / "facts"
        self.out = self.root / "tse_analytics_cube_v1.duckdb"
        self.report = self.root / "cube_stage_report.json"
        self.manifest_path = self.root / "manifest.json"
        self.main.write_bytes(b"approved-Gold-before-staging")
        self.preview.write_bytes(b"approved-preview-before-staging")
        for db in (self.main, self.preview):
            Path(str(db) + ".build.lock").write_bytes(b"")
        self.cycles = {}
        for i, cycle in enumerate(evidence.CYCLES):
            partition = evidence.partition_for(self.fact, cycle)
            partition.mkdir(parents=True)
            (partition / ".publish.lock").write_bytes(b"")
            (partition / "data.parquet").write_bytes((cycle + "payload").encode())
            self.cycles[cycle] = {
                "rows": i + 1, "nominal_votes": (i+1)*10,
                "mismatch_rows": 0, "columns": 29,
                "source_signature": evidence.publication_signature(partition),
            }
        self.manifest = {
            "result": "PASS", "scope": mod.READ_SCOPE,
            "main": str(self.main), "preview": str(self.preview),
            "source_root": str(self.fact),
            "main_sha256": evidence.digest(self.main),
            "preview_sha256": evidence.digest(self.preview),
            "cycles": self.cycles, "cycle_count": 5,
            "total_rows": 15, "total_nominal_votes": 150,
        }
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.manifest_sha = evidence.digest(self.manifest_path)

    def kwargs(self):
        return {
            "main": self.main, "preview": self.preview,
            "root": self.fact, "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha,
            "output": self.out, "report": self.report,
            "memory_limit": "4GB", "lock_timeout": 2.0,
            "acknowledgements": True,
        }

    def fake_import(self, *_args, **_kwargs):
        return "test", {
            c: {"rows": entry["rows"], "nominal_votes": entry["nominal_votes"],
                "missing_rows": 0, "extra_rows": 0}
            for c, entry in self.cycles.items()
        }


class StageContractTests(StageFixture):
    def test_manifest_matches_five_cycle_totals(self):
        mod.check_identity(self.main, self.preview, self.fact, self.manifest)
        self.assertEqual(sum(x["rows"] for x in self.manifest["cycles"].values()), 15)

    def test_incomplete_or_corrupt_manifest_rejected(self):
        self.manifest["cycles"].pop("2026:general")
        with self.assertRaisesRegex(mod.StageError, "five-cycle"):
            mod.check_identity(self.main, self.preview, self.fact, self.manifest)

    def test_wrong_totals_or_schema_rejected(self):
        self.manifest["total_nominal_votes"] += 1
        with self.assertRaisesRegex(mod.StageError, "totals"):
            mod.check_identity(self.main, self.preview, self.fact, self.manifest)
        self.manifest["total_nominal_votes"] -= 1
        self.manifest["cycles"]["2024:municipal"]["columns"] = 28
        with self.assertRaisesRegex(mod.StageError, "bad manifest cycle"):
            mod.check_identity(self.main, self.preview, self.fact, self.manifest)

    def test_wrong_scope_or_database_identity_rejected(self):
        self.manifest["scope"] = "untrusted"
        with self.assertRaisesRegex(mod.StageError, "identity"):
            mod.check_identity(self.main, self.preview, self.fact, self.manifest)

    def test_output_must_not_be_main_preview_or_existing(self):
        for output in (self.main, self.preview):
            with self.assertRaises(mod.StageError):
                mod.assert_destination(self.main, self.preview, output,
                                       self.report, self.manifest_path)
        self.out.write_bytes(b"occupied")
        with self.assertRaisesRegex(mod.StageError, "already exists"):
            mod.assert_destination(self.main, self.preview, self.out,
                                   self.report, self.manifest_path)

    def test_invalid_manifest_sha_rejected_before_stage(self):
        with self.assertRaisesRegex(mod.StageError, "manifest SHA-256 mismatch"):
            mod.stage_cube(**{**self.kwargs(), "manifest_sha256": "0"*64})
        self.assertFalse(self.out.exists())

    def test_ack_required_before_stage(self):
        with self.assertRaisesRegex(mod.StageError, "ack-quiescent"):
            mod.stage_cube(**{**self.kwargs(), "acknowledgements": False})
        self.assertFalse(self.out.exists())

    def test_wal_and_missing_stable_lock_fail_closed(self):
        Path(str(self.main) + ".wal").write_bytes(b"WAL")
        with self.assertRaisesRegex(mod.StageError, "WAL"):
            mod.verify_sources(self.main, self.preview, self.fact, self.manifest)
        Path(str(self.main) + ".wal").unlink()
        (evidence.partition_for(self.fact,"2018:general") / ".publish.lock").unlink()
        with self.assertRaises(evidence.EvidenceError):
            with evidence.noncreating_shared_locks(
                mod.locks_for(self.main,self.preview,self.fact), 0.2):
                pass
        self.assertFalse((evidence.partition_for(self.fact,"2018:general") / ".publish.lock").exists())

    def test_parquet_drift_denied(self):
        (evidence.partition_for(self.fact,"2020:municipal")/"data.parquet").write_bytes(b"DIFFERENT")
        with self.assertRaisesRegex(mod.StageError, "Parquet publication drift"):
            mod.verify_sources(self.main,self.preview,self.fact,self.manifest)

    def test_exact_sql_includes_duplicate_sensitive_comparison(self):
        q=mod.exact_delta_query("main.cube_candidate_performance_municipality", "preview_source.main.cube_candidate_performance_municipality")
        self.assertIn("EXCEPT ALL",q)
        self.assertIn("WHERE cycle_id = ?",q)
        self.assertEqual(q.count("?"),2)
        self.assertNotIn("hash(",q.lower())

    def test_copy_is_durable_and_verified(self):
        dest=self.root/".staged.duckdb"
        mod.durable_copy(self.main,dest,evidence.digest(self.main))
        self.assertEqual(dest.read_bytes(), self.main.read_bytes())
        with self.assertRaisesRegex(mod.StageError,"staged original"):
            mod.durable_copy(self.preview, self.root/".bad.duckdb", "0"*64)

    def test_success_publication_does_not_touch_sources(self):
        old_main,old_preview=self.main.read_bytes(),self.preview.read_bytes()
        with mock.patch.object(mod,"import_and_verify",side_effect=self.fake_import),\
             mock.patch.dict(sys.modules,{"duckdb":mock.Mock()}):
            record=mod.stage_cube(**self.kwargs())
        self.assertEqual(record["result"],"PASS")
        self.assertEqual(record["scope"],mod.STAGE_SCOPE)
        self.assertEqual(record["manifest_sha256"],self.manifest_sha)
        self.assertEqual(record["total_rows"],15)
        self.assertTrue(self.report.is_file())
        self.assertEqual(self.out.read_bytes(),old_main) # synthetic import does not mutate clone
        self.assertEqual(self.main.read_bytes(),old_main)
        self.assertEqual(self.preview.read_bytes(),old_preview)
        self.assertEqual(json.loads(self.report.read_text())["output_sha256"], evidence.digest(self.out))
        self.assertFalse(list(self.root.glob(".*incoming-*.duckdb")))
        with self.assertRaisesRegex(mod.StageError,"already exists"):
            mod.stage_cube(**self.kwargs())

    def test_failed_import_leaves_only_hidden_orphan_not_main(self):
        old=self.main.read_bytes()
        with mock.patch.object(mod,"import_and_verify",side_effect=RuntimeError("injected")),\
             mock.patch.dict(sys.modules,{"duckdb":mock.Mock()}):
            with self.assertRaisesRegex(RuntimeError,"injected"):
                mod.stage_cube(**self.kwargs())
        self.assertFalse(self.out.exists())
        self.assertFalse(self.report.exists())
        self.assertEqual(self.main.read_bytes(),old)
        self.assertEqual(len(list(self.root.glob(".*incoming-*.duckdb"))),1)

    @unittest.skipUnless(os.name=="posix", "POSIX SIGKILL required")
    def test_real_sigkill_before_publication_does_not_touch_gold(self):
        # Kill child after durable clone and before the fake import/publish.
        source=textwrap.dedent('''
            import os,signal,sys
            from pathlib import Path
            from unittest import mock
            from scripts import stage_candidate_cube as mod
            mod.import_and_verify=lambda *args,**kwargs:os.kill(os.getpid(),signal.SIGKILL)
            with mock.patch.dict(sys.modules,{'duckdb':mock.Mock()}):
                mod.stage_cube(main=Path(sys.argv[1]),preview=Path(sys.argv[2]),
                    root=Path(sys.argv[3]),manifest_path=Path(sys.argv[4]),
                    manifest_sha256=sys.argv[5],output=Path(sys.argv[6]),
                    report=Path(sys.argv[7]),memory_limit='4GB',
                    lock_timeout=3,acknowledgements=True)
        ''')
        before_main=self.main.read_bytes()
        result=subprocess.run([sys.executable,"-c",source,str(self.main),str(self.preview),
                               str(self.fact),str(self.manifest_path),self.manifest_sha,
                               str(self.out),str(self.report)],
                              capture_output=True,timeout=10)
        self.assertEqual(result.returncode,-signal.SIGKILL,result.stderr)
        self.assertEqual(self.main.read_bytes(),before_main)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.report.exists())
        self.assertTrue(list(self.root.glob(".*incoming-*.duckdb")))
        # Killed worker releases flock; next read lock succeeds.
        with evidence.noncreating_shared_locks(mod.locks_for(self.main,self.preview,self.fact), 1):
            pass


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "DuckDB unavailable in test runtime")
class RealDuckDBCloneTests(StageFixture):
    def test_small_five_cycle_copy_and_exact_schema(self):
        import duckdb
        # Replace synthetic bytes with real databases before manifest rehash.
        self.main.unlink()
        self.preview.unlink()
        con = duckdb.connect(str(self.main))
        try:
            con.execute("CREATE TABLE existing_gold AS SELECT 1 AS id")
        finally:
            con.close()
        col_def = "cycle_id VARCHAR, nominal_valid_votes BIGINT, " + ", ".join(
            f"attribute_{i} VARCHAR" for i in range(27))
        con = duckdb.connect(str(self.preview))
        try:
            con.execute(f"CREATE TABLE {mod.TABLE} ({col_def})")
            for i,c in enumerate(evidence.CYCLES):
                params=[c,(i+1)*10]+[f"v{i}_{k}" for k in range(27)]
                con.execute(f"INSERT INTO {mod.TABLE} VALUES ({','.join('?' for _ in params)})",params)
        finally:
            con.close()
        self.manifest["main_sha256"]=evidence.digest(self.main)
        self.manifest["preview_sha256"]=evidence.digest(self.preview)
        for i,cycle in enumerate(evidence.CYCLES):
            self.manifest["cycles"][cycle]["rows"] = 1
        self.manifest["total_rows"] = 5
        self.save_manifest()
        record=mod.stage_cube(**self.kwargs())
        self.assertEqual(record["result"],"PASS")
        self.assertEqual(record["total_rows"],5)
        self.assertEqual(record["total_nominal_votes"],150)
        con = duckdb.connect(str(self.out),read_only=True)
        try:
            self.assertEqual(con.execute(f"SELECT count(*) FROM {mod.TABLE}").fetchone()[0],5)
            self.assertEqual(con.execute("SELECT id FROM existing_gold").fetchone()[0],1)
        finally:
            con.close()


if __name__ == '__main__':
    unittest.main()
