"""Offline tests for five-cycle, read-only candidate-cube evidence closeout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# Repository root is on sys.path under `python -m unittest discover`.
from scripts import consolidate_candidate_cube_transfer as mod


class CandidateCubeCloseoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.main = self.base / "main.duckdb"
        self.preview = self.base / "preview.duckdb"
        self.root = self.base / "facts"
        self.root.mkdir()
        self.main.write_bytes(b"main")
        self.preview.write_bytes(b"preview")
        self.main_sha = mod.digest(self.main)
        self.preview_sha = mod.digest(self.preview)
        for db in (self.main, self.preview):
            mod.lock_path(db).write_bytes(b"")
        self.reports = {}
        self.counts = {cycle: 12 + i for i, cycle in enumerate(mod.CYCLES)}
        for i, cycle in enumerate(mod.CYCLES):
            part = mod.partition_for(self.root, cycle)
            part.mkdir(parents=True)
            (part / ".publish.lock").write_bytes(b"")
            (part / "data.parquet").write_bytes(f"file-{cycle}".encode())
            record = {
                "result": "PASS", "scope": mod.SCOPE, "cycle": cycle,
                "main": str(self.main), "preview": str(self.preview),
                "main_sha256": self.main_sha, "preview_sha256": self.preview_sha,
                "physical_parquet": str(part / "data.parquet"),
                "source_signature": mod.publication_signature(part),
                "published_cycles": list(mod.CYCLES),
                "cube_rows_by_cycle": self.counts,
                "selected_cube_rows": self.counts[cycle],
                "selected_nominal_votes": 123 + i,
                "selected_schema_columns": 29,
                "multiset_mismatch_rows": 0,
                "comparison_mode": "signed" if i >= 3 else "except-all",
            }
            path = self.base / f"evidence-{i}.json"
            path.write_text(json.dumps(record))
            self.reports[cycle] = path

    def run_check(self, **overrides):
        kwargs = {"main": self.main, "preview": self.preview, "root": self.root,
                  "report_paths": list(self.reports.values()),
                  "main_sha256": self.main_sha, "preview_sha256": self.preview_sha,
                  "lock_timeout": 0.5}
        kwargs.update(overrides)
        return mod.consolidate(**kwargs)

    def mutate(self, cycle, field, value):
        path = self.reports[cycle]
        data = json.loads(path.read_text())
        data[field] = value
        path.write_text(json.dumps(data))

    def test_five_matching_reports_pass_without_data_mutation(self):
        before = {p:mod.digest(p) for p in (self.main, self.preview)}
        result = self.run_check()
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["cycle_count"], 5)
        self.assertEqual(result["total_rows"], sum(self.counts.values()))
        self.assertEqual(set(result["cycles"]), set(mod.CYCLES))
        for p, sha in before.items():
            self.assertEqual(mod.digest(p), sha)

    def test_wrong_hash_denied(self):
        with self.assertRaisesRegex(mod.EvidenceError, "hash"):
            self.run_check(main_sha256="a" * 64)

    def test_missing_and_duplicate_cycles_denied(self):
        with self.assertRaisesRegex(mod.EvidenceError, "exactly"):
            self.run_check(report_paths=list(self.reports.values())[:4])
        with self.assertRaisesRegex(mod.EvidenceError, "duplicate"):
            paths = list(self.reports.values())
            self.run_check(report_paths=paths[:4] + paths[:1])

    def test_single_cycle_mismatch_denied(self):
        self.mutate("2024:municipal", "multiset_mismatch_rows", 1)
        with self.assertRaisesRegex(mod.EvidenceError, "multiset"):
            self.run_check()

    def test_wrong_columns_denied(self):
        self.mutate("2018:general", "selected_schema_columns", 28)
        with self.assertRaisesRegex(mod.EvidenceError, "multiset"):
            self.run_check()

    def test_source_mutation_denied(self):
        (mod.partition_for(self.root, "2026:general") / "data.parquet").write_bytes(b"new")
        with self.assertRaisesRegex(mod.EvidenceError, "Parquet changed"):
            self.run_check()

    def test_row_inventory_drift_denied(self):
        self.mutate("2020:municipal", "cube_rows_by_cycle", {**self.counts, "2018:general": 999})
        with self.assertRaisesRegex(mod.EvidenceError, "inventory"):
            self.run_check()

    def test_report_identity_denied(self):
        self.mutate("2022:general", "preview_sha256", "a"*64)
        with self.assertRaisesRegex(mod.EvidenceError, "identity"):
            self.run_check()

    def test_warehouse_wal_denied(self):
        Path(str(self.preview) + ".wal").write_bytes(b"uncheckpointed")
        with self.assertRaisesRegex(mod.EvidenceError, "WAL"):
            self.run_check()

    def test_lock_missing_never_created(self):
        missing = mod.partition_for(self.root, "2018:general") / ".publish.lock"
        missing.unlink()
        with self.assertRaisesRegex(mod.EvidenceError, "missing"):
            self.run_check()
        self.assertFalse(missing.exists())

    def test_create_only_manifest_and_main_cli(self):
        output = self.base / "manifest.json"
        argv = ["--main-db", str(self.main), "--preview-db", str(self.preview),
                "--source-root", str(self.root), "--main-sha256", self.main_sha,
                "--preview-sha256", self.preview_sha, "--output", str(output)]
        for p in self.reports.values():
            argv.extend(["--report", str(p)])
        self.assertEqual(mod.main(argv), 0)
        self.assertEqual(json.loads(output.read_text())["total_rows"], sum(self.counts.values()))
        self.assertEqual(mod.main(argv), 2)

    def test_symlink_lock_denied(self):
        lock = mod.partition_for(self.root, "2022:general") / ".publish.lock"
        lock.unlink()
        lock.symlink_to(self.main)
        with self.assertRaisesRegex(mod.EvidenceError, "symlinked"):
            self.run_check()

    def test_busy_writer_lock_denied(self):
        import fcntl
        with self.main.with_name(self.main.name + ".build.lock").open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            # POSIX flock locks are tracked on independently opened file descriptions.
            with self.assertRaisesRegex(mod.EvidenceError, "timed out"):
                self.run_check(lock_timeout=0.01)


if __name__ == "__main__":
    unittest.main()
