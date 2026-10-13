"""Read-only storage report tests with local fixtures, no CKAN or real datasets."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import report_local_storage as report


class LocalStorageReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "data" / "tse"
        self.meta = self.root / "_metadata"
        self.meta.mkdir(parents=True)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.version = (self.raw / "election_type=general" / "year=2026" /
                        "domain=vote_result" / "resource=r1" / "sha256=version1")
        self.source = self.version / "source" / "r1.zip"
        self.extracted = self.version / "extracted" / "votos.csv"
        self.source.parent.mkdir(parents=True)
        self.extracted.parent.mkdir(parents=True)
        self.source.write_bytes(b"source-bytes")
        self.extracted.write_bytes(b"large-unpacked-csv")
        self.source_rel = self.source.relative_to(self.root).as_posix()
        self.extracted_rel = self.extracted.relative_to(self.root).as_posix()
        self.state = {"r1": {
            "source_object": self.source_rel,
            "source_sha256": hashlib.sha256(b"source-bytes").hexdigest(),
            "source_size_bytes": len(b"source-bytes"),
            "extracted_objects": [self.extracted_rel],
            "materialization_state": "present",
            "selection_complete": True,
        }}
        self.pending = {}
        self.index = [{"source_object": self.source_rel, "object": self.extracted_rel}]
        self.dump()

    def dump(self):
        (self.meta / "resource_state.json").write_text(json.dumps(self.state))
        (self.meta / "pending_resources.json").write_text(json.dumps(self.pending))
        (self.meta / "current_objects.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.index)
        )

    def audit(self, **kw):
        return report.produce_report(self.root, self.base / "warehouse.duckdb",
                                     self.base / "candidate-facts", **kw)

    def test_read_only_accounting_and_source_backed_candidate(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        data = self.audit()
        self.assertEqual(data["issues"], [])
        self.assertEqual(data["active_source_bytes"], len(b"source-bytes"))
        self.assertEqual(data["source_backed_cache_candidate_bytes"], len(b"large-unpacked-csv"))
        self.assertEqual(data["currently_referenced_cache_bytes"], len(b"large-unpacked-csv"))
        self.assertEqual(data["source_hashes_verified"], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_pending_journal_is_named_pending_resources(self):
        p = (self.raw / "election_type=general" / "year=2026" /
             "domain=electorate" / "resource=r2" / "sha256=version2" /
             "source" / "r2.zip")
        p.parent.mkdir(parents=True)
        p.write_bytes(b"pending")
        self.pending = {"r2": {"source_object": p.relative_to(self.root).as_posix(),
                                "source_sha256": hashlib.sha256(b"pending").hexdigest(),
                                "source_size_bytes": 7}}
        self.dump()
        data = self.audit(verify_hashes=True)
        self.assertEqual(data["pending_resources"], 1)
        self.assertEqual(data["pending_source_bytes"], 7)
        self.assertEqual(data["source_hashes_verified"], 2)
        self.assertEqual(data["issues"], [])

    def test_evicted_materialization_does_not_become_a_missing_source(self):
        self.extracted.unlink()
        self.index = []
        self.state["r1"]["materialization_state"] = "evicted"
        self.dump()
        data = self.audit()
        self.assertEqual(data["issues"], [])
        self.assertEqual(data["evicted_missing_objects"], 1)
        self.assertEqual(data["active_source_bytes"], len(b"source-bytes"))

    def test_unexpected_missing_cache_is_counted_not_labelled_as_evicted(self):
        self.extracted.unlink()
        self.index = []
        self.dump()
        data = self.audit()
        self.assertEqual(data["unexpected_missing_objects"], 1)
        self.assertEqual(data["evicted_missing_objects"], 0)
        self.assertEqual(data["unavailable_source_objects"], 0)
        self.assertEqual(data["diagnostics"][0]["missing_extracted_objects"],
                         [self.extracted_rel])
        self.assertEqual(data["diagnostics"][0]["source_status"], "file")

    def test_missing_source_reports_concrete_path_and_siblings(self):
        self.source.unlink()
        self.dump()
        data = self.audit()
        self.assertEqual(data["unavailable_source_objects"], 1)
        self.assertEqual(data["unexpected_missing_objects"], 0)
        self.assertIn(self.source_rel, " ".join(data["issues"]))
        self.assertEqual(data["diagnostics"][0]["source_status"], "missing")
        self.assertEqual(data["diagnostics"][0]["source_siblings"], [])

    def test_unrecognized_source_path_is_not_misreported_as_missing(self):
        alternate = self.version / "prepared" / "r1.zip"
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes(self.source.read_bytes())
        self.state["r1"]["source_object"] = alternate.relative_to(self.root).as_posix()
        self.dump()
        data = self.audit()
        self.assertEqual(data["unavailable_source_objects"], 0)
        self.assertEqual(data["unrecognized_source_objects"], 1)
        self.assertEqual(data["diagnostics"][0]["source_status"], "file")
        self.assertEqual(len(data["diagnostics"][0]["source_siblings"]), 1)

    def test_missing_source_and_expected_cache_have_one_diagnostic(self):
        self.source.unlink()
        self.extracted.unlink()
        self.index = []
        self.dump()
        data = self.audit()
        self.assertEqual(data["unavailable_source_objects"], 1)
        self.assertEqual(data["unexpected_missing_objects"], 1)
        self.assertEqual(len(data["diagnostics"]), 1)

    def test_missing_cache_without_evicted_flag_is_detected(self):
        self.extracted.unlink()
        self.index = []
        self.dump()
        self.assertTrue(any("required materialization missing" in x for x in self.audit()["issues"]))

    def test_dangling_physical_index_is_detected(self):
        self.source.unlink()
        self.assertTrue(any("dangling source_object" in x for x in self.audit()["issues"]))

    def test_hash_verification_is_explicit_and_detects_corruption(self):
        self.source.write_bytes(b"source-SPOOF")  # same length as original
        self.state["r1"]["source_size_bytes"] = len(b"source-SPOOF")
        self.dump()
        self.assertFalse(any("SHA-256 mismatch" in x for x in self.audit()["issues"]))
        self.assertTrue(any("SHA-256 mismatch" in x for x in self.audit(verify_hashes=True)["issues"]))

    def test_missing_raw_directory_fails_instead_of_zero_report(self):
        (self.root / "raw").rename(self.root / "no-raw")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.audit()

    def test_unbacked_cache_not_counted_as_candidate(self):
        self.source.unlink()
        data = self.audit()
        self.assertEqual(data["source_backed_cache_candidate_bytes"], 0)
        self.assertEqual(data["unbacked_cache_bytes"], len(b"large-unpacked-csv"))

    def test_absolute_or_traversal_references_rejected(self):
        self.state["r1"]["source_object"] = "../../etc/passwd"
        self.dump()
        self.assertTrue(any("unsafe path" in x for x in self.audit()["issues"]))

    def test_nested_extracted_path_uses_version_source(self):
        nested = self.version / "extracted" / "folder" / "votes.csv"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b"nested")
        self.assertEqual(report.sibling_source(nested, self.raw, {self.source}), self.source)

    def test_make_storage_report_is_read_only_and_visible_in_advanced_help(self):
        help_result = subprocess.run(
            ["make", "--no-print-directory", "help-all"], cwd=PROJECT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("storage-report", help_result.stdout)
        dry = subprocess.run(
            ["make", "--no-print-directory", "-n", "storage-report",
             f"DATA_ROOT={self.root}", "FORMAT=json"], cwd=PROJECT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("--raw-root", dry.stdout)
        self.assertNotIn("uv run", dry.stdout)
        live = subprocess.run(
            ["make", "--no-print-directory", "storage-report",
             f"DATA_ROOT={self.root}", "FORMAT=json", "STRICT=1"], cwd=PROJECT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(live.returncode, 0, live.stderr)
        self.assertEqual(json.loads(live.stdout)["issues"], [])

    def test_make_diagnose_lists_missing_cache_without_modifying_state(self):
        self.extracted.unlink()
        self.index = []
        self.dump()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = subprocess.run(
            ["make", "--no-print-directory", "storage-report",
             f"DATA_ROOT={self.root}", "DIAGNOSE=1"],
            cwd=PROJECT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Unexpectedly missing cache objects   1", result.stdout)
        self.assertIn("Anomalous resources (metadata only)", result.stdout)
        self.assertIn(self.extracted_rel, result.stdout)
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_verify_hashes_cannot_certify_source_without_digest(self):
        self.state["r1"]["source_sha256"] = ""
        self.dump()
        result = self.audit(verify_hashes=True)
        self.assertTrue(any("missing/invalid source_sha256" in issue
                            for issue in result["issues"]))

    def test_strict_detects_complete_active_object_missing_from_index(self):
        # A valid active materialization must be present in the physical index.
        self.state["r1"]["selection_complete"] = True
        self.index = []
        self.dump()
        data = self.audit()
        self.assertTrue(any("missing from current index" in issue for issue in data["issues"]))

    def test_strict_detects_indexed_object_not_owned_by_active_state(self):
        self.state["r1"]["selection_complete"] = True
        stray = self.version / "extracted" / "stray.csv"
        stray.write_text("x")
        self.index.append({"source_object": self.source_rel,
                           "object": stray.relative_to(self.root).as_posix()})
        self.dump()
        data = self.audit()
        self.assertTrue(any("not backed by an active state" in issue for issue in data["issues"]))

    def test_strict_detects_duplicate_current_index_rows(self):
        self.state["r1"]["selection_complete"] = True
        self.index.append(dict(self.index[0]))
        self.dump()
        data = self.audit()
        self.assertTrue(any("duplicate current index" in issue for issue in data["issues"]))

    def test_does_not_index_partially_materialized_multi_member_version(self):
        self.state["r1"]["selection_complete"] = True
        self.state["r1"]["selected_members"] = ["votos.csv", "other.csv"]
        self.state["r1"]["extracted_objects"] = [
            self.extracted_rel,
            self.extracted_rel.replace("votos.csv", "other.csv"),
        ]
        self.index = []  # save_state would omit the whole incomplete version.
        self.dump()
        data = self.audit()
        self.assertTrue(any("required materialization missing" in issue for issue in data["issues"]))
        self.assertFalse(any("missing from current index" in issue for issue in data["issues"]))

    def test_machine_json_and_strict_exit_status(self):
        args = [sys.executable, str(PROJECT / "scripts" / "report_local_storage.py"),
                "--raw-root", str(self.root), "--duckdb", str(self.base / "db"),
                "--candidate-fact-root", str(self.base / "facts"), "--json", "--strict"]
        ok = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(json.loads(ok.stdout)["schema_version"], 1)
        self.source.unlink()
        bad = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertEqual(bad.returncode, 2)
        self.assertTrue(json.loads(bad.stdout)["issues"])


if __name__ == "__main__":
    unittest.main()
