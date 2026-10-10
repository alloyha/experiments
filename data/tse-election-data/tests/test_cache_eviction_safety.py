"""Safety gates for offline extracted-cache eviction."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import prune_extracted_raw as prune

SCRIPT = Path(prune.__file__).resolve()


class CacheEvictionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_root = Path(self.tmp.name) / "tse"
        self.raw_root = self.data_root / "raw"
        self.raw_root.mkdir(parents=True)
        self.version = (
            self.raw_root / "election_type=general" / "year=2026"
            / "domain=vote_result" / "dataset=resultados_2026"
            / "resource=r1" / "sha256=abc"
        )
        self.source = self.version / "source" / "archive.zip"
        self.extracted = self.version / "extracted" / "votes.csv"
        self.extracted.parent.mkdir(parents=True)
        self.extracted.write_bytes(b"cache")

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.raw_root), *args],
            text=True, capture_output=True, check=False,
        )

    def persist_metadata(self):
        md = self.data_root / "_metadata"
        md.mkdir(parents=True, exist_ok=True)
        relative = str(self.extracted.relative_to(self.data_root))
        source_rel = str(self.source.relative_to(self.data_root))
        state = {"r1": {
            "extracted_objects": [relative],
            "source_object": source_rel,
            "selection_complete": True,
            "materialization_state": "present",
            "source_sha256": "abc",
        }}
        (md / "resource_state.json").write_text(json.dumps(state))
        (md / "current_objects.jsonl").write_text(
            json.dumps({"object": relative}) + "\n"
        )
        return md, relative

    def test_invalid_root_is_not_silent_empty_success(self):
        missing = self.raw_root / "does-not-exist"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(missing),
             "--domain", "vote_result"],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_apply_refuses_empty_domain_filter(self):
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source")
        self.persist_metadata()
        result = self.run_cli("--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.extracted.is_file())

    def test_apply_refuses_missing_control_plane(self):
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source")
        result = self.run_cli("--apply", "--domain", "vote_result")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.extracted.is_file())

    def test_prepared_without_immutable_source_is_not_evicted(self):
        prepared = self.version / "prepared" / "votes.parquet"
        prepared.parent.mkdir(parents=True)
        prepared.write_bytes(b"derived")
        result = self.run_cli("--domain", "vote_result")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("candidate files: 0", result.stdout)
        self.assertTrue(self.extracted.is_file())

    def test_interrupted_unlink_leaves_no_phantom_index(self):
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source")
        md, relative = self.persist_metadata()
        argv = [str(SCRIPT), "--root", str(self.raw_root),
                "--domain", "vote_result", "--apply"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(Path, "unlink", side_effect=OSError("injected unlink failure")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "injected unlink failure"):
                prune.main()

        # Crash-safe offline invariant: no index reference to a file that
        # may be deleted, while authoritative source and state remain.
        self.assertEqual((md / "current_objects.jsonl").read_text(), "")
        self.assertTrue(self.extracted.is_file())
        self.assertTrue(self.source.is_file())
        state = json.loads((md / "resource_state.json").read_text())
        self.assertEqual(state["r1"]["materialization_state"], "evicted")
        self.assertEqual(state["r1"]["extracted_objects"], [relative])

        # A rerun after interruption finishes pruning without remote fetch.
        result = self.run_cli("--apply", "--domain", "vote_result")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.extracted.exists())
        self.assertTrue(self.source.is_file())
        self.assertEqual((md / "current_objects.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
