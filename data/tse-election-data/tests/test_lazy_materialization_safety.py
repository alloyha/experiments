"""PR34 hardening: source integrity, staging, and index completeness."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import tse_ingest
from scripts import ensure_materialized as ensure


class LazyMaterializationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "tse"
        self.version = (self.root / "raw" / "election_type=general" / "year=2026"
                        / "domain=vote_result" / "dataset=resultados_2026"
                        / "resource=r1" / "sha256=abc")
        self.source = self.version / "source" / "archive.zip"
        self.source.parent.mkdir(parents=True)
        with zipfile.ZipFile(self.source, "w") as zf:
            zf.writestr("votes_BRASIL.csv", "id,votes\n1,10\n")
        self.extracted = self.version / "extracted" / "votes_BRASIL.csv"
        self.source_rel = str(self.source.relative_to(self.root))
        self.extracted_rel = str(self.extracted.relative_to(self.root))
        self.row = {
            "year": 2026, "election_type": "general", "resource_id": "r1",
            "resource_name": "votes", "dataset_id": "resultados_2026",
            "dataset_title": "Votes", "domain": "vote_result", "partition": "BRASIL",
            "source_object": self.source_rel,
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "source_size_bytes": self.source.stat().st_size,
            "extracted_objects": [self.extracted_rel],
            "selected_members": ["votes_BRASIL.csv"],
            "selection_complete": True, "materialization_state": "evicted",
            "checked_at": "2026-10-10T00:00:00+00:00", "resource_format": "ZIP",
            "granularity": "brasil", "uf": "",
        }

    def test_corrupt_source_rejected_before_materialization(self):
        self.source.write_bytes(self.source.read_bytes() + b"tampered")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            ensure.materialize_state_row(self.root, self.row, extractor="python")
        self.assertFalse(self.extracted.exists())

    def test_member_selection_drift_rejected_without_publish(self):
        self.row["selected_members"] = ["different_BRASIL.csv"]
        with self.assertRaisesRegex(RuntimeError, "selected archive members differ"):
            ensure.materialize_state_row(self.root, self.row, extractor="python")
        self.assertFalse(self.extracted.exists())

    def test_extraction_interruption_keeps_final_cache_unpublished(self):
        def partial_extract(_source, dest, *_args):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "votes_BRASIL.csv").write_bytes(b"partial")
            raise RuntimeError("injected extraction crash")

        with mock.patch.object(tse_ingest, "extract_selected_members", side_effect=partial_extract):
            with self.assertRaisesRegex(RuntimeError, "injected extraction crash"):
                ensure.materialize_state_row(self.root, self.row, extractor="python")
        self.assertFalse(self.extracted.exists())
        self.assertTrue(self.source.exists())

    def test_partial_materialization_not_advertised_by_save_state(self):
        self.row["selected_members"] = ["votes_BRASIL.csv", "other_BRASIL.csv"]
        self.row["extracted_objects"] = [self.extracted_rel, str(
            (self.version / "extracted" / "other_BRASIL.csv").relative_to(self.root)
        )]
        self.extracted.parent.mkdir(parents=True)
        self.extracted.write_bytes(b"existing")
        tse_ingest.save_state(self.root, {"r1": self.row})
        self.assertEqual(tse_ingest.current_objects_path(self.root).read_text(), "")


if __name__ == "__main__":
    unittest.main()
