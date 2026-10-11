"""PR34 isolated round-trip: durable eviction -> offline rehydration -> identity.

Run in data/tse-election-data with:
    .venv/bin/python -m unittest discover -s tests -p 'test_lazy_materialization_e2e.py' -v

All destructive operations are confined to TemporaryDirectory, never DATA_ROOT.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import socket
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import tse_ingest
from scripts import ensure_materialized, prune_extracted_raw


class OfflineEvictionRehydrationE2ETests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="pr34-e2e-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "tse"
        self.version = (
            self.root / "raw" / "election_type=general" / "year=2026"
            / "domain=vote_result" / "dataset=resultados_2026"
            / "resource=synthetic-r1" / "sha256=fixture"
        )
        self.source = self.version / "source" / "fixture.zip"
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.members = {
            "votacao_candidato_munzona_2026_BRASIL.csv": b"id,votes\n1,10\n2,20\n",
            "votacao_partido_munzona_2026_BRASIL.csv": b"id,votes\n3,30\n4,40\n",
        }
        with zipfile.ZipFile(self.source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, data in self.members.items():
                zf.writestr(name, data)

        initial, _, selected = tse_ingest.extract_selected_members(
            self.source, self.version / "extracted", "brasil", None, "python"
        )
        self.assertEqual(set(selected), set(self.members))
        self.object_rels = [str(p.relative_to(self.root)) for p in initial]
        self.row = {
            "year": 2026,
            "election_type": "general",
            "election_scope": "national",
            "dataset_id": "resultados_2026",
            "dataset_title": "Fixture E2E",
            "resource_id": "synthetic-r1",
            "resource_name": "Synthetic fixture",
            "resource_format": "ZIP",
            "resource_fingerprint": "fixture",
            "domain": "vote_result",
            "partition": "BRASIL",
            "source_url": "https://example.invalid/never-download.zip",
            "source_object": str(self.source.relative_to(self.root)),
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "source_size_bytes": self.source.stat().st_size,
            "extracted_objects": self.object_rels,
            "selected_members": selected,
            "selection_complete": True,
            "materialization_state": "present",
            "granularity": "brasil",
            "uf": "",
            "extractor": "python",
            "checked_at": "2026-10-10T00:00:00+00:00",
        }
        tse_ingest.save_state(self.root, {"general|2026|synthetic-r1|brasil|": self.row})

    def invoke(self, entrypoint, argv):
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out:
            result = entrypoint()
        return result, out.getvalue()

    def snapshot(self):
        blobs = {rel: (self.root / rel).read_bytes() for rel in self.object_rels}
        counts = {}
        for rel, content in blobs.items():
            rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
            counts[rel] = (len(rows), sum(int(row["votes"]) for row in rows))
        index_path = tse_ingest.current_objects_path(self.root)
        index = sorted(
            json.loads(line)["object"]
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return blobs, counts, index

    def test_isolated_roundtrip_offline_and_idempotent(self):
        before_blobs, before_counts, before_index = self.snapshot()
        self.assertEqual(before_index, sorted(self.object_rels))
        immutable_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

        # An unexpected HTTP/socket connection must fail the test.
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
             mock.patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden")):
            _, eviction_log = self.invoke(prune_extracted_raw.main, [
                "prune_extracted_raw.py", "--root", str(self.root / "raw"),
                "--apply", "--domain", "vote_result",
            ])
            self.assertIn("Deleted 2 file(s).", eviction_log)
            self.assertTrue(self.source.is_file())
            self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), immutable_hash)
            self.assertFalse(any((self.root / rel).exists() for rel in self.object_rels))
            self.assertEqual(tse_ingest.current_objects_path(self.root).read_text(), "")

            result, materialization_log = self.invoke(ensure_materialized.main, [
                "ensure_materialized.py", "--root", str(self.root),
                "--year", "2026", "--election-type", "general",
                "--domain", "vote_result", "--extractor", "python",
            ])
            self.assertEqual(result, 0, materialization_log)
            self.assertIn("rehydrated=1", materialization_log)
            self.assertEqual(self.snapshot(), (before_blobs, before_counts, before_index))

            # A second call should be a no-op, not a second extraction.
            with mock.patch.object(
                tse_ingest, "extract_selected_members",
                side_effect=AssertionError("unexpected re-extraction"),
            ):
                result, second_log = self.invoke(ensure_materialized.main, [
                    "ensure_materialized.py", "--root", str(self.root),
                    "--year", "2026", "--election-type", "general",
                    "--domain", "vote_result", "--extractor", "python",
                ])
            self.assertEqual(result, 0, second_log)
            self.assertIn("already-present=1", second_log)
            self.assertEqual(self.snapshot(), (before_blobs, before_counts, before_index))


if __name__ == "__main__":
    unittest.main()
