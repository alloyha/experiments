"""PR39: fail-closed rehydration and no skip for incomplete active versions.

All data are synthetic under TemporaryDirectory; no network or real TSE data.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest as ingest
from test_publication_recovery import _prepare
from test_worker_lifecycle import PACKAGE, RESOURCE, _complete_previous

PAYLOAD = b"a;b\n1;2\n"


def _download(session, url, destination, **kwargs):
    destination.write_bytes(PAYLOAD)
    return hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD), 0.0, 1, 0.0, 0, 0


def _worker(root: Path, *, previous: dict | None = None, pending_row: dict | None = None):
    return ingest.download_resource_worker(
        root=root, year=2026, election_type="general", mode="incremental",
        package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
        previous=previous, pending_row=pending_row, force=False,
    )


class RehydrationIntegrityTests(unittest.TestCase):
    def test_corrupt_journaled_source_is_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory(prefix="pr39-journal-hash-") as tmp:
            root = Path(tmp)
            item, _, _, _, journal = _prepare(root)
            item.source_target.write_bytes(b"corrupt bytes\n")

            with patch.object(
                ingest, "download_resilient",
                side_effect=AssertionError("unexpected network access"),
            ) as download:
                with self.assertRaises(ingest.ResourceFailed) as ctx:
                    _worker(root, pending_row=journal[item.key])
            self.assertEqual(ctx.exception.from_state, "rehydrating")
            self.assertIn("integrity mismatch", str(ctx.exception))
            self.assertEqual(download.call_count, 0)
            self.assertEqual(ingest.load_state(root), {})

    def test_same_hash_does_not_content_skip_missing_extracted_object(self):
        with tempfile.TemporaryDirectory(prefix="pr39-incomplete-") as tmp:
            root = Path(tmp)
            previous = _complete_previous(root, PAYLOAD, "stale-fingerprint")
            previous["extracted_objects"] = ["raw/missing.csv"]
            self.assertFalse(ingest.materialization_complete(root, previous))
            with patch.object(ingest, "download_resilient", side_effect=_download):
                outcome = _worker(root, previous=previous)
            self.assertEqual(outcome["kind"], "pending")
            row, _, _ = ingest.prepare_resource_worker_guarded(
                pending=outcome["pending"], extractor="python"
            )
            self.assertTrue(ingest.materialization_complete(root, row))

    def test_same_hash_restores_missing_source_instead_of_skipping(self):
        with tempfile.TemporaryDirectory(prefix="pr39-missing-source-") as tmp:
            root = Path(tmp)
            previous = _complete_previous(root, PAYLOAD, "stale-fingerprint")
            source = root / previous["source_object"]
            source.unlink()  # Synthetic fixture only.
            self.assertFalse(ingest.materialization_complete(root, previous))
            with patch.object(ingest, "download_resilient", side_effect=_download):
                outcome = _worker(root, previous=previous)
            self.assertEqual(outcome["kind"], "pending")
            self.assertEqual(source.read_bytes(), PAYLOAD)
            row, _, _ = ingest.prepare_resource_worker_guarded(
                pending=outcome["pending"], extractor="python"
            )
            self.assertTrue(ingest.materialization_complete(root, row))


if __name__ == "__main__":
    unittest.main()
