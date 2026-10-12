"""Restart and publication failure-injection contract for the TSE ingest boundary.

All files and downloaded content stay under tempfile.TemporaryDirectory.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest
from resource_lifecycle import state_name
from test_worker_lifecycle import PACKAGE, RESOURCE


def _prepare(root: Path):
    payload = b"a;b\n1;2\n"

    def fake_download(session, url, destination, **kwargs):
        destination.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload), 0.0, 1, 0.0, 0, 0

    with patch.object(tse_ingest, "download_resilient", side_effect=fake_download):
        outcome = tse_ingest.download_resource_worker(
            root=root, year=2026, election_type="general", mode="incremental",
            package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
            previous=None, force=False,
        )
    item = outcome["pending"]
    row, record, profile = tse_ingest.prepare_resource_worker_guarded(
        pending=item, extractor="python",
    )
    journal = tse_ingest.remember_pending_preparation(root, {}, item)
    return item, row, record, profile, journal


def _manifest_rows(root: Path):
    path = tse_ingest.manifest_path(root)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class PreparedPublicationRecoveryTests(unittest.TestCase):
    def test_manifest_write_failure_keeps_active_and_journal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            active = {}
            with patch.object(tse_ingest, "ensure_manifest_record", side_effect=OSError("disk error")):
                with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                    tse_ingest.publish_prepared_resource(
                        root=root, state=active, pending_journal=journal,
                        item=item, state_row=row, record=rec, profile=prof,
                    )
            self.assertEqual(active, {})
            self.assertIn(item.key, journal)
            self.assertFalse(tse_ingest.state_path(root).exists())
            self.assertEqual(ctx.exception.from_state, "materialized")
            self.assertEqual(state_name(item.lifecycle), "failed")

    def test_retry_after_manifest_commit_deduplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            active = {}
            with patch.object(tse_ingest, "save_state", side_effect=OSError("database unavailable")):
                with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                    tse_ingest.publish_prepared_resource(
                        root=root, state=active, pending_journal=journal,
                        item=item, state_row=row, record=rec, profile=prof,
                    )
            self.assertEqual(active, {})
            self.assertEqual(len(_manifest_rows(root)), 1)
            self.assertIn(item.key, journal)
            self.assertEqual(ctx.exception.from_state, "materialized")
            self.assertEqual(state_name(item.lifecycle), "failed")
            # A new process reconstructs its in-memory machine from the journal.
            with patch.object(
                tse_ingest, "download_resilient",
                side_effect=AssertionError("retry must use durable raw source"),
            ):
                outcome = tse_ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
                    previous=None, pending_row=journal[item.key], force=False,
                )
            recovered_item = outcome["pending"]
            recovered_row, recovered_record, recovered_profile = (
                tse_ingest.prepare_resource_worker_guarded(
                    pending=recovered_item, extractor="python",
                )
            )
            active, prof = tse_ingest.publish_prepared_resource(
                root=root, state=active, pending_journal=journal,
                item=recovered_item, state_row=recovered_row,
                record=recovered_record, profile=recovered_profile,
            )
            self.assertEqual(len(_manifest_rows(root)), 1)
            self.assertIn(item.key, active)
            self.assertNotIn(item.key, journal)
            self.assertEqual(state_name(recovered_item.lifecycle), "published")
            self.assertEqual(prof.lifecycle, "published")

    def test_failed_journal_cleanup_is_recovered_after_restart(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            with patch.object(tse_ingest, "save_pending", side_effect=OSError("disk error")):
                active, _ = tse_ingest.publish_prepared_resource(
                    root=root, state={}, pending_journal=journal,
                    item=item, state_row=row, record=rec, profile=prof,
                )
            self.assertIn(item.key, journal)
            self.assertIn(item.key, tse_ingest.load_pending(root))
            recovered = tse_ingest.reconcile_ingest_recovery(
                root, tse_ingest.load_state(root), tse_ingest.load_pending(root),
            )
            self.assertNotIn(item.key, recovered)
            self.assertNotIn(item.key, tse_ingest.load_pending(root))
            self.assertIn(item.key, active)

    def test_rebuild_torn_current_objects_from_active_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            tse_ingest.publish_prepared_resource(
                root=root, state={}, pending_journal=journal,
                item=item, state_row=row, record=rec, profile=prof,
            )
            objects = tse_ingest.current_objects_path(root)
            self.assertTrue(objects.read_text().strip())
            objects.write_text("", encoding="utf-8")
            tse_ingest.reconcile_ingest_recovery(root, tse_ingest.load_state(root), {})
            self.assertTrue(objects.read_text().strip())

    def test_corrupt_manifest_does_not_publish(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            path = tse_ingest.manifest_path(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{"incomplete": ')
            with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                tse_ingest.publish_prepared_resource(
                    root=root, state={}, pending_journal=journal,
                    item=item, state_row=row, record=rec, profile=prof,
                )
            self.assertEqual(path.read_bytes(), b'{"incomplete": ')
            self.assertFalse(tse_ingest.state_path(root).exists())
            self.assertEqual(ctx.exception.from_state, "materialized")
            self.assertIn(item.key, journal)

    def test_failed_journal_write_does_not_change_in_memory_journal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            before = dict(journal)
            with patch.object(tse_ingest, "save_pending", side_effect=OSError("journal unavailable")):
                with self.assertRaises(OSError):
                    tse_ingest.remember_pending_preparation(root, journal, item)
            self.assertEqual(journal, before)

    def test_pending_source_wins_over_incomplete_active_row(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            previous = {
                "resource_fingerprint": item.fingerprint,
                "source_object": "raw/missing-source.csv",
                "source_sha256": item.source_sha256,
                "selection_complete": False,
                "extracted_objects": [],
            }
            with patch.object(
                tse_ingest, "download_resilient",
                side_effect=AssertionError("recover from journal without network"),
            ):
                outcome = tse_ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
                    previous=previous, pending_row=journal[item.key], force=False,
                )
            self.assertEqual(outcome["kind"], "pending")
            self.assertEqual(outcome["pending"].attempts, 0)

    def test_recovery_keeps_unpublished_journal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item, row, rec, prof, journal = _prepare(root)
            recovered = tse_ingest.reconcile_ingest_recovery(root, {}, journal)
            self.assertIn(item.key, recovered)
            self.assertFalse(tse_ingest.current_objects_path(root).read_text().strip())


if __name__ == "__main__":
    unittest.main()
