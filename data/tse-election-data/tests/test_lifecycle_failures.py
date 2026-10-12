"""Failure-path tests: every exception ends in a recorded lifecycle state."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest
from resource_lifecycle import ResourceLifecycle, drive, fail_lifecycle, state_name
from test_worker_lifecycle import PACKAGE, RESOURCE, _complete_previous


def _fake_download(content: bytes):
    def fake(session, url, destination, **_):
        destination.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()
        return sha, len(content), 0.0, 1, 0.0, 0, 0
    return fake


def _run_worker(root: Path, **overrides):
    kwargs = dict(
        root=root, year=2026, election_type="general", mode="incremental",
        package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
        previous=None, force=False,
    )
    kwargs.update(overrides)
    return tse_ingest.download_resource_worker(**kwargs)


def _profile_rows(root: Path) -> list[dict]:
    path = tse_ingest.profile_path(root)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DownloadFailureTests(unittest.TestCase):
    def test_download_exception_is_recorded_as_failed_from_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                tse_ingest, "download_resilient",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                    _run_worker(Path(tmp))
            self.assertEqual(ctx.exception.from_state, "downloading")
            self.assertIn("boom", str(ctx.exception))


class OutcomeLifecycleTests(unittest.TestCase):
    def test_metadata_skip_profile_carries_final_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = tse_ingest.resource_fingerprint(RESOURCE)
            previous = _complete_previous(root, b"a;b\n1;2\n", fp)
            outcome = _run_worker(root, previous=previous)
            self.assertEqual(outcome["kind"], "metadata-skip")
            self.assertEqual(outcome["lifecycle"], "metadata_skipped")
            self.assertEqual(outcome["profile"].lifecycle, "metadata_skipped")

    def test_pending_outcome_carries_live_machine_in_materialized_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = b"a;b\n1;2\n"
            with patch.object(tse_ingest, "download_resilient", side_effect=_fake_download(content)):
                outcome = _run_worker(Path(tmp))
            self.assertEqual(outcome["kind"], "pending")
            machine = outcome["pending"].lifecycle
            self.assertIsInstance(machine, ResourceLifecycle)
            self.assertEqual(state_name(machine), "materialized")


class PreparationFailureTests(unittest.TestCase):
    def _pending(self, root: Path):
        content = b"a;b\n1;2\n"
        with patch.object(tse_ingest, "download_resilient", side_effect=_fake_download(content)):
            return _run_worker(root)["pending"]

    def test_prepare_exception_is_recorded_as_failed_from_materialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = self._pending(Path(tmp))
            with patch.object(
                tse_ingest, "prepare_resource_worker",
                side_effect=RuntimeError("extract broke"),
            ):
                with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                    tse_ingest.prepare_resource_worker_guarded(
                        pending=item, extractor="python",
                    )
            self.assertEqual(ctx.exception.from_state, "materialized")
            self.assertEqual(state_name(item.lifecycle), "failed")

    def test_publish_moves_materialized_to_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = self._pending(Path(tmp))
            tse_ingest.prepare_resource_worker_guarded(pending=item, extractor="python")
            drive(item.lifecycle, "publish")
            self.assertEqual(state_name(item.lifecycle), "published")

    def test_fail_after_publish_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = self._pending(Path(tmp))
            drive(item.lifecycle, "publish")
            self.assertEqual(fail_lifecycle(item.lifecycle), "published")
            self.assertEqual(state_name(item.lifecycle), "published")


class VanishedSourceTests(unittest.TestCase):
    def test_source_vanishing_after_plan_fails_then_next_run_downloads(self):
        content = b"a;b\n1;2\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp = tse_ingest.resource_fingerprint(RESOURCE)
            seeded = _complete_previous(root, content, fp)
            source = root / seeded["source_object"]
            # Only the pending journal knows about the source, so the plan is a rehydrate.
            pending_row = {**seeded, "extracted_objects": [], "selection_complete": False}

            real_plan = tse_ingest.initial_plan_event

            def plan_then_vanish(**kwargs):
                # The plan is decided from the source's existence; the file then
                # disappears before the rehydrate reads it. This removes the race
                # deterministically and does not depend on how many is_file calls
                # the worker makes.
                event = real_plan(**kwargs)
                source.unlink()
                return event

            with patch.object(tse_ingest, "initial_plan_event", side_effect=plan_then_vanish):
                with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                    _run_worker(root, previous=None, pending_row=pending_row)

            self.assertEqual(ctx.exception.from_state, "rehydrating")
            self.assertIn("vanished", str(ctx.exception))

            # Recovery: the journal still points at the missing file, so the next
            # run must plan a download and must not loop on rehydrate.
            with patch.object(
                tse_ingest, "download_resilient", side_effect=_fake_download(content),
            ) as download:
                outcome = _run_worker(root, previous=None, pending_row=pending_row)
            self.assertEqual(outcome["kind"], "pending")
            self.assertEqual(download.call_count, 1)


class FailureAuditTests(unittest.TestCase):
    def test_download_failure_is_appended_to_profile_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(tse_ingest, "download_resilient", side_effect=RuntimeError("boom")):
                with self.assertRaises(tse_ingest.ResourceFailed) as ctx:
                    _run_worker(root)
            tse_ingest.record_failure(
                root, no_profile=False, exc=ctx.exception, year=2026,
                election_type="general", package=PACKAGE, resource=RESOURCE, worker="w",
            )
            rows = _profile_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["result"], "failed")
            self.assertEqual(rows[0]["lifecycle"], "downloading")
            self.assertIn("boom", rows[0]["error"])
            self.assertEqual(rows[0]["resource_id"], "r1")

    def test_no_profile_suppresses_failure_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tse_ingest.record_failure(
                root, no_profile=True, exc=RuntimeError("x"), year=2026,
                election_type="general", package=PACKAGE, resource=RESOURCE, worker="w",
            )
            self.assertEqual(_profile_rows(root), [])


if __name__ == "__main__":
    unittest.main()
