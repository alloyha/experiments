"""PR39: coordinator must not roll back a partially committed active snapshot.

Uses a synthetic pending CSV and a temporary root. No network or canonical TSE data.
"""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest as ingest
from test_publication_recovery import _prepare
from test_worker_lifecycle import PACKAGE, RESOURCE


class CoordinatorRollbackSafetyTests(unittest.TestCase):
    def test_projection_failure_does_not_rollback_committed_active_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="pr39-coordinator-") as temporary:
            root = Path(temporary)
            item, expected, _record, _profile, _pending = _prepare(root)
            args = argparse.Namespace(
                root=root, year=[2026], from_year=None, to_year=None,
                election_type=None, datasets=None, mode="incremental",
                profile="mirror", granularity="all", uf=None,
                resource_regex=None, list_only=False, force=False,
                workers=1, extract_workers=1, checkpoint_every=1,
                extractor="python", heartbeat_seconds=0, no_profile=True,
            )
            package = {**PACKAGE, "resources": [RESOURCE]}
            original_write = ingest.atomic_json
            interrupted = False

            def fail_after_active_snapshot(path, value):
                nonlocal interrupted
                original_write(path, value)
                if (
                    Path(path) == ingest.state_path(root)
                    and isinstance(value, dict)
                    and item.key in value
                    and not interrupted
                ):
                    interrupted = True
                    raise OSError("injected failure after active snapshot replace")

            with (
                patch.object(ingest, "parse_args", return_value=args),
                patch.object(ingest, "discover_packages", return_value=[package]),
                patch.object(
                    ingest, "download_resilient",
                    side_effect=AssertionError("should recover locally without network"),
                ),
                patch.object(ingest, "atomic_json", side_effect=fail_after_active_snapshot),
            ):
                with self.assertRaises(ingest.ResourceFailed):
                    ingest.main()

            self.assertTrue(interrupted, "fault must reach the active snapshot")
            active = ingest.load_state(root)
            self.assertIn(item.key, active, "stale checkpoint rolled back committed state")
            self.assertEqual(active[item.key]["source_sha256"], expected["source_sha256"])
            self.assertIn(item.key, ingest.load_pending(root))

            remaining = ingest.reconcile_ingest_recovery(
                root, active, ingest.load_pending(root)
            )
            self.assertFalse(remaining, "committed journal must be reconciled")
            current = ingest.current_objects_path(root).read_text(encoding="utf-8")
            self.assertIn('"resource_id": "r1"', current)


if __name__ == "__main__":
    unittest.main()
