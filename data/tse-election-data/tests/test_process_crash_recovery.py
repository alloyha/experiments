"""PR39: real SIGKILL fault injection across a local publication transaction.

Tests only synthetic CSV content in tempfile.TemporaryDirectory. No TSE raw or
network access. Linux/WSL only; subprocesses die through the kernel's SIGKILL.

Run from data/tse-election-data:
  .venv/bin/python -m unittest discover -s tests -p test_process_crash_recovery.py -v
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tse_ingest as ingest
from test_publication_recovery import _prepare  # Fixture shares the exact same production path.
from test_worker_lifecycle import PACKAGE, RESOURCE


_CHILD = r'''
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

root, point, tests_dir, project_dir = sys.argv[1:]
sys.path.insert(0, tests_dir)
sys.path.insert(0, project_dir)
import tse_ingest as ingest
from test_publication_recovery import _prepare

root = Path(root)
item, row, record, profile, journal = _prepare(root)

def die():
    os.kill(os.getpid(), signal.SIGKILL)
    os._exit(127)  # Unreachable if SIGKILL is delivered.

if point == 'before_manifest':
    with patch.object(ingest, 'ensure_manifest_record', side_effect=lambda *_a, **_kw: die()):
        ingest.publish_prepared_resource(root=root, state={}, pending_journal=journal,
                                         item=item, state_row=row, record=record, profile=profile)
elif point == 'after_manifest':
    original = ingest.ensure_manifest_record
    def after_manifest(*args, **kwargs):
        result = original(*args, **kwargs)
        die()
        return result
    with patch.object(ingest, 'ensure_manifest_record', side_effect=after_manifest):
        ingest.publish_prepared_resource(root=root, state={}, pending_journal=journal,
                                         item=item, state_row=row, record=record, profile=profile)
elif point == 'after_state_snapshot':
    original = ingest.atomic_json
    def after_state_json(path, data):
        result = original(path, data)
        if Path(path) == ingest.state_path(root):
            die()
        return result
    with patch.object(ingest, 'atomic_json', side_effect=after_state_json):
        ingest.publish_prepared_resource(root=root, state={}, pending_journal=journal,
                                         item=item, state_row=row, record=record, profile=profile)
elif point == 'after_journal_cleanup':
    original = ingest.save_pending
    def after_pending_save(*args, **kwargs):
        result = original(*args, **kwargs)
        new_rows = args[1] if len(args) > 1 else kwargs['pending']
        if not new_rows:
            die()
        return result
    with patch.object(ingest, 'save_pending', side_effect=after_pending_save):
        ingest.publish_prepared_resource(root=root, state={}, pending_journal=journal,
                                         item=item, state_row=row, record=record, profile=profile)
else:
    raise ValueError('unknown crash point: ' + point)

raise AssertionError('expected actual SIGKILL; publication unexpectedly returned')
'''


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires Linux/WSL SIGKILL")
class RealWorkerDeathRecoveryTests(unittest.TestCase):
    """Kill a separate process after each concrete filesystem boundary."""

    def _kill(self, root: Path, point: str) -> None:
        base = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, "-c", _CHILD, str(root), point,
               str(Path(__file__).resolve().parent), str(base)]
        completed = subprocess.run(
            cmd, cwd=base, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(
            completed.returncode, -signal.SIGKILL,
            msg=f"expected child SIGKILL at {point}; exit={completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )

    def _recover_and_assert(self, root: Path, *, expect_initial_active: bool) -> None:
        state = ingest.load_state(root)
        journal = ingest.load_pending(root)
        self.assertEqual(bool(state), expect_initial_active)
        self.assertEqual(len(state), 1 if state else 0)
        self.assertLessEqual(len(_jsonl(ingest.manifest_path(root))), 1)

        # Model a separate process restarting: repair projected index and reconcile
        # only journal entries that have genuinely become authoritative.
        journal = ingest.reconcile_ingest_recovery(root, state, journal)
        if not state:
            self.assertEqual(len(journal), 1)
            key, pending_row = next(iter(journal.items()))
            with patch.object(
                ingest, "download_resilient",
                side_effect=AssertionError("restart unexpectedly downloaded from network"),
            ):
                result = ingest.download_resource_worker(
                    root=root, year=2026, election_type="general", mode="incremental",
                    package=PACKAGE, resource=RESOURCE, granularity="all", uf=None,
                    previous=None, force=False, pending_row=pending_row,
                )
            self.assertEqual(result["kind"], "pending")
            item = result["pending"]
            row, record, profile = ingest.prepare_resource_worker_guarded(
                pending=item, extractor="python",
            )
            state, profile = ingest.publish_prepared_resource(
                root=root, state=state, pending_journal=journal,
                item=item, state_row=row, record=record, profile=profile,
            )
            self.assertEqual(profile.lifecycle, "published")
        else:
            self.assertFalse(journal, "committed journal must be reconciled")

        # Repeat restart to prove stable active truth and no duplicate manifest.
        state2 = ingest.load_state(root)
        journal2 = ingest.reconcile_ingest_recovery(root, state2, ingest.load_pending(root))
        self.assertEqual(state2, state)
        self.assertFalse(journal2)
        self.assertEqual(len(_jsonl(ingest.manifest_path(root))), 1)
        current = _jsonl(ingest.current_objects_path(root))
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["resource_id"], "r1")

    def test_worker_death_before_manifest_commit(self):
        with tempfile.TemporaryDirectory(prefix="pr39-kill-") as td:
            root = Path(td)
            self._kill(root, "before_manifest")
            self.assertFalse(ingest.state_path(root).exists())
            self.assertFalse(ingest.manifest_path(root).exists())
            self._recover_and_assert(root, expect_initial_active=False)

    def test_worker_death_after_manifest_before_active_state(self):
        with tempfile.TemporaryDirectory(prefix="pr39-kill-") as td:
            root = Path(td)
            self._kill(root, "after_manifest")
            self.assertFalse(ingest.state_path(root).exists())
            self.assertEqual(len(_jsonl(ingest.manifest_path(root))), 1)
            self._recover_and_assert(root, expect_initial_active=False)

    def test_worker_death_after_active_snapshot_before_projection(self):
        with tempfile.TemporaryDirectory(prefix="pr39-kill-") as td:
            root = Path(td)
            self._kill(root, "after_state_snapshot")
            self.assertTrue(ingest.state_path(root).exists())
            self.assertFalse(ingest.current_objects_path(root).exists())
            self._recover_and_assert(root, expect_initial_active=True)

    def test_worker_death_after_pending_journal_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="pr39-kill-") as td:
            root = Path(td)
            self._kill(root, "after_journal_cleanup")
            self.assertTrue(ingest.state_path(root).exists())
            self.assertFalse(ingest.load_pending(root))
            self._recover_and_assert(root, expect_initial_active=True)


if __name__ == "__main__":
    unittest.main()
