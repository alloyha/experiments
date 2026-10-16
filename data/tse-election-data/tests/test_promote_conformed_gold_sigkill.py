"""Real OS-process death across promotion boundary, isolated files only.

Unlike exception injection, SIGKILL cannot execute in-process cleanup or
except/finally clauses. Run with:
  .venv/bin/python -m unittest discover -s tests \
    -p 'test_promote_conformed_gold_sigkill.py' -v
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest

from scripts import promote_conformed_gold as promotion
from scripts.warehouse_lifecycle import lifecycle_epoch


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix", "requires POSIX SIGKILL/flock")
class RealKillPromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tse-gold-sigkill-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.original = self.folder / "tse_analytics.duckdb"
        self.candidate = self.folder / "tse_analytics_conformed_v2.duckdb"
        # Synthetic bytes isolate the file/journal/epoch mechanism from SQL.
        # The separate WSL rehearsal with real DuckDB files covers SQL smoke.
        self.old_data = b"stable-original" * 8192
        self.new_data = b"conformed-v2" * 8192
        self.original.write_bytes(self.old_data)
        self.candidate.write_bytes(self.new_data)
        self.old_hash = hashlib.sha256(self.old_data).hexdigest()
        self.new_hash = hashlib.sha256(self.new_data).hexdigest()

    def crash_and_recover(self, boundary: str) -> None:
        # Every subprocess acquires the *actual* POSIX fcntl sidecar locks.
        # No monkeypatching of the lifecycle lock or fsync implementation.
        child = textwrap.dedent('''
            import os, signal, sys
            from pathlib import Path
            from scripts import promote_conformed_gold as mod

            original, candidate = Path(sys.argv[1]), Path(sys.argv[2])
            old_sha, new_sha, boundary = sys.argv[3:6]
            if boundary == 'before_swap':
                real_replace = mod.os.replace
                def interrupted_replace(src, dst):
                    if Path(dst) == original:
                        os.kill(os.getpid(), signal.SIGKILL)
                    return real_replace(src, dst)
                mod.os.replace = interrupted_replace
            elif boundary == 'after_swap_before_journal':
                real_write = mod.write_journal
                def interrupted_write(target, record, **kwargs):
                    if record.get('phase') == 'SWAPPED_UNCERTIFIED':
                        os.kill(os.getpid(), signal.SIGKILL)
                    return real_write(target, record, **kwargs)
                mod.write_journal = interrupted_write
            else:
                raise AssertionError('unsupported injection boundary')
            mod.promote(
                original, candidate, old_sha, new_sha,
                timeout=5, expected_events=198, expected_cycles=5,
                contracts=original.parent / 'unused.json',
                post_validator=lambda _: None,
            )
            raise AssertionError('SIGKILL fault was not reached')
        ''')
        completed = subprocess.run(
            [sys.executable, "-c", child,
             str(self.original), str(self.candidate),
             self.old_hash, self.new_hash, boundary],
            cwd=str(ROOT), text=True, capture_output=True, timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode, -signal.SIGKILL,
            msg=f"child returned {completed.returncode}: {completed.stderr}",
        )
        journal = promotion.read_journal(self.original)
        self.assertEqual(journal["phase"], "EPOCH_ROTATED")
        self.assertEqual(journal["original_sha256"], self.old_hash)
        self.assertEqual(journal["candidate_sha256"], self.new_hash)
        self.assertEqual(hashlib.sha256(Path(journal["backup"]).read_bytes()).hexdigest(), self.old_hash)
        self.assertEqual(hashlib.sha256(self.candidate.read_bytes()).hexdigest(), self.new_hash)
        epoch_after_kill = lifecycle_epoch(self.original)
        self.assertNotEqual(epoch_after_kill, "unrotated")
        self.assertTrue(Path(str(self.original) + ".build.lock").exists())

        # The journal may lag the rename; status must disclose actual bytes.
        diagnostic = promotion.status(self.original, timeout=5)
        self.assertEqual(diagnostic['phase'], 'EPOCH_ROTATED')
        self.assertTrue(diagnostic['recovery_required'])
        self.assertEqual(
            diagnostic['observed_content'],
            'ORIGINAL' if boundary == 'before_swap' else 'CANDIDATE',
        )
        self.assertEqual(
            diagnostic['journal_phase_matches_content'],
            boundary == 'before_swap',
        )
        active_hash = hashlib.sha256(self.original.read_bytes()).hexdigest()
        expected_active = self.old_hash if boundary == "before_swap" else self.new_hash
        self.assertEqual(active_hash, expected_active)

        # The parent process recovers via *the real lock implementation*.
        # A leaked lock in the killed process would cause timeout/failure.
        result = promotion.rollback(self.original, timeout=5)
        self.assertEqual(result["phase"], "ROLLED_BACK")
        self.assertEqual(hashlib.sha256(self.original.read_bytes()).hexdigest(), self.old_hash)
        self.assertEqual(hashlib.sha256(self.candidate.read_bytes()).hexdigest(), self.new_hash)
        self.assertNotEqual(lifecycle_epoch(self.original), epoch_after_kill)
        self.assertEqual(promotion.read_journal(self.original)["phase"], "ROLLED_BACK")

    def test_sigkill_after_epoch_before_swap(self):
        self.crash_and_recover("before_swap")

    def test_sigkill_after_swap_before_journal(self):
        self.crash_and_recover("after_swap_before_journal")


if __name__ == "__main__":
    unittest.main()
