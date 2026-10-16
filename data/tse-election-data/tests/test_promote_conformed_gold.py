"""Offline, no-network recovery tests for the explicit Gold promoter.

Run in the project venv with:
  .venv/bin/python -m unittest discover -s tests -p test_promote_conformed_gold.py -v
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import promote_conformed_gold as mod
from scripts.warehouse_lifecycle import lifecycle_epoch


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class GoldPromotionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original = self.root / 'tse_analytics.duckdb'
        self.candidate = self.root / 'tse_analytics_conformed_v2.duckdb'
        self.old = b'old-warehouse-fixed-test-data'
        self.new = b'candidate-warehouse-fixed-test-data'
        self.original.write_bytes(self.old)
        self.candidate.write_bytes(self.new)
        self.oh = digest(self.old)
        self.ch = digest(self.new)

    def activate(self, post_validator=lambda _: None):
        return mod.promote(
            self.original, self.candidate, self.oh, self.ch, timeout=2,
            expected_events=198, expected_cycles=5,
            contracts=self.root / 'unused.json', post_validator=post_validator,
        )

    def test_preflight_never_mutates_data_or_epoch(self):
        result = mod.preflight(self.original, self.candidate, self.oh, self.ch)
        self.assertEqual(result['mode'], 'read_only_preflight')
        self.assertFalse(mod.journal_path(self.original).exists())
        self.assertEqual(lifecycle_epoch(self.original), 'unrotated')
        self.assertEqual(self.original.read_bytes(), self.old)

    def test_explicit_activation_keeps_backup_and_candidate(self):
        before_epoch = lifecycle_epoch(self.original)
        result = self.activate()
        self.assertEqual(result['phase'], 'ACTIVE_UNCERTIFIED')
        self.assertEqual(self.original.read_bytes(), self.new)
        self.assertEqual(self.candidate.read_bytes(), self.new)
        self.assertEqual(Path(result['backup']).read_bytes(), self.old)
        self.assertNotEqual(lifecycle_epoch(self.original), before_epoch)
        self.assertEqual(mod.read_journal(self.original)['phase'], 'ACTIVE_UNCERTIFIED')
        self.assertTrue(Path(str(self.original) + '.build.lock').exists())
        with self.assertRaisesRegex(mod.PromotionError, 'journal'):
            mod.preflight(self.original, self.candidate, self.ch, self.ch)

    def test_explicit_rollback_invalidates_epoch_and_restores_original(self):
        self.activate()
        before = lifecycle_epoch(self.original)
        result = mod.rollback(self.original, timeout=2)
        self.assertEqual(result['phase'], 'ROLLED_BACK')
        self.assertEqual(self.original.read_bytes(), self.old)
        self.assertNotEqual(lifecycle_epoch(self.original), before)
        self.assertEqual(mod.rollback(self.original, timeout=2)['phase'], 'ROLLED_BACK')

    def test_failed_post_validation_rolls_back(self):
        def fail(_: Path):
            raise ValueError('intentional post-promotion validation failure')

        with self.assertRaisesRegex(ValueError, 'intentional'):
            self.activate(post_validator=fail)
        self.assertEqual(self.original.read_bytes(), self.old)
        self.assertEqual(mod.read_journal(self.original)['phase'], 'ROLLED_BACK')

    def test_interruption_after_epoch_rotation_recovers(self):
        real_replace = mod.os.replace

        def crash_on_swap(src, dst):
            if str(dst) == str(self.original):
                raise KeyboardInterrupt('simulated SIGKILL before atomic swap')
            return real_replace(src, dst)

        with mock.patch.object(mod.os, 'replace', side_effect=crash_on_swap):
            with self.assertRaises(KeyboardInterrupt):
                self.activate()
        self.assertEqual(mod.read_journal(self.original)['phase'], 'EPOCH_ROTATED')
        self.assertEqual(self.original.read_bytes(), self.old)
        record = mod.rollback(self.original, timeout=2)
        self.assertEqual(record['phase'], 'ROLLED_BACK')
        self.assertEqual(self.original.read_bytes(), self.old)

    def test_interruption_after_swap_recovers_even_if_journal_is_stale(self):
        real_write = mod.write_journal

        def crash_after_swap(original, record, **kwargs):
            if record['phase'] == 'SWAPPED_UNCERTIFIED':
                raise KeyboardInterrupt('simulated SIGKILL after atomic swap')
            return real_write(original, record, **kwargs)

        with mock.patch.object(mod, 'write_journal', side_effect=crash_after_swap):
            with self.assertRaises(KeyboardInterrupt):
                self.activate()
        self.assertEqual(self.original.read_bytes(), self.new)
        self.assertEqual(mod.read_journal(self.original)['phase'], 'EPOCH_ROTATED')
        record = mod.rollback(self.original, timeout=2)
        self.assertEqual(record['phase'], 'ROLLED_BACK')
        self.assertEqual(self.original.read_bytes(), self.old)

    def test_rollback_denied_after_unknown_external_mutation(self):
        self.activate()
        self.original.write_bytes(b'unknown-post-swap-state')
        with self.assertRaisesRegex(mod.PromotionError, 'changed since activation'):
            mod.rollback(self.original, timeout=2)
        self.assertEqual(self.original.read_bytes(), b'unknown-post-swap-state')
        self.assertEqual(mod.read_journal(self.original)['phase'], 'ACTIVE_UNCERTIFIED')

    def test_wal_and_hash_drift_fail_closed(self):
        Path(str(self.original) + '.wal').write_text('writer active')
        with self.assertRaisesRegex(mod.PromotionError, 'WAL'):
            self.activate()
        Path(str(self.original) + '.wal').unlink()
        self.candidate.write_bytes(b'unexpected-data')
        with self.assertRaisesRegex(mod.PromotionError, 'SHA-256 drift'):
            self.activate()
        self.assertFalse(mod.journal_path(self.original).exists())
        self.assertEqual(self.original.read_bytes(), self.old)

    def test_no_promotion_without_explicit_acknowledgement(self):
        rc = mod.main([
            'promote', '--original', str(self.original),
            '--candidate', str(self.candidate),
            '--original-sha256', self.oh,
            '--candidate-sha256', self.ch,
        ])
        self.assertEqual(rc, 2)
        self.assertEqual(self.original.read_bytes(), self.old)
        self.assertFalse(mod.journal_path(self.original).exists())


if __name__ == '__main__':
    unittest.main()
