"""File-backed V2 promotion/recovery falsifiers with genuine Linux SIGKILL."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import preflight_candidate_cube_promotion_v2 as gate
from scripts import promote_candidate_cube_v2 as v2
import test_preflight_candidate_cube_promotion_v2 as gate_tests


def die_on(boundary):
    def kill(label):
        if label == boundary:
            os.kill(os.getpid(), signal.SIGKILL)
    return kill


def child_activate(db: str, boundary: str):
    v2.activate(main=Path(db), timeout=1.0,
                validator=lambda db, anchor: None, hook=die_on(boundary))


class CandidateCubeV2StateTests(unittest.TestCase):
    def setUp(self):
        self.f = gate_tests.CubePromotionPreflightTests('test_full_legacy_anchor_and_candidate_provenance_pass_readonly')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.main = self.f.main
        self.orig = evidence.digest(self.main)
        self.v1journalhash = evidence.digest(self.f.journal)
        self.v1backuphash = evidence.digest(self.f.backup)
        self.clonehash = evidence.digest(self.f.clone)
        self.approval = self.f.dir / 'verified-approval.json'
        gate_record = self.f.run_gate()
        evidence.write_create_only(self.approval, gate_record)
        self.approvalhash = evidence.digest(self.approval)

    def prepare(self):
        return v2.prepare(main=self.main, approval_path=self.approval,
                          approval_sha=self.approvalhash, timeout=1.0)

    def assert_history(self):
        self.assertEqual(evidence.digest(self.f.journal), self.v1journalhash)
        self.assertEqual(evidence.digest(self.f.backup), self.v1backuphash)
        self.assertEqual(evidence.digest(self.f.clone), self.clonehash)

    def test_prepare_never_mutates_main_or_epoch_and_keeps_v1(self):
        epoch = gate.lifecycle_epoch(self.main)
        self.assertEqual(self.prepare()['phase'], 'PREPARED')
        self.assertEqual(evidence.digest(self.main), self.orig)
        self.assertEqual(gate.lifecycle_epoch(self.main), epoch)
        self.assertEqual(v2.status(self.main)['phase'], 'PREPARED')
        self.assert_history()
        with self.assertRaisesRegex(v2.PromotionV2Error, 'existing V2 state'):
            self.prepare()

    def test_actives_are_not_freshness_certificates(self):
        self.prepare()
        out = v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        self.assertEqual(out['phase'], 'ACTIVE_UNCERTIFIED')
        self.assertFalse(out['certified'])
        self.assertEqual(evidence.digest(self.main), self.clonehash)
        s = v2.status(self.main)
        self.assertFalse(s['recovery_required'])
        self.assertEqual(s['events'], 7)
        self.assert_history()

    def test_explicit_recover_from_success_restores_baseline_and_revokes(self):
        initial_epoch = gate.lifecycle_epoch(self.main)
        self.prepare()
        v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        self.assertEqual(v2.recover(main=self.main, timeout=1)['phase'], 'ROLLED_BACK')
        self.assertEqual(evidence.digest(self.main), self.orig)
        self.assertNotEqual(gate.lifecycle_epoch(self.main), initial_epoch)
        self.assertEqual(v2.status(self.main)['phase'], 'ROLLED_BACK')
        self.assertFalse(v2.status(self.main)['recovery_required'])
        self.assert_history()
        self.assertTrue(v2.recover(main=self.main, timeout=1)['already_done'])

    def test_smoke_failure_automatically_restores_baseline(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'smoke intentionally failed'):
            v2.activate(main=self.main, timeout=1,
                        validator=lambda *_: (_ for _ in ()).throw(ValueError('smoke intentionally failed')))
        self.assertEqual(evidence.digest(self.main), self.orig)
        self.assertEqual(v2.status(self.main)['phase'], 'ROLLED_BACK')
        self.assert_history()

    def test_reject_unapproved_warehouse_mutation_before_epoch(self):
        self.prepare()
        self.main.write_bytes(b'unapproved')
        with self.assertRaisesRegex(v2.PromotionV2Error, 'unknown'):
            v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        self.assertEqual(v2.read_ledger(v2.state_dir(self.main))[-1][1]['phase'], 'PREPARED')

    def test_reject_changed_immutable_v1_journal(self):
        self.prepare()
        self.f.journal.write_bytes(b'tampered')
        with self.assertRaisesRegex(v2.PromotionV2Error, 'V1 history'):
            v2.status(self.main)

    def test_reject_corrupted_ledger_before_epoch(self):
        self.prepare()
        (v2.state_dir(self.main) / '0002-malicious.json').write_text('{}')
        with self.assertRaisesRegex(v2.PromotionV2Error, 'corrupt'):
            v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        self.assertEqual(evidence.digest(self.main), self.orig)

    def test_reject_missing_immutable_backup(self):
        self.prepare()
        ledger, anchor = v2._load_and_check(self.main)
        Path(anchor['backup']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(v2.PromotionV2Error, 'durable promotion copy drift'):
            v2.activate(main=self.main, timeout=1, validator=lambda *_: None)

    def test_status_flags_wal_even_when_active_hash_is_correct(self):
        self.prepare()
        v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        Path(str(self.main) + '.wal').write_bytes(b'not a valid WAL')
        observed = v2.status(self.main)
        self.assertTrue(observed['wal_present'])
        self.assertTrue(observed['recovery_required'])

    def test_rollback_denies_unexpected_clone_before_intent(self):
        self.prepare()
        self.main.write_bytes(self.f.clone.read_bytes())
        with self.assertRaisesRegex(v2.PromotionV2Error, 'unexpected active bytes'):
            v2.recover(main=self.main, timeout=1)

    def test_reject_wal_before_activation(self):
        self.prepare()
        Path(str(self.main) + '.wal').write_bytes(b'wal')
        with self.assertRaises(Exception):
            v2.activate(main=self.main, timeout=1, validator=lambda *_: None)

    def test_reject_approval_hash_drift(self):
        self.approval.write_text('tampered')
        with self.assertRaisesRegex(v2.PromotionV2Error, 'approval SHA'):
            self.prepare()
        self.assertFalse(v2.state_dir(self.main).exists())

    def test_cli_does_not_activate_without_explicit_ack(self):
        self.prepare()
        self.assertEqual(v2.main(['activate','--main-db',str(self.main)]),2)
        self.assertEqual(evidence.digest(self.main),self.orig)

    def test_cancel_prepared_does_not_rotate_epoch(self):
        original_epoch = gate.lifecycle_epoch(self.main)
        self.prepare()
        self.assertEqual(v2.recover(main=self.main, timeout=1)['phase'], 'ROLLED_BACK')
        self.assertEqual(gate.lifecycle_epoch(self.main), original_epoch)
        self.assertEqual(evidence.digest(self.main), self.orig)
        self.assert_history()

    def test_kill_during_preparation_recover_without_backup(self):
        # Fail after ADMITTED ledger but before durable baseline copy finishes.
        context = multiprocessing.get_context('fork')
        def child():
            with patch.object(v2, 'durable_copy', lambda *_args: os.kill(os.getpid(), signal.SIGKILL)):
                self.prepare()
        proc = context.Process(target=child)
        proc.start()
        proc.join(timeout=15)
        self.assertEqual(proc.exitcode, -signal.SIGKILL)
        self.assertEqual(v2.status(self.main)['phase'], 'ADMITTED')
        self.assertEqual(v2.recover(main=self.main, timeout=1)['phase'], 'ROLLED_BACK')
        self.assertEqual(evidence.digest(self.main), self.orig)
        self.assertEqual(gate.lifecycle_epoch(self.main), 'a' * 32)
        self.assert_history()

    def test_activation_denies_stale_lifecycle_and_preserves_main(self):
        self.prepare()
        from scripts.warehouse_lifecycle import rotate_epoch
        rotate_epoch(self.main)
        with self.assertRaisesRegex(v2.PromotionV2Error, 'lifecycle drift'):
            v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        self.assertEqual(evidence.digest(self.main), self.orig)
        self.assert_history()

    def test_activation_denies_approval_mutation_without_epoch_change(self):
        self.prepare()
        self.approval.write_text('changed')
        with self.assertRaisesRegex(v2.PromotionV2Error, 'preflight changed'):
            v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        self.assertEqual(evidence.digest(self.main), self.orig)

    def test_crash_recovery_each_durable_boundary(self):
        boundaries = ('after_epoch_rotation_intent', 'after_epoch_rotation',
                      'after_epoch_rotated_event', 'after_swap_intent',
                      'after_swap', 'before_live_smoke', 'after_live_smoke')
        # One independent POSIX filesystem fixture per subcase.
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                if v2.state_dir(self.main).exists():
                    # complete previous test scenario, then recreate fixture
                    self.f.doCleanups()
                    self.f = gate_tests.CubePromotionPreflightTests('test_full_legacy_anchor_and_candidate_provenance_pass_readonly')
                    self.f.setUp()
                    self.main = self.f.main
                    self.approval = self.f.dir / 'verified-approval.json'
                    evidence.write_create_only(self.approval, self.f.run_gate())
                    self.approvalhash = evidence.digest(self.approval)
                    self.orig = evidence.digest(self.main)
                    self.v1journalhash = evidence.digest(self.f.journal)
                    self.v1backuphash = evidence.digest(self.f.backup)
                    self.clonehash = evidence.digest(self.f.clone)
                self.prepare()
                ctx = multiprocessing.get_context('fork')
                proc = ctx.Process(target=child_activate, args=(str(self.main), boundary))
                proc.start()
                proc.join(timeout=15)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=3)
                    self.fail('subprocess did not reach injected SIGKILL boundary')
                self.assertEqual(proc.exitcode, -signal.SIGKILL)
                observed = v2.status(self.main)
                self.assertTrue(observed['recovery_required'])
                restored = v2.recover(main=self.main, timeout=2)
                self.assertEqual(restored['phase'], 'ROLLED_BACK')
                self.assertEqual(evidence.digest(self.main), self.orig)
                self.assert_history()

    def test_rollback_killed_after_intent_can_be_resumed(self):
        self.prepare()
        v2.activate(main=self.main, timeout=1, validator=lambda *_: None)
        ctx = multiprocessing.get_context('fork')
        def child():
            v2.recover(main=self.main, timeout=1, hook=die_on('after_rollback_epoch'))
        p = ctx.Process(target=child)
        p.start()
        p.join(timeout=15)
        self.assertEqual(p.exitcode, -signal.SIGKILL)
        self.assertEqual(v2.status(self.main)['phase'], 'ROLLBACK_INTENT')
        result = v2.recover(main=self.main, timeout=1)
        self.assertEqual(result['phase'], 'ROLLED_BACK')
        self.assertEqual(evidence.digest(self.main), self.orig)


if __name__ == '__main__':
    unittest.main()
