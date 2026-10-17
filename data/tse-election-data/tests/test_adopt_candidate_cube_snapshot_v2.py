"""Prepared vs activated snapshot adoption; synthetic provenance and filesystem.

The real DuckDB SQL checks already ran in the offline contract gate. These
falsifiers test V2 authority, certificate publication atomicity and fail closed
behaviors. Output signatures are faked here; WSL runs a real-DB integration
through the existing offline and candidate-cube freshness adapters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import adopt_candidate_cube_snapshot_v2 as adoption
from scripts import candidate_cube_freshness as freshness
from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import preflight_candidate_cube_promotion_v2 as gate
from scripts import promote_candidate_cube_v2 as promoter
import test_preflight_candidate_cube_promotion_v2 as fixture


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.CubePromotionPreflightTests('test_full_legacy_anchor_and_candidate_provenance_pass_readonly')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.main = self.f.main
        self.approval = self.f.dir / 'approval.json'
        evidence.write_create_only(self.approval, self.f.run_gate())
        self.old_hash = evidence.digest(self.main)
        promoter.prepare(main=self.main, approval_path=self.approval,
                         approval_sha=evidence.digest(self.approval), timeout=2)
        self.certdir = self.f.dir / 'certifications-v2'

    def activate(self):
        promoter.activate(main=self.main, timeout=2,
                          validator=lambda _path, _anchor: None)

    def fake_freshness(self, fail_cycle=None):
        def signature(db, root, cycle):
            if cycle == fail_cycle:
                raise RuntimeError('injected failed fingerprint')
            return {'source': self.f.cycles[cycle]['source_signature'], 'dimensions': {'synthetic': True}}
        def output(db, cycle):
            return {'cube': {'rows': 3, 'nominal_votes': 11,
                             'columns': [f'col{i}' for i in range(29)]}}
        def certpath(folder, cycle):
            return Path(folder) / (cycle.replace(':', '-') + '.json')
        def issue(folder, db, root, cycle, inputs, outputs):
            certpath(folder, cycle).write_text(json.dumps({'cycle': cycle, 'db': str(db),
                                    'input': inputs, 'output': outputs}), encoding='utf8')
        # Patch the actual, already-imported module rather than sys.modules.
        # A full unittest discovery run may have cached it on the scripts
        # package, so swapping sys.modules alone does not intercept imports.
        return patch.multiple(
            freshness,
            input_signature=signature,
            output_signature=output,
            certificate_path=certpath,
            issue_certificate=issue,
        )

    def test_freshness_mock_handles_preimported_module_and_restores(self):
        # Full unittest discovery imports candidate_cube_freshness before our tests.
        from scripts import candidate_cube_freshness as cached
        self.assertIs(cached, freshness)
        original = cached.input_signature
        with self.fake_freshness():
            self.assertIsNot(cached.input_signature, original)
            actual = cached.input_signature(self.main, self.f.root, '2018:general')
            self.assertEqual(actual['source'], self.f.cycles['2018:general']['source_signature'])
        self.assertIs(cached.input_signature, original)

    def test_prepared_readiness_passes_but_adoption_denied(self):
        report = adoption.readiness(self.main, timeout=2)
        self.assertEqual(report['phase'], 'PREPARED')
        self.assertFalse(report['ready_for_adoption'])
        with self.assertRaisesRegex(adoption.AdoptionError, 'phase'):
            adoption.adopt(main=self.main, cert_dir=self.certdir, timeout=2,
                           ack_quiescent=True, confirm_adopt=True)
        self.assertEqual(evidence.digest(self.main), self.old_hash)
        self.assertFalse(self.certdir.exists())

    def test_requires_explicit_ack(self):
        self.activate()
        with self.assertRaisesRegex(adoption.AdoptionError, 'requires'):
            adoption.adopt(main=self.main, cert_dir=self.certdir, timeout=2,
                           ack_quiescent=False, confirm_adopt=True)
        self.assertFalse(self.certdir.exists())

    def test_activated_readiness_and_atomic_full_certificate_bundle(self):
        self.activate()
        self.assertTrue(adoption.readiness(self.main, timeout=2)['ready_for_adoption'])
        with self.fake_freshness():
            result = adoption.adopt(main=self.main, cert_dir=self.certdir, timeout=2,
                                    ack_quiescent=True, confirm_adopt=True)
        self.assertEqual(result['result'], 'PASS')
        self.assertEqual(result['cycle_count'], 5)
        self.assertEqual(len(list(self.certdir.glob('*.json'))), 6)
        self.assertEqual(json.loads((self.certdir/'adoption.json').read_text())['certificates_complete'], True)
        self.assertEqual(evidence.digest(self.main), self.f.contract_data['clone_sha256'])
        self.assertFalse(result['graph_freshness_certified'])

    def test_create_only_destination_not_overwritten(self):
        self.activate()
        self.certdir.mkdir()
        (self.certdir / 'kept.txt').write_text('do not touch')
        with self.assertRaisesRegex(adoption.AdoptionError, 'destination'):
            adoption.adopt(main=self.main, cert_dir=self.certdir, timeout=2,
                           ack_quiescent=True, confirm_adopt=True)
        self.assertEqual((self.certdir/'kept.txt').read_text(), 'do not touch')

    def test_midcycle_failure_does_not_publish_partial_certificates(self):
        self.activate()
        with self.fake_freshness(fail_cycle='2022:general'):
            with self.assertRaisesRegex(RuntimeError, 'injected failed'):
                adoption.adopt(main=self.main, cert_dir=self.certdir, timeout=2,
                               ack_quiescent=True, confirm_adopt=True)
        self.assertFalse(self.certdir.exists())
        self.assertTrue(list(self.f.dir.glob('.certifications-v2.*.building')))

    def test_prepared_tampered_manifest_fails_closed(self):
        self.f.manifest.write_text('{"result":"tampered"}')
        with self.assertRaises(Exception):
            adoption.readiness(self.main, timeout=2)

    def test_activated_tampered_sql_fails_closed(self):
        self.activate()
        sql = next(iter(self.f.sql_root.rglob('assert_cube_candidate_vote_conservation.sql')))
        sql.write_text('select 1')
        with self.assertRaises(Exception):
            adoption.readiness(self.main, timeout=2)

    def test_activated_parquet_drift_fails_closed(self):
        self.activate()
        part = evidence.partition_for(self.f.root, '2026:general')
        (part / 'data.parquet').write_bytes(b'changed bytes')
        with self.assertRaisesRegex(adoption.AdoptionError, 'Parquet drift'):
            adoption.readiness(self.main, timeout=2)

    def test_activated_epoch_reversal_fails_closed(self):
        self.activate()
        (self.f.dir / 'main.duckdb.lifecycle').write_text('a'*32+'\n')
        with self.assertRaisesRegex(adoption.AdoptionError, 'not rotated'):
            adoption.readiness(self.main, timeout=2)

    def test_after_recover_denies_adoption(self):
        self.activate()
        promoter.recover(main=self.main, timeout=2)
        with self.assertRaisesRegex(adoption.AdoptionError, 'phase'):
            adoption.readiness(self.main, timeout=2)

    @unittest.skipUnless(os.name=='posix', 'requires Linux')
    def test_createonly_rename_preserves_existing(self):
        src = self.f.dir/'src-dir'; src.mkdir()
        dst = self.f.dir/'target-dir'; dst.mkdir()
        (dst/'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):
            adoption._linux_rename_noreplace(src, dst)
        self.assertTrue(src.is_dir())
        self.assertEqual((dst/'sentinel').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
