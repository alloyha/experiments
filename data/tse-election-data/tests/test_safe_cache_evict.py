"""Operational gate: fail-closed planner, byte match, journal and restart."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import safe_cache_evict as gate


class SafeEvictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'tse'
        self.meta = self.root / '_metadata'
        self.meta.mkdir(parents=True)
        self.version = (self.root / 'raw/election_type=general/year=2026/domain=vote_result/'
                        'dataset=x/resource=r1/sha256=placeholder')
        self.source = self.version / 'source' / 'v.zip'
        self.source.parent.mkdir(parents=True)
        self.member = 'votes_BRASIL.csv'
        self.payload = b'ANO_ELEICAO;VOTOS\n2026;3\n'
        with zipfile.ZipFile(self.source, 'w') as z:
            z.writestr(self.member,self.payload)
        self.digest = gate.base.sha256(self.source)
        newdir=self.version.with_name('sha256='+self.digest)
        self.version.rename(newdir)
        self.version=newdir
        self.source=self.version/'source/v.zip'
        self.output=self.version/'extracted'/self.member
        self.output.parent.mkdir()
        self.output.write_bytes(self.payload)
        self.srel=self.source.relative_to(self.root).as_posix()
        self.orel=self.output.relative_to(self.root).as_posix()
        self.row=dict(year=2026,election_type='general',domain='vote_result',
                      resource_id='r1',source_sha256=self.digest,source_size_bytes=self.source.stat().st_size,
                      source_object=self.srel,extracted_objects=[self.orel],selected_members=[self.member],
                      selection_complete=True,materialization_state='present')
        self.state={'cycle|2026|r1':self.row}
        (self.meta / gate.STATE).write_text(json.dumps(self.state))
        (self.meta / gate.INDEX).write_text(json.dumps(dict(domain='vote_result',year=2026,election_type='general',
            resource_id='r1',source_object=self.srel,object=self.orel))+'\n')
        (self.meta / gate.MANIFEST).write_text(json.dumps(dict(source_object=self.srel,
            resource_id='r1',source_sha256=self.digest))+'\n')

    def plan(self):
        return gate.build_plan(self.root, {'vote_result'}, {2026}, 1, 1048576, 1048576)

    def test_plan_is_read_only_and_has_exact_byte_proof(self):
        before={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        p=self.plan()
        self.assertEqual(p['resource_count'],1)
        self.assertEqual(p['reclaimable_bytes'],len(self.payload))
        self.assertEqual(p['resources'][0]['outputs'][0]['sha256'],hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(before,{p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_second_operation_fails_fast_on_lock_contention(self):
        with gate.exclusive(self.root):
            with self.assertRaisesRegex(gate.Error, 'holding the lock'):
                with gate.exclusive(self.root):
                    self.fail('should not acquire the lock twice')

    def test_plan_prints_progress_to_stderr(self):
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = self.plan()
        self.assertEqual(result['resource_count'], 1)
        self.assertIn('[verify 1/1]', output.getvalue())
        self.assertIn('[verified 1]', output.getvalue())

    def test_corrupted_source_rejected(self):
        self.source.write_bytes(b'untrusted')
        with self.assertRaises(Exception):self.plan()

    def test_corrupted_csv_rejected(self):
        self.output.write_bytes(b'not the archive member')
        with self.assertRaisesRegex(gate.Error,'differs'):self.plan()

    def test_stale_plan_rejected(self):
        p=self.plan()
        self.row['checked_at']='new'
        (self.meta/gate.STATE).write_text(json.dumps(self.state))
        with self.assertRaisesRegex(gate.Error,'stale plan'):gate.validate_plan(self.root,p)

    def test_missing_index_rejected(self):
        (self.meta/gate.INDEX).write_text('')
        with self.assertRaises(Exception):self.plan()

    def test_eviction_keeps_immutable_source_and_journals_metadata(self):
        p=self.plan(); j=gate.build_journal(self.root,p)
        gate.atomic_bytes(self.meta/gate.JOURNAL,gate.canonical(j))
        removed=gate.recover_pending(self.root)
        self.assertEqual(removed,1)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.source.exists())
        self.assertFalse((self.meta/gate.JOURNAL).exists())
        self.assertEqual(json.loads((self.meta/gate.STATE).read_text())['cycle|2026|r1']['materialization_state'],'evicted')
        self.assertEqual((self.meta/gate.INDEX).read_text(),'')

    def test_restart_after_partial_metadata_commit(self):
        p=self.plan();j=gate.build_journal(self.root,p)
        gate.atomic_bytes(self.meta/gate.JOURNAL,gate.canonical(j))
        gate.atomic_bytes(self.meta/gate.STATE,j['post'][gate.STATE].encode())
        self.assertEqual(gate.recover_pending(self.root),1)
        self.assertEqual((self.meta/gate.INDEX).read_text(),'')
        self.assertEqual(gate.recover_pending(self.root),0)

    def test_restart_after_deletion(self):
        p=self.plan();j=gate.build_journal(self.root,p)
        gate.atomic_bytes(self.meta/gate.JOURNAL,gate.canonical(j))
        gate.atomic_bytes(self.meta/gate.STATE,j['post'][gate.STATE].encode())
        gate.atomic_bytes(self.meta/gate.INDEX,j['post'][gate.INDEX].encode())
        self.output.unlink()
        self.assertEqual(gate.recover_pending(self.root),0)
        self.assertTrue(self.source.exists())

    def test_abort_if_concurrent_metadata_writer_after_journal(self):
        p=self.plan();j=gate.build_journal(self.root,p)
        gate.atomic_bytes(self.meta/gate.JOURNAL,gate.canonical(j))
        (self.meta/gate.STATE).write_text('{}')
        with self.assertRaisesRegex(gate.Error,'another writer'):gate.recover_pending(self.root)
        self.assertTrue(self.output.exists())

    def test_abort_if_target_modified_after_journal(self):
        p=self.plan();j=gate.build_journal(self.root,p)
        gate.atomic_bytes(self.meta/gate.JOURNAL,gate.canonical(j))
        self.output.write_text('corrupted')
        with self.assertRaisesRegex(gate.Error,'target changed'):gate.recover_pending(self.root)
        self.assertTrue(self.output.exists())

    def test_apply_requires_ack_and_proof(self):
        rc=gate.main(['--root',str(self.root),'--apply'])
        self.assertEqual(rc,2)
        self.assertTrue(self.output.exists())

    def test_apply_refuses_canary_from_different_plan(self):
        p=self.plan()
        with tempfile.TemporaryDirectory() as td:
            plan_path=Path(td)/'plan.json';canary=Path(td)/'canary.json'
            plan_path.write_bytes(gate.canonical(p))
            canary.write_text(json.dumps({'result':'PASS','scratch_deleted':True,'plan_sha256':'wrong'}))
            rc=gate.main(['--root',str(self.root),'--apply','--ack-quiescent',
                          '--plan-file',str(plan_path),'--canary-report',str(canary)])
            self.assertEqual(rc,2)
            self.assertTrue(self.output.exists())

    def test_scoped_canary_reusable_and_bounded_apply(self):
        p=self.plan()
        with tempfile.TemporaryDirectory() as td:
            plan_path=Path(td)/'plan.json'; canary=Path(td)/'canary.json'
            plan_path.write_bytes(gate.canonical(p))
            canary.write_bytes(gate.canonical({
                'result':'PASS','schema_version':1,'scratch_deleted':True,
                'canonical_read_only':True,'root':str(self.root),
                'tool_sha256':gate.base.sha256(Path(gate.__file__)),
                'certified_scopes':[{'year':2026,'domain':'vote_result'}],
                'plan_sha256':'different-plan-can-still-be-authorized-when-source-verified'}))
            rc=gate.main(['--root',str(self.root),'--apply','--ack-quiescent',
                '--plan-file',str(plan_path),'--canary-report',str(canary)])
            self.assertEqual(rc,0)
            self.assertFalse(self.output.exists())
            self.assertTrue(self.source.exists())

    def test_scope_mismatch_canary_rejected(self):
        p=self.plan()
        with tempfile.TemporaryDirectory() as td:
            plan_path=Path(td)/'plan.json'; canary=Path(td)/'canary.json'
            plan_path.write_bytes(gate.canonical(p))
            canary.write_bytes(gate.canonical({
                'result':'PASS','schema_version':1,'scratch_deleted':True,
                'canonical_read_only':True,'root':str(self.root),
                'tool_sha256':gate.base.sha256(Path(gate.__file__)),
                'certified_scopes':[{'year':2024,'domain':'candidate'}]}))
            rc=gate.main(['--root',str(self.root),'--apply','--ack-quiescent',
                '--plan-file',str(plan_path),'--canary-report',str(canary)])
            self.assertEqual(rc,2)
            self.assertTrue(self.output.exists())

    def test_manifest_is_required_for_authorization(self):
        (self.meta/gate.MANIFEST).unlink()
        with self.assertRaises(Exception):self.plan()


    def verified_run(self, *, plan=None, expected=None, ack=True, extra=()):
        plan = plan if plan is not None else self.plan()
        with tempfile.TemporaryDirectory() as td:
            plan_file = Path(td) / 'plan.json'
            plan_file.write_bytes(gate.canonical(plan))
            cmd = ['--root', str(self.root), '--apply-verified',
                   '--plan-file', str(plan_file)]
            if ack:
                cmd.append('--ack-quiescent')
            if expected is not None:
                cmd.extend(['--expect-plan-sha256', expected])
            cmd.extend(extra)
            return gate.main(cmd)

    def test_verified_only_eviction_without_canary_or_scratch(self):
        p = self.plan()
        digest = gate.hash_bytes(gate.canonical(p))
        # The temp directory used for the plan is created outside the guarded call.
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / 'plan.json'
            plan_path.write_bytes(gate.canonical(p))
            with mock.patch.object(gate, 'canary', side_effect=AssertionError('canary')):
                with mock.patch.object(gate.tempfile, 'TemporaryDirectory',
                                       side_effect=AssertionError('scratch')):
                    rc = gate.main(['--root', str(self.root), '--apply-verified',
                                    '--plan-file', str(plan_path), '--ack-quiescent',
                                    '--expect-plan-sha256', digest])
        self.assertEqual(rc, 0)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.source.exists())
        self.assertFalse((self.meta / gate.JOURNAL).exists())
        self.assertEqual(json.loads((self.meta / gate.STATE).read_text())['cycle|2026|r1']['materialization_state'], 'evicted')
        self.assertEqual((self.meta / gate.INDEX).read_text(), '')

    def test_verified_only_requires_reviewed_plan_sha(self):
        p = self.plan()
        self.assertEqual(self.verified_run(plan=p), 2)
        self.assertEqual(self.verified_run(plan=p, expected='0' * 64), 2)
        self.assertTrue(self.output.exists())

    def test_verified_only_requires_quiescence(self):
        p = self.plan()
        sha = gate.hash_bytes(gate.canonical(p))
        self.assertEqual(self.verified_run(plan=p, expected=sha, ack=False), 2)
        self.assertTrue(self.output.exists())

    def test_verified_only_rejects_stale_metadata(self):
        p = self.plan()
        sha = gate.hash_bytes(gate.canonical(p))
        self.row['extra'] = 'concurrent writer'
        (self.meta / gate.STATE).write_text(json.dumps(self.state))
        self.assertEqual(self.verified_run(plan=p, expected=sha), 2)
        self.assertTrue(self.output.exists())

    def test_verified_only_rejects_changed_source_or_csv(self):
        p = self.plan()
        sha = gate.hash_bytes(gate.canonical(p))
        self.output.write_bytes(b'changed extracted content')
        self.assertEqual(self.verified_run(plan=p, expected=sha), 2)
        self.assertTrue(self.output.exists())
        self.output.write_bytes(self.payload)
        self.source.write_bytes(b'broken source')
        self.assertEqual(self.verified_run(plan=p, expected=sha), 2)
        self.assertTrue(self.output.exists())

    def test_verified_only_rejects_pending_journal(self):
        p = self.plan()
        sha = gate.hash_bytes(gate.canonical(p))
        (self.meta / gate.JOURNAL).write_text('pending')
        self.assertEqual(self.verified_run(plan=p, expected=sha), 2)
        self.assertTrue(self.output.exists())

    def test_verified_only_rejects_tampered_plan_totals(self):
        p = self.plan()
        p['resource_count'] = 2
        sha = gate.hash_bytes(gate.canonical(p))
        self.assertEqual(self.verified_run(plan=p, expected=sha), 2)
        self.assertTrue(self.output.exists())

    def test_verified_only_rejects_canary_report_argument(self):
        p = self.plan()
        sha = gate.hash_bytes(gate.canonical(p))
        self.assertEqual(self.verified_run(plan=p, expected=sha,
                                           extra=['--canary-report', '/dev/null']), 2)
        self.assertTrue(self.output.exists())


if __name__ == '__main__':unittest.main()
