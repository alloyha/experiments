"""PR42 safety-first fixtures. No network, no original raw data access."""
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
import certify_cache_roundtrip as c


class CacheRoundtripContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='pr42-fixture-')
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / 'canonical'
        self.meta = self.root / '_metadata'
        self.meta.mkdir(parents=True)
        self.resource_id = 'synthetic-resource'
        self.year, self.domain = 2026, 'candidate_history'
        self.csv = b'ID;VOTOS\n1;10\n2;20\n'
        prepared_zip = self.base / 'scratch.zip'
        with zipfile.ZipFile(prepared_zip, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('historico_candidatura_2026_BRASIL.csv', self.csv)
        self.sha = c.sha256(prepared_zip)
        version = (self.root / 'raw/election_type=general/year=2026/'
                   'domain=candidate_history/dataset=candidatos_2026/'
                   f'resource=synthetic/sha256={self.sha}')
        self.source = version / 'source' / 'historico_candidatura_2026.zip'
        self.extracted = version / 'extracted' / 'historico_candidatura_2026_BRASIL.csv'
        self.source.parent.mkdir(parents=True)
        self.extracted.parent.mkdir(parents=True)
        self.source.write_bytes(prepared_zip.read_bytes())
        self.extracted.write_bytes(self.csv)
        self.src_rel = self.source.relative_to(self.root).as_posix()
        self.obj_rel = self.extracted.relative_to(self.root).as_posix()
        self.row = {
            'year': 2026, 'domain': self.domain, 'resource_id': self.resource_id,
            'election_type': 'general', 'dataset_id': 'candidatos_2026',
            'partition': 'BRASIL', 'election_scope': 'national',
            'source_object': self.src_rel, 'source_sha256': self.sha,
            'source_size_bytes': self.source.stat().st_size,
            'extracted_objects': [self.obj_rel],
            'selected_members': ['historico_candidatura_2026_BRASIL.csv'],
            'materialization_state': 'present', 'selection_complete': True,
            'granularity': 'brasil', 'uf': '',
        }
        self.key = f'general|{self.year}|{self.resource_id}|brasil|'
        self.index = [{'source_object': self.src_rel, 'object': self.obj_rel}]
        self.write_meta()
        self.project = self.base / 'project'
        scripts = self.project / 'scripts'
        scripts.mkdir(parents=True)
        for name in ('prune_extracted_raw.py', 'ensure_materialized.py',
                     'check_raw_contracts.py', 'report_local_storage.py'):
            (scripts / name).write_text('# synthetic command placeholder\n')

    def write_meta(self):
        (self.meta / 'resource_state.json').write_text(json.dumps({self.key: self.row}))
        (self.meta / 'current_objects.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in self.index))
        (self.meta / 'ingest_manifest.jsonl').write_text(json.dumps({'resource_id': self.resource_id, 'source_sha256': self.sha, 'source_object': self.src_rel}) + '\n')
        (self.meta / 'pending_resources.json').write_text('{}\n')

    def selected(self):
        return c.validate_resource(self.root, self.resource_id, self.year,
                                   self.domain, 64 * 1024**2, 512 * 1024**2)

    def test_selection_and_integrity(self):
        result = self.selected()
        self.assertEqual(result['outputs'], [self.obj_rel])
        self.assertEqual(result['source_bytes'], self.source.stat().st_size)

    def test_legacy_missing_materialization_field_requires_optin(self):
        self.row.pop('materialization_state')
        self.write_meta()
        original = self.meta.joinpath('resource_state.json').read_bytes()
        with self.assertRaisesRegex(c.CertificationError, 'not completely materialized'):
            self.selected()
        chosen = c.validate_resource(
            self.root, self.resource_id, self.year, self.domain,
            1_000_000, 1_000_000, allow_legacy_materialization=True,
        )
        self.assertTrue(chosen['legacy_materialization_state_missing'])
        self.assertEqual(original, self.meta.joinpath('resource_state.json').read_bytes())

    def test_explicit_null_or_evicted_not_accepted_as_legacy(self):
        for value in (None, 'evicted', 'not_required'):
            with self.subTest(value=value):
                self.row['materialization_state'] = value
                self.write_meta()
                with self.assertRaisesRegex(c.CertificationError, 'not completely materialized'):
                    c.validate_resource(self.root, self.resource_id, self.year,
                        self.domain, 1_000_000, 1_000_000,
                        allow_legacy_materialization=True)

    def test_legacy_missing_index_or_payload_is_never_accepted(self):
        self.row.pop('materialization_state')
        self.index.clear()
        self.write_meta()
        with self.assertRaisesRegex(c.CertificationError, 'current index'):
            c.validate_resource(self.root, self.resource_id, self.year,
                self.domain, 1_000_000, 1_000_000,
                allow_legacy_materialization=True)
        self.index.append({'source_object': self.src_rel, 'object': self.obj_rel})
        self.write_meta()
        self.extracted.unlink()
        with self.assertRaisesRegex(c.CertificationError, 'extracted object'):
            c.validate_resource(self.root, self.resource_id, self.year,
                self.domain, 1_000_000, 1_000_000,
                allow_legacy_materialization=True)

    def test_deny_copy_to_path_outside_sandbox(self):
        with self.assertRaisesRegex(c.CertificationError, 'unsafe'):
            c.require_path(self.root, '../../secret', kind='test')
        with self.assertRaisesRegex(c.CertificationError, 'unsafe'):
            c.require_path(self.root, '/etc/passwd', kind='test')

    def test_symlink_source_denied(self):
        self.source.unlink()
        self.source.symlink_to(self.base / 'scratch.zip')
        with self.assertRaises(c.CertificationError):
            self.selected()

    def test_source_must_be_published_in_append_only_manifest(self):
        (self.meta / 'ingest_manifest.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(c.CertificationError, 'absent from append-only manifest'):
            self.selected()

    def test_digest_mismatch_is_hard_failure(self):
        self.source.write_bytes(b'bad ZIP bytes')
        with self.assertRaisesRegex(c.CertificationError, 'invalid ZIP'):
            self.selected()

    def test_digest_path_mismatch_is_hard_failure(self):
        self.row['source_sha256'] = '0' * 64
        self.write_meta()
        (self.meta / 'ingest_manifest.jsonl').write_text(json.dumps({'resource_id': self.resource_id, 'source_sha256': '0' * 64, 'source_object': self.src_rel})+'\n')
        with self.assertRaisesRegex(c.CertificationError, 'digest metadata'):
            self.selected()

    def test_wrong_domain_denied(self):
        with self.assertRaisesRegex(c.CertificationError, 'exactly one'):
            c.validate_resource(self.root, self.resource_id, self.year, 'candidate',
                                1_000_000, 1_000_000)

    def test_duplicate_current_index_denied(self):
        self.index.append(self.index[0])
        self.write_meta()
        with self.assertRaisesRegex(c.CertificationError, 'current index'):
            self.selected()

    def test_missing_extracted_denied(self):
        self.extracted.unlink()
        with self.assertRaisesRegex(c.CertificationError, 'extracted object'):
            self.selected()

    def test_source_size_limit_denied(self):
        with self.assertRaisesRegex(c.CertificationError, 'source exceeds'):
            c.validate_resource(self.root, self.resource_id, self.year, self.domain,
                                1, 1_000_000)

    def test_csv_reader_signature_detects_change(self):
        before = c.csv_fingerprint(self.extracted)
        self.assertEqual(before['rows'], 2)
        self.extracted.write_bytes(b'ID;VOTOS\n1;10\n2;99\n')
        after = c.csv_fingerprint(self.extracted)
        self.assertNotEqual(before['ordered_row_sha256'], after['ordered_row_sha256'])

    def test_full_protocol_with_simulated_offline_children(self):
        real_baseline = c.fingerprints([self.source, self.extracted, *self.meta.iterdir()])

        def fake_run(command, *, policy, project):
            self.assertTrue((policy / 'sitecustomize.py').exists())
            sandbox = Path(command[command.index('--root')+1]) if '--root' in command else Path(command[command.index('--raw-root')+1])
            # prune --root refers to the raw subdirectory.
            if command[1].endswith('prune_extracted_raw.py'):
                sandbox = sandbox.parent
                state = c.read_json(sandbox / '_metadata/resource_state.json')
                state[self.key]['materialization_state'] = 'evicted'
                (sandbox / '_metadata/resource_state.json').write_text(json.dumps(state))
                (sandbox / '_metadata/current_objects.jsonl').write_text('')
                (sandbox / self.obj_rel).unlink()
                return 'Deleted 1 file(s).\n'
            if command[1].endswith('ensure_materialized.py'):
                state = c.read_json(sandbox / '_metadata/resource_state.json')
                with zipfile.ZipFile(sandbox / self.src_rel) as archive:
                    (sandbox / self.obj_rel).parent.mkdir(parents=True, exist_ok=True)
                    (sandbox / self.obj_rel).write_bytes(archive.read('historico_candidatura_2026_BRASIL.csv'))
                state[self.key]['materialization_state'] = 'present'
                (sandbox / '_metadata/resource_state.json').write_text(json.dumps(state))
                (sandbox / '_metadata/current_objects.jsonl').write_text(json.dumps(self.index[0]) + '\n')
                return 'Materialization: selected=1, rehydrated=1, errors=0\n'
            if command[1].endswith('report_local_storage.py'):
                sandbox = Path(command[command.index('--raw-root') + 1])
                evicted = not (sandbox / self.obj_rel).exists()
                return json.dumps({'issues': [], 'evicted_missing_objects': 1 if evicted else 0,
                                   'unexpected_missing_objects': 0,
                                   'current_index_rows': 0 if evicted else 1,
                                   'source_hashes_verified': 1})
            return 'current object references .... PASS (1 active object(s))\n'

        with mock.patch.object(c, 'run_command', side_effect=fake_run):
            result = c.run_certification(self.root, self.project, self.resource_id,
                                         self.year, self.domain, reader='csv')
        self.assertEqual(result['result'], 'PASS')
        self.assertTrue(result['offline_enforced'])
        self.assertEqual(real_baseline, c.fingerprints([self.source, self.extracted, *self.meta.iterdir()]))

    def test_tampering_after_rehydrate_fails(self):
        def fake_run(command, *, policy, project):
            sandbox = Path(command[command.index('--root')+1]) if '--root' in command else Path(command[command.index('--raw-root')+1])
            if command[1].endswith('prune_extracted_raw.py'):
                sandbox = sandbox.parent
                state = c.read_json(sandbox / '_metadata/resource_state.json')
                state[self.key]['materialization_state'] = 'evicted'
                (sandbox / '_metadata/resource_state.json').write_text(json.dumps(state))
                (sandbox / '_metadata/current_objects.jsonl').write_text('')
                (sandbox / self.obj_rel).unlink()
            elif command[1].endswith('ensure_materialized.py'):
                state = c.read_json(sandbox / '_metadata/resource_state.json')
                state[self.key]['materialization_state'] = 'present'
                (sandbox / '_metadata/resource_state.json').write_text(json.dumps(state))
                (sandbox / '_metadata/current_objects.jsonl').write_text(json.dumps(self.index[0]) + '\n')
                (sandbox / self.obj_rel).write_bytes(b'ID;VOTOS\n1;666\n')
            if command[1].endswith('report_local_storage.py'):
                sandbox = Path(command[command.index('--raw-root') + 1])
                evicted = not (sandbox / self.obj_rel).exists()
                return json.dumps({'issues': [], 'evicted_missing_objects': int(evicted),
                                   'unexpected_missing_objects': 0,
                                   'current_index_rows': 0 if evicted else 1,
                                   'source_hashes_verified': 1})
            return 'rehydrated=1 errors=0\n'
        with mock.patch.object(c, 'run_command', side_effect=fake_run):
            with self.assertRaisesRegex(c.CertificationError, 'payload or downstream reader'):
                c.run_certification(self.root, self.project, self.resource_id,
                                    self.year, self.domain, reader='csv')


if __name__ == '__main__':
    unittest.main()
