"""Offline, no-mutation contract for 2026 historical-source forensic diagnosis."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from research.diagnose_history_source import RESOURCE_ID, audit


class HistorySourceDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name) / 'data/tse'
        self.root.joinpath('_metadata').mkdir(parents=True)
        self.payload = b'fake archive for matching SHA only'
        self.expected_sha = hashlib.sha256(self.payload).hexdigest()
        self.src_rel = 'raw/election_type=general/year=2026/domain=candidate/dataset=candidatos_2026/resource=r1/sha256=' + self.expected_sha + '/source/historico_candidatura_2026.zip'
        self.out_rel = self.src_rel.replace('/source/historico_candidatura_2026.zip', '/extracted/historico_candidatura_2026_BRASIL.csv')
        self.row = {'year': 2026, 'resource_id': RESOURCE_ID, 'domain': 'candidate',
                    'resource_name': 'Histórico de candidaturas',
                    'source_object': self.src_rel, 'source_sha256': self.expected_sha,
                    'source_size_bytes': len(self.payload),
                    'selected_members': ['historico_candidatura_2026_BRASIL.csv'],
                    'extracted_objects': [self.out_rel], 'selection_complete': True,
                    'source_url': 'https://example.test/history.zip'}
        self.key = f'general|2026|{RESOURCE_ID}|brasil|'
        self.write_json('_metadata/resource_state.json', {self.key: self.row})
        self.write_json('_metadata/pending_resources.json', {})
        self.write_jsonl('_metadata/ingest_manifest.jsonl', [
            {'resource_id': RESOURCE_ID, 'source_sha256': self.expected_sha,
             'source_object': self.src_rel, 'domain': 'candidate', 'ingested_at': '2026-10-10T00:00:00Z'}])
        self.write_jsonl('_metadata/ingest_profile.jsonl', [
            {'resource_id': RESOURCE_ID, 'result': 'ingested', 'started_at': '2026-10-10T00:00:00Z'}])
        self.write_jsonl('_metadata/current_objects.jsonl', [])

    def write_json(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding='utf-8')

    def write_jsonl(self, name, values):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(v)+'\n' for v in values), encoding='utf-8')

    def test_missing_source_and_extracted_are_reported(self):
        r = audit(self.root)
        self.assertEqual(r['conclusion'], 'source_not_found_in_scanned_locations')
        self.assertFalse(r['source_exists'])
        self.assertEqual(r['logical_domain'], 'candidate_history')
        self.assertEqual(r['stored_domain'], 'candidate')
        self.assertEqual(r['missing_outputs'], [self.out_rel])
        self.assertEqual(r['physical_index_rows'], 0)
        self.assertEqual(len(r['manifest_recent']), 1)
        self.assertEqual(len(r['profiles_recent']), 1)
        self.assertFalse(r['metadata_changed_while_scanning'])

    def test_identical_relocated_source_detected_by_sha(self):
        sibling = self.root / 'raw/different/resource/source/historico_candidatura_2026.zip'
        sibling.parent.mkdir(parents=True)
        sibling.write_bytes(self.payload)
        r = audit(self.root)
        self.assertEqual(r['conclusion'], 'source_relocated_exact_sha_found')
        self.assertEqual(len(r['matching_hash_elsewhere']), 1)
        self.assertEqual(r['matching_hash_elsewhere'][0]['path'], str(sibling))

    def test_renamed_source_found_by_size_only_when_enabled(self):
        p = self.root / 'raw/archive/renamed.bin'
        p.parent.mkdir(parents=True)
        p.write_bytes(self.payload)
        self.assertEqual(audit(self.root)['matching_hash_elsewhere'], [])
        r = audit(self.root, scan_by_size=True)
        self.assertEqual(r['conclusion'], 'source_relocated_exact_sha_found')
        self.assertEqual(len(r['matching_hash_elsewhere']), 1)

    def test_same_name_wrong_hash_cannot_be_recovered(self):
        p = self.root / 'raw/archive/historico_candidatura_2026.zip'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'x' * len(self.payload))
        r = audit(self.root)
        self.assertEqual(r['conclusion'], 'source_not_found_in_scanned_locations')
        self.assertEqual(len(r['possible_alternatives']), 1)
        self.assertFalse(r['possible_alternatives'][0]['match_sha256'])

    def test_invalid_recorded_path_is_rejected(self):
        self.row['source_object'] = '../../other'
        self.write_json('_metadata/resource_state.json', {self.key: self.row})
        with self.assertRaisesRegex(ValueError, 'invalid or escapes'):
            audit(self.root)

    def test_source_present_has_verified_hash(self):
        path = self.root / self.src_rel
        path.parent.mkdir(parents=True)
        path.write_bytes(self.payload)
        r = audit(self.root)
        self.assertEqual(r['conclusion'], 'source_exists_and_sha_matches')
        self.assertTrue(r['source_hash_matches'])


if __name__ == '__main__':
    unittest.main()
