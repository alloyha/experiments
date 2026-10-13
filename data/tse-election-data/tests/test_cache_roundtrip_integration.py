"""Genuine PR42 integration: official prune + offline ensure_materialized + raw contracts.

Entire test dataset is inside TemporaryDirectory. No canonical TSE files or CKAN
network calls. Run with the repo's installed .venv to exercise real scripts.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import tse_ingest
import certify_cache_roundtrip as c


class RealOfflineRoundtripTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='tse-pr42-integration-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'tse'
        version_stem = (self.root / 'raw/election_type=general/year=2026/'
                        'domain=candidate_history/dataset=candidatos_2026/'
                        'resource=fixture-history')
        zip_stage = Path(temp.name) / 'source-staging.zip'
        payload = b'ANO_ELEICAO;SQ_CANDIDATO;NOME_CANDIDATO\n2026;1;Ada\n2026;2;Beto\n'
        with zipfile.ZipFile(zip_stage, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('historico_candidatura_2026_BRASIL.csv', payload)
        hash_value = hashlib.sha256(zip_stage.read_bytes()).hexdigest()
        version = version_stem / f'sha256={hash_value}'
        source = version / 'source/historico_candidatura_2026.zip'
        source.parent.mkdir(parents=True)
        source.write_bytes(zip_stage.read_bytes())
        output_files, _, members = tse_ingest.extract_selected_members(
            source, version / 'extracted', 'brasil', None, 'python')
        self.assertEqual(members, ['historico_candidatura_2026_BRASIL.csv'])
        self.assertEqual(len(output_files), 1)
        self.resource_id = 'fixture-history'
        self.row = {
            'year': 2026, 'election_type': 'general', 'election_scope': 'national',
            'dataset_id': 'candidatos_2026', 'dataset_title': 'synthetic fixture',
            'resource_id': self.resource_id, 'resource_name': 'Histórico de candidaturas',
            'resource_format': 'ZIP', 'resource_fingerprint': 'synthetic',
            'domain': 'candidate_history', 'partition': 'BRASIL',
            'source_url': 'https://example.invalid/never-download.zip',
            'source_object': source.relative_to(self.root).as_posix(),
            'source_sha256': hash_value, 'source_size_bytes': source.stat().st_size,
            'extracted_objects': [x.relative_to(self.root).as_posix() for x in output_files],
            'selected_members': members, 'selection_complete': True,
            'materialization_state': 'present', 'granularity': 'brasil', 'uf': '',
            'checked_at': '2026-10-13T00:00:00Z',
        }
        tse_ingest.save_state(
            self.root,
            {'general|2026|fixture-history|brasil|': self.row},
        )
        (self.root / '_metadata/ingest_manifest.jsonl').write_text(
            json.dumps({'resource_id': self.resource_id, 'source_sha256': hash_value, 'source_object': self.row['source_object']}) + '\n',
            encoding='utf-8')

    def test_real_prune_rehydrate_raw_contracts_offline(self):
        before_manifest = (self.root / '_metadata/ingest_manifest.jsonl').read_bytes()
        result = c.run_certification(self.root, PROJECT, self.resource_id, 2026,
                                     'candidate_history', reader='duckdb')
        self.assertEqual(result['result'], 'PASS')
        self.assertTrue(result['index_rebuilt'])
        self.assertEqual(result['objects'], 1)
        self.assertTrue(result['payload_and_reader_equivalent'])
        self.assertEqual((self.root / '_metadata/ingest_manifest.jsonl').read_bytes(), before_manifest)
        # The synthetic original root has never been pruned.
        self.assertEqual(tse_ingest.load_state(self.root)['general|2026|fixture-history|brasil|'], self.row)
        self.assertEqual(len((self.root / '_metadata/current_objects.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
