"""C3 certification matrix contracts, part of the normal make unit gate."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.c3_certification_matrix import digest, run_scenario

class C3CertificationMatrixTests(unittest.TestCase):
    def test_digest_changes_with_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'evidence'
            path.write_bytes(b'one')
            before = digest(path)
            path.write_bytes(b'two')
            self.assertNotEqual(before, digest(path))

    def test_rejected_scenario_must_preserve_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root/'baseline.duckdb'
            baseline.write_bytes(b'fake database')
            evidence = root/'evidence.json'
            evidence.write_text(json.dumps({'reference_sha256':'0'*64}))
            with patch('scripts.c3_certification_matrix.snapshot', side_effect=[{'silver': 1},{'silver':0}]), \
                 patch('scripts.c3_certification_matrix.subprocess.run') as run:
                run.return_value.returncode = 1
                run.return_value.stdout = 'error'
                run.return_value.stderr = ''
                value = run_scenario(gateway=root/'gateway.py',python=root/'python',dbt=root/'dbt',
                    baseline=baseline,reference=baseline,evidence=evidence,raw_root=root,
                    workspace=root,scenario='stale')
            self.assertEqual(value['status'],'FAIL')

    def test_accepted_scenario_requires_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=root/'baseline.duckdb'
            base.write_bytes(b'fake database')
            e=root/'evidence.json'
            e.write_text('{}')
            with patch('scripts.c3_certification_matrix.snapshot',return_value={'silver':1}), \
                 patch('scripts.c3_certification_matrix.subprocess.run') as run:
                run.return_value.returncode=0
                run.return_value.stdout='success'
                run.return_value.stderr=''
                value=run_scenario(gateway=root/'gateway.py',python=root/'python',dbt=root/'dbt',
                    baseline=base,reference=base,evidence=e,raw_root=root,
                    workspace=root,scenario='intact')
            self.assertEqual(value['status'],'PASS')

if __name__ == '__main__':
    unittest.main()
