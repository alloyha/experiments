"""C3 gateway preflight fails closed before executing a dbt writer."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / 'scripts' / 'run_coordinated_dbt.py'

class C3GatewayPreflightTests(unittest.TestCase):
    def test_preflight_rejects_before_dbt_executes(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            marker=base/'executed'
            fake=base/'fake_dbt'
            fake.write_text('#!/bin/sh\ntouch "'+str(marker)+'"\n')
            fake.chmod(0o755)
            result=subprocess.run([
                sys.executable,str(GATEWAY),'--db',str(base/'candidate.duckdb'),
                '--dbt-executable',str(fake),'--artifacts-root',str(base/'artifacts'),
                '--c3-reference-db',str(base/'missing-reference.duckdb'),
                '--c3-reference-raw-root',str(base),
                '--c3-candidate-raw-root',str(base),
                '--','run','--select','silver_tally_munzona',
            ],capture_output=True,text=True,check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(marker.exists(), result.stdout + result.stderr)

    def test_rejects_incomplete_configuration_before_dbt(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            fake=base/'fake_dbt'
            fake.write_text('#!/bin/sh\nexit 0\n')
            fake.chmod(0o755)
            result=subprocess.run([
                sys.executable,str(GATEWAY),'--db',str(base/'candidate.duckdb'),
                '--dbt-executable',str(fake),'--artifacts-root',str(base/'artifacts'),
                '--c3-reference-db',str(base/'reference.duckdb'),
                '--','run',
            ],capture_output=True,text=True,check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('both raw roots', result.stderr)

if __name__=='__main__':
    unittest.main()
