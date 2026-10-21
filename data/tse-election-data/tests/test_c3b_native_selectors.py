import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.run_coordinated_dbt import _resolve_selected_models, coordinated_dbt

class NativeSelectionTests(unittest.TestCase):
    def test_multiple_select_and_exclude_forwarded(self):
        args=['run','--select','silver_tally_munzona','fact_tally_munzona','--exclude','other','--vars','{"x":1}']
        observed=[]
        def run(command,**kwargs):
            observed.append(command)
            return subprocess.CompletedProcess(command,0,stdout='{"resource_type":"model","name":"silver_tally_munzona"}\n{"resource_type":"model","name":"fact_tally_munzona"}\n',stderr='')
        with patch('scripts.run_coordinated_dbt.subprocess.run',side_effect=run):
            self.assertEqual(_resolve_selected_models(Path('/fake/dbt'),args,os.environ.copy()),{'silver_tally_munzona','fact_tally_munzona'})
        self.assertEqual(observed[0][6:],args[1:])

    def test_missing_selector_rejected_before_resolver(self):
        with patch('scripts.run_coordinated_dbt.subprocess.run') as runner:
            with self.assertRaisesRegex(ValueError,'Missing dbt option value'):
                _resolve_selected_models(Path('/fake/dbt'),['run','--select','--vars','{}'],os.environ.copy())
            runner.assert_not_called()

    def test_multi_select_requires_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); dbt=root/'fake-dbt'; dbt.write_text('#!/usr/bin/env python3\n'); dbt.chmod(0o755)
            def run(command,**kwargs):
                if command[1]=='ls':
                    return subprocess.CompletedProcess(command,0,stdout='{"resource_type":"model","name":"silver_tally_munzona"}\n',stderr='')
                self.fail('dbt writer invoked')
            with patch('scripts.run_coordinated_dbt.subprocess.run',side_effect=run):
                with self.assertRaisesRegex(ValueError,'mandatory reference evidence'):
                    coordinated_dbt(db=root/'db.duckdb', dbt_executable=dbt,artifacts_root=root/'artifacts',dbt_args=['run','--select','silver_tally_munzona','fact_tally_munzona'])
