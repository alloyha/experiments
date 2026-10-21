"""Regression of generic GG-009 entrypoints after scoped C3 enforcement."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from scripts import run_coordinated_dbt as gateway
import subprocess
import sys

class C3GenericCompatibility(TestCase):
    def test_external_project_does_not_resolve_c3(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            dbt=root/'dbt'
            dbt.write_text('#!'+sys.executable+'\nimport sys\nsys.exit(0)\n')
            dbt.chmod(0o755)
            with mock.patch.object(gateway,'_resolve_selected_models',side_effect=AssertionError('must not resolve')):
                self.assertEqual(gateway.coordinated_dbt(db=root/'scratch.duckdb',dbt_executable=dbt,
                    artifacts_root=root/'runs',dbt_args=['build','--project-dir',str(root)]),0)
    def test_tse_project_still_requires_evidence(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            dbt=root/'dbt'
            dbt.write_text('#!'+sys.executable+'\nimport sys\nsys.exit(0)\n')
            dbt.chmod(0o755)
            tse=Path(gateway.__file__).resolve().parents[1]/'tse_dbt'
            with mock.patch.object(gateway,'_resolve_selected_models',return_value={'silver_tally_munzona'}):
                with self.assertRaisesRegex(ValueError,'mandatory reference evidence'):
                    gateway.coordinated_dbt(db=root/'scratch.duckdb',dbt_executable=dbt,
                        artifacts_root=root/'runs',dbt_args=['run','--project-dir',str(tse)])
