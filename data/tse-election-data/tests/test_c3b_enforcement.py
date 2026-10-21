"""C3-B model-graph resolution and mandatory attestation tests."""
import os
from pathlib import Path
import tempfile
import unittest
from scripts.run_coordinated_dbt import _resolve_selected_models, coordinated_dbt

class EnforcementTests(unittest.TestCase):
    def fake_dbt(self, root, selected):
        f = root / 'fake-dbt'
        import json
        lines = '\n'.join(json.dumps({'resource_type':'model','name':name}) for name in selected)
        f.write_text('#!/usr/bin/env python3\nimport sys\nif sys.argv[1] == "ls":\n print('+repr(lines)+')\nelse:\n open('+repr(str(root / 'ran'))+',"w").write("ran")\n')
        f.chmod(0o755)
        return f

    def test_indirect_gold_requires_evidence_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dbt=self.fake_dbt(root, ['fact_tally_munzona','silver_tally_munzona'])
            with self.assertRaisesRegex(ValueError,'mandatory reference evidence'):
                coordinated_dbt(db=root/'warehouse.duckdb',dbt_executable=dbt,
                                artifacts_root=root/'artifacts',dbt_args=['build','--select','+tag:gold'])
            self.assertFalse((root/'ran').exists())

    def test_unselected_build_requires_evidence_when_c3_in_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dbt=self.fake_dbt(root, ['silver_tally_munzona'])
            with self.assertRaisesRegex(ValueError,'mandatory reference evidence'):
                coordinated_dbt(db=root/'warehouse.duckdb',dbt_executable=dbt,
                                artifacts_root=root/'artifacts',dbt_args=['build'])
            self.assertFalse((root/'ran').exists())

    def test_unknown_selector_flag_denied_before_dbt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dbt=self.fake_dbt(root, ['other_model'])
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                coordinated_dbt(db=root/'warehouse.duckdb',dbt_executable=dbt,
                                artifacts_root=root/'artifacts',dbt_args=['run','--selectors','myselector'])
            self.assertFalse((root/'ran').exists())

    def test_dbt_ls_resolves_excluded_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dbt=self.fake_dbt(root, ['other_model'])
            actual=_resolve_selected_models(dbt,['build','--select','+tag:gold','--exclude','fact_tally_munzona'],os.environ.copy())
            self.assertEqual(actual,{'other_model'})

if __name__ == '__main__': unittest.main()
