import unittest
from pathlib import Path
class CostAwareGateTests(unittest.TestCase):
    def test_expensive_tests_are_tagged_and_pr_gate_excludes_them(self):
        root=Path(__file__).resolve().parents[1]
        singular=(root/'tse_dbt/tests/assert_stg_electorate_municipality_code_canonical.sql').read_text()
        schema=(root/'tse_dbt/models/schema.yml').read_text()
        make=(root/'Makefile').read_text()
        self.assertIn("tags=['expensive']", singular)
        self.assertIn("tags: ['expensive']", schema)
        self.assertIn('pr-gate:', make)
        self.assertIn("--exclude 'tag:expensive'", make)
if __name__=='__main__': unittest.main()
