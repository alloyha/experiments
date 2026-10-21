import unittest
from pathlib import Path

import yaml


class CostAwareGateTests(unittest.TestCase):
    def test_expensive_tests_are_tagged_and_pr_gate_excludes_them(self):
        root = Path(__file__).resolve().parents[1]

        singular = (
            root / "tse_dbt/tests/assert_stg_electorate_municipality_code_canonical.sql"
        ).read_text()

        metadata = yaml.safe_load(
            (
                root / "tse_dbt/models/gold/facts/fact_candidate_votes.yml"
            ).read_text()
        )

        makefile = (root / "Makefile").read_text()

        self.assertIn("tags=['expensive']", singular)

        models = [
            model
            for model in metadata["models"]
            if model["name"] == "fact_candidate_votes"
        ]
        self.assertEqual(len(models), 1)

        tests = models[0]["tests"]

        uniqueness = [
            value
            for test in tests
            if isinstance(test, dict)
            for name, value in test.items()
            if name == "dbt_utils.unique_combination_of_columns"
        ]

        self.assertEqual(len(uniqueness), 1)
        self.assertIn(
            "expensive",
            uniqueness[0]["config"]["tags"],
        )

        self.assertIn("pr-gate:", makefile)
        self.assertIn("--exclude 'tag:expensive'", makefile)


if __name__ == "__main__":
    unittest.main()
