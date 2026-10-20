"""Public dimension obligations stay valid as the registry grows."""
from __future__ import annotations
import unittest
from pathlib import Path
import yaml
from scripts.check_odcs_contract_version import check_contract_version
from scripts.check_odcs_contract_registry import check_registry
from scripts.check_contract_coverage import audit

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "dim_candidate": ["election_year", "election_type", "election_code", "candidate_id"],
    "dim_geography": ["election_year", "election_type", "uf", "municipality_code"],
}


class PublicDimensionCoverage(unittest.TestCase):
    def test_baselines_owners_grain(self):
        registry = yaml.safe_load((ROOT / "contracts/odcs/registry.yaml").read_text())
        names = check_registry(ROOT / "contracts/odcs/registry.yaml", ROOT)
        self.assertEqual(len(names), len(registry["contracts"]))
        self.assertTrue(set(EXPECTED).issubset(names))
        policy = yaml.safe_load((ROOT / "contracts/coverage_policy.yml").read_text())["resources"]
        domain = yaml.safe_load((ROOT / "tse_dbt/model_contracts.yml").read_text())["models"]
        shared = yaml.safe_load((ROOT / "tse_dbt/models/schema.yml").read_text())
        for name, grain in EXPECTED.items():
            with self.subTest(model=name):
                record = next(x for x in registry["contracts"] if x["name"] == name)
                check_contract_version(ROOT / record["baseline"], ROOT / record["contract"], contract_name=name)
                metadata = yaml.safe_load((ROOT / record["dbt_metadata"]).read_text())["models"][0]
                self.assertEqual(metadata["name"], name)
                self.assertEqual(metadata["config"]["meta"]["datacontract_cli"]["contract_id"],
                                 f"urn:datacontract:tse:gold:{name}")
                self.assertEqual(policy[f"model:{name}"]["visibility"], "public")
                self.assertEqual(policy[f"model:{name}"]["grain"], grain)
                self.assertEqual(domain[name]["grain"], grain)
                self.assertNotIn(name, {m["name"] for m in shared["models"]})
                tests = metadata.get("tests", []) + metadata.get("data_tests", [])
                composites = [
                    t["dbt_utils.unique_combination_of_columns"]["arguments"]["combination_of_columns"]
                    for t in tests if isinstance(t, dict) and "dbt_utils.unique_combination_of_columns" in t
                ]
                self.assertEqual(composites, [grain])

    def test_coverage_counts_reconcile_without_freezing_pilot_size(self):
        report = audit(ROOT)
        stats = report["stats"]
        self.assertEqual(stats["total"], sum(stats[key] for key in
                         ("public", "internal", "excluded", "unclassified")))
        self.assertGreaterEqual(stats["public"], len(EXPECTED) + 2)
        self.assertEqual(stats["invalid_contracts"], 0)
        self.assertEqual(stats["unmapped_semantic_obligations"], 0)
        # Deliberately do not assert FAIL or a fixed unclassified count.
        # This test must keep passing as CCC closes its remaining gaps.


if __name__ == "__main__":
    unittest.main()
