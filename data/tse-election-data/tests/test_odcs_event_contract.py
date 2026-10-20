"""Certified event grain, isolated ownership and immutable ODCS public surface."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.check_odcs_contract_registry import check_registry
from scripts.check_odcs_contract_version import check_contract_version
from scripts.check_odcs_dbt_sync import compare_model_trees

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "contracts/odcs/baselines/dim_election.v1.0.0.odcs.yaml"
CURRENT = ROOT / "contracts/odcs/dim_election.odcs.yaml"
DBT = ROOT / "tse_dbt/models/gold/dimensions/dim_election.yml"


def _checks(column: dict) -> list[str]:
    checks = column.get("data_tests", []) + column.get("tests", [])
    return [next(iter(item)) if isinstance(item, dict) else item for item in checks]


class EventContractTests(unittest.TestCase):
    def setUp(self):
        self.doc = yaml.safe_load(CURRENT.read_text())
        self.meta = yaml.safe_load(DBT.read_text())["models"][0]

    def test_exact_public_identity_and_event_grain(self):
        self.assertEqual(self.doc["id"], "urn:datacontract:tse:gold:dim_election")
        self.assertEqual(self.doc["version"], "1.0.0")
        fields = {f["name"]: f for f in self.doc["schema"][0]["properties"]}
        self.assertEqual(set(fields), {
            "election_id", "cycle_id", "election_year", "election_type",
            "election_code", "election_scope",
        })
        self.assertEqual([x for x in fields if fields[x].get("primaryKey")], ["election_id"])
        self.assertTrue(all(f.get("required") for f in fields.values()))
        self.assertEqual(fields["election_year"]["physicalType"], "integer")
        self.assertEqual(fields["election_code"]["physicalType"], "varchar")
        grain = yaml.safe_load((ROOT / "tse_dbt/model_contracts.yml").read_text())
        self.assertEqual(grain["models"]["dim_election"]["grain"],
                         ["election_year", "election_type", "election_code"])
        self.assertEqual(grain["models"]["dim_election_cycle"]["grain"],
                         ["election_year", "election_type"])

    def test_ownership_and_no_duplicate_model(self):
        shared = yaml.safe_load((ROOT / "tse_dbt/models/schema.yml").read_text())
        self.assertNotIn("dim_election", {m["name"] for m in shared["models"]})
        self.assertEqual(self.meta["name"], "dim_election")
        self.assertEqual(self.meta["config"]["meta"]["datacontract_cli"]["contract_id"], self.doc["id"])
        names = check_registry(ROOT / "contracts/odcs/registry.yaml", ROOT)
        self.assertIn("dim_election", names)
        self.assertIn("dim_election_cycle", names)

    def test_dbt_retains_strict_business_tests(self):
        cols = {x["name"]: x for x in self.meta["columns"]}
        self.assertTrue({"election_id", "cycle_id", "election_year", "election_type",
                         "election_code", "election_scope"}.issubset(cols))
        self.assertIn("unique", _checks(cols["election_id"]))
        self.assertNotIn("unique", _checks(cols["cycle_id"]))
        for name in ("election_id", "cycle_id", "election_year", "election_type", "election_code", "election_scope"):
            self.assertIn("not_null", _checks(cols[name]), name)
        self.assertIn("accepted_values", _checks(cols["election_type"]))
        tests = self.meta.get("data_tests", []) + self.meta.get("tests", [])
        combinations = [t for t in tests if isinstance(t, dict) and "dbt_utils.unique_combination_of_columns" in t]
        self.assertEqual(len(combinations), 1)
        combo = combinations[0]["dbt_utils.unique_combination_of_columns"]["arguments"]["combination_of_columns"]
        self.assertEqual(combo, ["election_year", "election_type", "election_code"])

    def test_sql_event_identity_uses_code_not_only_cycle(self):
        sql = (ROOT / "tse_dbt/models/gold/dimensions/dim_election.sql").read_text().lower()
        self.assertIn("|| ':' || election_code", sql)
        self.assertIn("as election_id", sql)
        self.assertIn("as cycle_id", sql)
        self.assertIn("from {{ ref('fact_tally_munzona') }}", sql)

    def test_frozen_event_baseline_and_version_regression(self):
        check_contract_version(BASE, CURRENT, contract_name="dim_election")
        with tempfile.TemporaryDirectory() as td:
            candidate = Path(td) / "candidate.yaml"
            broken = yaml.safe_load(CURRENT.read_text())
            broken["schema"][0]["properties"] = [p for p in broken["schema"][0]["properties"]
                                                  if p["name"] != "election_code"]
            candidate.write_text(yaml.safe_dump(broken, sort_keys=False))
            with self.assertRaisesRegex(ValueError, "without a version increase"):
                check_contract_version(BASE, candidate, contract_name="dim_election")

    def test_event_owned_model_byte_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            from shutil import copytree
            base = ROOT / "tse_dbt/models"
            altered = Path(td) / "models"
            copytree(base, altered)
            (altered / "gold/dimensions/dim_election.yml").write_bytes(DBT.read_bytes() + b"# spurious\n")
            self.assertTrue(any("byte drift" in issue for issue in compare_model_trees(
                base, altered, managed_models={Path("gold/dimensions/dim_election.yml")})))


if __name__ == "__main__":
    unittest.main()
