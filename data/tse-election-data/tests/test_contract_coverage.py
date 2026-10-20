"""Fail-closed falsifiers for CCC inventory, semantics and classifications."""
from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.check_contract_coverage import audit, discover, manifest_members


class ContractCoverageTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="ccc-falsifier-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.models = self.root / "tse_dbt/models"
        for directory in ("bronze", "gold/dimensions", "semantic"):
            (self.models / directory).mkdir(parents=True)
        (self.root / "tse_dbt/seeds").mkdir()
        (self.models / "bronze/bronze_one.sql").write_text("select 1")
        (self.models / "gold/dimensions/dim_election.sql").write_text("select 1")
        (self.root / "tse_dbt/seeds/election_calendar.csv").write_text("year\n2026\n")
        self.odcs_dir = self.root / "contracts/odcs"
        self.odcs_dir.mkdir(parents=True)
        self.metadata = self.models / "gold/dimensions/dim_election.yml"
        self.metadata.write_text(yaml.safe_dump({"version": 2, "models": [{"name": "dim_election", "tests": [
            {"dbt_utils.unique_combination_of_columns": {"arguments": {
                "combination_of_columns": ["election_year", "election_type", "election_code"]}}}]}]}))
        self.registry = {"version": 1, "contracts": [{"name": "dim_election", "dbt_metadata":
                         "tse_dbt/models/gold/dimensions/dim_election.yml"}]}
        (self.odcs_dir / "registry.yaml").write_text(yaml.safe_dump(self.registry))
        self.policy = {"version": 1, "internal_defaults": {
            "bronze": "Raw inputs are not a supported consumer interface.",
            "silver": "Staging is not externally exposed to consumers.",
            "physical": "Physical files have their own GG certification fence.",
            "seed": "Repository reference seeds have no consumer SLA.",
            "source": "External raw inputs are not stable Gold outputs.",
        }, "resources": {"model:dim_election": {"visibility": "public", "contract": "dim_election",
                                            "grain": ["election_year", "election_type", "election_code"],
                                            "semantic_test": "dbt_utils.unique_combination_of_columns"}}}
        self.domain = {"models": {"dim_election": {"status": "implemented",
                                                   "grain": ["election_year", "election_type", "election_code"]}}}
        self.save()

    def save(self):
        (self.root / "contracts/coverage_policy.yml").write_text(yaml.safe_dump(self.policy))
        (self.root / "tse_dbt/model_contracts.yml").write_text(yaml.safe_dump(self.domain))
        (self.odcs_dir / "registry.yaml").write_text(yaml.safe_dump(self.registry))

    def verify(self):
        return audit(self.root, strict_odcs=False, today=dt.date(2026, 10, 19))

    def test_closed_classification_covers_public_and_technical_layers(self):
        report = self.verify()
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["stats"]["public"], 1)
        self.assertEqual(report["stats"]["internal"], 2)

    def test_new_gold_model_fails_unclassified(self):
        (self.models / "gold/dimensions/new_public_candidate.sql").write_text("select 1")
        report = self.verify()
        self.assertIn("model:new_public_candidate", report["unclassified"])
        self.assertEqual(report["status"], "FAIL")

    def test_new_semantic_model_fails_unclassified(self):
        (self.models / "semantic/external_analytics.sql").write_text("select 1")
        self.assertIn("model:external_analytics", self.verify()["unclassified"])

    def test_registered_public_contract_cannot_be_misclassified_internal(self):
        self.policy["resources"]["model:dim_election"] = {
            "visibility": "internal", "reason": "Hidden although it is an ODCS published interface"}
        self.save()
        self.assertTrue(any("no public classified" in e for e in self.verify()["errors"]))

    def test_stale_internal_classification_fails(self):
        self.policy["resources"]["model:ghost"] = {
            "visibility": "internal", "reason": "This deleted node used to be a private model"}
        self.save()
        self.assertTrue(any("Stale policy entry" in e for e in self.verify()["errors"]))

    def test_expired_exclusion_is_not_accepted(self):
        (self.models / "semantic/temporary.sql").write_text("select 1")
        self.policy["resources"]["model:temporary"] = {
            "visibility": "excluded", "reason": "Approved temporary exception", "owner": "analytics",
            "expires": "2026-10-18"}
        self.save()
        self.assertTrue(any("Expired" in e for e in self.verify()["errors"]))

    def test_public_grain_must_match_policy(self):
        self.policy["resources"]["model:dim_election"]["grain"] = ["election_year"]
        self.save()
        self.assertTrue(any("grain disagrees" in e for e in self.verify()["errors"]))

    def test_public_grain_must_have_executable_dbt_test(self):
        self.metadata.write_text("version: 2\nmodels:\n  - name: dim_election\n")
        self.assertTrue(any("no dbt composite" in e for e in self.verify()["errors"]))

    def test_unknown_exposure_requires_classification(self):
        (self.models / "exposures.yml").write_text("exposures:\n  - name: public_dashboard\n")
        self.assertIn("exposure:public_dashboard", self.verify()["unclassified"])

    def test_implemented_policy_must_have_discovered_model(self):
        self.domain["models"]["missing_gold"] = {"status": "implemented"}
        self.save()
        self.assertTrue(any("no dbt SQL" in e for e in self.verify()["errors"]))

    def test_superseded_domain_alias_has_a_real_successor(self):
        self.domain["models"]["stg_bronze_one"] = {
            "status": "superseded", "superseded_by": "bronze_one",
            "supersession_reason": "Old staging model renamed into the bronze layer.",
        }
        self.save()
        self.assertEqual(self.verify()["status"], "PASS")

    def test_superseded_domain_alias_without_successor_fails(self):
        self.domain["models"]["stg_bronze_one"] = {
            "status": "superseded", "superseded_by": "bronze_not_found",
            "supersession_reason": "Old staging model renamed into the bronze layer.",
        }
        self.save()
        self.assertTrue(any("successor missing" in e for e in self.verify()["errors"]))

    def test_superseded_domain_alias_cannot_still_be_active(self):
        (self.models / "bronze/stg_bronze_one.sql").write_text("select 1")
        self.domain["models"]["stg_bronze_one"] = {
            "status": "superseded", "superseded_by": "bronze_one",
            "supersession_reason": "Old staging model renamed into the bronze layer.",
        }
        self.save()
        self.assertTrue(any("still an active" in e for e in self.verify()["errors"]))

    def test_superseded_alias_must_have_reviewed_rationale(self):
        self.domain["models"]["stg_bronze_one"] = {
            "status": "superseded", "superseded_by": "bronze_one",
            "supersession_reason": "old",
        }
        self.save()
        self.assertTrue(any("lacks reviewed reason" in e for e in self.verify()["errors"]))

    def test_unknown_domain_status_fails_closed(self):
        self.domain["models"]["dim_election"]["status"] = "unknown"
        self.save()
        self.assertTrue(any("Unknown domain policy status" in e for e in self.verify()["errors"]))

    def test_manifest_crosscheck_rejects_missing_nodes(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"nodes": {"model.x.tse": {"package_name": "tse_analytics",
            "resource_type": "model", "name": "dim_election"}}, "sources": {}, "exposures": {}}))
        self.assertEqual(manifest_members(path), {"model:dim_election"})
        report = audit(self.root, manifest=path, strict_odcs=False)
        self.assertTrue(any("Source-only resource" in e for e in report["errors"]))

    def test_duplicate_model_names_are_rejected(self):
        (self.models / "semantic/dim_election.sql").write_text("select 1")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            discover(self.root)


if __name__ == "__main__":
    unittest.main()
