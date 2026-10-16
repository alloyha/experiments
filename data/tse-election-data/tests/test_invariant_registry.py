"""Registry's coverage must fail on missing falsifiers, drift or duplicate IDs."""
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.check_invariant_registry import DEFAULT, ROOT, check_registry


class InvariantRegistryTests(unittest.TestCase):
    def test_every_implemented_invariant_has_a_positive_and_negative_oracle(self):
        result = check_registry()
        self.assertEqual(result["implemented"], 12)
        self.assertEqual(result["planned"], 0)
        self.assertTrue(result["critical_coverage_complete"])

    def test_strict_gate_passes_when_every_critical_contract_has_evidence(self):
        result = check_registry(strict=True)
        self.assertEqual(result["implemented"], 12)
        self.assertEqual(result["planned"], 0)
        self.assertTrue(result["critical_coverage_complete"])

    def modified(self, apply):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invariants.yml"
            document = yaml.safe_load(DEFAULT.read_text())
            apply(document)
            path.write_text(yaml.safe_dump(document))
            with self.assertRaises(ValueError):
                check_registry(path, root=ROOT)

    def test_missing_negative_oracle_is_rejected(self):
        self.modified(lambda d: d["invariants"][0].update(
            falsifier_test="tests/test_gold_graph_invariants.py::GoldGraphTests.test_absent"))

    def test_duplicate_invariant_ids_are_rejected(self):
        self.modified(lambda d: d["invariants"][1].update(id=d["invariants"][0]["id"]))

    def test_missing_sql_assertion_is_rejected(self):
        self.modified(lambda d: d["invariants"][0].update(sql_assertion="tse_dbt/tests/nonexistent.sql"))

    def test_missing_runtime_check_is_rejected(self):
        self.modified(lambda d: next(r for r in d["invariants"] if r["id"] == "GG-010")
                      .update(runtime_check="scripts/graph_freshness.py::not_an_auditor"))

    def test_implemented_invariant_requires_exactly_one_oracle_type(self):
        self.modified(lambda d: next(r for r in d["invariants"] if r["id"] == "GG-010")
                      .update(sql_assertion="tse_dbt/tests/assert_graph_node_identity.sql"))

    def test_unsafe_runtime_reference_is_rejected(self):
        self.modified(lambda d: next(r for r in d["invariants"] if r["id"] == "GG-010")
                      .update(runtime_check="../scripts/graph_freshness.py::audit_certificate"))


if __name__ == "__main__":
    unittest.main()
