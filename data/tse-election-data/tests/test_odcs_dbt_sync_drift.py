"""Falsifiers for semantic ODCS/dbt drift (synthetic model trees only)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_odcs_dbt_sync import MANAGED_MODEL, compare_model_trees


class OdcsDbtDriftTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="odcs-drift-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.committed = root / "committed"
        self.generated = root / "generated"
        self.committed.mkdir()
        self.generated.mkdir()
        self.write("schema.yml", "version: 2\nmodels:\n- name: other\n  description:\n    Same description across\n    two lines.\n  columns:\n  - name: id\n    tests: [not_null]\n")
        self.write(MANAGED_MODEL, "version: 2\nmodels:\n- name: dim_election_cycle\n  columns:\n  - name: cycle_id\n    data_type: varchar\n    data_tests: [not_null, unique]\n")

    def write(self, relative: str | Path, source: str, target: str | None = None):
        for root, text in ((self.committed, source), (self.generated, source if target is None else target)):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf8")

    def check(self):
        return compare_model_trees(self.committed, self.generated)

    def test_matching_trees_pass(self):
        self.assertEqual(self.check(), [])

    def test_folded_scalar_reformat_is_not_drift(self):
        (self.generated / "schema.yml").write_text(
            "version: 2\nmodels:\n- name: other\n  description: Same description across two lines.\n  columns:\n  - name: id\n    tests: [not_null]\n", encoding="utf8"
        )
        self.assertEqual(self.check(), [])

    def test_unrelated_dbt_test_change_is_drift(self):
        p = self.generated / "schema.yml"
        p.write_text(p.read_text().replace("[not_null]", "[unique]"))
        self.assertIn("Semantic dbt YAML drift: schema.yml", self.check())

    def test_managed_column_type_change_is_drift(self):
        p = self.generated / MANAGED_MODEL
        p.write_text(p.read_text().replace("varchar", "integer"))
        self.assertIn(f"ODCS-managed model has byte drift: {MANAGED_MODEL}", self.check())

    def test_managed_reformatting_is_not_silently_accepted(self):
        p = self.generated / MANAGED_MODEL
        p.write_text(p.read_text() + "\n")
        self.assertIn(f"ODCS-managed model has byte drift: {MANAGED_MODEL}", self.check())

    def test_unexpected_generated_yaml_rejected(self):
        (self.generated / "extra.yml").write_text("version: 2\nmodels: []\n")
        self.assertIn("Unexpected generated file: extra.yml", self.check())

    def test_deleted_original_yaml_rejected(self):
        (self.generated / "schema.yml").unlink()
        self.assertIn("Deleted by sync: schema.yml", self.check())

    def test_nonyaml_change_rejected(self):
        self.write("readme.txt", "unchanged", "changed")
        self.assertIn("Non-YAML byte drift: readme.txt", self.check())

    def test_invalid_generated_yaml_rejected(self):
        (self.generated / "schema.yml").write_text("models: [\n")
        self.assertTrue(any("Cannot parse schema.yml" in x for x in self.check()))


if __name__ == "__main__":
    unittest.main()
