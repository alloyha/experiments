"""Verify shared ODCS orchestration never omits a registered contract."""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.check_odcs_dbt_sync import compare_model_trees
from scripts.run_odcs_checks import CLI_COMPAT, CLI_SYNC, run_odcs


class RegistryDrivenGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="odcs-runner-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commands = []
        models = self.root / "tse_dbt/models/gold/dimensions"
        models.mkdir(parents=True)
        (self.root / "tse_dbt/dbt_project.yml").write_text("name: tse\n")
        self.entries = []
        self.add_contract("dim_election_cycle")

    def add_contract(self, name):
        relative = f"tse_dbt/models/gold/dimensions/{name}.yml"
        path = self.root / relative
        path.write_text(f"version: 2\nmodels:\n- name: {name}\n")
        self.entries.append({
            "name": name,
            "contract": f"contracts/odcs/{name}.odcs.yaml",
            "baseline": f"contracts/odcs/baselines/{name}.v1.0.0.odcs.yaml",
            "dbt_metadata": relative,
        })
        reg = self.root / "contracts/odcs/registry.yaml"
        reg.parent.mkdir(parents=True, exist_ok=True)
        reg.write_text(yaml.safe_dump({"version": 1, "contracts": self.entries}))

    def invoke(self, command, *, cwd, check):
        self.assertEqual(cwd, self.root)
        self.assertFalse(check)
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def run_mode(self, mode):
        with patch("scripts.run_odcs_checks.check_registry", return_value=sorted(e["name"] for e in self.entries)):
            run_odcs(mode, root=self.root, invoke=self.invoke)

    def test_all_modes_one_contract_has_no_redundant_plan(self):
        self.run_mode("all")
        self.assertEqual(len(self.commands), 4)  # current lint, baseline lint, breaking, sandbox sync
        self.assertEqual(sum("--dry-run" in cmd for cmd in self.commands), 0)
        self.assertEqual(sum("breaking" in cmd for cmd in self.commands), 1)
        self.assertEqual(sum("sync" in cmd for cmd in self.commands), 1)
        self.assertIn(CLI_SYNC, " ".join(self.commands[0]))
        self.assertIn(CLI_COMPAT, " ".join(self.commands[1]))

    def test_two_contracts_use_two_isolated_owned_models(self):
        self.add_contract("dim_election")
        self.run_mode("all")
        self.assertEqual(len(self.commands), 8)
        self.assertEqual(sum("breaking" in cmd for cmd in self.commands), 2)
        self.assertEqual(sum("sync" in cmd for cmd in self.commands), 2)

    def test_plan_is_dry_run_only(self):
        self.run_mode("plan")
        self.assertEqual(len(self.commands), 2)
        self.assertIn("--dry-run", self.commands[-1])
        self.assertIn("--no-prune", self.commands[-1])

    def test_external_cli_error_is_fatal(self):
        def fail(command, **kwargs):
            return subprocess.CompletedProcess(command, 1)
        with patch("scripts.run_odcs_checks.check_registry", return_value=["dim_election_cycle"]):
            with self.assertRaisesRegex(RuntimeError, "exit=1"):
                run_odcs("all", root=self.root, invoke=fail)

    def test_one_managed_model_byte_drift_is_fatal(self):
        def drift(command, **kwargs):
            if "sync" in command:
                output = Path(command[command.index("--project-dir") + 1])
                file = output / "models/gold/dimensions/dim_election_cycle.yml"
                file.write_bytes(file.read_bytes() + b"# extra\n")
            return subprocess.CompletedProcess(command, 0)
        with patch("scripts.run_odcs_checks.check_registry", return_value=["dim_election_cycle"]):
            with self.assertRaisesRegex(ValueError, "byte drift"):
                run_odcs("dbt-check", root=self.root, invoke=drift)

    def test_missing_registered_model_in_sync_tree_rejected(self):
        source = self.root / "tse_dbt/models"
        with tempfile.TemporaryDirectory() as other:
            errors = compare_model_trees(source, Path(other), managed_models={Path("gold/dimensions/dim_election_cycle.yml")})
            self.assertTrue(any("Missing ODCS-owned model" in err for err in errors))

    def test_legacy_default_managed_model_remains_supported(self):
        source = self.root / "tse_dbt/models"
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "gold/dimensions"
            target.mkdir(parents=True)
            (target / "dim_election_cycle.yml").write_bytes((source / "gold/dimensions/dim_election_cycle.yml").read_bytes())
            self.assertEqual(compare_model_trees(source, Path(other)), [])


if __name__ == "__main__":
    unittest.main()
