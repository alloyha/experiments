"""Closed-world falsifiers for the ODCS contract registry."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.check_odcs_contract_registry import check_registry

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "contracts/odcs/baselines/dim_election_cycle.v1.0.0.odcs.yaml"
CURRENT = ROOT / "contracts/odcs/dim_election_cycle.odcs.yaml"
METADATA = ROOT / "tse_dbt/models/gold/dimensions/dim_election_cycle.yml"


class OdcsRegistryTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="tse-odcs-registry-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        source_registry = yaml.safe_load((ROOT / "contracts/odcs/registry.yaml").read_text())
        self.data = source_registry
        for entry in source_registry["contracts"]:
            for field in ("baseline", "contract", "dbt_metadata"):
                relative = entry[field]
                target = self.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
        self.registry = self.root / "contracts/odcs/registry.yaml"
        self.save()

    def save(self):
        self.registry.write_text(yaml.safe_dump(self.data, sort_keys=False))

    def validate(self):
        return check_registry(self.registry, self.root)

    def test_valid_registry(self):
        self.assertEqual(self.validate(), sorted(x["name"] for x in self.data["contracts"]))

    def test_duplicate_registration_rejected(self):
        self.data["contracts"].append(dict(self.data["contracts"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate()

    def test_unregistered_public_contract_rejected(self):
        (self.root / "contracts/odcs/unlisted.odcs.yaml").write_text("kind: DataContract\n")
        with self.assertRaisesRegex(ValueError, "Unregistered ODCS contract"):
            self.validate()

    def test_unregistered_baseline_rejected(self):
        (self.root / "contracts/odcs/baselines/unlisted.odcs.yaml").write_text("kind: DataContract\n")
        with self.assertRaisesRegex(ValueError, "Unregistered ODCS baselines"):
            self.validate()

    def test_wrong_baseline_digest_rejected(self):
        self.data["contracts"][0]["baseline_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "source-pinned trust anchor"):
            self.validate()

    def test_path_traversal_rejected(self):
        self.data["contracts"][0]["contract"] = "contracts/odcs/../../etc/hosts"
        self.save()
        with self.assertRaisesRegex(ValueError, "outside permitted"):
            self.validate()

    def test_foreign_managed_model_rejected(self):
        meta = self.root / self.data["contracts"][0]["dbt_metadata"]
        obj = yaml.safe_load(meta.read_text())
        obj["models"][0]["name"] = "not_the_owner"
        meta.write_text(yaml.safe_dump(obj))
        with self.assertRaisesRegex(ValueError, "ownership mismatch"):
            self.validate()

    def test_wrong_contract_id_rejected(self):
        meta = self.root / self.data["contracts"][0]["dbt_metadata"]
        obj = yaml.safe_load(meta.read_text())
        obj["models"][0]["config"]["meta"]["datacontract_cli"]["contract_id"] = "foreign"
        meta.write_text(yaml.safe_dump(obj))
        with self.assertRaisesRegex(ValueError, "contract_id mismatch"):
            self.validate()

    def test_missing_managed_yaml_rejected(self):
        (self.root / self.data["contracts"][0]["dbt_metadata"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing registered file"):
            self.validate()

    def test_symlinked_managed_yaml_rejected(self):
        target = self.root / self.data["contracts"][0]["dbt_metadata"]
        original = target.read_bytes()
        target.unlink()
        other = self.root / "temp-metadata.yml"
        other.write_bytes(original)
        target.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "Symlink forbidden"):
            self.validate()

    def test_unversioned_change_rejected(self):
        current = self.root / self.data["contracts"][0]["contract"]
        current.write_bytes(current.read_bytes().replace(b"physicalType: varchar", b"physicalType: integer"))
        with self.assertRaisesRegex(ValueError, "without a version increase"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
