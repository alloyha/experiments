"""Adversarial regressions for the immutable ODCS baseline and version gate."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.check_odcs_contract_version import check_contract_version

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "contracts/odcs/baselines/dim_election_cycle.v1.0.0.odcs.yaml"


class OdcsCompatibilityVersionTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="tse-odcs-version-")
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        self.baseline = root / "baseline.yaml"
        self.candidate = root / "candidate.yaml"
        self.baseline.write_bytes(BASELINE.read_bytes())
        self.candidate.write_bytes(BASELINE.read_bytes())

    def edit(self, old: str, new: str, *, baseline: bool = False):
        path = self.baseline if baseline else self.candidate
        source = path.read_text(encoding="utf-8")
        assert old in source, old
        path.write_text(source.replace(old, new), encoding="utf-8")

    def verify(self):
        check_contract_version(self.baseline, self.candidate)

    def test_unchanged_published_contract_passes(self):
        self.verify()

    def test_reformatting_without_semantic_change_passes(self):
        self.edit("version: 1.0.0", 'version: "1.0.0"')
        self.verify()

    def test_frozen_baseline_cannot_change(self):
        self.edit("version: 1.0.0", "version: 1.0.1", baseline=True)
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            self.verify()

    def test_schema_change_without_version_bump_rejected(self):
        self.edit("physicalType: varchar", "physicalType: integer")
        with self.assertRaisesRegex(ValueError, "without a version increase"):
            self.verify()

    def test_documented_additive_change_with_version_bump_passes_version_gate(self):
        self.edit("version: 1.0.0", "version: 1.0.1")
        self.edit("Election class, general or municipal.", "Election kind, general or municipal.")
        self.verify()  # CLI separately classifies compatibility.

    def test_regressive_version_rejected(self):
        self.edit("version: 1.0.0", "version: 0.9.9")
        with self.assertRaisesRegex(ValueError, "regressed"):
            self.verify()

    def test_public_identity_change_rejected(self):
        self.edit("name: dim_election_cycle", "name: another_model")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.verify()

    def test_invalid_semver_rejected(self):
        self.edit("version: 1.0.0", "version: 1.0")
        with self.assertRaisesRegex(ValueError, "MAJOR.MINOR.PATCH"):
            self.verify()

    def test_invalid_yaml_rejected(self):
        self.candidate.write_text("schema: [\n", encoding="utf-8")
        with self.assertRaises(yaml.YAMLError):
            self.verify()

    def test_missing_baseline_rejected(self):
        self.baseline.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
