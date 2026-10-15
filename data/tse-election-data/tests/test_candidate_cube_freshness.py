"""Fail-closed TDD for source-generation and scoped-dimension freshness."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "candidate_cube_freshness", ROOT / "scripts/candidate_cube_freshness.py"
)
fresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fresh)


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.root = Path(handle.name)
        self.source = self.root / "source"
        self.db = self.root / "canary.duckdb"
        self.db.write_bytes(b"canary database fixture")
        self.certs = self.root / "certificates"
        self.cycle = "2024:municipal"
        self.partition = self.source / "election_type=municipal/year=2024"
        self.partition.mkdir(parents=True)
        self.pointer = self.partition / "data.parquet"
        self.pointer.write_bytes(b"fixture-parquet-content")
        self.inputs = {"source": fresh.publication_signature(self.source, self.cycle),
                       "dimensions": {"dim_candidate": {"rows": 2, "sha256": "a"},
                                      "dim_geography": {"rows": 1, "sha256": "b"}}}
        self.output = {"duckdb_version": "1.5.6", "cube": {"rows": 7, "hash": "x"}}

    def publish(self):
        return fresh.issue_certificate(self.certs, self.db, self.source,
                                       self.cycle, self.inputs, self.output)

    def verify(self):
        with patch.object(fresh, "input_signature", return_value=self.inputs), \
             patch.object(fresh, "output_signature", return_value=self.output):
            return fresh.audit_freshness(self.db, self.source, self.certs, {self.cycle})

    def test_passes_only_matching_verified_certificate(self):
        self.publish()
        self.assertEqual(self.verify()["fresh_cycles"], [self.cycle])

    def test_missing_certificate_denies_freshness(self):
        with self.assertRaisesRegex(fresh.FreshnessError, "missing attestation"):
            self.verify()

    def test_replaced_generation_changes_identity(self):
        old = fresh.publication_signature(self.source, self.cycle)
        generation = self.partition / "generation-1.parquet"
        generation.write_bytes(b"new generation")
        self.pointer.unlink()
        self.pointer.symlink_to(generation.name)
        new = fresh.publication_signature(self.source, self.cycle)
        self.assertNotEqual(old, new)
        self.assertEqual(new["generation"], generation.name)
        self.publish()
        with patch.object(fresh, "input_signature", return_value={
            **self.inputs, "source": new
        }), patch.object(fresh, "output_signature", return_value=self.output):
            with self.assertRaisesRegex(fresh.FreshnessError, "Parquet or dimensions changed"):
                fresh.audit_freshness(self.db, self.source, self.certs, {self.cycle})

    def test_in_place_legacy_content_change_is_detected(self):
        before = fresh.publication_signature(self.source, self.cycle)
        self.pointer.write_bytes(b"mutated legacy content")
        self.assertNotEqual(before["sha256"],
                            fresh.publication_signature(self.source, self.cycle)["sha256"])

    def test_invalid_symlink_or_dangling_generation_fails_closed(self):
        self.pointer.unlink()
        self.pointer.symlink_to("not-present.parquet")
        with self.assertRaisesRegex(fresh.FreshnessError, "Broken"):
            fresh.publication_signature(self.source, self.cycle)
        self.pointer.unlink()
        self.pointer.symlink_to("../escaped.parquet")
        with self.assertRaisesRegex(fresh.FreshnessError, "Unsafe"):
            fresh.publication_signature(self.source, self.cycle)

    def test_dimension_change_invalidates(self):
        self.publish()
        changed = dict(self.inputs)
        changed["dimensions"] = {**self.inputs["dimensions"],
                                 "dim_candidate": {"rows": 2, "sha256": "changed"}}
        with patch.object(fresh, "input_signature", return_value=changed), \
             patch.object(fresh, "output_signature", return_value=self.output):
            with self.assertRaisesRegex(fresh.FreshnessError, "dimensions changed"):
                fresh.audit_freshness(self.db, self.source, self.certs, {self.cycle})

    def test_cube_content_change_invalidates(self):
        self.publish()
        with patch.object(fresh, "input_signature", return_value=self.inputs), \
             patch.object(fresh, "output_signature", return_value={"changed": True}):
            with self.assertRaisesRegex(fresh.FreshnessError, "cube contents changed"):
                fresh.audit_freshness(self.db, self.source, self.certs, {self.cycle})

    def test_certificate_cannot_be_reused_for_another_warehouse(self):
        self.publish()
        with self.assertRaisesRegex(fresh.FreshnessError, "different database"):
            fresh.audit_freshness(self.root / "another.duckdb", self.source,
                                  self.certs, {self.cycle})

    def test_certification_revoke_archives_authority(self):
        self.publish()
        fresh.revoke_certificate(self.certs, self.cycle)
        self.assertFalse(fresh.certificate_path(self.certs, self.cycle).exists())
        self.assertEqual(len(list((self.certs / "revoked").glob("*.json"))), 1)
        with self.assertRaisesRegex(fresh.FreshnessError, "missing attestation"):
            self.verify()

    def test_issue_refuses_to_clobber_certificate(self):
        self.publish()
        with self.assertRaisesRegex(fresh.FreshnessError, "overwrite"):
            self.publish()

    def test_invalid_cycle_rejected_without_touching_files(self):
        with self.assertRaisesRegex(fresh.FreshnessError, "Invalid election cycle"):
            fresh.publication_signature(self.source, "../../etc/passwd")

    def test_output_proof_fails_closed_if_any_cycle_unattested(self):
        self.publish()
        with self.assertRaisesRegex(fresh.FreshnessError, "2018:general"):
            self.verify_all({"2018:general", self.cycle})

    def verify_all(self, cycles):
        with patch.object(fresh, "input_signature", return_value=self.inputs), \
             patch.object(fresh, "output_signature", return_value=self.output):
            return fresh.audit_freshness(self.db, self.source, self.certs, cycles)


if __name__ == "__main__":
    unittest.main()
