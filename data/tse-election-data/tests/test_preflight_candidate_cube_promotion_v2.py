"""Synthetic, filesystem-backed falsifiers for the second-promotion admission gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import consolidate_candidate_cube_transfer as evidence
from scripts import stage_candidate_cube as staging
from scripts import audit_staged_candidate_cube_contracts as offline
from scripts import preflight_candidate_cube_promotion_v2 as gate


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf8")


class CubePromotionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.root = self.dir / "fact_candidate_votes"
        self.root.mkdir()
        self.main = self.dir / "main.duckdb"
        self.preview = self.dir / "preview.duckdb"
        self.clone = self.dir / "cube.duckdb"
        self.main.write_bytes(b"main-baseline")
        self.preview.write_bytes(b"preview-baseline")
        self.clone.write_bytes(b"candidate-validated-clone")
        self.backup = self.dir / "main.backup-v1.duckdb"
        self.backup.write_bytes(b"original-legacy")
        (self.dir / "main.duckdb.lifecycle").write_text("a" * 32 + "\n")
        self.journal = self.dir / "main.duckdb.promotion.json"
        write_json(self.journal, {
            "version": 1, "id": "v1-unique-id", "phase": "ACTIVE_UNCERTIFIED",
            "original": str(self.main), "candidate": str(self.main),
            "original_sha256": evidence.digest(self.backup),
            "candidate_sha256": evidence.digest(self.main),
            "backup": str(self.backup), "lifecycle_epoch": "a" * 32,
        })
        (self.dir / "main.duckdb.build.lock").touch()
        (self.dir / "preview.duckdb.build.lock").touch()
        self.cycles = {}
        for cycle in evidence.CYCLES:
            part = evidence.partition_for(self.root, cycle)
            part.mkdir(parents=True)
            (part / ".publish.lock").touch()
            (part / "data.parquet").write_bytes(("PARQUET-" + cycle).encode())
            self.cycles[cycle] = {
                "rows": 3, "nominal_votes": 11, "mismatch_rows": 0,
                "columns": 29, "source_signature": evidence.publication_signature(part)
            }
        self.manifest = self.dir / "manifest.json"
        self.manifest_data = {
            "result": "PASS", "scope": staging.READ_SCOPE,
            "main": str(self.main), "preview": str(self.preview), "source_root": str(self.root),
            "main_sha256": evidence.digest(self.main),
            "preview_sha256": evidence.digest(self.preview),
            "cycles": self.cycles, "cycle_count": 5,
            "total_rows": 15, "total_nominal_votes": 55,
        }
        write_json(self.manifest, self.manifest_data)
        self.stage = self.dir / "stage.json"
        self.stage_data = {
            "result": "PASS", "scope": staging.STAGE_SCOPE,
            "main": str(self.main), "preview": str(self.preview),
            "source_root": str(self.root), "manifest": str(self.manifest),
            "manifest_sha256": evidence.digest(self.manifest),
            "main_sha256": evidence.digest(self.main),
            "preview_sha256": evidence.digest(self.preview),
            "output_db": str(self.clone), "output_sha256": evidence.digest(self.clone),
            "schema_columns": 29, "total_rows": 15, "total_nominal_votes": 55,
            "cycles": {cycle: {"rows": 3, "nominal_votes": 11,
                               "missing_rows": 0, "extra_rows": 0}
                       for cycle in evidence.CYCLES},
        }
        write_json(self.stage, self.stage_data)
        self.sql_root = self.dir / "project" / "tse_dbt" / "tests"
        self.sql_root.mkdir(parents=True)
        sqlhashes = {}
        for name in offline.TESTS:
            sql = self.sql_root / f"{name}.sql"
            sql.write_text(f"SELECT '{name}';\n")
            sqlhashes[name] = evidence.digest(sql)
        self.contracts = self.dir / "contracts.json"
        self.contract_data = {
            "result": "PASS", "scope": offline.SCOPE,
            "main": str(self.main), "preview": str(self.preview), "clone": str(self.clone),
            "source_root": str(self.root),
            "main_sha256": evidence.digest(self.main),
            "preview_sha256": evidence.digest(self.preview),
            "clone_sha256": evidence.digest(self.clone),
            "manifest": str(self.manifest),
            "manifest_sha256": evidence.digest(self.manifest),
            "stage_report": str(self.stage),
            "stage_report_sha256": evidence.digest(self.stage),
            "total_rows": 15, "total_nominal_votes": 55,
            "cycles": [
                {"cycle": c, "not_null": "PASS", "grain_unique": "PASS",
                 "tests": [{"test": n, "pass": True, "sha256": sqlhashes[n]}
                           for n in offline.TESTS]}
                for c in evidence.CYCLES
            ]
        }
        write_json(self.contracts, self.contract_data)
        self.output = self.dir / "promotion-preflight.json"
        p = patch.object(offline, "PROJECT", self.sql_root.parents[1])
        p.start()
        self.addCleanup(p.stop)

    def run_gate(self):
        return gate.preflight(
            main=self.main, preview=self.preview, clone=self.clone,
            root=self.root, manifest_path=self.manifest,
            manifest_hash=evidence.digest(self.manifest),
            stage_path=self.stage, stage_hash=evidence.digest(self.stage),
            contracts_path=self.contracts,
            contracts_hash=evidence.digest(self.contracts),
            clone_hash=evidence.digest(self.clone), report_path=self.output,
            lock_timeout=0.05)

    def test_full_legacy_anchor_and_candidate_provenance_pass_readonly(self):
        originals = {p: evidence.digest(p) for p in
                     (self.main, self.preview, self.clone, self.backup, self.journal)}
        result = self.run_gate()
        self.assertEqual(result["scope"], gate.SCOPE)
        self.assertEqual(result["legacy_anchor"]["legacy_promotion_id"], "v1-unique-id")
        self.assertEqual(result["cycle_count"], 5)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.dir / "main.duckdb.promotion-v2").exists())
        for path, expected in originals.items():
            self.assertEqual(evidence.digest(path), expected)

    def test_create_only_report_does_not_overwrite(self):
        result = self.run_gate()
        evidence.write_create_only(self.output, result)
        with self.assertRaises(evidence.EvidenceError):
            evidence.write_create_only(self.output, result)

    def test_denies_existing_v2_state(self):
        (self.dir / "main.duckdb.promotion-v2").mkdir()
        with self.assertRaisesRegex(gate.PreflightError, "v2 state"):
            self.run_gate()

    def test_denies_wrong_legacy_phase(self):
        d = json.loads(self.journal.read_text())
        d["phase"] = "EPOCH_ROTATED"
        write_json(self.journal, d)
        with self.assertRaisesRegex(gate.PreflightError, "v1 promotion"):
            self.run_gate()

    def test_denies_legacy_epoch_change(self):
        (self.dir / "main.duckdb.lifecycle").write_text("b" * 32 + "\n")
        with self.assertRaisesRegex(gate.PreflightError, "epoch"):
            self.run_gate()

    def test_denies_bad_legacy_backup(self):
        self.backup.write_bytes(b"tampered")
        with self.assertRaisesRegex(gate.PreflightError, "backup SHA"):
            self.run_gate()

    def test_denies_clone_wal(self):
        (self.dir / "cube.duckdb.wal").touch()
        with self.assertRaises(staging.StageError):
            self.run_gate()

    def test_denies_missing_partition_lock(self):
        (evidence.partition_for(self.root, "2018:general") / ".publish.lock").unlink()
        with self.assertRaises(evidence.EvidenceError):
            self.run_gate()

    def test_denies_source_parquet_drift(self):
        (evidence.partition_for(self.root, "2018:general") / "data.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(staging.StageError, "Parquet publication drift"):
            self.run_gate()

    def test_denies_offline_sql_failure(self):
        self.contract_data["cycles"][0]["tests"][0]["pass"] = False
        write_json(self.contracts, self.contract_data)
        with self.assertRaisesRegex(gate.PreflightError, "invalid SQL proof"):
            self.run_gate()

    def test_denies_offline_test_code_change(self):
        (self.sql_root / f"{offline.TESTS[0]}.sql").write_text("-- changed")
        with self.assertRaisesRegex(gate.PreflightError, "SQL test changed"):
            self.run_gate()

    def test_denies_missing_cycle(self):
        self.contract_data["cycles"].pop()
        write_json(self.contracts, self.contract_data)
        with self.assertRaisesRegex(gate.PreflightError, "missing offline cycle"):
            self.run_gate()

    def test_denies_stage_mismatch(self):
        self.stage_data["output_sha256"] = "f" * 64
        write_json(self.stage, self.stage_data)
        with self.assertRaises(offline.OfflineAuditError):
            self.run_gate()

    def test_denies_existing_output(self):
        self.output.write_text("should-never-be-replaced")
        with self.assertRaisesRegex(gate.PreflightError, "already exists"):
            self.run_gate()
        self.assertEqual(self.output.read_text(), "should-never-be-replaced")


if __name__ == "__main__":
    unittest.main()
