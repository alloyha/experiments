"""Tests for the fail-closed, offline-only candidate-cube contract gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_staged_candidate_cube_contracts as gate
from scripts import consolidate_candidate_cube_transfer as evidence


class OfflineCubeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.main = self.base / "tse_analytics.duckdb"
        self.preview = self.base / "tse_event_registry_preview.duckdb"
        self.clone = self.base / "tse_analytics_cube_v1.duckdb"
        for p, content in [(self.main, b"main"), (self.preview, b"preview"),
                           (self.clone, b"clone")]:
            p.write_bytes(content)
        self.root = self.base / "fact_candidate_votes"
        self.root.mkdir()
        self.manifest_path = self.base / "manifest.json"
        self.stage_path = self.base / "stage.json"
        self.report = self.base / "contracts.json"
        self.manifest = {
            "result": "PASS",
            "scope": gate.staging.READ_SCOPE,
            "main": str(self.main), "preview": str(self.preview),
            "source_root": str(self.root),
            "main_sha256": evidence.digest(self.main),
            "preview_sha256": evidence.digest(self.preview),
            "cycle_count": 5, "total_rows": 50, "total_nominal_votes": 500,
            "cycles": {c: {
                "rows": 10, "nominal_votes": 100, "mismatch_rows": 0,
                "columns": 29, "source_signature": {"generation": "legacy-regular-file", "sha256": "a"*64, "bytes": 10}
            } for c in evidence.CYCLES},
        }
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.manifest_hash = evidence.digest(self.manifest_path)
        self.clone_hash = evidence.digest(self.clone)
        self.stage = {
            "result": "PASS", "scope": gate.staging.STAGE_SCOPE,
            "main": str(self.main), "preview": str(self.preview),
            "source_root": str(self.root), "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_hash,
            "main_sha256": self.manifest["main_sha256"],
            "preview_sha256": self.manifest["preview_sha256"],
            "output_db": str(self.clone), "output_sha256": self.clone_hash,
            "schema_columns": 29,
            "total_rows": 50, "total_nominal_votes": 500,
            "cycles": {c: {"rows": 10, "nominal_votes": 100,
                           "missing_rows": 0, "extra_rows": 0} for c in evidence.CYCLES},
        }
        self.save_stage()

    def save_stage(self):
        self.stage_path.write_text(json.dumps(self.stage), encoding="utf-8")

    def check(self):
        return gate.validate_metadata(
            main=self.main, preview=self.preview, clone=self.clone,
            root=self.root, manifest_path=self.manifest_path,
            manifest_hash=self.manifest_hash, stage_path=self.stage_path,
            clone_hash=self.clone_hash, report=self.report,
        )

    def test_valid_provenance_and_full_cycle_inventory(self):
        manifest, stage = self.check()
        self.assertEqual(manifest["total_rows"], 50)
        self.assertEqual(len(stage["cycles"]), 5)

    def test_reject_clone_tampering(self):
        self.clone.write_bytes(b"changed")
        with self.assertRaisesRegex(gate.OfflineAuditError, "clone SHA-256"):
            self.check()

    def test_reject_stage_coverage_drift(self):
        self.stage["cycles"]["2022:general"]["extra_rows"] = 1
        self.save_stage()
        with self.assertRaisesRegex(gate.OfflineAuditError, "2022:general"):
            self.check()

    def test_reject_manifest_tampering(self):
        self.manifest_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(gate.OfflineAuditError, "manifest SHA"):
            self.check()

    def test_reject_wrong_stage_output(self):
        self.stage["output_db"] = str(self.main)
        self.save_stage()
        with self.assertRaisesRegex(gate.OfflineAuditError, "identity/coverage"):
            self.check()

    def test_reject_wal_even_if_database_hash_matches(self):
        Path(str(self.clone) + ".wal").write_bytes(b"unfinished")
        with self.assertRaisesRegex(gate.staging.StageError, "WAL"):
            self.check()

    def test_reject_existing_output_no_overwrite(self):
        self.report.write_text("original")
        with self.assertRaisesRegex(gate.OfflineAuditError, "overwrite"):
            self.check()
        self.assertEqual(self.report.read_text(), "original")

    def test_reject_symlinked_clone(self):
        self.clone.unlink()
        self.clone.symlink_to(self.main)
        with self.assertRaises(evidence.EvidenceError):
            self.check()

    def test_reject_ack_absent_before_any_locks_or_sql(self):
        with self.assertRaisesRegex(gate.OfflineAuditError, "acknowledge"):
            gate.audit(main=self.main, preview=self.preview, clone=self.clone,
                       root=self.root, manifest_path=self.manifest_path,
                       manifest_hash=self.manifest_hash, stage_path=self.stage_path,
                       clone_hash=self.clone_hash, report=self.report,
                       memory="4GB", timeout=0, ack_quiescent=False)
        self.assertFalse(self.report.exists())

    def test_render_rejects_unknown_name_and_cycle(self):
        with self.assertRaisesRegex(gate.OfflineAuditError, "unsupported"):
            gate.render_test("unknown", "2024:municipal")
        with self.assertRaisesRegex(gate.OfflineAuditError, "unsupported"):
            gate.render_test(gate.TESTS[0], "2023:general")

    def test_render_strict_ref_and_cycle_pruning(self):
        path = self.base / "tse_dbt/tests"
        path.mkdir(parents=True)
        (path / f"{gate.TESTS[0]}.sql").write_text(
            "select * from {{ ref('fact_candidate_votes') }} "
            "where {{ incremental_partition_predicate('year', 'election_type') }}",
            encoding="utf8",
        )
        with patch.object(gate, "PROJECT", self.base):
            sql, sha = gate.render_test(gate.TESTS[0], "2022:general")
            self.assertIn('"main"."fact_candidate_votes"', sql)
            self.assertIn("year = 2022 AND election_type = 'general'", sql)
            self.assertEqual(sha, evidence.digest(path / f"{gate.TESTS[0]}.sql"))
            (path / f"{gate.TESTS[0]}.sql").write_text(
                "select * from {{ ref('unknown_table') }}")
            with self.assertRaises(gate.OfflineAuditError):
                gate.render_test(gate.TESTS[0], "2022:general")

    def test_reject_non_sane_output_memory(self):
        with self.assertRaisesRegex(gate.OfflineAuditError, "memory-limit"):
            gate.audit(main=self.main, preview=self.preview, clone=self.clone,
                       root=self.root, manifest_path=self.manifest_path,
                       manifest_hash=self.manifest_hash, stage_path=self.stage_path,
                       clone_hash=self.clone_hash, report=self.report,
                       memory="8GB; DROP TABLE", timeout=0, ack_quiescent=True)

    def test_render_both_real_dbt_tests_if_present(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2 not installed in this environment")
        if not all((ROOT / f"tse_dbt/tests/{test}.sql").exists() for test in gate.TESTS):
            self.skipTest("TSE dbt tests missing in this standalone patch workspace")
        for name in gate.TESTS:
            for cycle in evidence.CYCLES:
                sql, source_sha = gate.render_test(name, cycle)
                self.assertIn("main", sql)
                self.assertEqual(len(source_sha), 64)

    def test_real_duckdb_contracts_small_fixture_if_installed(self):
        try:
            import duckdb
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("DuckDB/Jinja2 unavailable in this container")
        if not all((ROOT / f"tse_dbt/tests/{test}.sql").is_file() for test in gate.TESTS):
            self.skipTest("dbt test SQL sources not present in patch workspace")
        db = self.base / "tiny.duckdb"
        con = duckdb.connect(str(db))
        try:
            con.execute("""create table fact_candidate_votes as select
                2024 as election_year, 2024 as year, 'municipal' as election_type,
                '2024:municipal:1' as election_id, 1 as round_number,
                'GO' as uf, '00001' as municipality_code, '11' as office_code,
                false as is_transit_vote, '42' as candidate_id,
                100::BIGINT as nominal_valid_votes, 1 as zone""")
            con.execute("""create table cube_candidate_performance_municipality as select
                2024 as election_year, 'municipal' as election_type, '2024:municipal' as cycle_id,
                '2024:municipal:1' as election_id, 1 as round_number,
                'GO' as uf, '00001' as municipality_code, '11' as office_code,
                false as is_transit_vote, '42' as candidate_id,
                100::BIGINT as nominal_valid_votes, 1::BIGINT as zone_rows,
                1::BIGINT as zone_count,
                100::BIGINT as observed_candidate_nominal_valid_votes,
                1::BIGINT as municipal_candidate_rank,
                1.0::DOUBLE as candidate_share_of_observed_nominal_votes""")
            for name in gate.TESTS:
                sql, _ = gate.render_test(name, "2024:municipal")
                self.assertIsNone(con.execute(f"SELECT * FROM ({sql}) t LIMIT 1").fetchone())
            con.execute("update cube_candidate_performance_municipality "
                        "set candidate_share_of_observed_nominal_votes = 0.5")
            sql, _ = gate.render_test(gate.TESTS[1], "2024:municipal")
            self.assertIsNotNone(con.execute(f"SELECT * FROM ({sql}) t LIMIT 1").fetchone())
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
