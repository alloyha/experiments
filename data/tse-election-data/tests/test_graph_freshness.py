"""Pure fast falsifiers; real-cycle validation is an explicit heavy gate."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import graph_freshness as fresh


class GraphFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.warehouse = self.root / "canary.duckdb"
        self.warehouse.write_bytes(b"fixture")
        self.source = self.root / "facts"
        self.source.mkdir()
        self.certs = self.root / "certificates"
        self.cycle = "2024:municipal"
        self.proof = {"publication": {"sha256": "old"},
                      "gold_sources": {"dim_party": {"xor": "1"}},
                      "graph": {"graph_edges": {"xor": "2"}}}

    def write(self):
        return fresh.issue_certificate(self.certs, self.warehouse,
                                       self.source, self.cycle, self.proof)

    def audit(self, proof=None):
        return fresh.audit_certificate(self.certs, self.warehouse, self.source,
                                       self.cycle, self.proof if proof is None else proof)

    def test_missing_fails_closed(self):
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "Missing"):
            self.audit()

    def test_matching_evidence_passes(self):
        self.write()
        self.assertEqual(self.audit()["snapshot"], self.proof)

    def test_source_generation_or_content_change_invalidates(self):
        self.write()
        different = {**self.proof, "publication": {"sha256": "changed"}}
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "freshness failure"):
            self.audit(different)

    def test_gold_dimension_change_invalidates(self):
        self.write()
        different = {**self.proof, "gold_sources": {"dim_party": {"xor": "99"}}}
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "freshness failure"):
            self.audit(different)

    def test_derived_edge_change_invalidates(self):
        self.write()
        different = {**self.proof, "graph": {"graph_edges": {"xor": "changed"}}}
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "freshness failure"):
            self.audit(different)

    def test_different_warehouse_rejected(self):
        self.write()
        another = self.root / "other.duckdb"
        another.write_bytes(b"fixture")
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "different warehouse"):
            fresh.audit_certificate(self.certs, another, self.source, self.cycle, self.proof)

    def test_revoke_archives_and_invalidates(self):
        self.write()
        fresh.revoke_certificate(self.certs, self.cycle)
        self.assertEqual(len(list((self.certs / "revoked").glob("*.json"))), 1)
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "Missing"):
            self.audit()

    def test_no_clobber_even_if_evidence_matches(self):
        self.write()
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "exists"):
            self.write()

    def test_tampered_assertion_list_rejected(self):
        p = self.write()
        record = json.loads(p.read_text())
        record["assertions"] = ["node_identity"]
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "invariant assertion set"):
            self.audit()

    def test_invalid_cycle_is_rejected_without_files(self):
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "Invalid cycle"):
            fresh.certificate_path(self.certs, "../../secrets")
        self.assertFalse(self.certs.exists())

    def test_failed_diagnostics_leave_previous_certificate_revoked(self):
        self.write()
        before = self.proof
        args = ["--db", str(self.warehouse), "--source-root", str(self.source),
                "--attestations-dir", str(self.certs), "--cycle", self.cycle,
                "certify"]
        with patch.object(fresh, "snapshot", return_value=before), \
             patch.object(fresh, "run_diagnostic", side_effect=AssertionError("bad conservation")):
            self.assertEqual(fresh.main(args), 1)
        with self.assertRaisesRegex(fresh.GraphFreshnessError, "Missing"):
            self.audit()

    def test_missing_audit_is_fast_and_never_scans(self):
        args = ["--db", str(self.warehouse), "--source-root", str(self.source),
                "--attestations-dir", str(self.certs), "--cycle", self.cycle,
                "audit"]
        with patch.object(fresh, "snapshot") as scan:
            self.assertEqual(fresh.main(args), 1)
            scan.assert_not_called()

    def test_snapshot_failure_during_certify_revokes_old_proof(self):
        self.write()
        args = ["--db", str(self.warehouse), "--source-root", str(self.source),
                "--attestations-dir", str(self.certs), "--cycle", self.cycle,
                "certify"]
        with patch.object(fresh, "snapshot", side_effect=fresh.GraphFreshnessError("failed source")):
            self.assertEqual(fresh.main(args), 1)
        self.assertFalse(fresh.certificate_path(self.certs, self.cycle).exists())

    def test_changed_mid_certification_does_not_publish(self):
        args = ["--db", str(self.warehouse), "--source-root", str(self.source),
                "--attestations-dir", str(self.certs), "--cycle", self.cycle,
                "certify"]
        result = {"assertions": [{"pass": True} for _ in fresh.ASSERTIONS]}
        with patch.object(fresh, "snapshot", side_effect=[self.proof, {**self.proof, "graph": {"changed": 1}}]), \
             patch.object(fresh, "run_diagnostic", return_value=result):
            self.assertEqual(fresh.main(args), 1)
        self.assertFalse(fresh.certificate_path(self.certs, self.cycle).exists())


    def test_relation_fingerprint_is_json_canonical(self):
        """Live schema and certificate schema compare identically."""
        connection = Mock()
        connection.execute.side_effect = [
            Mock(fetchall=Mock(return_value=[
                (0, "cycle_id", "VARCHAR"),
                (1, "party_number", "VARCHAR"),
            ])),
            Mock(fetchone=Mock(return_value=(2, 7, 11))),
        ]
        fingerprint = fresh._relation_fingerprint(connection, "dim_party", self.cycle)
        self.assertEqual(fingerprint["columns"], [
            ["cycle_id", "VARCHAR"], ["party_number", "VARCHAR"]
        ])
        self.assertEqual(fingerprint, json.loads(json.dumps(fingerprint)))

        proof = {**self.proof, "gold_sources": {"dim_party": fingerprint}}
        fresh.issue_certificate(self.certs, self.warehouse, self.source,
                                self.cycle, proof)
        self.assertEqual(self.audit(proof)["snapshot"], proof)

    def test_cross_process_fact_republication_invalidates(self):
        """A real Parquet replacement must fail audit across process boundaries."""
        import os
        from test_gold_graph_invariants import GoldGraphTests
        from scripts.certify_gold_graph_cycle import SCOPED_SOURCES, quote_sql

        fixture = GoldGraphTests("test_graph_projection_conserves_relationships")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "actual-canary.duckdb"
            partition = root / "sources/election_type=municipal/year=2024"
            partition.mkdir(parents=True)
            parquet = partition / "data.parquet"

            fixture.db.execute(f"ATTACH {quote_sql(str(db))} AS evidence")
            try:
                for name in SCOPED_SOURCES:
                    fixture.db.execute(
                        f"CREATE TABLE evidence.{name} AS SELECT * FROM main.{name}"
                    )
            finally:
                fixture.db.execute("DETACH evidence")
            fixture.db.execute(
                f"COPY (SELECT * EXCLUDE (cycle_id) FROM fact_candidate_votes) "
                f"TO {quote_sql(str(parquet))} (FORMAT PARQUET)"
            )
            original = parquet.read_bytes()

            base = [sys.executable, str(fresh.ROOT / "scripts/graph_freshness.py"),
                    "--db", str(db), "--source-root", str(root / "sources"),
                    "--attestations-dir", str(root / "proofs"),
                    "--cycle", self.cycle, "--memory-limit", "512MB"]

            def invoke(action):
                return subprocess.run(base + [action], cwd=fresh.ROOT,
                                      text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=120)

            first = invoke("certify")
            self.assertEqual(first.returncode, 0, first.stdout)
            initial = invoke("audit")
            self.assertEqual(initial.returncode, 0, initial.stdout)

            # Publish a legitimate, different Parquet with the same schema.
            changed = partition / "generation-changed.parquet"
            fixture.db.execute(
                "COPY (SELECT * EXCLUDE (cycle_id) "
                "REPLACE (nominal_valid_votes + 1 AS nominal_valid_votes) "
                "FROM fact_candidate_votes) "
                f"TO {quote_sql(str(changed))} (FORMAT PARQUET)"
            )
            os.replace(changed, parquet)
            rejected = invoke("audit")
            self.assertEqual(rejected.returncode, 1, rejected.stdout)
            self.assertIn("freshness failure", rejected.stdout)

            # Restore the exact bytes: freshness must recover without reissue.
            parquet.write_bytes(original)
            restored = invoke("audit")
            self.assertEqual(restored.returncode, 0, restored.stdout)

    def test_cross_process_certify_audit_and_dimension_invalidation(self):
        """Exercise real scoped DuckDB proofs across independent processes."""
        import duckdb
        from test_gold_graph_invariants import GoldGraphTests
        from scripts.certify_gold_graph_cycle import SCOPED_SOURCES, quote_sql

        fixture = GoldGraphTests("test_graph_projection_conserves_relationships")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "actual-canary.duckdb"
            publication = root / "sources/election_type=municipal/year=2024"
            publication.mkdir(parents=True)
            parquet = publication / "data.parquet"

            fixture.db.execute(f"ATTACH {quote_sql(str(db))} AS evidence")
            try:
                for name in SCOPED_SOURCES:
                    fixture.db.execute(
                        f"CREATE TABLE evidence.{name} AS SELECT * FROM main.{name}"
                    )
            finally:
                fixture.db.execute("DETACH evidence")
            fixture.db.execute(
                f"COPY (SELECT * EXCLUDE (cycle_id) FROM fact_candidate_votes) "
                f"TO {quote_sql(str(parquet))} (FORMAT PARQUET)"
            )

            base = [
                sys.executable, str(fresh.ROOT / "scripts/graph_freshness.py"),
                "--db", str(db),
                "--source-root", str(root / "sources"),
                "--attestations-dir", str(root / "proofs"),
                "--cycle", self.cycle,
                "--memory-limit", "512MB",
            ]

            def invoke(action):
                return subprocess.run(base + [action], cwd=fresh.ROOT,
                                      text=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, timeout=120)

            issued = invoke("certify")
            self.assertEqual(issued.returncode, 0, issued.stdout)
            self.assertIn("graph proof created", issued.stdout)

            verified = invoke("audit")
            self.assertEqual(verified.returncode, 0, verified.stdout)
            self.assertIn("freshness matches certificate", verified.stdout)

            # A real Gold update must still invalidate the proof.
            with duckdb.connect(str(db)) as connection:
                connection.execute(
                    "UPDATE dim_party SET party_name = party_name || '_altered' "
                    "WHERE cycle_id = '2024:municipal'"
                )
            rejected = invoke("audit")
            self.assertEqual(rejected.returncode, 1, rejected.stdout)
            self.assertIn("freshness failure", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
