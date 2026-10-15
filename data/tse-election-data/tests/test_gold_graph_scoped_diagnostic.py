"""Scoped Gold graph diagnostic: trusted test SQL, isolated read-only canary."""
from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "certify_gold_graph_cycle", ROOT / "scripts/certify_gold_graph_cycle.py"
)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class ScopedGraphDiagnosticTests(unittest.TestCase):
    def test_publication_is_bound_to_one_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "election_type=municipal/year=2024"
            ignored = root / "election_type=general/year=2022"
            selected.mkdir(parents=True)
            ignored.mkdir(parents=True)
            (selected / "data.parquet").write_bytes(b"2024-file")
            (ignored / "data.parquet").write_bytes(b"2022-file")
            chosen, proof = diag.selected_publication(root, 2024, "municipal")
            self.assertEqual(chosen, selected / "data.parquet")
            self.assertEqual(proof["bytes"], len(b"2024-file"))

    def test_escaping_symlink_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "election_type=municipal/year=2024"
            selected.mkdir(parents=True)
            external = root / "outside.parquet"
            external.write_bytes(b"untrusted")
            (selected / "data.parquet").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "outside partition"):
                diag.selected_publication(root, 2024, "municipal")

    def test_model_and_assertion_sql_are_unmodified_and_scoped(self):
        sql = diag.render(diag.TESTS / "assert_graph_candidate_vote_conservation.sql",
                          year=2024, election_type="municipal")
        self.assertIn("year in (2024)", sql)
        self.assertIn("election_type in ('municipal')", sql)
        self.assertNotIn("{{", sql)
        model = diag.render(diag.MODELS / "graph_edges.sql",
                            year=2024, election_type="municipal")
        self.assertIn("from fact_candidate_votes", model)
        self.assertIn("from fact_party_votes", model)

    def test_synthetic_cycle_executes_existing_sql_without_main_writes(self):
        import duckdb
        from test_gold_graph_invariants import GoldGraphTests

        fixture = GoldGraphTests("test_graph_projection_conserves_relationships")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "canary.duckdb"
            selected = root / "sources/election_type=municipal/year=2024"
            selected.mkdir(parents=True)
            parquet = selected / "data.parquet"
            fixture.db.execute(f"ATTACH {diag.quote_sql(str(db))} AS evidence")
            try:
                for name in diag.SCOPED_SOURCES:
                    fixture.db.execute(
                        f"CREATE TABLE evidence.{name} AS SELECT * FROM main.{name}"
                    )
            finally:
                fixture.db.execute("DETACH evidence")
            fixture.db.execute(f"""COPY (
                SELECT * EXCLUDE (cycle_id) FROM fact_candidate_votes
            ) TO {diag.quote_sql(str(parquet))} (FORMAT PARQUET)""")
            result = diag.run(argparse.Namespace(
                db=db, source_root=root / "sources", cycle="2024:municipal",
                memory_limit="512MB", temp_directory=None,
                execute=True, tests=["node_identity", "edge_measures",
                                     "candidate_vote_conservation", "party_vote_conservation"],
            ))
            self.assertEqual(len(result["assertions"]), 4)
            self.assertTrue(all(row["pass"] for row in result["assertions"]))
            self.assertIn("not_a_freshness_certificate", result["status"])
            # A separate read-only connection has not mutated source tables.
            with duckdb.connect(str(db), read_only=True) as con:
                tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
                self.assertNotIn("graph_edges", tables)
                self.assertNotIn("graph_vertices", tables)


if __name__ == "__main__":
    unittest.main()
