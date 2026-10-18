import subprocess
import sys
import unittest
from pathlib import Path


class MedallionDAGContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.checker = cls.root / "scripts/check_medallion_dag.py"

    def test_checker_exists_and_compiles(self):
        self.assertTrue(self.checker.is_file())
        subprocess.run([sys.executable, "-m", "py_compile", str(self.checker)], check=True)

    def test_detail_and_mermaid_derive_from_validated_dag(self):
        detail = subprocess.run(
            [sys.executable, str(self.checker), "--format", "detail"], cwd=self.root,
            check=True, capture_output=True, text=True,
        ).stdout
        for text in ("Medallion DAG contract: PASS", "=== MODELS ===",
                     "=== PHYSICAL LINEAGE ===", "dim_election_cycle"):
            self.assertIn(text, detail)

        mermaid = subprocess.run(
            [sys.executable, str(self.checker), "--format", "mermaid"], cwd=self.root,
            check=True, capture_output=True, text=True,
        ).stdout
        self.assertTrue(mermaid.startswith("flowchart LR\n"))
        self.assertIn("subgraph GOLD", mermaid)
        self.assertIn("model_dim_election_cycle", mermaid)
        self.assertIn("physical_candidate_fact_partition", mermaid)

    def test_invalid_report_format_fails(self):
        result = subprocess.run(
            [sys.executable, str(self.checker), "--format", "unknown"], cwd=self.root,
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_current_dag_contract_passes(self):
        result = subprocess.run(
            [sys.executable, str(self.checker)], cwd=self.root,
            check=True, capture_output=True, text=True,
        )
        self.assertIn("Medallion DAG contract: PASS", result.stdout)
        self.assertIn("physical lineages: 1", result.stdout)
        self.assertIn("declared orphans: 1", result.stdout)

    def test_candidate_physical_lineage_is_declared(self):
        publisher = (self.root / "tse_dbt/models/physical/int_candidate_fact_partition.sql").read_text()
        consumer = (self.root / "tse_dbt/models/gold/facts/fact_candidate_votes.sql").read_text()
        self.assertIn("'physical_publish': 'candidate_fact_partition'", publisher)
        self.assertIn("'physical_source': 'candidate_fact_partition'", consumer)

    def test_disabled_electorate_section_is_declared_orphan(self):
        model = (self.root / "tse_dbt/models/bronze/electorate/bronze_electorate_section.sql").read_text()
        self.assertIn("'architecture_status': 'orphan'", model)
        self.assertIn("'architecture_reason':", model)


if __name__ == "__main__":
    unittest.main()
