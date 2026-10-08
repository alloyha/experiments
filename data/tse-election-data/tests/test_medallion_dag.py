import subprocess
import sys
import unittest
from pathlib import Path


class MedallionDAGContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.checker = cls.root / "scripts/check_medallion_dag.py"
        cls.printer = cls.root / "scripts/print_medallion_dag.py"

    def test_tools_exist_and_compile(self):
        for script in (self.checker, self.printer):
            self.assertTrue(script.is_file())
            subprocess.run([sys.executable, "-m", "py_compile", str(script)], check=True)

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
