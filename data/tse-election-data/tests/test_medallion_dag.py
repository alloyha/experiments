import subprocess
import sys
import unittest
from pathlib import Path

class MedallionDAGContractTests(unittest.TestCase):
    def test_checker_exists_and_compiles(self):
        root=Path(__file__).resolve().parents[1]
        checker=root/'scripts/check_medallion_dag.py'
        self.assertTrue(checker.is_file())
        subprocess.run([sys.executable,'-m','py_compile',str(checker)],check=True)

if __name__=='__main__':
    unittest.main()
