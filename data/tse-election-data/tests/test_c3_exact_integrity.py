"""C3 integration regression, no historical raw data required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

from scripts.c3_certification_matrix import assert_relations_equivalent
from scripts.run_coordinated_dbt import _resolve_selected_models


class C3ExactTargetEquivalence(unittest.TestCase):
    def create_db(self, path, *, value=12, second_row=False):
        import duckdb
        db = duckdb.connect(str(path))
        for table in ("silver_tally_munzona", "fact_tally_munzona"):
            db.execute(f"CREATE TABLE {table} (election_year INTEGER, total_votes INTEGER, token VARCHAR)")
            db.execute(f"INSERT INTO {table} VALUES (2022, ?, 'one')", [value])
            if second_row:
                db.execute(f"INSERT INTO {table} VALUES (2022, 3, 'two')")
        db.close()

    def test_full_parity(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'before.duckdb', Path(folder)/'after.duckdb'
            self.create_db(a, second_row=True)
            self.create_db(b, second_row=True)
            assert_relations_equivalent(a,b)

    def test_same_count_and_sum_but_different_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'before.duckdb', Path(folder)/'after.duckdb'
            self.create_db(a, value=12)
            self.create_db(b, value=12)
            import duckdb
            conn = duckdb.connect(str(b))
            conn.execute("UPDATE silver_tally_munzona SET token='changed'")
            conn.close()
            with self.assertRaisesRegex(AssertionError, 'EXCEPT ALL'):
                assert_relations_equivalent(a,b)

    def test_multiplicity_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'before.duckdb', Path(folder)/'after.duckdb'
            self.create_db(a, second_row=True)
            self.create_db(b, second_row=True)
            import duckdb
            conn = duckdb.connect(str(b))
            conn.execute("DELETE FROM silver_tally_munzona WHERE token='two'")
            conn.execute("INSERT INTO silver_tally_munzona VALUES (2022, 3, 'one')")
            conn.close()
            with self.assertRaises(AssertionError):
                assert_relations_equivalent(a,b)


class C3AlternativeWritersFailClosed(unittest.TestCase):
    def test_writing_subcommands_are_rejected_before_dbt_ls(self):
        for op in ('run-operation','seed','snapshot','clone','retry'):
            with self.subTest(op=op), patch('scripts.run_coordinated_dbt.subprocess.run') as runner:
                with self.assertRaisesRegex(ValueError, 'Uncertified dbt writer subcommand'):
                    _resolve_selected_models(Path('/nonexistent/dbt'), [op], {})
                runner.assert_not_called()
