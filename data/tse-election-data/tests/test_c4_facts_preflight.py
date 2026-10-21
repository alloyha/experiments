import tempfile
import unittest
from pathlib import Path
from scripts.c4_facts_preflight import audit

class PreflightTests(unittest.TestCase):
    def test_missing_sources_fail_without_false_pass(self):
        with tempfile.TemporaryDirectory() as path:
            finding = audit(Path(path))
            self.assertEqual(finding['summary']['FAIL'], 2)
            self.assertEqual(finding['summary']['NOT_EVALUATED'], 12)

    def test_static_check_and_runtime_not_evaluated(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            facts = root / 'tse_dbt/models/gold/facts'
            facts.mkdir(parents=True)
            (facts / 'fact_candidate_votes.sql').write_text('select * from read_parquet("x", hive_partitioning = true)')
            (facts / 'fact_turnout.sql').write_text("{{ config(pre_hook='{{ partition_replace_pre_hook() }}', incremental_strategy='delete+insert') }}")
            finding = audit(root)
            self.assertEqual(finding['summary']['FAIL'], 0)
            self.assertEqual(finding['summary']['PASS'], 4)
            self.assertEqual(finding['summary']['NOT_EVALUATED'], 10)


    def test_real_duckdb_schema(self):
        import duckdb
        from scripts.c4_facts_preflight import FACTS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = root / "tse_dbt/models/gold/facts"
            facts.mkdir(parents=True)

            (facts / "fact_candidate_votes.sql").write_text(
                "select * from read_parquet('x', hive_partitioning = true)"
            )
            (facts / "fact_turnout.sql").write_text(
                "{{ config(pre_hook='{{ partition_replace_pre_hook() }}', "
                "incremental_strategy='delete+insert') }}"
            )

            db = root / "test.duckdb"
            with duckdb.connect(str(db)) as con:
                for name, grain in FACTS.items():
                    fields = ", ".join(
                        f'"{column}" VARCHAR' for column in grain
                    )
                    con.execute(
                        f'CREATE TABLE "{name}" ({fields})'
                    )

            report = audit(root, db)
            self.assertEqual(report["summary"]["FAIL"], 0)
            self.assertEqual(report["summary"]["PASS"], 8)
            self.assertEqual(
                report["summary"]["NOT_EVALUATED"], 6
            )

            for name, grain in FACTS.items():
                checks = report["facts"][name]["checks"]
                self.assertEqual(
                    checks["grain_columns"]["status"], "PASS"
                )
                self.assertEqual(
                    checks["grain_columns"]["columns"], list(grain)
                )

    def test_missing_grain_column_fails_closed(self):
        import duckdb
        from scripts.c4_facts_preflight import FACTS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "test.duckdb"

            with duckdb.connect(str(db)) as con:
                for name, grain in FACTS.items():
                    fields = ", ".join(
                        f'"{column}" VARCHAR'
                        for column in grain
                        if column != "election_year"
                    )
                    con.execute(
                        f'CREATE TABLE "{name}" ({fields})'
                    )

            report = audit(root, db)

            for name in FACTS:
                checks = report["facts"][name]["checks"]
                self.assertEqual(
                    checks["relation"]["status"], "PASS"
                )
                self.assertEqual(
                    checks["grain_columns"]["status"], "FAIL"
                )
                self.assertEqual(
                    checks["grain_columns"]["missing"],
                    ["election_year"]
                )
                for check in ("cycles", "null_grain", "duplicate_grain"):
                    self.assertEqual(
                        checks[check]["status"], "NOT_EVALUATED"
                    )


    def test_partial_history_fails_closed(self):
        import duckdb
        from scripts.c4_facts_preflight import FACTS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = root / "tse_dbt/models/gold/facts"
            facts.mkdir(parents=True)

            (facts / "fact_candidate_votes.sql").write_text(
                "select * from read_parquet('x', hive_partitioning = true)"
            )
            (facts / "fact_turnout.sql").write_text(
                "{{ config(pre_hook='{{ partition_replace_pre_hook() }}', "
                "incremental_strategy='delete+insert') }}"
            )

            db = root / "test.duckdb"

            with duckdb.connect(str(db)) as con:
                for name, grain in FACTS.items():
                    fields = ", ".join(
                        f'"{column}" VARCHAR' for column in grain
                    )
                    con.execute(f'CREATE TABLE "{name}" ({fields})')
                    con.execute(
                        f'INSERT INTO "{name}" '
                        '(election_year, election_type) '
                        "VALUES ('2026', 'general')"
                    )

            report = audit(root, db, scan=True)

            for name in FACTS:
                cycles = report["facts"][name]["checks"]["cycles"]
                self.assertEqual(cycles["status"], "FAIL")
                self.assertEqual(len(cycles["missing"]), 4)
                self.assertEqual(cycles["unexpected"], [])

if __name__ == '__main__': unittest.main()
