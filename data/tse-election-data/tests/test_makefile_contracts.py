import unittest
from pathlib import Path


class MakefileContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "Makefile"
        ).read_text(encoding="utf-8")

    def test_make_is_globally_serial(self):
        self.assertIn(".NOTPARALLEL:\n", self.text)

    def test_environment_is_stamped(self):
        self.assertIn("ENV_STAMP :=", self.text)
        self.assertIn("DBT_DEPS_STAMP :=", self.text)

    def test_dbt_deps_is_cached(self):
        self.assertIn(
            "dbt-deps: $(DBT_DEPS_STAMP)",
            self.text,
        )

    def test_python_runtime_is_canonical(self):
        self.assertIn("PY := $(RUN) python", self.text)

    def test_scope_contract_runs_as_module(self):
        self.assertIn(
            "$(PY) -m scripts.check_scope_contracts",
            self.text,
        )
        self.assertNotIn(
            "$(PY) scripts/check_scope_contracts.py",
            self.text,
        )

    def test_non_incremental_paths_do_not_require_incremental_scope(self):
        self.assertIn(
            "dbt-full-refresh: dbt-deps",
            self.text,
        )
        self.assertIn(
            "compile: dbt-deps",
            self.text,
        )
        self.assertNotIn(
            "dbt-full-refresh: scope-contracts",
            self.text,
        )

    def test_clean_all_requires_confirmation(self):
        self.assertIn(
            "CONFIRM=1",
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
