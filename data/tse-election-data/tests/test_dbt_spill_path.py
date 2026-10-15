"""Regressions for DuckDB/dbt spill defaults and strict profiles.yml YAML.

dbt parses profiles.yml as YAML, so Jinja statements cannot be YAML nodes.
Explicit temp_directory SET is unsafe across cursors after heavy disk spill.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / 'tse_dbt' / 'profiles.yml'
COORDINATOR = ROOT / 'scripts' / 'build_candidate_fact_partition.py'


class DbtSpillPathTests(unittest.TestCase):
    def parsed_profile(self):
        # Parse BEFORE any Jinja interpolation, as dbt does.
        return yaml.safe_load(PROFILES.read_text(encoding='utf-8'))

    def settings(self):
        return self.parsed_profile()['tse_analytics']['outputs']['dev']['settings']

    def test_profile_is_valid_yaml_before_jinja(self):
        self.assertEqual(self.parsed_profile()['tse_analytics']['target'], 'dev')
        self.assertNotIn('{%', PROFILES.read_text())

    def test_canary_never_reapplies_temp_directory(self):
        self.assertNotIn('temp_directory', self.settings())
        self.assertEqual(self.settings()['threads'], 1)

    def test_publication_does_not_export_ignored_spill_override(self):
        self.assertNotIn('TSE_DUCKDB_TEMP_DIRECTORY', PROFILES.read_text())
        runner = COORDINATOR.read_text(encoding='utf-8')
        self.assertIn('env.pop("TSE_DUCKDB_TEMP_DIRECTORY", None)', runner)
        self.assertNotIn('env["TSE_DUCKDB_TEMP_DIRECTORY"]', runner)

    def test_profile_preserves_memory_and_spill_bounds(self):
        settings = self.settings()
        self.assertIn('2GB', settings['memory_limit'])
        self.assertIn('50GB', settings['max_temp_directory_size'])
        self.assertNotIn('data/warehouse/.duckdb_tmp', PROFILES.read_text())

    def test_duckdb_default_uses_database_sibling(self):
        try:
            import duckdb
        except ImportError:
            self.skipTest('DuckDB is installed in the project environment')
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / 'warehouse.duckdb'
            with duckdb.connect(str(db)) as con:
                configured = con.execute("select current_setting('temp_directory')").fetchone()[0]
                self.assertEqual(Path(configured).resolve(), Path(str(db) + '.tmp').resolve())


if __name__ == '__main__':
    unittest.main()
