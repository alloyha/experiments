"""Public Makefile years API and inferred dbt scope; no downloads or warehouse writes."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make(*args: str) -> subprocess.CompletedProcess[str]:
    # These tests also run under `make unit`, where MAKELEVEL causes nested
    # make to print directory banners that would corrupt stdout JSON.
    # Force that environment even when the test is run directly.
    env = {**os.environ, "MAKELEVEL": "1"}
    return subprocess.run(
        ["make", "--no-print-directory", *args],
        cwd=ROOT, capture_output=True, text=True, check=False, env=env,
    )


class MakefileScopeTests(unittest.TestCase):
    def test_single_year_updates_itself_and_infers_general(self):
        result = make("print-dbt-vars", "YEARS=2026")
        self.assertEqual(result.returncode, 0, result.stderr)
        scope = json.loads(result.stdout)
        self.assertEqual(scope["election_years"], [2026])
        self.assertEqual(scope["incremental_years"], [2026])
        self.assertEqual(scope["election_types"], ["general"])
        self.assertEqual(scope["incremental_election_types"], ["general"])

    def test_history_and_update_subset_infer_both_types(self):
        result = make(
            "print-dbt-vars", "YEARS=2018 2020 2022 2024 2026", "UPDATE_YEARS=2026",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        scope = json.loads(result.stdout)
        self.assertEqual(scope["election_years"], [2018, 2020, 2022, 2024, 2026])
        self.assertEqual(scope["incremental_years"], [2026])
        self.assertEqual(scope["election_types"], ["general", "municipal"])
        self.assertEqual(scope["incremental_election_types"], ["general"])

    def test_mixed_update_subset_infers_both_types(self):
        result = make(
            "print-dbt-vars", "YEARS=2020 2022 2026", "UPDATE_YEARS=2020 2026",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        scope = json.loads(result.stdout)
        self.assertEqual(scope["incremental_election_types"], ["general", "municipal"])

    def test_ingest_uses_update_years(self):
        result = make("-n", "ingest", "YEARS=2018 2026", "UPDATE_YEARS=2026")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--year 2026", result.stdout)
        self.assertNotIn("--year 2018", result.stdout)

    def test_invalid_scope_fails_before_build_or_download(self):
        for arguments in (
            ("YEARS=2018", "UPDATE_YEARS=2026"),
            ("YEARS=2030",),
            ("YEARS=",),
            ("YEARS=2026", "UPDATE_YEARS="),
        ):
            with self.subTest(arguments=arguments):
                result = make("print-dbt-vars", *arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_removed_public_flags_fail_instead_of_being_ignored(self):
        for assignment in (
            "ELECTION_TYPES=municipal",
            "INCREMENTAL_YEARS=2026",
            "INCREMENTAL_ELECTION_TYPES=general",
        ):
            with self.subTest(assignment=assignment):
                result = make("help", assignment)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
