from __future__ import annotations

import unittest
from pathlib import Path


class RuntimePreHookContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.models = cls.root / "tse_dbt/models"

    def test_partition_replace_hooks_are_deferred(self):
        offenders = []

        for path in self.models.rglob("*.sql"):
            text = path.read_text(encoding="utf-8")

            if "pre_hook=partition_replace_pre_hook()" in text:
                offenders.append(str(path))

        self.assertEqual(
            offenders,
            [],
            "partition_replace_pre_hook must be deferred: "
            + ", ".join(offenders),
        )

    def test_source_aware_hooks_are_deferred(self):
        paths = (
            self.models
            / "bronze/party/bronze_party_votes_raw.sql",
            self.models
            / "bronze/tally/bronze_tally_raw.sql",
            self.models
            / "silver/electorate/"
              "silver_electorate_municipality.sql",
        )

        for path in paths:
            text = path.read_text(encoding="utf-8")

            self.assertIn(
                '"{{ ensure_source_snapshot_column_pre_hook() }}"',
                text,
            )

            self.assertIn(
                '"{{ source_aware_partition_replace_pre_hook(',
                text,
            )


if __name__ == "__main__":
    unittest.main()
