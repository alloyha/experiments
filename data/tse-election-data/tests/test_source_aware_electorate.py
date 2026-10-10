from __future__ import annotations

import unittest
from pathlib import Path


class SourceAwareElectorateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

        cls.macro = (
            cls.root
            / "tse_dbt/macros/source_snapshot.sql"
        ).read_text(encoding="utf-8")

        cls.model = (
            cls.root
            / "tse_dbt/models/silver/electorate/"
              "silver_electorate_municipality.sql"
        ).read_text(encoding="utf-8")

        cls.dbt_test = (
            cls.root
            / "tse_dbt/tests/"
              "assert_silver_electorate_source_snapshot_current.sql"
        ).read_text(encoding="utf-8")

    def test_snapshot_identity_uses_immutable_source_sha(self):
        self.assertIn("source_sha256", self.macro)
        self.assertIn("resource_id", self.macro)
        self.assertIn("snapshot_component", self.macro)
        self.assertIn("md5(", self.macro)

    def test_incremental_migration_adds_snapshot_column(self):
        self.assertIn(
            "add column if not exists",
            self.macro.lower(),
        )
        self.assertIn(
            "source_snapshot_id",
            self.model,
        )

    def test_unchanged_path_never_references_bronze(self):
        marker = (
            "{% if is_incremental() "
            "and not should_refresh %}"
        )

        self.assertIn(marker, self.model)

        branch = self.model.split(
            marker,
            1,
        )[1]

        noop, refresh = branch.split(
            "{% else %}",
            1,
        )

        self.assertIn(
            "from {{ this }}",
            noop,
        )
        self.assertIn(
            "where false",
            noop,
        )

        # Test executable dependency, not prose/comments.
        self.assertNotIn(
            "ref('bronze_electorate')",
            noop,
        )

        self.assertIn(
            "ref('bronze_electorate')",
            refresh,
        )

    def test_changed_partition_is_replaced_atomically(self):
        self.assertIn(
            "source_aware_partition_replace_pre_hook",
            self.model,
        )
        self.assertIn(
            "delete from {{ this }}",
            self.macro,
        )

    def test_current_snapshot_is_asserted(self):
        self.assertIn(
            "expected_source_snapshot_id",
            self.dbt_test,
        )
        self.assertIn(
            "actual_min_source_snapshot_id",
            self.dbt_test,
        )
        self.assertIn(
            "actual_max_source_snapshot_id",
            self.dbt_test,
        )
        self.assertIn(
            "distinct_snapshot_count",
            self.dbt_test,
        )
        self.assertIn(
            "non_null_snapshot_count",
            self.dbt_test,
        )


if __name__ == "__main__":
    unittest.main()
