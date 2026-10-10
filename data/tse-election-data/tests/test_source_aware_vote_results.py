from __future__ import annotations

import unittest
from pathlib import Path


class SourceAwareVoteResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]

        cls.party = (
            root
            / "tse_dbt/models/bronze/party/"
              "bronze_party_votes_raw.sql"
        ).read_text(encoding="utf-8")

        cls.tally = (
            root
            / "tse_dbt/models/bronze/tally/"
              "bronze_tally_raw.sql"
        ).read_text(encoding="utf-8")

        cls.raw_macro = (
            root
            / "tse_dbt/macros/read_raw_csv.sql"
        ).read_text(encoding="utf-8")

        cls.files_macro = (
            root
            / "tse_dbt/macros/current_raw_files.sql"
        ).read_text(encoding="utf-8")

        cls.snapshot_macro = (
            root
            / "tse_dbt/macros/source_snapshot.sql"
        ).read_text(encoding="utf-8")

        cls.makefile = (
            root / "Makefile"
        ).read_text(encoding="utf-8")

    def test_raw_reader_supports_incremental_scope(self):
        self.assertIn(
            "incremental_scope=false",
            self.raw_macro,
        )
        self.assertIn(
            "incremental_scope=false",
            self.files_macro,
        )
        self.assertIn(
            "incremental_years",
            self.files_macro,
        )

    def test_party_has_single_pre_hook_contract(self):
        self.assertEqual(
            self.party.count("pre_hook="),
            1,
        )
        self.assertNotIn(
            "replace_selected_cycle_partitions",
            self.party,
        )

    def test_bronze_models_are_source_aware(self):
        for model in (self.party, self.tally):
            self.assertIn(
                "source_snapshot_changed(",
                model,
            )
            self.assertIn(
                "source_aware_partition_replace_pre_hook(",
                model,
            )
            self.assertIn(
                "source_snapshot_id",
                model,
            )

    def test_noop_paths_do_not_read_raw_csv(self):
        marker = (
            "{% if is_incremental() "
            "and not should_refresh %}"
        )

        for model in (self.party, self.tally):
            branch = model.split(marker, 1)[1]
            noop, refresh = branch.split(
                "{% else %}",
                1,
            )

            self.assertIn(
                "from {{ this }}",
                noop,
            )
            self.assertNotIn(
                "read_raw_csv(",
                noop,
            )
            self.assertIn(
                "read_raw_csv(",
                refresh,
            )

    def test_refresh_reads_only_incremental_scope(self):
        for model in (self.party, self.tally):
            self.assertIn(
                "incremental_scope=is_incremental()",
                model,
            )

    def test_partition_replacement_is_whole_partition(self):
        start = self.snapshot_macro.index(
            "{% macro source_aware_partition_replace_pre_hook("
        )
        end = self.snapshot_macro.index(
            "{% macro source_snapshot_changed(",
            start,
        )
        hook = self.snapshot_macro[start:end]

        self.assertIn(
            "delete from {{ this }} as target",
            hook,
        )
        self.assertIn(
            "incremental_scope=true",
            hook,
        )
        self.assertIn(
            "target.{{ year_column }}",
            hook,
        )
        self.assertIn(
            "= current_snapshot.election_year",
            hook,
        )
        self.assertIn(
            "target.{{ type_column }}",
            hook,
        )
        self.assertIn(
            "= current_snapshot.election_type",
            hook,
        )
        self.assertIn(
            "stored.{{ snapshot_column }} is null",
            hook,
        )
        self.assertIn(
            "<> current_snapshot.source_snapshot_id",
            hook,
        )

    def test_pre_hook_decides_replacement_at_sql_runtime(self):
        start = self.snapshot_macro.index(
            "{% macro source_aware_partition_replace_pre_hook("
        )
        end = self.snapshot_macro.index(
            "{% macro source_snapshot_changed(",
            start,
        )
        hook = self.snapshot_macro[start:end]

        self.assertIn(
            "delete from {{ this }} as target",
            hook,
        )
        self.assertIn(
            "and exists (",
            hook,
        )
        self.assertIn(
            "stored.{{ snapshot_column }} is null",
            hook,
        )
        self.assertNotIn(
            "source_snapshot_changed(",
            hook,
        )

    def test_snapshot_metadata_stops_at_source_aware_boundary(self):
        root = Path(__file__).resolve().parents[1]

        silver_tally = (
            root
            / "tse_dbt/models/silver/tally/"
              "silver_tally_munzona.sql"
        ).read_text(encoding="utf-8")

        fact_electorate = (
            root
            / "tse_dbt/models/gold/facts/"
              "fact_electorate_municipality.sql"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "source_snapshot_id",
            silver_tally,
        )
        self.assertIn(
            "exclude (",
            silver_tally,
        )

        self.assertNotIn(
            "select *",
            fact_electorate,
        )
        self.assertNotIn(
            "source_snapshot_id",
            fact_electorate,
        )

    def test_makefile_is_dbt_core_compatible(self):
        self.assertIn(
            "DBT_PROJECT_ARGS :=",
            self.makefile,
        )
        self.assertIn(
            "$(DBT) compile $(DBT_PROJECT_ARGS)",
            self.makefile,
        )
        self.assertIn(
            "$(UV) sync --inexact",
            self.makefile,
        )


if __name__ == "__main__":
    unittest.main()
