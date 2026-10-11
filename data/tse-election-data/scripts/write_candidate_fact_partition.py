#!/usr/bin/env python3
"""Publish the dbt physical candidate table as one validated Parquet partition.

Only the authoritative ``int_candidate_fact_partition`` table is an allowed
source. Verify the staged file before atomically replacing any live partition.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import duckdb

SOURCE_TABLE = "main.int_candidate_fact_partition"


def validate_staged_parquet(con: duckdb.DuckDBPyConnection, staged: Path, expected_rows: int) -> None:
    """Reject incomplete exports before they become visible as live partitions."""
    if not staged.is_file():
        raise RuntimeError("Temporary parquet was not created")

    rows = con.execute(
        "select count(*) from read_parquet(?, hive_partitioning = false)", [str(staged)]
    ).fetchone()[0]
    if rows != expected_rows:
        raise RuntimeError(
            f"Staged parquet row-count mismatch: source={expected_rows}, parquet={rows}"
        )

    # The published data must retain the dbt model's column names and types.
    source_schema = con.execute(f"select * from {SOURCE_TABLE} limit 0").description
    staged_schema = con.execute(
        "select * from read_parquet(?, hive_partitioning = false) limit 0", [str(staged)]
    ).description
    source_fields = [(col[0], str(col[1])) for col in source_schema]
    staged_fields = [(col[0], str(col[1])) for col in staged_schema]
    if source_fields != staged_fields:
        raise RuntimeError(
            f"Staged parquet schema mismatch: source={source_fields}, parquet={staged_fields}"
        )


def publish_partition(db_path: Path, root: Path, year: int, election_type: str) -> Path:
    if year < 1 or election_type not in {"general", "municipal"}:
        raise ValueError("A positive year and general/municipal election type are required")
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB warehouse not found: {db_path}")

    partition = root / f"election_type={election_type}" / f"year={year}"
    partition.mkdir(parents=True, exist_ok=True)
    final_path = partition / "data.parquet"
    # Unique staging names avoid reusing a crashed or concurrent export's bytes.
    temp_path = partition / f"data.parquet.{uuid.uuid4().hex}.tmp"

    try:
        with duckdb.connect(str(db_path)) as con:
            try:
                cycles = con.execute(f"""
                    select distinct election_year, election_type
                    from {SOURCE_TABLE}
                    order by 1, 2
                """).fetchall()
            except duckdb.CatalogException as exc:
                raise RuntimeError(
                    f"Missing dbt physical source table {SOURCE_TABLE}; "
                    "run the enabled int_candidate_fact_partition model first"
                ) from exc

            row_count = con.execute(
                f"select count(*) from {SOURCE_TABLE}"
            ).fetchone()[0]
            if not row_count:
                raise RuntimeError("Refusing to publish an empty candidate fact partition")

            expected = [(year, election_type)]
            if cycles != expected:
                raise RuntimeError(
                    f"{SOURCE_TABLE} is not scoped to the requested partition: "
                    f"expected={expected}, actual={cycles}"
                )

            escaped = str(temp_path).replace("'", "''")
            con.execute(f"""
                copy (select * from {SOURCE_TABLE})
                to '{escaped}' (format parquet, compression zstd)
            """)
            validate_staged_parquet(con, temp_path, row_count)

            # Same-directory replacement is atomic for readers opening the final
            # path, provided the filesystem supports atomic rename semantics.
            os.replace(temp_path, final_path)

        print(f"candidate fact partition replaced: {year}/{election_type}")
        print(f"rows: {row_count}")
        print(f"path: {final_path}")
        return final_path
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/warehouse/tse_analytics.duckdb")
    parser.add_argument("--root", default="data/warehouse/fact_candidate_votes")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--election-type", choices=["general", "municipal"], required=True
    )
    args = parser.parse_args()
    publish_partition(
        Path(args.db).resolve(), Path(args.root).resolve(),
        args.year, args.election_type,
    )


if __name__ == "__main__":
    main()
