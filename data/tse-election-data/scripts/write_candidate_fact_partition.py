#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="data/warehouse/tse_analytics.duckdb",
    )
    parser.add_argument(
        "--root",
        default="data/warehouse/fact_candidate_votes",
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--election-type", required=True)
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    root = Path(args.root).resolve()

    partition = (
        root
        / f"election_type={args.election_type}"
        / f"year={args.year}"
    )
    partition.mkdir(parents=True, exist_ok=True)

    final_path = partition / "data.parquet"
    temp_path = partition / "data.parquet.tmp"

    if temp_path.exists():
        temp_path.unlink()

    con = duckdb.connect(str(db_path))

    try:
        # Critical guard: the upstream views must actually be scoped
        # to the partition we intend to write.
        cycles = con.execute("""
            select distinct
                election_year,
                election_type
            from main.int_candidate_votes
            order by 1,2
        """).fetchall()

        expected = [(args.year, args.election_type)]

        if cycles != expected:
            raise RuntimeError(
                "int_candidate_votes is not scoped to the requested "
                f"partition. expected={expected}, actual={cycles}"
            )

        row_count = con.execute("""
            select count(*)
            from main.int_candidate_votes
        """).fetchone()[0]

        if row_count == 0:
            raise RuntimeError(
                "Refusing to replace candidate fact partition with "
                "an empty result."
            )

        escaped = str(temp_path).replace("'", "''")

        con.execute(f"""
            copy (
                select *
                from main.int_candidate_votes
            )
            to '{escaped}'
            (
                format parquet,
                compression zstd
            )
        """)

    finally:
        con.close()

    if not temp_path.exists():
        raise RuntimeError("Temporary parquet was not created")

    # Atomic replacement on the same filesystem.
    os.replace(temp_path, final_path)

    check = duckdb.connect()
    try:
        written_rows = check.execute(
            "select count(*) from read_parquet(?)",
            [str(final_path)],
        ).fetchone()[0]
    finally:
        check.close()

    if written_rows != row_count:
        raise RuntimeError(
            f"Row-count mismatch: source={row_count}, "
            f"parquet={written_rows}"
        )

    print(
        "candidate fact partition replaced:",
        f"{args.year}/{args.election_type}",
    )
    print("rows:", written_rows)
    print("path:", final_path)


if __name__ == "__main__":
    main()
