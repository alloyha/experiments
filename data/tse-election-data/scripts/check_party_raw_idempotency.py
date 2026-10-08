#!/usr/bin/env python3

import argparse
import subprocess
import duckdb


def count_rows(db_path, year, election_type):
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(
            """
            select count(*)
            from main.stg_party_votes_raw
            where election_year = ?
              and election_type = ?
            """,
            [year, election_type],
        ).fetchone()[0]
    finally:
        con.close()


def run_dbt(root, db_path, year, election_type):
    variables = (
        "{"
        f'"tse_raw_root": "{root}", '
        f'"election_years": [{year}], '
        f'"election_types": ["{election_type}"], '
        f'"incremental_years": [{year}], '
        f'"incremental_election_types": ["{election_type}"]'
        "}"
    )

    subprocess.run(
        [
            "env",
            "-u",
            "VIRTUAL_ENV",
            "uv",
            "run",
            "--python",
            ".venv/bin/python",
            "dbt",
            "run",
            "--project-dir",
            "tse_dbt",
            "--profiles-dir",
            "tse_dbt",
            "--select",
            "stg_party_votes_raw",
            "--vars",
            variables,
        ],
        check=True,
        env={
            **__import__("os").environ,
            "TSE_DUCKDB_PATH": db_path,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tse")
    parser.add_argument(
        "--db",
        default="data/warehouse/tse_analytics.duckdb",
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--election-type", required=True)
    args = parser.parse_args()

    before = count_rows(
        args.db,
        args.year,
        args.election_type,
    )

    run_dbt(
        args.root,
        args.db,
        args.year,
        args.election_type,
    )

    after = count_rows(
        args.db,
        args.year,
        args.election_type,
    )

    print(f"before={before}")
    print(f"after={after}")

    if before != after:
        raise SystemExit(
            f"FAIL: partition grew/shrank: {before} -> {after}"
        )

    print("PASS: selected partition is idempotent")


if __name__ == "__main__":
    main()
