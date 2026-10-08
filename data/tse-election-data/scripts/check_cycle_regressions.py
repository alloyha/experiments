#!/usr/bin/env python3
"""Warehouse-wide regression gate for historically validated election cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def scalar(con, sql: str, params: list[object]) -> int:
    value = con.execute(sql, params).fetchone()[0]
    return int(value or 0)


def party_gap_counts(con, year: int, election_type: str) -> dict[str, int]:
    rows = con.execute(
        """
        select gap_reason, count(*)
        from party_tally_coverage_gaps
        where election_year = ? and election_type = ?
        group by 1
        order by 1
        """,
        [year, election_type],
    ).fetchall()
    return {str(reason): int(count) for reason, count in rows}


def check_cycle(con, cycle: dict) -> list[str]:
    year = int(cycle["year"])
    election_type = str(cycle["election_type"])
    label = f"{year}/{election_type}"
    failures: list[str] = []

    expected_rows = cycle.get("candidate_fact_rows")
    if expected_rows is not None:
        actual = scalar(
            con,
            """
            select count(*)
            from fact_candidate_votes
            where election_year = ? and election_type = ?
            """,
            [year, election_type],
        )
        if actual != int(expected_rows):
            failures.append(
                f"{label}: candidate_fact_rows expected={expected_rows} actual={actual}"
            )

    expected_dim = cycle.get("dim_candidate_rows")
    if expected_dim is not None:
        actual = scalar(
            con,
            """
            select count(*)
            from dim_candidate
            where election_year = ? and election_type = ?
            """,
            [year, election_type],
        )
        if actual != int(expected_dim):
            failures.append(
                f"{label}: dim_candidate_rows expected={expected_dim} actual={actual}"
            )

    expected_candidate_gap_count = cycle.get("candidate_gap_count")
    if expected_candidate_gap_count is not None:
        actual = scalar(
            con,
            """
            select count(*)
            from candidate_tally_coverage_gaps
            where election_year = ? and election_type = ?
            """,
            [year, election_type],
        )
        if actual != int(expected_candidate_gap_count):
            failures.append(
                f"{label}: candidate_gap_count "
                f"expected={expected_candidate_gap_count} actual={actual}"
            )

    expected_candidate_gap_nominal = cycle.get("candidate_gap_nominal_votes")
    if expected_candidate_gap_nominal is not None:
        actual = scalar(
            con,
            """
            select coalesce(sum(nominal_valid_votes), 0)
            from candidate_tally_coverage_gaps
            where election_year = ? and election_type = ?
            """,
            [year, election_type],
        )
        if actual != int(expected_candidate_gap_nominal):
            failures.append(
                f"{label}: candidate_gap_nominal_votes "
                f"expected={expected_candidate_gap_nominal} actual={actual}"
            )

    expected_party_gaps = cycle.get("party_gap_counts")
    if expected_party_gaps is not None:
        actual = party_gap_counts(con, year, election_type)
        expected = {str(k): int(v) for k, v in expected_party_gaps.items()}
        if actual != expected:
            failures.append(
                f"{label}: party_gap_counts expected={expected} actual={actual}"
            )

    expected_candidate_mismatches = cycle.get("candidate_reconciliation_mismatches")
    if expected_candidate_mismatches is not None:
        actual = scalar(
            con,
            """
            select count(*)
            from candidate_tally_reconciliation
            where election_year = ?
              and election_type = ?
              and nominal_valid_delta <> 0
            """,
            [year, election_type],
        )
        if actual != int(expected_candidate_mismatches):
            failures.append(
                f"{label}: candidate reconciliation mismatches "
                f"expected={expected_candidate_mismatches} actual={actual}"
            )

    expected_party_mismatches = cycle.get("party_reconciliation_mismatches")
    if expected_party_mismatches is not None:
        actual = scalar(
            con,
            """
            select count(*)
            from party_tally_reconciliation
            where election_year = ?
              and election_type = ?
              and (
                  nominal_valid_delta <> 0
                  or total_legend_valid_delta <> 0
                  or total_valid_delta <> 0
              )
            """,
            [year, election_type],
        )
        if actual != int(expected_party_mismatches):
            failures.append(
                f"{label}: party reconciliation mismatches "
                f"expected={expected_party_mismatches} actual={actual}"
            )

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("data/warehouse/tse_analytics.duckdb"),
    )
    ap.add_argument(
        "--contracts",
        type=Path,
        default=Path("contracts/election_cycles.json"),
    )
    ap.add_argument(
        "--exclude-provisional",
        action="store_true",
        help="Skip cycles marked provisional.",
    )
    args = ap.parse_args()

    payload = json.loads(args.contracts.read_text(encoding="utf-8"))
    cycles = payload["cycles"]
    if args.exclude_provisional:
        cycles = [c for c in cycles if c.get("status") != "provisional"]

    con = duckdb.connect(str(args.db), read_only=True)
    failures: list[str] = []
    try:
        for cycle in cycles:
            cycle_failures = check_cycle(con, cycle)
            label = f'{cycle["year"]}/{cycle["election_type"]}'
            status = cycle.get("status", "unknown")
            if cycle_failures:
                print(f"{label:16} {status:11} FAIL")
                failures.extend(cycle_failures)
            else:
                print(f"{label:16} {status:11} PASS")
    finally:
        con.close()

    if failures:
        print("\nRegression failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nCycle regressions: PASS ({len(cycles)} cycle(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
