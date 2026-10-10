from __future__ import annotations

import argparse

from tse_ingest import ELECTION_CALENDAR

ALLOWED_TYPES = {"general", "municipal"}


def tokens(value: str) -> set[str]:
    return {item for item in value.split() if item}


def validate_scope(
    years: set[str],
    election_types: set[str],
    incremental_years: set[str],
    incremental_types: set[str],
) -> list[str]:
    errors: list[str] = []

    if not years:
        errors.append("YEARS must not be empty")
    if not election_types:
        errors.append("ELECTION_TYPES must not be empty")
    if not incremental_years:
        errors.append("INCREMENTAL_YEARS must not be empty")
    if not incremental_types:
        errors.append(
            "INCREMENTAL_ELECTION_TYPES must not be empty"
        )

    extra_years = incremental_years - years
    if extra_years:
        errors.append(
            "incremental years outside YEARS: "
            + ", ".join(sorted(extra_years))
        )

    extra_types = incremental_types - election_types
    if extra_types:
        errors.append(
            "incremental election types outside ELECTION_TYPES: "
            + ", ".join(sorted(extra_types))
        )

    unknown = (
        election_types | incremental_types
    ) - ALLOWED_TYPES
    if unknown:
        errors.append(
            "unknown election types: "
            + ", ".join(sorted(unknown))
        )

    try:
        selected_years = {int(year) for year in years}
        selected_incremental_years = {
            int(year) for year in incremental_years
        }
    except ValueError:
        errors.append("election years must be integers")
        return errors

    unknown_years = {
        year
        for year in selected_years
        if year not in ELECTION_CALENDAR
    }

    if unknown_years:
        errors.append(
            "unknown election years: "
            + ", ".join(
                str(year)
                for year in sorted(unknown_years)
            )
        )
        return errors

    expected_types = {
        ELECTION_CALENDAR[year]
        for year in selected_years
    }

    if election_types != expected_types:
        errors.append(
            "ELECTION_TYPES does not match YEARS calendar: "
            f"expected={sorted(expected_types)}, "
            f"actual={sorted(election_types)}"
        )

    expected_incremental_types = {
        ELECTION_CALENDAR[year]
        for year in selected_incremental_years
    }

    if incremental_types != expected_incremental_types:
        errors.append(
            "INCREMENTAL_ELECTION_TYPES does not match "
            "INCREMENTAL_YEARS calendar: "
            f"expected={sorted(expected_incremental_types)}, "
            f"actual={sorted(incremental_types)}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", required=True)
    parser.add_argument("--election-types", required=True)
    parser.add_argument("--incremental-years", required=True)
    parser.add_argument(
        "--incremental-election-types",
        required=True,
    )
    args = parser.parse_args()

    errors = validate_scope(
        tokens(args.years),
        tokens(args.election_types),
        tokens(args.incremental_years),
        tokens(args.incremental_election_types),
    )

    if errors:
        for error in errors:
            print(f"Scope contract: FAIL: {error}")
        return 1

    print("Scope contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
