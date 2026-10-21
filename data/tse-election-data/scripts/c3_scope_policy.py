"""Fail-closed C3 dbt scope binding against a validated reference manifest.

This policy verifies scope/identity only. It does NOT authenticate a reference
or replace the existing source-byte and Bronze parity preflights.
"""
from __future__ import annotations

import json
from pathlib import Path


def _option_value(args: list[str], flag: str) -> str:
    hits = []
    for i, arg in enumerate(args):
        if arg == flag:
            if i + 1 >= len(args):
                raise ValueError(f"C3 {flag} requires a value")
            hits.append(args[i + 1])
        elif arg.startswith(flag + "="):
            hits.append(arg[len(flag) + 1:])
    if len(hits) != 1 or not hits[0]:
        raise ValueError(f"C3 requires exactly one {flag}")
    return hits[0]


def _integers(value: object, name: str) -> set[int]:
    if not isinstance(value, list) or not value or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in value
    ):
        raise ValueError(f"C3 invalid {name}")
    if len(value) != len(set(value)):
        raise ValueError(f"C3 duplicated {name}")
    return set(value)


def _types(value: object, name: str) -> set[str]:
    if not isinstance(value, list) or not value or any(
        x not in ("general", "municipal") for x in value
    ):
        raise ValueError(f"C3 invalid {name}")
    if len(value) != len(set(value)):
        raise ValueError(f"C3 duplicated {name}")
    return set(value)


def validate_c3_scope(dbt_args: list[str], *, evidence_path: Path,
                      raw_root: Path, calendar_csv: Path) -> set[str]:
    """Bind the effective incremental cycle scope to exactly one raw root.

    Reject any unsupported scope-changing option rather than trusting inferred
    defaults. The canonical calendar determines valid (year,type) pairs.
    """
    import csv

    raw = _option_value(dbt_args, "--vars")
    try:
        variables = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("C3 --vars must be a JSON object") from exc
    if not isinstance(variables, dict):
        raise ValueError("C3 --vars must be a JSON object")
    required = ("tse_raw_root", "election_years", "election_types",
                "incremental_years", "incremental_election_types")
    if any(k not in variables for k in required):
        raise ValueError("C3 requires explicit raw root and all scope variables")
    root_value = variables["tse_raw_root"]
    if not isinstance(root_value, str) or not root_value or not Path(root_value).is_absolute():
        raise ValueError("C3 tse_raw_root must be an absolute path")
    if Path(root_value).resolve() != Path(raw_root).resolve():
        raise ValueError("C3 dbt tse_raw_root differs from certified raw root")

    years = _integers(variables["election_years"], "election_years")
    types = _types(variables["election_types"], "election_types")
    update_years = _integers(variables["incremental_years"], "incremental_years")
    update_types = _types(variables["incremental_election_types"], "incremental_election_types")
    if not update_years <= years or not update_types <= types:
        raise ValueError("C3 incremental scope not contained in analysis scope")

    calendar = {}
    with Path(calendar_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            year = int(row["election_year"])
            kind = row["election_type"]
            if year in calendar and calendar[year] != kind:
                raise ValueError("C3 conflicting election calendar")
            calendar[year] = kind
    if any(y not in calendar for y in years):
        raise ValueError("C3 year not in election calendar")
    effective = {f"{year}:{calendar[year]}" for year in update_years
                 if calendar[year] in update_types}
    if not effective or len(effective) != len(update_years):
        raise ValueError("C3 scope omits or mismatches an update year")

    try:
        manifest = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        certified = manifest["cycles"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("C3 evidence cycle coverage unavailable") from exc
    if not isinstance(certified, dict) or not effective <= set(certified):
        raise ValueError("C3 publication contains uncertified election cycles")
    return effective
