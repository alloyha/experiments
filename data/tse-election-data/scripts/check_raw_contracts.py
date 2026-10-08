#!/usr/bin/env python3
"""Validate raw-lake control-plane contracts before dbt reads active objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing current-object index: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def object_years(con: duckdb.DuckDBPyConnection, path: Path) -> set[int | None]:
    suffix = path.suffix.lower()
    src = sql_literal(str(path.resolve()))

    if suffix == ".parquet":
        relation = f"read_parquet({src})"
    elif suffix in {".csv", ".txt"}:
        relation = (
            f"read_csv({src}, delim=';', quote='\"', header=true, "
            "all_varchar=true, encoding='latin-1', sample_size=20480)"
        )
    else:
        raise ValueError(f"unsupported active tabular object: {path}")

    rows = con.execute(
        f"""
        select distinct try_cast("ANO_ELEICAO" as integer)
        from {relation}
        order by 1
        """
    ).fetchall()
    return {row[0] for row in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/tse"))
    args = ap.parse_args()

    root = args.root.resolve()
    current_path = root / "_metadata" / "current_objects.jsonl"
    rows = load_jsonl(current_path)

    missing: list[dict] = []
    candidate_scope: list[dict] = []

    for row in rows:
        for field in ("source_object", "object"):
            value = row.get(field)
            if not value:
                missing.append({
                    "resource_id": row.get("resource_id"),
                    "field": field,
                    "path": value,
                    "reason": "empty reference",
                })
                continue
            path = root / value
            if not path.is_file():
                missing.append({
                    "resource_id": row.get("resource_id"),
                    "field": field,
                    "path": value,
                    "reason": "missing file",
                })

    con = duckdb.connect()
    try:
        for row in rows:
            if row.get("domain") != "candidate":
                continue
            value = row.get("object")
            if not value:
                continue
            path = root / value
            if not path.is_file():
                continue

            expected = int(row["year"])
            try:
                years = object_years(con, path)
            except Exception as exc:
                candidate_scope.append({
                    "resource_id": row.get("resource_id"),
                    "object": value,
                    "expected_year": expected,
                    "error": str(exc),
                })
                continue

            if years != {expected}:
                candidate_scope.append({
                    "resource_id": row.get("resource_id"),
                    "object": value,
                    "expected_year": expected,
                    "actual_years": sorted(
                        years, key=lambda x: (-1 if x is None else x)
                    ),
                })
    finally:
        con.close()

    if missing:
        print("current object references .... FAIL")
        for failure in missing:
            print("  ", json.dumps(failure, ensure_ascii=False))
    else:
        print(f"current object references .... PASS ({len(rows)} active object(s))")

    if candidate_scope:
        print("candidate cycle scope ......... FAIL")
        for failure in candidate_scope:
            print("  ", json.dumps(failure, ensure_ascii=False))
    else:
        candidate_count = sum(row.get("domain") == "candidate" for row in rows)
        print(f"candidate cycle scope ......... PASS ({candidate_count} active object(s))")

    return 1 if missing or candidate_scope else 0


if __name__ == "__main__":
    raise SystemExit(main())
