#!/usr/bin/env python3
"""
Graph integrity and semantic consistency validation for metric_catalog.duckdb.

Usage:
    python validate.py <catalog.duckdb>

Exits with code 1 if any check fails.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb


def _check_dependency_cycles(con) -> list[str]:
    """Detect cycles in metric_dependency via iterative DFS."""
    edges = con.execute(
        "SELECT metric_id, depends_on_metric_id FROM metric_dependency"
    ).fetchall()
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    in_path: set[str] = set()
    visited: set[str] = set()
    issues: list[str] = []

    def dfs(node: str, path: list[str]) -> None:
        if node in in_path:
            cycle = path[path.index(node):] + [node]
            issues.append(f"Cycle: {' -> '.join(cycle)}")
            return
        if node in visited:
            return
        in_path.add(node)
        path.append(node)
        for neighbor in adj[node]:
            dfs(neighbor, path)
        path.pop()
        in_path.discard(node)
        visited.add(node)

    for node in list(adj):
        if node not in visited:
            dfs(node, [])
    return issues


def _check_dangling_superseded_by(con) -> list[str]:
    return [
        f"superseded_by references unknown metric '{r[0]}' (from '{r[1]}')"
        for r in con.execute("""
            SELECT m.superseded_by, m.metric_id
            FROM metric_definition m
            WHERE m.superseded_by IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM metric_definition t WHERE t.metric_id = m.superseded_by
              )
        """).fetchall()
    ]


def _check_formula_dep_consistency(con) -> list[str]:
    """Warn when a declared dep's key token does not appear in the expression."""
    token_re = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    issues: list[str] = []
    rows = con.execute("""
        SELECT md.metric_id, md.depends_on_metric_id, i.expression
        FROM metric_dependency md
        JOIN metric_implementation i
          ON i.metric_id = md.metric_id AND i.is_current = true
        WHERE md.origin = 'declared' AND i.expression IS NOT NULL
    """).fetchall()
    for metric_id, dep_id, expr in rows:
        dep_key = dep_id.split(".", 1)[1]
        tokens = {t.lower() for t in token_re.findall(expr)}
        if dep_key.lower() not in tokens and dep_key.upper() not in token_re.findall(expr):
            issues.append(
                f"Dep declared but not in formula: '{metric_id}' declares '{dep_id}' "
                f"but token '{dep_key}' absent in: {expr[:80]}"
            )
    return issues


def _check_duplicate_aliases(con) -> list[str]:
    return [
        f"Alias '{r[0]}' shared by multiple metrics: {r[1]}"
        for r in con.execute("""
            SELECT alias, list(metric_id ORDER BY metric_id)
            FROM metric_alias GROUP BY alias HAVING count(*) > 1
        """).fetchall()
    ]


def _check_missing_implementations(con) -> list[str]:
    return [
        f"No implementation for metric '{r[0]}'"
        for r in con.execute("""
            SELECT m.metric_id FROM metric_definition m
            WHERE NOT EXISTS (
                SELECT 1 FROM metric_implementation i WHERE i.metric_id = m.metric_id
            )
        """).fetchall()
    ]


def _check_orphan_datasets(con) -> list[str]:
    return [
        f"Dataset '{r[0]}' not referenced by any impl_column or impl_join"
        for r in con.execute("""
            SELECT ds.dataset_id FROM dataset ds
            WHERE NOT EXISTS (SELECT 1 FROM impl_column c WHERE c.dataset_id = ds.dataset_id)
              AND NOT EXISTS (
                SELECT 1 FROM impl_join j
                WHERE j.left_dataset_id = ds.dataset_id
                   OR j.right_dataset_id = ds.dataset_id
              )
        """).fetchall()
    ]


def _check_deprecated_without_supersession(con) -> list[str]:
    return [
        f"'{r[0]}' is deprecated but has no superseded_by"
        for r in con.execute("""
            SELECT metric_id FROM metric_definition
            WHERE status = 'deprecated' AND superseded_by IS NULL
        """).fetchall()
    ]


def _check_entity_coverage(con) -> list[str]:
    """Warn on active metrics with no entity_id (unresolved grain)."""
    rows = con.execute("""
        SELECT metric_id FROM metric_definition
        WHERE entity_id IS NULL AND status != 'deprecated'
    """).fetchall()
    if not rows:
        return []
    ids = ", ".join(r[0] for r in rows[:5])
    suffix = f" (+ {len(rows) - 5} more)" if len(rows) > 5 else ""
    return [f"{len(rows)} active metrics have no entity_id: {ids}{suffix}"]


def _check_inferred_lineage_coverage(con) -> list[str]:
    """Report ratio of declared vs inferred lineage — informational."""
    row = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE origin = 'declared')  AS declared,
            COUNT(*) FILTER (WHERE origin = 'inferred')  AS inferred,
            COUNT(*) FILTER (WHERE origin = 'generated') AS generated,
            COUNT(*) AS total
        FROM impl_column
    """).fetchone()
    if not row or row[3] == 0:
        return []
    declared, inferred, generated, total = row
    pct_declared = round(100 * declared / total, 1)
    if pct_declared < 5:
        return [
            f"Only {pct_declared}% of column lineage is declared; "
            f"{inferred} inferred (regex), {generated} generated out of {total} total. "
            "Consider adding explicit lineage to high-value metrics."
        ]
    return []


CHECKS = [
    ("dependency_cycles",              _check_dependency_cycles),
    ("dangling_superseded_by",         _check_dangling_superseded_by),
    ("formula_dep_consistency",        _check_formula_dep_consistency),
    ("duplicate_aliases",              _check_duplicate_aliases),
    ("missing_implementations",        _check_missing_implementations),
    ("orphan_datasets",                _check_orphan_datasets),
    ("deprecated_without_supersession", _check_deprecated_without_supersession),
    ("entity_coverage",                _check_entity_coverage),
    ("inferred_lineage_coverage",      _check_inferred_lineage_coverage),
]


def validate(db_path: Path) -> int:
    """Run all checks. Returns total number of issues found."""
    con = duckdb.connect(str(db_path), read_only=True)
    total = 0
    try:
        for name, fn in CHECKS:
            issues = fn(con)
            status = "FAIL" if issues else "pass"
            print(f"  [{status:4s}] {name}  ({len(issues)} issues)")
            for msg in issues[:5]:
                print(f"         {msg}")
            if len(issues) > 5:
                print(f"         ... and {len(issues) - 5} more")
            total += len(issues)
    finally:
        con.close()
    return total


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Validate metric_catalog.duckdb integrity.")
    p.add_argument("db", help="Path to metric_catalog.duckdb")
    args = p.parse_args()

    print(f"Validating {args.db}\n")
    total = validate(Path(args.db))
    print(f"\n{'─' * 50}")
    if total:
        print(f"  {total} issue(s) found")
        raise SystemExit(1)
    print("  All checks passed")


if __name__ == "__main__":
    main()
