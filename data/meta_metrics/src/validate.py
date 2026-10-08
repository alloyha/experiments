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


def _check_entity_relation_cycles(con) -> list[str]:
    """Detect cycles in rollup-safe entity_relation edges (would cause infinite rollup loops)."""
    from collections import defaultdict, deque
    edges = [
        (r[0], r[1])
        for r in con.execute(
            "SELECT from_entity_id, to_entity_id FROM entity_relation WHERE rollup_safe = true"
        ).fetchall()
    ]
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    in_path: set[str] = set()
    visited: set[str] = set()
    issues: list[str] = []

    def dfs(node: str, path: list[str]) -> None:
        if node in in_path:
            cycle = path[path.index(node):] + [node]
            issues.append("Rollup cycle: " + " -> ".join(cycle))
            return
        if node in visited:
            return
        in_path.add(node)
        path.append(node)
        for nxt in adj[node]:
            dfs(nxt, path)
        path.pop()
        in_path.discard(node)
        visited.add(node)

    for node in list(adj):
        if node not in visited:
            dfs(node, [])
    return issues


def _check_cube_orphan_metrics(con) -> list[str]:
    """Active metrics not assigned to any cube."""
    rows = con.execute("""
        SELECT m.metric_id FROM metric_definition m
        WHERE m.status != 'deprecated'
          AND NOT EXISTS (
            SELECT 1 FROM cube_metric cm WHERE cm.metric_id = m.metric_id
          )
    """).fetchall()
    if not rows:
        return []
    ids = ", ".join(r[0] for r in rows[:5])
    suffix = f" (+ {len(rows) - 5} more)" if len(rows) > 5 else ""
    return [f"{len(rows)} active metrics not in any cube: {ids}{suffix}"]


def _check_cube_missing_dep_closure(con) -> list[str]:
    """Cubes containing a derived metric but missing ≥1 dep that isn't in any cube at all."""
    issues: list[str] = []
    rows = con.execute("""
        SELECT cm.cube_id, cm.metric_id, md.depends_on_metric_id
        FROM cube_metric cm
        JOIN metric_dependency md ON md.metric_id = cm.metric_id
        WHERE NOT EXISTS (
            SELECT 1 FROM cube_metric cm2 WHERE cm2.metric_id = md.depends_on_metric_id
        )
    """).fetchall()
    for cube_id, metric_id, dep_id in rows:
        issues.append(
            f"Cube '{cube_id}' contains '{metric_id}' but dep '{dep_id}' is in no cube"
        )
    return issues


def _check_entity_missing_identifier(con) -> list[str]:
    """Every non-virtual business entity must have exactly one canonical
    identifier attribute — this is the semantic replacement for ENTITY_PK."""
    return [
        f"Entity '{r[0]}' has no declared identifier attribute (expected "
        f"semantic_attribute '{r[0]}.identifier')"
        for r in con.execute("""
            SELECT e.entity_id FROM entity e
            WHERE NOT EXISTS (
                SELECT 1 FROM semantic_attribute sa
                WHERE sa.entity_id = e.entity_id AND sa.semantic_type = 'identifier'
            )
        """).fetchall()
    ]


def _check_entity_multiple_identifiers(con) -> list[str]:
    return [
        f"Entity '{r[0]}' has {r[1]} identifier attributes (expected exactly 1): {r[2]}"
        for r in con.execute("""
            SELECT entity_id, count(*), list(attribute_id ORDER BY attribute_id)
            FROM semantic_attribute
            WHERE semantic_type = 'identifier'
            GROUP BY entity_id HAVING count(*) > 1
        """).fetchall()
    ]


def _check_attribute_missing_binding(con) -> list[str]:
    """Informational: semantic attributes with zero bindings at all (not even
    a candidate) — nothing physical has been proposed for them yet."""
    rows = con.execute("""
        SELECT sa.attribute_id FROM semantic_attribute sa
        WHERE NOT EXISTS (
            SELECT 1 FROM attribute_binding ab WHERE ab.attribute_id = sa.attribute_id
        )
    """).fetchall()
    if not rows:
        return []
    ids = ", ".join(r[0] for r in rows[:5])
    suffix = f" (+ {len(rows) - 5} more)" if len(rows) > 5 else ""
    return [f"{len(rows)} semantic attributes have no binding at all: {ids}{suffix}"]


def _check_resolved_binding_ambiguity(con) -> list[str]:
    """Two or more resolution_state='resolved' bindings for the same
    attribute with no way to disambiguate (same dataset_id, or both NULL) —
    resolve_attribute_binding would raise AmbiguousBindingError even after
    dataset_id filtering."""
    return [
        f"Attribute '{r[0]}' has {r[1]} resolved bindings sharing dataset_id={r[2]!r}: {r[3]}"
        for r in con.execute("""
            SELECT attribute_id, count(*), dataset_id, list(binding_id ORDER BY binding_id)
            FROM attribute_binding
            WHERE resolution_state = 'resolved'
            GROUP BY attribute_id, dataset_id HAVING count(*) > 1
        """).fetchall()
    ]


def _check_binding_unknown_attribute(con) -> list[str]:
    return [
        f"Binding '{r[0]}' references unknown attribute_id '{r[1]}'"
        for r in con.execute("""
            SELECT ab.binding_id, ab.attribute_id FROM attribute_binding ab
            WHERE NOT EXISTS (
                SELECT 1 FROM semantic_attribute sa WHERE sa.attribute_id = ab.attribute_id
            )
        """).fetchall()
    ]


def _check_binding_unknown_dataset(con) -> list[str]:
    return [
        f"Binding '{r[0]}' references unknown dataset_id '{r[1]}'"
        for r in con.execute("""
            SELECT ab.binding_id, ab.dataset_id FROM attribute_binding ab
            WHERE ab.dataset_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM dataset d WHERE d.dataset_id = ab.dataset_id)
        """).fetchall()
    ]


def _check_entity_reference_missing_target_identifier(con) -> list[str]:
    """An entity_reference attribute must point at an entity that itself has
    a declared identifier — otherwise the reference dangles semantically."""
    return [
        f"Attribute '{r[0]}' references entity '{r[1]}' which has no identifier attribute"
        for r in con.execute("""
            SELECT sa.attribute_id, sa.references_entity_id
            FROM semantic_attribute sa
            WHERE sa.semantic_type = 'entity_reference'
              AND sa.references_entity_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM semantic_attribute t
                WHERE t.entity_id = sa.references_entity_id AND t.semantic_type = 'identifier'
              )
        """).fetchall()
    ]


def _check_entity_relation_unresolvable(con) -> list[str]:
    """Informational: structural relations not yet decomposed into
    from_attribute_id/to_attribute_id — resolve_entity_relation() will raise
    UnresolvedBindingError for these until they are.

    Relations with no join_expression at all (typically many_to_many bridge
    relations, which don't have a simple two-column equality to decompose)
    are excluded: there was never information to derive attributes from, so
    flagging them here would be noise, not an actionable gap.
    """
    rows = con.execute("""
        SELECT relation_id FROM entity_relation
        WHERE (from_attribute_id IS NULL OR to_attribute_id IS NULL)
          AND join_expression IS NOT NULL
    """).fetchall()
    if not rows:
        return []
    ids = ", ".join(r[0] for r in rows[:5])
    suffix = f" (+ {len(rows) - 5} more)" if len(rows) > 5 else ""
    return [f"{len(rows)} entity_relations have a join_expression but no decomposed "
            f"attribute references: {ids}{suffix}"]


def _check_dimension_missing_attribute(con) -> list[str]:
    """Informational: dimensions still relying on default_expr (legacy) with
    no attribute_id link."""
    rows = con.execute(
        "SELECT dimension_id FROM dimension WHERE attribute_id IS NULL"
    ).fetchall()
    if not rows:
        return []
    ids = ", ".join(r[0] for r in rows[:5])
    suffix = f" (+ {len(rows) - 5} more)" if len(rows) > 5 else ""
    return [f"{len(rows)} dimensions have no semantic attribute_id (still default_expr-only): "
            f"{ids}{suffix}"]


def _check_metric_attribute_unresolved(con) -> list[str]:
    """Informational: metrics with physical lineage (impl_column) but no
    corresponding semantic lineage (metric_attribute) yet — i.e. physical
    lineage that hasn't been promoted to the semantic layer."""
    rows = con.execute("""
        SELECT DISTINCT mi.metric_id
        FROM impl_column ic
        JOIN metric_implementation mi ON mi.impl_id = ic.impl_id AND mi.is_current = true
        WHERE NOT EXISTS (
            SELECT 1 FROM metric_attribute ma WHERE ma.metric_id = mi.metric_id
        )
    """).fetchall()
    if not rows:
        return []
    ids = ", ".join(r[0] for r in rows[:5])
    suffix = f" (+ {len(rows) - 5} more)" if len(rows) > 5 else ""
    return [f"{len(rows)} metrics have physical lineage but no semantic attribute lineage: "
            f"{ids}{suffix}"]


def _check_candidate_bindings_used_as_canonical(con) -> list[str]:
    """Hard invariant: cube construction must never depend on unresolved
    candidate bindings. Flags any cube_metric whose metric's semantic
    attributes have ONLY candidate (never resolved) bindings — informational
    today (expected right after a bootstrap migration), but must trend to
    zero for cubes that claim to be executable."""
    rows = con.execute("""
        SELECT DISTINCT cm.cube_id, ma.attribute_id
        FROM cube_metric cm
        JOIN metric_attribute ma ON ma.metric_id = cm.metric_id
        WHERE NOT EXISTS (
            SELECT 1 FROM attribute_binding ab
            WHERE ab.attribute_id = ma.attribute_id AND ab.resolution_state = 'resolved'
        )
    """).fetchall()
    if not rows:
        return []
    by_cube: dict = defaultdict(list)
    for cube_id, attr_id in rows:
        by_cube[cube_id].append(attr_id)
    return [
        f"Cube '{cube_id}' uses {len(attrs)} attribute(s) with only candidate "
        f"bindings (none resolved): {', '.join(attrs[:3])}{' ...' if len(attrs) > 3 else ''}"
        for cube_id, attrs in sorted(by_cube.items())
    ]


# Each entry is (name, fn, severity). severity="error" counts toward the
# process exit code; severity="info" is reported but never fails the build.
# The semantic-layer checks below are mostly "info" because a freshly
# bootstrap-migrated catalog is EXPECTED to have unresolved/candidate state —
# that's the whole point of not auto-promoting inferred bindings. Only
# structural integrity problems (dangling references, impossible states) are
# "error".
CHECKS = [
    ("dependency_cycles",              _check_dependency_cycles,              "error"),
    ("dangling_superseded_by",         _check_dangling_superseded_by,         "error"),
    ("formula_dep_consistency",        _check_formula_dep_consistency,        "error"),
    ("duplicate_aliases",              _check_duplicate_aliases,              "error"),
    ("missing_implementations",        _check_missing_implementations,        "error"),
    ("orphan_datasets",                _check_orphan_datasets,                "error"),
    ("deprecated_without_supersession", _check_deprecated_without_supersession, "error"),
    ("entity_coverage",                _check_entity_coverage,                "info"),
    ("inferred_lineage_coverage",      _check_inferred_lineage_coverage,      "info"),
    ("entity_relation_cycles",         _check_entity_relation_cycles,         "error"),
    ("cube_orphan_metrics",            _check_cube_orphan_metrics,            "info"),
    ("cube_missing_dep_closure",       _check_cube_missing_dep_closure,       "error"),
    # ── Semantic binding layer ──────────────────────────────────────────────
    ("entity_missing_identifier",              _check_entity_missing_identifier,              "error"),
    ("entity_multiple_identifiers",            _check_entity_multiple_identifiers,            "error"),
    ("attribute_missing_binding",              _check_attribute_missing_binding,              "info"),
    ("resolved_binding_ambiguity",             _check_resolved_binding_ambiguity,             "error"),
    ("binding_unknown_attribute",              _check_binding_unknown_attribute,              "error"),
    ("binding_unknown_dataset",                _check_binding_unknown_dataset,                "error"),
    ("entity_reference_missing_target_identifier", _check_entity_reference_missing_target_identifier, "error"),
    ("entity_relation_unresolvable",           _check_entity_relation_unresolvable,           "info"),
    ("dimension_missing_attribute",            _check_dimension_missing_attribute,            "info"),
    ("metric_attribute_unresolved",            _check_metric_attribute_unresolved,            "info"),
    ("candidate_bindings_used_as_canonical",   _check_candidate_bindings_used_as_canonical,   "info"),
]


def validate(db_path: Path) -> tuple[int, int]:
    """Run all checks. Returns (error_issue_count, info_issue_count).
    Only error_issue_count should affect a caller's exit code — info-severity
    checks are expected to be non-zero on a freshly bootstrap-migrated
    catalog and must not fail the build."""
    con = duckdb.connect(str(db_path), read_only=True)
    error_total, info_total = 0, 0
    try:
        for name, fn, severity in CHECKS:
            issues = fn(con)
            if issues:
                status = "FAIL" if severity == "error" else "warn"
            else:
                status = "pass"
            print(f"  [{status:4s}] {name}  ({len(issues)} issues, severity={severity})")
            for msg in issues[:5]:
                print(f"         {msg}")
            if len(issues) > 5:
                print(f"         ... and {len(issues) - 5} more")
            if severity == "error":
                error_total += len(issues)
            else:
                info_total += len(issues)
    finally:
        con.close()
    return error_total, info_total


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Validate metric_catalog.duckdb integrity.")
    p.add_argument("db", help="Path to metric_catalog.duckdb")
    args = p.parse_args()

    print(f"Validating {args.db}\n")
    error_total, info_total = validate(Path(args.db))
    print(f"\n{'─' * 50}")
    if info_total:
        print(f"  {info_total} informational issue(s) — expected on a fresh "
              "bootstrap migration, does not fail the build")
    if error_total:
        print(f"  {error_total} error-severity issue(s) found")
        raise SystemExit(1)
    print("  All error-severity checks passed")


if __name__ == "__main__":
    main()