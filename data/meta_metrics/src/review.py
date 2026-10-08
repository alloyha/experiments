#!/usr/bin/env python3
"""
Review queue for candidate attribute_binding rows.

This tool NEVER promotes a binding on its own. It does three things:
  1. Surfaces WHICH candidates need a human decision, prioritized by blast
     radius (how many cubes/metrics depend on that attribute).
  2. Flags CONFLICTS explicitly — attributes with more than one competing
     candidate — since those need a decision, not just a rubber stamp.
  3. Optionally cross-checks a candidate's proposed column against a real
     physical schema (if you attach one), as evidence — never as an
     automatic verdict. A "match" still requires a human to promote it.

Applying a decision always requires the human to name the winning
binding_id explicitly, via --decide or a --decisions-file. There is no mode
that picks a winner for you.

Usage:
    # Just look at the queue, most impactful first
    python3 src/review.py data/metric_catalog.duckdb

    # Only one attribute
    python3 src/review.py data/metric_catalog.duckdb --attribute order.identifier

    # Cross-check candidates against a real schema (any DuckDB-queryable
    # source exposing information_schema.columns — e.g. an attached
    # Postgres/MySQL scanner extension, or a DuckDB file with the same
    # table/column names mirrored in for inspection)
    python3 src/review.py data/metric_catalog.duckdb --schema-db real_warehouse.duckdb

    # Apply explicit decisions: promote the named binding, reject its
    # competitors for that same attribute
    python3 src/review.py data/metric_catalog.duckdb \\
        --decide order.identifier=order.identifier:bootstrap \\
        --decide order.gross_value=order.gross_value:from:order

    # Same, from a file: {"order.identifier": "order.identifier:bootstrap", ...}
    python3 src/review.py data/metric_catalog.duckdb --decisions-file decisions.json

    # Machine-readable queue for building your own UI over it
    python3 src/review.py data/metric_catalog.duckdb --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

import duckdb

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
import bindings as _bindings  # noqa: E402


# ── Data shapes ──────────────────────────────────────────────────────────────

@dataclass
class CandidateInfo:
    binding_id: str
    dataset_id: str | None
    column_name: str | None
    expression: str | None
    inference_rule: str | None
    schema_check: str  # "match" | "no_match" | "unknown"


@dataclass
class ReviewItem:
    attribute_id: str
    entity_id: str
    semantic_type: str
    cubes_affected: int
    metrics_affected: int
    cube_ids: list[str]
    conflict: bool
    candidates: list[CandidateInfo]

    @property
    def priority_score(self) -> int:
        """Higher = more urgent. Cubes matter more than raw metric count
        (a cube blocked is a whole analytical surface unusable); a conflict
        adds urgency because it can't be resolved by a quick rubber stamp."""
        return self.cubes_affected * 10 + self.metrics_affected + (5 if self.conflict else 0)


# ── Optional physical-schema cross-check ─────────────────────────────────────

def _schema_check(
    schema_con: duckdb.DuckDBPyConnection | None, dataset_id: str | None, column_name: str | None
) -> str:
    """Best-effort check of whether (dataset_id, column_name) exists in an
    attached real schema. Returns 'unknown' whenever there isn't enough
    information to check, or no schema was attached — never guesses.

    Tries three tiers, in order:
      1. Exact table_name match.
      2. Fuzzy match: dataset_id as a substring of a real table name (handles
         warehouse naming conventions like 'fct_'/'dim_' prefixes that a
         pseudocode-derived dataset_id like 'invoice' wouldn't include).
      3. No dataset_id at all: search the column name across every table.
    A fuzzy or no-dataset match is reported distinctly from an exact one —
    it's weaker evidence, not a verdict.
    """
    if schema_con is None or not column_name:
        return "unknown"
    try:
        if dataset_id:
            table = dataset_id.split(".")[-1]
            row = schema_con.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = ? AND column_name = ? LIMIT 1",
                [table, column_name],
            ).fetchone()
            if row:
                return "match"

            # Kimball-prefix tier: try the dataset_id's most likely physical
            # name directly (fct_/dim_ + entity) before falling back to
            # generic substring matching. If exactly one prefixed form is an
            # EXACT table+column match, that's strong, unambiguous evidence —
            # e.g. dataset_id='order' -> 'fct_order' resolves cleanly even
            # though a raw substring search would also (wrongly) tie it with
            # 'dim_purchase_order', since 'order' really is a trailing word
            # of 'purchase_order' too and no string trick separates them.
            prefixed_hits = []
            for prefix in ("fct_", "dim_"):
                hit = schema_con.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = ? AND column_name = ? LIMIT 1",
                    [prefix + table, column_name],
                ).fetchone()
                if hit:
                    prefixed_hits.append(prefix + table)
            if len(prefixed_hits) == 1:
                return f"match_kimball_prefix (table '{prefixed_hits[0]}')"

            # Word-boundary-aware substring fallback: 'order' must appear as
            # its own '_'-delimited token, not merely contained in a longer
            # word. Genuinely still ties 'order' with 'purchase_order' (both
            # contain 'order' as a real trailing word) when the prefix tier
            # above didn't already resolve it — that tie is correct to flag,
            # not a bug to hide.
            fuzzy_tables = sorted({r[0] for r in schema_con.execute(
                "SELECT DISTINCT table_name FROM information_schema.columns "
                "WHERE column_name = ? AND "
                "('_' || table_name || '_') LIKE '%_' || ? || '_%'",
                [column_name, table],
            ).fetchall()})
            if len(fuzzy_tables) == 1:
                return f"match_fuzzy (table '{fuzzy_tables[0]}')"
            if len(fuzzy_tables) > 1:
                return f"ambiguous_fuzzy ({len(fuzzy_tables)} tables: {', '.join(fuzzy_tables)})"
            return "no_match"

        tables = schema_con.execute(
            "SELECT DISTINCT table_name FROM information_schema.columns WHERE column_name = ?",
            [column_name],
        ).fetchall()
        if len(tables) == 1:
            return f"match_no_dataset (found in table '{tables[0][0]}')"
        if len(tables) > 1:
            return f"ambiguous ({len(tables)} tables have this column)"
        return "no_match"
    except duckdb.Error:
        return "unknown"


# ── Building the queue ───────────────────────────────────────────────────────

def build_review_queue(
    con: duckdb.DuckDBPyConnection,
    schema_con: duckdb.DuckDBPyConnection | None = None,
    attribute_filter: str | None = None,
) -> list[ReviewItem]:
    where = "ab.resolution_state = 'candidate'"
    params: list = []
    if attribute_filter:
        where += " AND ab.attribute_id = ?"
        params.append(attribute_filter)

    attrs = con.execute(f"""
        SELECT DISTINCT ab.attribute_id, sa.entity_id, sa.semantic_type
        FROM attribute_binding ab
        JOIN semantic_attribute sa ON sa.attribute_id = ab.attribute_id
        WHERE {where}
        ORDER BY ab.attribute_id
    """, params).fetchall()

    items: list[ReviewItem] = []
    for attribute_id, entity_id, semantic_type in attrs:
        cand_rows = con.execute("""
            SELECT binding_id, dataset_id, column_name, expression, inference_rule
            FROM attribute_binding
            WHERE attribute_id = ? AND resolution_state = 'candidate'
            ORDER BY binding_id
        """, [attribute_id]).fetchall()

        candidates = [
            CandidateInfo(
                binding_id=b_id, dataset_id=ds, column_name=col, expression=expr,
                inference_rule=rule, schema_check=_schema_check(schema_con, ds, col),
            )
            for b_id, ds, col, expr, rule in cand_rows
        ]

        cube_ids = [r[0] for r in con.execute("""
            SELECT DISTINCT cm.cube_id
            FROM metric_attribute ma
            JOIN cube_metric cm ON cm.metric_id = ma.metric_id
            WHERE ma.attribute_id = ?
            ORDER BY cm.cube_id
        """, [attribute_id]).fetchall()]

        metrics_affected = con.execute(
            "SELECT COUNT(DISTINCT metric_id) FROM metric_attribute WHERE attribute_id = ?",
            [attribute_id],
        ).fetchone()[0]

        items.append(ReviewItem(
            attribute_id=attribute_id, entity_id=entity_id, semantic_type=semantic_type,
            cubes_affected=len(cube_ids), metrics_affected=metrics_affected,
            cube_ids=cube_ids, conflict=len(candidates) > 1, candidates=candidates,
        ))

    items.sort(key=lambda i: i.priority_score, reverse=True)
    return items


# ── Applying explicit human decisions ────────────────────────────────────────

def _confirmed_physical_table(schema_con, dataset_id: str, column_name: str) -> str | None:
    """Re-derive the single confirmed physical table name for (dataset_id,
    column_name) from the same evidence _schema_check already computes, so
    it can be written into dataset.full_ref at promotion time. Returns None
    for anything ambiguous or unmatched — never guesses."""
    if schema_con is None or not column_name:
        return None
    check = _schema_check(schema_con, dataset_id, column_name)
    if check == "match":
        return dataset_id.split(".")[-1]
    m = re.match(r"^(match_kimball_prefix|match_fuzzy|match_no_dataset) \(table '([^']+)'\)$", check)
    return m.group(2) if m else None


def apply_decision(con: duckdb.DuckDBPyConnection, attribute_id: str, winner_binding_id: str,
                    schema_con=None) -> dict:
    """Promote winner_binding_id to 'resolved' and reject every other
    candidate binding for the same attribute_id. Requires the caller to name
    the winner explicitly — this function makes no choice of its own.

    Idempotent: if winner_binding_id is already 'resolved', this is a no-op
    that reports success rather than erroring, so re-running a decisions
    file after a partial run (or after deciding the same attribute earlier
    in a different session) doesn't fail.
    """
    current = con.execute(
        "SELECT resolution_state FROM attribute_binding WHERE binding_id = ?",
        [winner_binding_id],
    ).fetchone()
    if current is None:
        existing = [
            f"{r[0]} ({r[1]})" for r in con.execute(
                "SELECT binding_id, resolution_state FROM attribute_binding WHERE attribute_id = ?",
                [attribute_id],
            ).fetchall()
        ]
        raise _bindings.UnresolvedBindingError(
            attribute_id,
            detail=f"'{winner_binding_id}' does not exist at all (check for a typo). "
                   f"Bindings that do exist for this attribute: {', '.join(existing) or '(none)'}",
        )

    state = current[0]
    if state == "resolved":
        return {"attribute_id": attribute_id, "resolved": winner_binding_id,
                 "rejected": [], "already_done": True}
    if state == "rejected":
        raise _bindings.BindingError(
            f"'{winner_binding_id}' was previously REJECTED, not just left as candidate. "
            "Promoting it now would contradict an earlier decision — if that's intentional, "
            "promote it directly with bindings.promote_binding() instead of --decide."
        )

    all_candidates = [
        r[0] for r in con.execute(
            "SELECT binding_id FROM attribute_binding "
            "WHERE attribute_id = ? AND resolution_state = 'candidate'",
            [attribute_id],
        ).fetchall()
    ]
    if winner_binding_id not in all_candidates:
        raise _bindings.UnresolvedBindingError(
            attribute_id,
            detail=f"'{winner_binding_id}' has resolution_state='{state}', not 'candidate' — "
                   "unexpected state, not handled by --decide.",
        )
    rejected = [b for b in all_candidates if b != winner_binding_id]
    _bindings.promote_binding(con, winner_binding_id, "resolved")
    for b in rejected:
        _bindings.promote_binding(con, b, "rejected")

    # Backfill dataset.full_ref with the CONFIRMED physical table name, not
    # the bare logical dataset_id it held until now. This is what makes the
    # catalog itself a queryable source of truth for physical location —
    # to_dbt.py / generate_erd.py / anything downstream can read
    # dataset.full_ref directly and never need to import build_warehouse.py
    # (or any other physical-generator's Python internals) to know where an
    # attribute actually lives.
    winner_row = con.execute(
        "SELECT dataset_id, column_name FROM attribute_binding WHERE binding_id = ?",
        [winner_binding_id],
    ).fetchone()
    if winner_row and winner_row[0]:
        table = _confirmed_physical_table(schema_con, winner_row[0], winner_row[1])
        if table:
            con.execute("UPDATE dataset SET full_ref = ?, table_name = ? WHERE dataset_id = ?",
                        [table, table, winner_row[0]])

    return {"attribute_id": attribute_id, "resolved": winner_binding_id,
             "rejected": rejected, "already_done": False}


# ── Policy-based batch decisions ─────────────────────────────────────────────
#
# Deciding 9 structurally-identical conflicts one at a time is pure toil, not
# careful review — the review effort should go into the *shape* of the
# decision once, not into repeating it. A PolicyRule is an explicit,
# human-authored rule ("for identifier attributes with a bootstrap-vs-lineage
# conflict, prefer the bootstrap"). This is NOT auto-promotion: the rule
# itself is a deliberate decision the human writes down, apply_policy() always
# shows exactly what it will do before committing (--dry-run), and every
# resolved binding gets an audit note naming the rule that resolved it — so
# it's traceable back to "who/what decided this and why", same as a manual
# decision would be.

@dataclass
class PolicyRule:
    name: str
    semantic_type: str | None = None   # only match attributes of this semantic_type, if set
    pattern: str | None = None         # currently only "bootstrap_vs_lineage" is supported
    choose: str = "bootstrap"          # "bootstrap" | "lineage" — which side wins


def _match_rule(item: ReviewItem, rule: PolicyRule) -> str | None:
    """Return the winning binding_id if `rule` applies to this conflict, else
    None. Deliberately narrow: only fires on the exact shape it names, never
    a closest-guess fallback."""
    if not item.conflict:
        return None
    if rule.semantic_type and item.semantic_type != rule.semantic_type:
        return None
    if rule.pattern == "bootstrap_vs_lineage":
        bootstrap = [c for c in item.candidates if c.binding_id.endswith(":bootstrap")]
        lineage = [c for c in item.candidates if not c.binding_id.endswith(":bootstrap")]
        if len(bootstrap) != 1 or not lineage:
            return None  # not this exact shape — e.g. 3+ lineage variants, no bootstrap at all
        if rule.choose == "bootstrap":
            return bootstrap[0].binding_id
        if rule.choose == "lineage" and len(lineage) == 1:
            return lineage[0].binding_id
        return None  # "lineage" chosen but multiple lineage candidates compete — don't guess which
    return None


def apply_policy(
    con: duckdb.DuckDBPyConnection,
    items: list[ReviewItem],
    rules: list[PolicyRule],
    dry_run: bool,
    schema_con=None,
) -> list[dict]:
    """Apply the first matching rule to each conflicting attribute. Returns
    what happened (or would happen, if dry_run) for every match — nothing is
    hidden. Attributes matching no rule are left untouched, still visible in
    the normal queue for manual review."""
    results = []
    for item in items:
        for rule in rules:
            winner = _match_rule(item, rule)
            if winner is None:
                continue
            rejected = [c.binding_id for c in item.candidates if c.binding_id != winner]
            results.append({
                "attribute_id": item.attribute_id, "rule": rule.name,
                "winner": winner, "rejected": rejected,
            })
            if not dry_run:
                apply_decision(con, item.attribute_id, winner, schema_con=schema_con)
                con.execute(
                    "UPDATE attribute_binding SET inference_rule = inference_rule || ? "
                    "WHERE binding_id = ?",
                    [f" | policy_applied:{rule.name}", winner],
                )
            break  # first matching rule wins; don't let a later rule re-decide it
    return results


def load_policy(path: str) -> list[PolicyRule]:
    with open(path) as f:
        raw = json.load(f)
    return [PolicyRule(**r) for r in raw.get("rules", [])]


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_candidate(c: CandidateInfo) -> str:
    loc = c.expression or (f"{c.dataset_id}.{c.column_name}" if c.dataset_id else c.column_name) or "?"
    if c.schema_check == "unknown":
        tag = ""
    elif c.schema_check == "no_match":
        tag = "✗ NOT found in real schema"
    elif c.schema_check == "match":
        tag = "✓ matches real schema (exact table+column)"
    else:
        # Everything else ('match_fuzzy (...)', 'match_no_dataset (...)',
        # 'ambiguous (...)', 'ambiguous_fuzzy (...)') is weaker evidence than
        # an exact table+column hit — a substring/name-only match, not a
        # confirmed physical location. Every one of these used to fall
        # through to the same '✓ matches real schema' tag as a true exact
        # match (the old code only explicitly special-cased 'no_match' and
        # the two 'ambiguous'/'match_no_dataset' prefixes, so 'match_fuzzy'
        # silently landed in the catch-all 'else' meant only for 'match').
        # That made fuzzy prefix matches like dataset_id='lead' found inside
        # table 'fct_lead' look exactly as trustworthy as a real exact
        # match — a false-confidence bug in the one tool whose entire job is
        # giving a human accurate signal before they promote a binding.
        tag = f"~ {c.schema_check}"
    return f"      [{c.binding_id}] {loc}" + (f"  {tag}" if tag else "")


def print_report(items: list[ReviewItem]) -> None:
    if not items:
        print("No candidate bindings pending review. Nothing to do.")
        return

    n_conflict = sum(1 for i in items if i.conflict)
    n_single = len(items) - n_conflict
    print(f"{len(items)} attribute(s) awaiting review "
          f"({n_conflict} with competing candidates, {n_single} single-candidate)\n")

    for item in items:
        flag = "CONFLICT — needs a decision" if item.conflict else "single candidate"
        print(f"  {item.attribute_id}  [{flag}]")
        print(f"    entity={item.entity_id}  type={item.semantic_type}  "
              f"cubes_affected={item.cubes_affected}  metrics_affected={item.metrics_affected}")
        if item.cube_ids:
            print(f"    cubes: {', '.join(item.cube_ids)}")
        for c in item.candidates:
            print(_fmt_candidate(c))
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("db", help="Path to metric_catalog.duckdb")
    p.add_argument("--attribute", help="Only show/decide this attribute_id")
    p.add_argument("--schema-db", help="Path to a DuckDB file exposing information_schema.columns "
                                        "for a real physical schema, used as evidence only")
    p.add_argument("--json", action="store_true", help="Print the queue as JSON instead of text")
    p.add_argument("--decide", action="append", default=[], metavar="ATTRIBUTE_ID=BINDING_ID",
                    help="Apply an explicit decision: promote BINDING_ID, reject its "
                         "competitors for ATTRIBUTE_ID. Repeatable.")
    p.add_argument("--decisions-file", help="JSON file of {attribute_id: winning_binding_id, ...}")
    p.add_argument("--policy", help="JSON file of {\"rules\": [...]} — apply a named, "
                                     "human-authored rule to every matching conflict at once. "
                                     "See PolicyRule in this file for the rule shape.")
    p.add_argument("--dry-run", action="store_true",
                    help="With --policy, show what would be decided without committing anything.")
    p.add_argument("--min-impact", type=int, default=0,
                    help="Hide items with cubes_affected + metrics_affected below this (default: "
                         "0, i.e. show everything). Use e.g. --min-impact 1 to hide the zero-impact "
                         "tail of attributes no cube currently uses.")
    args = p.parse_args()

    con = duckdb.connect(args.db)
    schema_con = duckdb.connect(args.schema_db, read_only=True) if args.schema_db else None
    try:
        decisions: dict[str, str] = {}
        for entry in args.decide:
            if "=" not in entry:
                raise SystemExit(f"--decide expects ATTRIBUTE_ID=BINDING_ID, got: {entry}")
            attr, binding = entry.split("=", 1)
            decisions[attr] = binding
        if args.decisions_file:
            with open(args.decisions_file) as f:
                decisions.update(json.load(f))

        if decisions:
            for attr, binding in decisions.items():
                result = apply_decision(con, attr, binding, schema_con=schema_con)
                if result["already_done"]:
                    print(f"'{attr}' -> {result['resolved']} (already resolved, no change)")
                else:
                    print(f"Resolved '{result['attribute_id']}' -> {result['resolved']}")
                    for r in result["rejected"]:
                        print(f"  rejected: {r}")
            print()

        if args.policy:
            rules = load_policy(args.policy)
            queue_for_policy = build_review_queue(con, schema_con=schema_con)
            results = apply_policy(con, queue_for_policy, rules, dry_run=args.dry_run, schema_con=schema_con)
            label = "WOULD resolve" if args.dry_run else "Resolved"
            if not results:
                print("Policy matched nothing (no conflict has the exact shape any rule targets).\n")
            for r in results:
                print(f"[{r['rule']}] {label} '{r['attribute_id']}' -> {r['winner']}")
                for rej in r["rejected"]:
                    print(f"  {'would reject' if args.dry_run else 'rejected'}: {rej}")
            if args.dry_run and results:
                print(f"\n{len(results)} decision(s) previewed, nothing committed. "
                      "Re-run without --dry-run to apply.")
            print()

        items = build_review_queue(con, schema_con=schema_con, attribute_filter=args.attribute)
        if args.min_impact:
            items = [i for i in items if (i.cubes_affected + i.metrics_affected) >= args.min_impact]
        if args.json:
            print(json.dumps([
                {**asdict(i), "priority_score": i.priority_score} for i in items
            ], indent=2))
        else:
            print_report(items)
    finally:
        con.close()
        if schema_con is not None:
            schema_con.close()


if __name__ == "__main__":
    main()