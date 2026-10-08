#!/usr/bin/env python3
"""
Generate dbt Semantic Layer (MetricFlow) YAML from metric_catalog.duckdb.

Targets the CURRENT dbt YAML spec (v1.12+ / Fusion): entities and dimensions
are declared at the column level, "measures" no longer exist as a separate
concept (simple metrics replace them and live *inside* the semantic model),
and agg_time_dimension is required at the model level wherever a semantic
model defines metrics. See:
  https://docs.getdbt.com/docs/build/semantic-models
  https://docs.getdbt.com/docs/build/latest-metrics-spec

Physical grounding: semantic models are grouped by the physical table each
attribute's REVIEWED binding resolves to (dataset.full_ref, confirmed by
review.py against whatever --schema-db a human pointed it at) — never
assumed to be a table literally named after the catalog's own dataset_id,
and never imported from any specific physical-warehouse generator's Python
internals. Point review.py at ANY queryable schema (a DuckDB file, or
anything DuckDB can attach to) and this script picks up wherever the
bindings ended up resolving. Several logical datasets can share one physical
table (e.g. 'closed_won' + 'open_opportunity' + ... all fold into one real
table, discriminated by a 'stage' column); where that's true, review.py
records the filter on the attribute_binding itself (filter_column/
filter_value) and the resulting simple metric gets an automatic `filter:` so
it keeps counting only the rows that specific attribute means.

IMPORTANT — this generator assumes each emitted model name (e.g.
'fct_opportunity') corresponds to an actual dbt model in your project (a
model materializing/reading whatever table dataset.full_ref names). That
model must exist for `dbt parse` to succeed; this script does not create it.
Any attribute whose binding hasn't been reviewed and promoted via review.py
is skipped with a warning, not guessed.

Known, deliberately-not-solved limitation: ratio metrics reference
numerator/denominator metric *names* that this script does not verify exist
anywhere (MetricFlow ratio metrics require two real metrics, not raw
expressions) — these are marked `meta.needs_review` rather than silently
assumed correct. Same spirit as bindings.py: never guess silently.

Usage:
    python metric_catalog_to_dbt.py <catalog.duckdb> [output_dir]

Output:
    <out>/_semantic_models.yml   — one dbt `models:` entry (with nested
                                    semantic_model + simple metrics) per
                                    consolidated physical warehouse table
    <out>/_sources.yml           — dbt source declarations for the raw
                                    (pre-warehouse) datasets
    <out>/<domain>/_metrics.yml  — derived/ratio metrics only (simple
                                    metrics now live inline, see above)
"""
from __future__ import annotations

import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import yaml


# dbt MetricFlow aggregation types
AGG_MAP: dict[str, str] = {
    "sum":            "sum",
    "count":          "count",
    "count_distinct": "count_distinct",
    "avg":            "average",
    "max":            "max",
    "min":            "min",
    "median":         "median",
}

_DURATION_UNIT = {"days": "day", "hours": "hour", "minutes": "minute"}
_DURATION_PATTERN = re.compile(
    r"^(MEDIAN|AVG|MIN|MAX)\(\s*(days|hours|minutes)_between\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\)$",
    re.IGNORECASE,
)


def _inject_duration_metrics(con: duckdb.DuckDBPyConnection, data: dict) -> None:
    """Some metrics (time_to_value, cycle_time, mean_time_to_recover, ...)
    were never derived-metric candidates at all — their 'dependency' isn't
    another metric, it's two TIMESTAMP columns on their OWN entity
    ('MEDIAN(days_between(start_at, value_at))'). classify_advanced() only
    ever looks for metric-to-metric deps, so these always fell through to
    an empty TODO regardless of dependency-graph coverage. Detect the
    pattern directly here and, only when BOTH referenced columns resolve to
    real, reviewed attributes on the metric's own entity, inject a
    synthetic measure computing the real date difference — never guessed,
    exactly the same 'only act on confirmed physical columns' standard as
    everywhere else in this pipeline. A metric whose columns don't both
    resolve (e.g. 'incident_start' was never declared/reviewed as an
    attribute — the real one is 'occurred_at') is left alone, still
    reported as a content gap rather than silently forced to match."""
    for mid, m in data["metrics"].items():
        if any(c["role"] == "measure" for c in data["cols"].get(mid, [])):
            continue  # already has a real measure linkage another mechanism found
        match = _DURATION_PATTERN.match((m.get("expression") or "").strip())
        if not match:
            continue
        agg_fn, unit, col1, col2 = match.groups()
        entity_id = m.get("entity_id")
        if not entity_id:
            continue
        rows = con.execute("""
            SELECT sa.name, ab.dataset_id, ab.column_name
            FROM semantic_attribute sa
            JOIN attribute_binding ab ON ab.attribute_id = sa.attribute_id
                                      AND ab.resolution_state = 'resolved'
            WHERE sa.entity_id = ? AND sa.name IN (?, ?)
        """, [entity_id, col1, col2]).fetchall()
        by_name = {r[0]: (r[1], r[2]) for r in rows}
        if col1 not in by_name or col2 not in by_name:
            continue  # genuine gap: one or both columns aren't reviewed attributes here
        ds1, phys1 = by_name[col1]
        ds2, phys2 = by_name[col2]
        if ds1 != ds2:
            continue  # the two timestamps live on different physical tables — can't diff them in one expr
        data["metrics"][mid]["aggregation"] = agg_fn.lower()
        data["cols"][mid].append({
            "source_id": ds1,
            "column": f"date_diff('{_DURATION_UNIT[unit.lower()]}', {phys1}, {phys2})",
            "role": "measure", "filter_column": None, "filter_value": None,
        })


def _slug(text: str) -> str:
    # Collapse repeated underscores too — MetricFlow forbids dunders in any
    # name, and any expression-derived name (e.g. a raw SQL snippet) is
    # guaranteed to produce runs of '_' from consecutive non-alnum chars.
    s = re.sub(r"[^a-z0-9_]", "_", text.lower(), flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_")


def _compact(d: dict) -> dict:
    """Drop None, empty list, empty dict values."""
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


def _translate_formula(expr: str, deps: list[dict]) -> str:
    """
    Replace known dep-metric tokens in the pseudocode expression with their
    MetricFlow-safe names so the output is as close to executable as possible.
    e.g. 'MRR * 12' + dep finance.mrr → 'finance_mrr * 12'
    """
    result = expr
    for dep in deps:
        dep_id  = dep["depends_on"]               # "finance.mrr"
        dep_key = dep_id.split(".", 1)[1]          # "mrr"
        safe    = _slug(dep_id.replace(".", "_"))  # "finance_mrr"
        # uppercase abbreviation (MRR, LTV, CAC, EBITDA …)
        abbrev = dep_key.upper()
        result = re.sub(rf"\b{re.escape(abbrev)}\b", safe, result)
        # snake_case name
        result = re.sub(rf"\b{re.escape(dep_key)}\b", safe, result, flags=re.IGNORECASE)
    return result


_IDENT = re.compile(r"\b[a-z][a-z0-9_]*\b")
_EXPR_STOPWORDS = frozenset({
    "nullif", "count", "sum", "avg", "median", "min", "max", "distinct",
    "where", "between", "not", "and", "or", "in", "null", "now",
})


def _unresolved_tokens(translated_expr: str, deps: list[dict]) -> list[str]:
    """
    Tokens left in a translated expression that are neither a SQL/operator
    keyword nor one of the safe dep names _translate_formula just substituted
    in. A partial translation — some tokens matched a known dependency,
    others didn't — looks identical to a clean one unless checked explicitly:
    `_translate_formula` only replaces tokens it recognizes and leaves
    everything else untouched, so 'net_revenue / NULLIF(distinct_paying_users,0)'
    with only 'net_revenue' known becomes 'finance_net_revenue / NULLIF(distinct_paying_users,0)'
    — a still-broken expr (distinct_paying_users resolves to nothing) that a
    naive "did the string change at all" check would wrongly call done.
    """
    safe_names = {_slug(d["depends_on"].replace(".", "_")) for d in deps}
    return sorted({
        t for t in _IDENT.findall(translated_expr.lower())
        if t not in _EXPR_STOPWORDS and t not in safe_names
    })


# ── physical table resolution ───────────────────────────────────────────────

# ── physical table resolution ───────────────────────────────────────────────

def resolve_dataset_tables(con: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """
    Map each RESOLVED dataset_id (as attribute_binding.dataset_id already
    stores it — the catalog's own normalized entity/variant name, e.g.
    'opportunity', 'survey_response__engagement_survey') to the CONFIRMED
    physical table dataset.full_ref carries. full_ref is only ever populated
    by review.py's apply_decision() at promotion time, backed by real
    information_schema evidence against whatever --schema-db was pointed at
    — never guessed here, never re-derived from a physical generator's
    internal Python logic. A dataset_id with no full_ref yet (never promoted,
    or promoted with no confirmable single table) is simply absent from the
    returned mapping — callers must treat that as "no usable physical
    source" and skip it loudly, not silently.
    """
    return {
        r[0]: {"table": r[1]}
        for r in con.execute(
            "SELECT dataset_id, full_ref FROM dataset WHERE full_ref IS NOT NULL "
            "AND full_ref != dataset_id"  # full_ref==dataset_id means never confirmed
        ).fetchall()
    }


# ── DB loading ────────────────────────────────────────────────────────────────

def load_data(con: duckdb.DuckDBPyConnection) -> dict:
    metrics = {
        r[0]: {
            "metric_id":         r[0],
            "name":              r[1],
            "department":        r[2],
            "description":       r[3],
            "aggregation":       r[4],
            "entity_id":         r[5],
            "unit":              r[6],
            "status":            r[7],
            "data_quality":      r[8],
            "refresh_frequency": r[9],
            "expression":        r[10],
            "source_table":      r[11],
        }
        for r in con.execute("""
            SELECT m.metric_id, m.name, m.department, m.description,
                   m.aggregation, m.entity_id, m.unit, m.status,
                   m.data_quality, m.refresh_frequency,
                   i.expression, i.source_table
            FROM metric_definition m
            LEFT JOIN metric_implementation i
              ON i.metric_id = m.metric_id AND i.is_current = true
            ORDER BY m.metric_id
        """).fetchall()
    }

    cols: dict[str, list] = defaultdict(list)
    # Sourced entirely from metric_attribute (metric -> attribute, with role)
    # joined to attribute_binding restricted to resolution_state='resolved'
    # — i.e. only human-reviewed, promoted bindings ever reach the generated
    # SQL. This replaces an earlier version that read impl_column, a
    # parallel table populated BEFORE alias-normalization and BEFORE any
    # review.py promotion — using it unchecked meant to_dbt.py silently
    # ignored every decision made through review.py. metric_attribute's role
    # vocabulary (measure|time) is coarser than impl_column's
    # (numerator|denominator|date_key|...); 'time' maps to 'date_key' below
    # so time attributes still get excluded from being used as a simple
    # metric's expr. There is currently no numerator/denominator role
    # anywhere in metric_attribute, so ratio-metric classification below
    # still never fires — same as before this change (0 ratio metrics
    # either way), not a regression.
    _ROLE_MAP = {"time": "date_key", "measure": "measure"}
    for metric_id, attribute_id, ma_role in con.execute(
        "SELECT metric_id, attribute_id, role FROM metric_attribute"
    ).fetchall():
        ab = con.execute(
            "SELECT dataset_id, column_name, filter_column, filter_value "
            "FROM attribute_binding WHERE attribute_id = ? AND resolution_state = 'resolved'",
            [attribute_id],
        ).fetchone()
        if not ab:
            continue  # not (yet) reviewed — see the warning below
        dataset_id, column, attr_filter_col, attr_filter_val = ab
        # Per-metric filter (metric_attribute) wins over per-attribute
        # filter (attribute_binding) when both are set — it's the more
        # specific of the two, needed whenever several metrics share one
        # attribute but mean different filtered slices of it.
        metric_filter_col, metric_filter_val = con.execute(
            "SELECT filter_column, filter_value FROM metric_attribute "
            "WHERE metric_id = ? AND attribute_id = ? AND role = ?",
            [metric_id, attribute_id, ma_role],
        ).fetchone()
        filter_column = metric_filter_col or attr_filter_col
        filter_value  = metric_filter_val or attr_filter_val
        ds = con.execute("SELECT full_ref FROM dataset WHERE dataset_id = ?", [dataset_id]).fetchone()
        table = ds[0] if ds else None
        cols[metric_id].append({
            "source_id": dataset_id, "column": column, "role": _ROLE_MAP.get(ma_role, ma_role),
            "table": table, "filter_column": filter_column, "filter_value": filter_value,
        })

    unresolved_attrs = con.execute("""
        SELECT DISTINCT ma.metric_id, ma.attribute_id FROM metric_attribute ma
        LEFT JOIN attribute_binding ab
          ON ab.attribute_id = ma.attribute_id AND ab.resolution_state = 'resolved'
        WHERE ab.binding_id IS NULL
        ORDER BY 1, 2
    """).fetchall()
    if unresolved_attrs:
        print(
            f"warning: {len(unresolved_attrs)} metric attribute reference(s) have no "
            "resolved (reviewed) binding and were skipped entirely — run review.py "
            "and promote them for these metrics to get real SQL:",
            file=sys.stderr,
        )
        for metric_id, attr_id in unresolved_attrs:
            print(f"  {metric_id}: {attr_id}", file=sys.stderr)

    dims: dict[str, list] = defaultdict(list)
    for r in con.execute("""
        SELECT md.metric_id, d.name, md.role, md.required, d.default_expr
        FROM metric_dimension md
        JOIN dimension d ON d.dimension_id = md.dimension_id
    """).fetchall():
        dims[r[0]].append({"name": r[1], "role": r[2], "required": r[3], "join_path": r[4]})

    deps: dict[str, list] = defaultdict(list)
    for r in con.execute(
        "SELECT metric_id, depends_on_metric_id, dependency_type FROM metric_dependency"
    ).fetchall():
        deps[r[0]].append({"depends_on": r[1], "type": r[2]})

    owners: dict[str, list] = defaultdict(list)
    for r in con.execute(
        "SELECT metric_id, owner_type, team, contact FROM metric_owner"
    ).fetchall():
        owners[r[0]].append(_compact({"type": r[1], "team": r[2], "contact": r[3]}))

    tags: dict[str, list] = defaultdict(list)
    for r in con.execute("SELECT metric_id, tag FROM metric_tag").fetchall():
        tags[r[0]].append(r[1])

    quality: dict[str, list] = defaultdict(list)
    for r in con.execute(
        "SELECT metric_id, dimension, rule, threshold, severity FROM quality_contract"
    ).fetchall():
        quality[r[0]].append(_compact({"dimension": r[1], "rule": r[2],
                                        "threshold": r[3], "severity": r[4]}))

    benchmarks: dict[str, dict] = {}
    for r in con.execute("""
        SELECT metric_id, benchmark_type, target, range_low, range_high,
               population, period, source
        FROM metric_benchmark
    """).fetchall():
        benchmarks[r[0]] = _compact({
            "type": r[1], "target": r[2], "range_low": r[3],
            "range_high": r[4], "population": r[5], "period": r[6], "source": r[7],
        })

    sources: dict[str, dict] = {
        r[0]: {"source_id": r[0], "warehouse": r[1], "db_schema": r[2],
               "table_name": r[3], "full_ref": r[4]}
        for r in con.execute(
            "SELECT dataset_id, warehouse, db_schema, table_name, full_ref FROM dataset"
        ).fetchall()
    }

    entities: dict[str, dict] = {
        r[0]: {"name": r[1], "pk_column": r[2]}
        for r in con.execute("SELECT entity_id, name, pk_column FROM entity").fetchall()
    }

    return dict(metrics=metrics, cols=cols, dims=dims, deps=deps,
                owners=owners, tags=tags, quality=quality,
                benchmarks=benchmarks, sources=sources, entities=entities)


# ── semantic model builder ────────────────────────────────────────────────────

def build_semantic_models(con: duckdb.DuckDBPyConnection, data: dict,
                           dataset_tables: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """
    One classic dbt `semantic_models:` entry per physical table a resolved
    binding actually points at (see resolve_dataset_tables()) — NOT the
    'semantic_model nested under models:' shape used previously, which
    dbt-core 1.12 rejects outright (that shape belongs to dbt Cloud's Fusion
    engine, not open-source dbt-core; verified against a real `dbt parse`).
    Each simple metric becomes its own top-level `metrics:` entry (returned
    separately, to be merged into build_metrics()'s output) pointing at a
    measure — never embedded inside the semantic model itself — so several
    metrics can share the same underlying column with DIFFERENT filters
    (e.g. bookings / pipeline_value / weighted_pipeline all aggregate
    opportunity.amount, filtered to a different stage each).
    """
    metrics  = data["metrics"]
    cols     = data["cols"]
    dims_map = data["dims"]

    # The table's PRIMARY entity is whichever entity's OWN identifier
    # attribute resolves onto it — NOT whichever entity most of the metrics
    # attached to this table happen to be classified under in
    # metric_definition. Those can disagree (e.g. most of fct_subscription's
    # metrics are business-classified under 'customer', but the table is
    # physically a subscription table) — using the majority-vote entity
    # instead of the table's own identifier caused MULTIPLE semantic models
    # to declare 'customer' as their primary entity with the SAME dimension
    # names, which MetricFlow rejects outright (a (primary_entity,
    # dimension) pair must be unique across the whole semantic manifest).
    table_to_entity = {
        r[1]: r[0] for r in con.execute("""
            SELECT sa.entity_id, ds.full_ref
            FROM semantic_attribute sa
            JOIN attribute_binding ab ON ab.attribute_id = sa.attribute_id
                                      AND ab.resolution_state = 'resolved'
            JOIN dataset ds ON ds.dataset_id = ab.dataset_id
            WHERE sa.semantic_type = 'identifier' AND ds.full_ref IS NOT NULL
        """).fetchall()
    }

    metrics_by_table: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unresolved: set[str] = set()
    for mid, col_list in cols.items():
        seen_tables_for_mid: set[str] = set()
        for c in col_list:
            info = dataset_tables.get(c["source_id"])
            if info is None:
                unresolved.add(c["source_id"])
                continue
            table = info["table"]
            if table in seen_tables_for_mid:
                continue
            seen_tables_for_mid.add(table)
            metrics_by_table[table].append((mid, c["source_id"]))

    if unresolved:
        print(
            f"warning: {len(unresolved)} dataset_id(s) have no resolvable physical "
            f"table and were skipped entirely: {sorted(unresolved)}",
            file=sys.stderr,
        )

    sem_models: list[dict] = []
    simple_metrics: list[dict] = []

    for table, pairs in sorted(metrics_by_table.items()):
        metric_ids = sorted({mid for mid, _ds in pairs})
        domain = metric_ids[0].split(".", 1)[0] if metric_ids else "misc"

        # MetricFlow forbids '__' (dunders) in semantic-model/measure names —
        # our split-variant physical tables use exactly that separator
        # (fct_survey_response__engagement_survey). The dbt `model: ref(...)`
        # further below still points at the real physical table unchanged;
        # only this semantic model's OWN identifier needs to satisfy
        # MetricFlow's naming rule. Computed here, first thing in the loop,
        # since measure names (built further down) need it too.
        sem_model_name = table.replace("__", "_v_")

        top_entity = table_to_entity.get(table)
        if top_entity is None:
            # No entity's identifier resolves to this table (shouldn't
            # normally happen once identifiers are reviewed) — fall back to
            # majority vote among the metrics attached, same as before, but
            # only as a last resort, loudly enough to notice in a diff.
            ent_counts: dict[str, int] = defaultdict(int)
            for mid in metric_ids:
                eid = metrics[mid].get("entity_id")
                if eid:
                    ent_counts[eid] += 1
            top_entity = max(ent_counts, key=ent_counts.get) if ent_counts else None
        if top_entity:
            edata = data.get("entities", {}).get(top_entity, {})
            entity_name = top_entity
            entity_col  = edata.get("pk_column") or (top_entity + "_id")
        else:
            entity_name, entity_col = "row", "id"

        seen_dims: set[str] = set()
        dimensions: list[dict[str, Any]] = []
        for mid in metric_ids:
            for d in dims_map.get(mid, []):
                dname = _slug(d["name"])
                if dname in seen_dims:
                    continue
                seen_dims.add(dname)
                dtype = "time" if d["role"] == "temporal" else "categorical"
                jp    = d.get("join_path") or ""
                expr  = jp.split(".")[-1] if "." in jp else (jp or dname)
                dimensions.append({"name": dname, "type": dtype, "expr": expr})

        # Time dimensions from metric_attribute's role='time' attributes
        # (mapped to 'date_key' in load_data(), see _ROLE_MAP) — a second,
        # independent source from the older metric_dimension/dimension
        # catalog above, which has zero role='temporal' rows anywhere in
        # this catalog. Without this, models whose only timestamp evidence
        # is a metric's own formula (e.g. 'created_at' in a duration calc)
        # would still show no time dimension even though the target schema
        # already has a real TIMESTAMP column for it.
        for mid, ds_id in pairs:
            for c in cols.get(mid, []):
                if c["source_id"] != ds_id or c["role"] != "date_key":
                    continue
                dname = _slug(c["column"])
                if dname in seen_dims:
                    continue
                seen_dims.add(dname)
                dimensions.append({"name": dname, "type": "time", "expr": c["column"]})

        # measures: one per unique (column, agg) actually used on this
        # table — unfiltered. Filtering (see 'bookings' vs 'pipeline_value'
        # above) happens on the METRIC that wraps a measure, not the measure
        # itself, so the same measure can back several differently-filtered
        # metrics without duplicating it.
        # MetricFlow forbids '__' (dunders) in semantic-model/measure names —
        # already handled above (sem_model_name), computed before the
        # measures loop so measure names can be prefixed with it too.

        measures: dict[str, dict[str, Any]] = {}
        for mid, ds_id in pairs:
            m   = metrics[mid]
            agg = m["aggregation"]
            if agg not in AGG_MAP:
                continue  # ratio/derived handled in build_metrics(), not here
            metric_cols = [c for c in cols[mid] if c["source_id"] == ds_id]
            measurable  = [c for c in metric_cols if c["role"] != "date_key"]
            if not measurable:
                continue  # see build_metrics()/classify_advanced fallback
            num_cols = [c for c in measurable if c["role"] == "numerator"]
            chosen   = num_cols[0] if num_cols else measurable[0]
            expr     = chosen["column"]
            measure_name = _slug(f"{sem_model_name}_{expr}_{AGG_MAP[agg]}")
            measures.setdefault(measure_name, {
                "name": measure_name, "agg": AGG_MAP[agg], "expr": expr,
            })

            disc_col, disc_val = chosen.get("filter_column"), chosen.get("filter_value")
            metric_filter = None
            if disc_col:
                metric_filter = f"{{{{ Dimension('{entity_name}__{disc_col}') }}}} = '{disc_val}'"
                if disc_col not in seen_dims:
                    seen_dims.add(disc_col)
                    dimensions.append({"name": disc_col, "type": "categorical", "expr": disc_col})

            simple_metrics.append(_compact({
                "name":        _slug(mid.replace(".", "_")),
                "label":       m["name"],
                "description": m["description"],
                "type":        "simple",
                "type_params": {"measure": {"name": measure_name}},
                "filter":      metric_filter,
                "_domain":     domain,
                "_table":      table,
            }))

        # entity + dimension columns, classic format. Two dimensions CAN
        # legitimately share the same physical column (e.g. a domain-level
        # 'status_lead' dimension and a discriminator-filter 'status'
        # dimension both pointing at the same 'status' column, under
        # different names) — that's fine in MetricFlow. What's NOT fine is
        # two dimensions with the SAME NAME, which would happen if the same
        # dimension got added twice (dims_map + discriminator loop, or a
        # genuine duplicate upstream) — dedupe on name, not on column.
        seen_dim_names: set[str] = set()
        classic_dims: list[dict[str, Any]] = []
        agg_time_dimension = None
        for d in dimensions:
            if d["name"] in seen_dim_names:
                continue
            seen_dim_names.add(d["name"])
            col_name = d["expr"]
            dim_entry: dict[str, Any] = {"name": d["name"]}
            if col_name != d["name"]:
                dim_entry["expr"] = col_name
            if d["type"] == "time":
                dim_entry["type"] = "time"
                dim_entry["type_params"] = {"time_granularity": "day"}
                if agg_time_dimension is None:
                    agg_time_dimension = d["name"]
            else:
                dim_entry["type"] = "categorical"
            classic_dims.append(dim_entry)

        if agg_time_dimension is None and measures:
            # MetricFlow requires agg_time_dimension whenever a semantic
            # model has measures, but several of our tables genuinely have
            # no timestamp column at all (pure reference/dimension tables
            # like cash_account, employee, vulnerability) — that's a real
            # gap in the physical model, not something to paper over with a
            # fake value (which MetricFlow rejects outright as an invalid
            # manifest, not just a warning). Surface it as a metric-level
            # problem instead: these measures still get emitted as
            # `measures:` (so they're visible/documented), but any simple
            # metric that would have used one is skipped, with a clear
            # reason, rather than shipping a semantic manifest that fails to
            # parse at all.
            agg_time_dimension = None
            no_time_dimension = True
        else:
            no_time_dimension = False

        sem_models.append(_compact({
            "name":    sem_model_name,
            "model":   f"ref('stg_{table}')",
            "defaults": {"agg_time_dimension": agg_time_dimension} if agg_time_dimension else None,
            "entities": [{"name": entity_name, "type": "primary", "expr": entity_col}],
            "dimensions": classic_dims or None,
            # A measure with no agg_time_dimension available is invalid on
            # its own in MetricFlow, independent of whether any metric
            # references it — so tables with no time dimension get no
            # measures declared at all, not just no metrics built from them.
            "measures": (list(measures.values()) or None) if not no_time_dimension else None,
        }))
        if no_time_dimension and measures:
            print(
                f"warning: '{table}' has no time dimension — {len(measures)} measure(s) "
                "defined for documentation, but no simple metric was emitted from them "
                "(MetricFlow requires agg_time_dimension for queryable metrics): "
                f"{sorted(measures)}",
                file=sys.stderr,
            )
            simple_metrics = [sm for sm in simple_metrics if sm.get("_table") != table]

    for sm in simple_metrics:
        sm.pop("_table", None)
    return sem_models, simple_metrics



# ── metric builder (derived / ratio only — simple metrics live inline) ───────

def classify_advanced(mid: str, m: dict, data: dict) -> tuple[str, dict, dict] | None:
    """Return (metricflow_type, extra_metric_keys, extra_meta) for non-simple
    metrics, or None if this metric is simple (already emitted inline by
    build_semantic_models — nothing left to do here)."""
    agg  = m["aggregation"]
    deps = data["deps"].get(mid, [])
    cols = data["cols"].get(mid, [])

    has_source = any(c["role"] != "date_key" for c in cols)  # a timestamp alone
                                                                # is never a valid simple-metric expr
    has_deps   = bool(deps)
    is_simple  = agg in AGG_MAP and has_source and not has_deps

    if is_simple:
        return None

    # ── derived: has explicit dependency edges (or custom agg with deps) ──
    if has_deps:
        raw_expr      = m.get("expression") or ""
        translated    = _translate_formula(raw_expr, deps)
        unresolved    = _unresolved_tokens(translated, deps)
        input_metrics = [{"name": _slug(d["depends_on"].replace(".", "_"))} for d in deps]
        if translated == raw_expr:
            # Not one single dep token was found in the expression at all.
            return "derived", {"type_params": {"expr": f"# TODO translate: {raw_expr}",
                                                "metrics": input_metrics}}, {}
        if unresolved:
            # Partial translation: some tokens matched a real dependency and
            # got substituted, but others didn't — the naive "did the string
            # change" check would call this done; it isn't. Keep the partial
            # progress visible (useful for whoever finishes it by hand) but
            # don't claim success, and name exactly what's still missing.
            return "derived", {
                "type_params": {
                    "expr": f"# TODO translate (unresolved: {', '.join(unresolved)}): {translated}",
                    "metrics": input_metrics,
                },
            }, {
                "needs_review": (
                    f"partially translated — no metric/column found for: {', '.join(unresolved)}. "
                    "Either add these as their own metrics/dependencies, or rewrite this expr by hand."
                ),
            }
        return "derived", {"type_params": {"expr": translated, "metrics": input_metrics}}, {}

    # ── ratio attempt: agg=ratio, find numerator/denominator columns ──
    if agg == "ratio" and has_source:
        num = [c for c in cols if c["role"] == "numerator"]
        den = [c for c in cols if c["role"] == "denominator"]
        if num and den:
            return "ratio", {
                "type_params": {
                    "numerator":   _slug(mid.split(".", 1)[1]) + "_num",
                    "denominator": _slug(mid.split(".", 1)[1]) + "_den",
                },
            }, {
                "needs_review": (
                    "numerator/denominator reference synthetic metric names that "
                    "are not materialized anywhere yet — wire these to real "
                    "metrics (or rebuild as a fraction-of-measures ratio) before use."
                ),
            }

    # ── fallback: derived with TODO, no input_metrics to avoid MetricFlow validation errors ──
    expr = m.get("expression") or "# TODO"
    return "derived", {"type_params": {"expr": f"# TODO translate: {expr}", "metrics": []}}, {}


def build_metrics(data: dict) -> list[dict]:
    out = []
    for mid, m in data["metrics"].items():
        domain, _ = mid.split(".", 1)
        classified = classify_advanced(mid, m, data)
        if classified is None:
            continue  # simple metric — already emitted inline in the semantic model
        mtype, extra_keys, extra_meta = classified

        meta: dict[str, Any] = {}
        if data["owners"].get(mid):
            meta["owners"]  = data["owners"][mid]
        if data["quality"].get(mid):
            meta["quality"] = data["quality"][mid]
        if data["benchmarks"].get(mid):
            meta["benchmarks"] = data["benchmarks"][mid]
        for field in ("unit", "data_quality", "refresh_frequency"):
            if m.get(field):
                meta[field] = m[field]
        meta.update(extra_meta)

        entry = _compact({
            "name":        _slug(mid.replace(".", "_")),
            "label":       m["name"],
            "description": m["description"],
            "type":        mtype,
            **extra_keys,
            "tags":        data["tags"].get(mid) or None,
            "meta":        meta or None,
        })
        entry["_domain"] = domain
        out.append(entry)
    return out


def build_sources(data: dict, schema: str = "TODO_replace_schema",
                  database: str = "TODO_replace_database") -> list[dict]:
    tables = [
        _compact({
            "name":        _slug(src["table_name"] or sid),
            "identifier":  src["table_name"] or sid,
            "description": f"Source table for catalog metrics using {sid}",
        })
        for sid, src in sorted(data["sources"].items())
    ]
    if not tables:
        return []
    return [{
        "name":        "metric_catalog",
        "description": "Auto-generated from metric_catalog.duckdb (raw, pre-warehouse sources)",
        "schema":      schema,
        "database":    database,
        "tables":      tables,
    }]


# ── YAML emission ─────────────────────────────────────────────────────────────

def _dump(obj: Any) -> str:
    return yaml.dump(obj, allow_unicode=True, default_flow_style=False,
                     sort_keys=False, indent=2, width=120)


def _neutralize_dangling_refs(metrics: list[dict]) -> list[dict]:
    """After everything is built, some derived/ratio metrics may reference
    another metric that ended up not being emitted at all (skipped for
    having no resolved binding, no time dimension, etc.) — dbt/MetricFlow
    treats a dangling metric reference as a hard parse error, not a
    warning. Rather than chase every individual cause upstream (there will
    always be another one), catch it here once, generically: any reference
    to a metric name that doesn't actually exist in the final set gets
    converted to a flagged TODO instead of silently breaking `dbt parse`.
    A derived metric left with ZERO real input metrics after that (i.e. its
    formula never matched any known dependency at all) is DROPPED entirely
    rather than kept as a hollow, input-less 'derived' metric — MetricFlow
    treats those as invalid too, not just unfinished."""
    emitted = {m["name"] for m in metrics}
    kept = []
    dropped = []
    for m in metrics:
        if m["type"] == "derived":
            refs = [d["name"] for d in m.get("type_params", {}).get("metrics", [])]
            missing = [r for r in refs if r not in emitted]
            if missing:
                m["type_params"] = {
                    "expr": f"# TODO translate (metric(s) not available: {', '.join(missing)}): "
                            f"{m['type_params'].get('expr', '')}",
                    "metrics": [],
                }
            if not m["type_params"].get("metrics"):
                dropped.append(m["name"])
                continue
        elif m["type"] == "ratio":
            tp = m.get("type_params", {})
            missing = [n for n in (tp.get("numerator"), tp.get("denominator"))
                       if n and n not in emitted]
            if missing:
                dropped.append(m["name"])
                continue
        kept.append(m)
    if dropped:
        print(
            f"warning: {len(dropped)} metric(s) dropped entirely — no real formula could be "
            f"resolved for them at all (not even partially), so there is nothing valid to "
            f"emit; fill these in by hand in the catalog and regenerate: {sorted(dropped)}",
            file=sys.stderr,
        )
    return kept


def emit_files(sem_models: list[dict], metrics: list[dict],
               sources_data: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "_semantic_models.yml").write_text(
        _dump({"semantic_models": sem_models}),
        encoding="utf-8",
    )

    if sources_data:
        (out_dir / "_sources.yml").write_text(
            _dump({"version": 2, "sources": sources_data}), encoding="utf-8",
        )

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for m in metrics:
        by_domain[m.pop("_domain")].append(m)

    for domain, dm in sorted(by_domain.items()):
        d = out_dir / domain
        d.mkdir(exist_ok=True)
        (d / "_metrics.yml").write_text(
            _dump({"metrics": dm}), encoding="utf-8",
        )

    n_simple = sum(1 for m in metrics if m["type"] == "simple")
    counts = {t: sum(1 for m in metrics if m["type"] == t) for t in ("derived", "ratio")}
    print(f"Semantic models : {len(sem_models)}  →  {out_dir / '_semantic_models.yml'}")
    print(f"Simple metrics  : {n_simple}")
    print(f"Other metrics   : {len(metrics) - n_simple}  (derived={counts['derived']} ratio={counts['ratio']})")
    for domain in sorted(by_domain):
        print(f"  {domain:20s} {len(by_domain[domain])} metrics  →  {domain}/_metrics.yml")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate dbt Semantic Layer YAML (classic dbt-core spec — "
                     "validated against a real `dbt parse`) from metric_catalog.duckdb."
    )
    p.add_argument("db",     help="Path to metric_catalog.duckdb")
    p.add_argument("out",    nargs="?", help="Output directory (default: <db_dir>/dbt_output)")
    p.add_argument("--schema",   default="TODO_replace_schema",   help="dbt source schema")
    p.add_argument("--database", default="TODO_replace_database", help="dbt source database")
    args = p.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out) if args.out else db_path.parent / "dbt_output"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        data           = load_data(con)
        _inject_duration_metrics(con, data)
        dataset_tables = resolve_dataset_tables(con)
        sem_models, simple_metrics = build_semantic_models(con, data, dataset_tables)
        metrics        = simple_metrics + build_metrics(data)
        # Fixed-point: dropping a metric with no real inputs can itself
        # dangle a metric that depended on IT — repeat until stable.
        for _ in range(5):
            before = len(metrics)
            metrics = _neutralize_dangling_refs(metrics)
            if len(metrics) == before:
                break
        sources        = build_sources(data, args.schema, args.database)
        emit_files(sem_models, metrics, sources, out_dir)
    finally:
        con.close()


if __name__ == "__main__":
    main()