#!/usr/bin/env python3
"""
Generate dbt Semantic Layer (MetricFlow) YAML from metric_catalog.duckdb.

Usage:
    python metric_catalog_to_dbt.py <catalog.duckdb> [output_dir]

Output:
    <out>/_semantic_models.yml   — one semantic model per distinct data source
    <out>/_sources.yml           — dbt source declarations
    <out>/<domain>/_metrics.yml  — metrics grouped by domain
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


# ── grain free-text → (entity_name, pk_column) ──────────────────────────────
GRAIN_TO_ENTITY: dict[str, tuple[str, str]] = {
    "fatura":           ("invoice",          "invoice_id"),
    "assinatura":       ("subscription",     "subscription_id"),
    "cliente":          ("customer",         "customer_id"),
    "oportunidade":     ("opportunity",      "opportunity_id"),
    "usuário":          ("user",             "user_id"),
    "usuario":          ("user",             "user_id"),
    "lead":             ("lead",             "lead_id"),
    "pedido":           ("order",            "order_id"),
    "ticket":           ("ticket",           "ticket_id"),
    "colaborador":      ("employee",         "employee_id"),
    "deploy":           ("deployment",       "deployment_id"),
    "incidente":        ("incident",         "incident_id"),
    "campanha":         ("campaign",         "campaign_id"),
    "sku":              ("sku",              "sku_id"),
    "contrato":         ("contract",         "contract_id"),
    "mês":              ("month",            "month_date"),
    "mes":              ("month",            "month_date"),
    "período":          ("period",           "period_date"),
    "periodo":          ("period",           "period_date"),
    "dia":              ("day",              "date_day"),
    "semana":           ("week",             "date_week"),
    "cohort":           ("cohort",           "cohort_id"),
    "resposta":         ("survey_response",  "response_id"),
    "sessão":           ("session",          "session_id"),
    "sessao":           ("session",          "session_id"),
    "release":          ("release",          "release_id"),
    "execução":         ("pipeline_run",     "run_id"),
    "execucao":         ("pipeline_run",     "run_id"),
    "dataset":          ("dataset",          "dataset_id"),
    "auditoria":        ("audit",            "audit_id"),
    "vulnerabilidade":  ("vulnerability",    "vulnerability_id"),
    "controle":         ("control",          "control_id"),
    "processo":         ("process",          "process_id"),
    "conta":            ("account",          "account_id"),
    "movimento":        ("movement",         "movement_id"),
    "vaga":             ("job",              "job_id"),
    "contratação":      ("hire",             "hire_id"),
    "contratacao":      ("hire",             "hire_id"),
    "entrega":          ("delivery",         "delivery_id"),
    "pedido de compra": ("purchase_order",   "po_id"),
    "meta":             ("target",           "target_id"),
    "kpi":              ("kpi",              "kpi_id"),
    "feature":          ("feature",          "feature_id"),
    "oferta":           ("offer",            "offer_id"),
    "trial":            ("trial",            "trial_id"),
    "ação":             ("action",           "action_id"),
    "acao":             ("action",           "action_id"),
    "item":             ("item",             "item_id"),
    "carrinho":         ("cart",             "cart_id"),
    "checkout":         ("checkout",         "checkout_id"),
    "recurso":          ("resource",         "resource_id"),
    "operação":         ("operation",        "operation_id"),
    "operacao":         ("operation",        "operation_id"),
    "linha de venda":   ("sale_line",        "sale_line_id"),
    "agente":           ("agent",            "agent_id"),
    "produto":          ("product",          "product_id"),
    "requisição":       ("request",          "request_id"),
    "requisicao":       ("request",          "request_id"),
    "serviço":          ("service",          "service_id"),
    "servico":          ("service",          "service_id"),
}

# dbt MetricFlow aggregation types
AGG_MAP: dict[str, str] = {
    "sum":            "sum",
    "count":          "count",
    "count_distinct": "count_distinct",
    "avg":            "average",
    "max":            "max",
    "min":            "min",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower(), flags=re.UNICODE).strip("_")


def _infer_entity(grain: str) -> tuple[str, str]:
    key = grain.lower().strip()
    if key in GRAIN_TO_ENTITY:
        return GRAIN_TO_ENTITY[key]
    s = _slug(key)
    return s, f"{s}_id"


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


# ── DB loading ────────────────────────────────────────────────────────────────

def load_data(con: duckdb.DuckDBPyConnection) -> dict:
    metrics = {
        r[0]: {
            "metric_id":         r[0],
            "name":              r[1],
            "department":        r[2],
            "description":       r[3],
            "aggregation":       r[4],
            "grain":             r[5],
            "unit":              r[6],
            "status":            r[7],
            "data_quality":      r[8],
            "refresh_frequency": r[9],
            "expression":        r[10],
            "source_table":      r[11],
        }
        for r in con.execute("""
            SELECT m.metric_id, m.name, m.department, m.description,
                   m.aggregation, m.grain, m.unit, m.status,
                   m.data_quality, m.refresh_frequency,
                   v.expression, v.source_table
            FROM metric m
            LEFT JOIN metric_version v
              ON v.metric_id = m.metric_id AND v.valid_to IS NULL
            ORDER BY m.metric_id
        """).fetchall()
    }

    cols: dict[str, list] = defaultdict(list)
    for r in con.execute("""
        SELECT mc.metric_id, mc.source_id, mc.column_name, mc.role, ds.table_name
        FROM metric_column mc
        JOIN data_source ds ON ds.source_id = mc.source_id
    """).fetchall():
        cols[r[0]].append({"source_id": r[1], "column": r[2], "role": r[3], "table": r[4]})

    dims: dict[str, list] = defaultdict(list)
    for r in con.execute(
        "SELECT metric_id, name, role, required, join_path FROM metric_dimension"
    ).fetchall():
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
        "SELECT metric_id, dimension, rule, threshold, severity FROM metric_quality"
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
            "SELECT source_id, warehouse, db_schema, table_name, full_ref FROM data_source"
        ).fetchall()
    }

    return dict(metrics=metrics, cols=cols, dims=dims, deps=deps,
                owners=owners, tags=tags, quality=quality,
                benchmarks=benchmarks, sources=sources)


# ── semantic model builder ────────────────────────────────────────────────────

def build_semantic_models(data: dict) -> dict[str, dict]:
    """
    One semantic model per data_source. Measures are one-per-metric that reads
    from that source via metric_column. Dimensions are collected from all metrics
    using the source and deduplicated.
    """
    metrics  = data["metrics"]
    cols     = data["cols"]
    dims_map = data["dims"]

    # map source_id → metric_ids that use it
    metrics_by_source: dict[str, list[str]] = defaultdict(list)
    for mid, col_list in cols.items():
        for sid in {c["source_id"] for c in col_list}:
            if mid not in metrics_by_source[sid]:
                metrics_by_source[sid].append(mid)

    sem_models: dict[str, dict] = {}
    for source_id, metric_ids in metrics_by_source.items():
        # infer entity from most common grain
        grain_counts: dict[str, int] = defaultdict(int)
        for mid in metric_ids:
            g = metrics[mid].get("grain") or ""
            if g:
                grain_counts[g] += 1
        best_grain = max(grain_counts, key=grain_counts.get) if grain_counts else "row"
        entity_name, entity_col = _infer_entity(best_grain)

        # one measure per metric
        measures = []
        for mid in sorted(metric_ids):
            m   = data["metrics"][mid]
            agg = m["aggregation"]
            if agg not in AGG_MAP:
                continue  # ratio/custom handled as derived metrics; skip measure
            metric_cols = [c for c in cols[mid] if c["source_id"] == source_id]
            num_cols    = [c for c in metric_cols if c["role"] == "numerator"]
            expr        = num_cols[0]["column"] if num_cols else metric_cols[0]["column"] if metric_cols else "1 -- TODO"
            measures.append({
                "name":        _slug(mid.split(".", 1)[1]),
                "agg":         AGG_MAP[agg],
                "expr":        expr,
                "description": m["description"],
            })

        # dimensions: collect + deduplicate across all metrics for this source
        seen_dims: set[str] = set()
        dimensions = []
        for mid in sorted(metric_ids):
            for d in dims_map.get(mid, []):
                dname = _slug(d["name"])
                if dname in seen_dims:
                    continue
                seen_dims.add(dname)
                dtype = "time" if d["role"] == "temporal" else "categorical"
                jp    = d.get("join_path") or ""
                # use only the column part of "table.col" join paths
                expr  = jp.split(".")[-1] if "." in jp else (jp or dname)
                entry: dict[str, Any] = {"name": dname, "type": dtype, "expr": expr}
                if dtype == "time":
                    entry["type_params"] = {"time_granularity": "day"}
                dimensions.append(entry)

        sem_models[source_id] = _compact({
            "name":        _slug(source_id),
            "description": f"Physical source: {source_id}",
            "model":       f"ref('{_slug(source_id)}')",
            "entities": [{"name": entity_name, "type": "primary", "expr": entity_col}],
            "dimensions": dimensions or None,
            "measures":   measures or None,
        })

    return sem_models


# ── metric builder ────────────────────────────────────────────────────────────

def _classify(mid: str, m: dict, data: dict) -> tuple[str, dict]:
    """Return (metricflow_type, type_params)."""
    agg  = m["aggregation"]
    deps = data["deps"].get(mid, [])
    cols = data["cols"].get(mid, [])

    has_source   = bool(cols)
    has_deps     = bool(deps)
    is_simple    = agg in AGG_MAP and has_source

    # ── simple: has lineage AND an aggregation MetricFlow understands ──
    if is_simple and not has_deps:
        measure_name = _slug(mid.split(".", 1)[1])
        return "simple", {"measure": {"name": measure_name}}

    # ── derived: has explicit dependency edges (or custom agg with deps) ──
    if has_deps:
        expr = _translate_formula(m.get("expression") or "", deps)
        # If translation made no progress, keep a TODO comment
        if expr == m.get("expression"):
            expr = f"# TODO translate: {expr}"
        input_metrics = [{"name": _slug(d["depends_on"].replace(".", "_"))} for d in deps]
        return "derived", _compact({"expr": expr, "metrics": input_metrics})

    # ── ratio attempt: agg=ratio, find numerator/denominator columns ──
    if agg == "ratio" and has_source:
        num = [c for c in cols if c["role"] == "numerator"]
        den = [c for c in cols if c["role"] == "denominator"]
        if num and den:
            return "ratio", {
                "numerator":   {"name": _slug(mid.split(".", 1)[1]) + "_num"},
                "denominator": {"name": _slug(mid.split(".", 1)[1]) + "_den"},
            }

    # ── fallback: derived with TODO, no metrics list to avoid MetricFlow validation errors ──
    expr = m.get("expression") or "# TODO"
    return "derived", {"expr": f"# TODO translate: {expr}"}


def build_metrics(data: dict, sem_models: dict) -> list[dict]:
    out = []
    for mid, m in data["metrics"].items():
        domain, _ = mid.split(".", 1)
        mtype, type_params = _classify(mid, m, data)

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

        entry = _compact({
            "name":        _slug(mid.replace(".", "_")),
            "label":       m["name"],
            "description": m["description"],
            "type":        mtype,
            "type_params": type_params,
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
        "description": "Auto-generated from metric_catalog.duckdb",
        "schema":      schema,
        "database":    database,
        "tables":      tables,
    }]


# ── YAML emission ─────────────────────────────────────────────────────────────

def _dump(obj: Any) -> str:
    return yaml.dump(obj, allow_unicode=True, default_flow_style=False,
                     sort_keys=False, indent=2, width=120)


def emit_files(sem_models: dict, metrics: list[dict],
               sources_data: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "_semantic_models.yml").write_text(
        _dump({"version": 2, "semantic_models": list(sem_models.values())}),
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
            _dump({"version": 2, "metrics": dm}), encoding="utf-8",
        )

    counts = {t: sum(1 for m in metrics if m["type"] == t)
              for t in ("simple", "derived", "ratio")}
    print(f"Semantic models : {len(sem_models)}  →  {out_dir / '_semantic_models.yml'}")
    print(f"Metrics total   : {len(metrics)}  "
          f"(simple={counts['simple']} derived={counts['derived']} ratio={counts['ratio']})")
    for domain in sorted(by_domain):
        print(f"  {domain:20s} {len(by_domain[domain])} metrics  →  {domain}/_metrics.yml")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate dbt Semantic Layer YAML from metric_catalog.duckdb."
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
        data       = load_data(con)
        sem_models = build_semantic_models(data)
        metrics    = build_metrics(data, sem_models)
        sources    = build_sources(data, args.schema, args.database)
        emit_files(sem_models, metrics, sources, out_dir)
    finally:
        con.close()


if __name__ == "__main__":
    main()
