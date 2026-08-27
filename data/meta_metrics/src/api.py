#!/usr/bin/env python3
"""
FastAPI semantic catalog API over metric_catalog.duckdb.

Usage:
    uvicorn catalog_api:app --reload
    python catalog_api.py [--db path/to/metric_catalog.duckdb] [--port 8000]
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query

_DB_PATH = "data/metric_catalog.duckdb"
_con: duckdb.DuckDBPyConnection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _con
    _con = duckdb.connect(_DB_PATH, read_only=True)
    yield
    if _con:
        _con.close()


app = FastAPI(
    title="Metric Catalog API",
    description="Semantic metadata service over metric_catalog.duckdb.",
    version="2.0.0",
    lifespan=lifespan,
)


def _q(sql: str, params: list | None = None) -> list[dict]:
    result = _con.execute(sql, params or [])
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def _q1(sql: str, params: list | None = None) -> dict | None:
    rows = _q(sql, params)
    return rows[0] if rows else None


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/metrics", summary="List and filter metrics")
def list_metrics(
    domain:     str | None = Query(None, description="Domain prefix, e.g. 'finance'"),
    kind:       str | None = Query(None, description="metric_kind: base|derived|ratio"),
    status:     str | None = Query(None, description="active|deprecated|under_review|experimental"),
    entity:     str | None = Query(None, description="Filter by entity_id"),
    additivity: str | None = Query(None, description="additive|semi_additive|non_additive"),
    search:     str | None = Query(None, description="Full-text search on name/description"),
    limit:  int = Query(100, le=1000),
    offset: int = Query(0),
) -> list[dict]:
    where, params = [], []
    if domain:
        where.append("metric_id LIKE ?"); params.append(f"{domain}.%")
    if kind:
        where.append("metric_kind = ?"); params.append(kind)
    if status:
        where.append("status = ?"); params.append(status)
    if entity:
        where.append("entity_id = ?"); params.append(entity)
    if additivity:
        where.append("additivity = ?"); params.append(additivity)
    if search:
        where.append("(lower(name) LIKE ? OR lower(description) LIKE ?)")
        params += [f"%{search.lower()}%", f"%{search.lower()}%"]
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    return _q(
        f"SELECT * FROM metric_definition {clause} ORDER BY metric_id LIMIT ? OFFSET ?",
        params + [limit, offset],
    )


@app.get("/metrics/{metric_id}", summary="Full metric detail")
def get_metric(metric_id: str) -> dict[str, Any]:
    m = _q1("SELECT * FROM metric_definition WHERE metric_id = ?", [metric_id])
    if not m:
        raise HTTPException(404, f"Metric '{metric_id}' not found")

    m["implementations"] = _q(
        "SELECT * FROM metric_implementation WHERE metric_id = ? ORDER BY is_current DESC, version",
        [metric_id],
    )
    m["dimensions"] = _q("""
        SELECT d.dimension_id, d.name, d.dimension_type, md.role, md.required
        FROM metric_dimension md JOIN dimension d USING (dimension_id)
        WHERE md.metric_id = ?
    """, [metric_id])
    m["quality_contracts"] = _q(
        "SELECT * FROM quality_contract WHERE metric_id = ? ORDER BY dimension, rule",
        [metric_id],
    )
    m["dependencies"] = _q(
        "SELECT depends_on_metric_id, dependency_type, origin FROM metric_dependency WHERE metric_id = ?",
        [metric_id],
    )
    m["dependents"] = _q(
        "SELECT metric_id, dependency_type FROM metric_dependency WHERE depends_on_metric_id = ?",
        [metric_id],
    )
    m["owners"]  = _q("SELECT * FROM metric_owner WHERE metric_id = ?", [metric_id])
    m["tags"]    = [r["tag"] for r in _q("SELECT tag FROM metric_tag WHERE metric_id = ?", [metric_id])]
    m["aliases"] = [r["alias"] for r in _q("SELECT alias FROM metric_alias WHERE metric_id = ?", [metric_id])]
    m["usage"]   = _q1("SELECT * FROM metric_usage WHERE metric_id = ?", [metric_id])
    m["benchmark"] = _q1(
        "SELECT * FROM metric_benchmark WHERE metric_id = ? AND benchmark_type = 'default'",
        [metric_id],
    )
    if m.get("entity_id"):
        m["entity"] = _q1("SELECT * FROM entity WHERE entity_id = ?", [m["entity_id"]])
    if m.get("superseded_by"):
        m["superseded_by_metric"] = _q1(
            "SELECT metric_id, name, status FROM metric_definition WHERE metric_id = ?",
            [m["superseded_by"]],
        )
    return m


@app.get("/metrics/{metric_id}/dependencies", summary="Upstream dependency graph")
def get_dependencies(metric_id: str, depth: int = Query(5, le=10)) -> dict:
    if not _q1("SELECT 1 FROM metric_definition WHERE metric_id = ?", [metric_id]):
        raise HTTPException(404, f"Metric '{metric_id}' not found")
    visited: dict[str, list] = {}
    queue = [(metric_id, 0)]
    while queue:
        mid, d = queue.pop(0)
        if mid in visited or d > depth:
            continue
        deps = _q(
            "SELECT depends_on_metric_id, dependency_type, origin FROM metric_dependency WHERE metric_id = ?",
            [mid],
        )
        visited[mid] = deps
        for dep in deps:
            queue.append((dep["depends_on_metric_id"], d + 1))
    return {"metric_id": metric_id, "depth": depth, "graph": visited}


@app.get("/metrics/{metric_id}/dependents", summary="Downstream metrics (impact analysis)")
def get_dependents(metric_id: str, depth: int = Query(5, le=10)) -> dict:
    if not _q1("SELECT 1 FROM metric_definition WHERE metric_id = ?", [metric_id]):
        raise HTTPException(404, f"Metric '{metric_id}' not found")
    visited: dict[str, list] = {}
    queue = [(metric_id, 0)]
    while queue:
        mid, d = queue.pop(0)
        if mid in visited or d > depth:
            continue
        dependents = _q(
            "SELECT metric_id, dependency_type FROM metric_dependency WHERE depends_on_metric_id = ?",
            [mid],
        )
        visited[mid] = dependents
        for dep in dependents:
            queue.append((dep["metric_id"], d + 1))
    return {"metric_id": metric_id, "depth": depth, "graph": visited}


@app.get("/metrics/{metric_id}/lineage", summary="Column-level lineage with provenance")
def get_lineage(metric_id: str) -> dict:
    if not _q1("SELECT 1 FROM metric_definition WHERE metric_id = ?", [metric_id]):
        raise HTTPException(404, f"Metric '{metric_id}' not found")
    return {
        "metric_id": metric_id,
        "columns": _q("""
            SELECT ic.impl_id, ic.dataset_id, ds.table_name, ic.column_name,
                   ic.role, ic.origin, ic.confidence, ic.inference_rule
            FROM impl_column ic
            JOIN metric_implementation mi ON mi.impl_id = ic.impl_id
            JOIN dataset ds ON ds.dataset_id = ic.dataset_id
            WHERE mi.metric_id = ?
        """, [metric_id]),
        "joins": _q("""
            SELECT ij.impl_id, ij.left_dataset_id, ij.right_dataset_id,
                   ij.join_type, ij.condition, ij.origin, ij.confidence
            FROM impl_join ij
            JOIN metric_implementation mi ON mi.impl_id = ij.impl_id
            WHERE mi.metric_id = ?
        """, [metric_id]),
    }


# ── Entities ──────────────────────────────────────────────────────────────────

@app.get("/entities", summary="List all entities")
def list_entities() -> list[dict]:
    return _q("SELECT * FROM entity ORDER BY entity_id")


@app.get("/entities/{entity_id}/metrics", summary="Metrics at this entity grain")
def get_entity_metrics(entity_id: str) -> list[dict]:
    if not _q1("SELECT 1 FROM entity WHERE entity_id = ?", [entity_id]):
        raise HTTPException(404, f"Entity '{entity_id}' not found")
    return _q("""
        SELECT metric_id, name, metric_kind, aggregation, additivity, status
        FROM metric_definition WHERE entity_id = ?
        ORDER BY metric_id
    """, [entity_id])


# ── Dimensions ────────────────────────────────────────────────────────────────

@app.get("/dimensions", summary="List all canonical dimensions")
def list_dimensions() -> list[dict]:
    return _q("SELECT * FROM dimension ORDER BY dimension_id")


@app.get("/dimensions/{dimension_id}/metrics", summary="Metrics that use this dimension")
def get_dimension_metrics(dimension_id: str) -> list[dict]:
    if not _q1("SELECT 1 FROM dimension WHERE dimension_id = ?", [dimension_id]):
        raise HTTPException(404, f"Dimension '{dimension_id}' not found")
    return _q("""
        SELECT md.metric_id, m.name, md.role, md.required
        FROM metric_dimension md
        JOIN metric_definition m ON m.metric_id = md.metric_id
        WHERE md.dimension_id = ?
        ORDER BY md.metric_id
    """, [dimension_id])


# ── Quality ───────────────────────────────────────────────────────────────────

@app.get("/quality/contracts", summary="List quality contracts")
def list_quality_contracts(
    metric_id: str | None = None,
    severity:  str | None = None,
    dimension: str | None = None,
    origin:    str | None = None,
) -> list[dict]:
    where, params = [], []
    if metric_id:
        where.append("metric_id = ?"); params.append(metric_id)
    if severity:
        where.append("severity = ?"); params.append(severity)
    if dimension:
        where.append("dimension = ?"); params.append(dimension)
    if origin:
        where.append("origin = ?"); params.append(origin)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    return _q(f"SELECT * FROM quality_contract {clause} ORDER BY metric_id, dimension, rule", params)


@app.post("/quality/runs", summary="Record a quality check result", status_code=201)
def record_quality_run(
    contract_id:        str,
    run_id:             str,
    run_at:             str,
    status:             str,
    observed_value:     str | None = None,
    expected_threshold: str | None = None,
    execution_context:  str | None = None,
) -> dict:
    if not _q1("SELECT 1 FROM quality_contract WHERE contract_id = ?", [contract_id]):
        raise HTTPException(404, f"Contract '{contract_id}' not found")
    _con.execute("""
        INSERT INTO quality_run VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [run_id, contract_id, run_at, observed_value, expected_threshold,
          status, execution_context])
    return {"run_id": run_id, "status": "recorded"}


# ── Validation ────────────────────────────────────────────────────────────────

@app.get("/validate", summary="Run graph integrity checks")
def run_validation() -> dict:
    import validate as v
    results = {}
    total = 0
    for name, fn in v.CHECKS:
        issues = fn(_con)
        results[name] = {"count": len(issues), "issues": issues[:10]}
        total += len(issues)
    return {"total_issues": total, "checks": results}


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats", summary="Catalog statistics")
def get_stats() -> dict:
    return {
        "metrics":           _q1("SELECT COUNT(*) AS n FROM metric_definition")["n"],
        "entities":          _q1("SELECT COUNT(*) AS n FROM entity")["n"],
        "dimensions":        _q1("SELECT COUNT(*) AS n FROM dimension")["n"],
        "datasets":          _q1("SELECT COUNT(*) AS n FROM dataset")["n"],
        "implementations":   _q1("SELECT COUNT(*) AS n FROM metric_implementation")["n"],
        "quality_contracts": _q1("SELECT COUNT(*) AS n FROM quality_contract")["n"],
        "dependency_edges":  _q1("SELECT COUNT(*) AS n FROM metric_dependency")["n"],
        "by_kind":       _q("SELECT metric_kind, COUNT(*) AS n FROM metric_definition GROUP BY 1 ORDER BY 1"),
        "by_additivity": _q("SELECT additivity, COUNT(*) AS n FROM metric_definition GROUP BY 1 ORDER BY 1"),
        "by_status":     _q("SELECT status, COUNT(*) AS n FROM metric_definition GROUP BY 1 ORDER BY 1"),
        "lineage_by_origin": _q("SELECT origin, COUNT(*) AS n FROM impl_column GROUP BY 1"),
    }


def main() -> None:
    import argparse
    import uvicorn
    p = argparse.ArgumentParser(description="Run the Metric Catalog API.")
    p.add_argument("--db",   default="data/metric_catalog.duckdb")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    global _DB_PATH
    _DB_PATH = args.db
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
