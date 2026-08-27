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

_CATALOG_PATH = "data/metric_catalog.duckdb"
_RUNTIME_PATH = "data/runtime.duckdb"
_con: duckdb.DuckDBPyConnection | None = None          # catalog — read-only
_runtime_con: duckdb.DuckDBPyConnection | None = None  # runtime observations — writable

_RUNTIME_DDL = """
CREATE TABLE IF NOT EXISTS quality_run (
    run_id             VARCHAR PRIMARY KEY,
    contract_id        VARCHAR NOT NULL,   -- FK validated at application level
    run_at             TIMESTAMP NOT NULL,
    observed_value     VARCHAR,
    expected_threshold VARCHAR,
    status             VARCHAR NOT NULL,
    execution_context  VARCHAR
);
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _con, _runtime_con
    _con         = duckdb.connect(_CATALOG_PATH, read_only=True)
    _runtime_con = duckdb.connect(_RUNTIME_PATH)
    _runtime_con.execute(_RUNTIME_DDL)
    yield
    for c in (_con, _runtime_con):
        if c:
            c.close()


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
    kind:       str | None = Query(None, description="derivation_type: base|derived"),
    metric_type: str | None = Query(None, description="metric_type: scalar|ratio|cumulative|snapshot"),
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
        where.append("derivation_type = ?"); params.append(kind)
    if metric_type:
        where.append("metric_type = ?"); params.append(metric_type)
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
        SELECT metric_id, name, derivation_type, metric_type, aggregation, additivity, status
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
    _runtime_con.execute("""
        INSERT INTO quality_run VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [run_id, contract_id, run_at, observed_value, expected_threshold,
          status, execution_context])
    return {"run_id": run_id, "status": "recorded"}


# ── Cubes ─────────────────────────────────────────────────────────────────────

@app.get("/cubes", summary="List all analytical cubes")
def list_cubes(cube_type: str | None = Query(None)) -> list[dict]:
    where = "WHERE cube_type = ?" if cube_type else ""
    params = [cube_type] if cube_type else []
    return _q(f"SELECT *, (SELECT COUNT(*) FROM cube_metric cm WHERE cm.cube_id = analytical_cube.cube_id) AS metric_count FROM analytical_cube {where} ORDER BY cube_id", params)


@app.get("/cubes/{cube_id}", summary="Cube detail with metrics, dimensions and datasets")
def get_cube(cube_id: str) -> dict:
    cube = _q1("SELECT * FROM analytical_cube WHERE cube_id = ?", [cube_id])
    if not cube:
        raise HTTPException(404, f"Cube '{cube_id}' not found")
    cube["metrics"] = _q("""
        SELECT cm.metric_id, cm.role, cm.rollup_entity_id, cm.reason,
               m.name, m.derivation_type, m.metric_type, m.additivity, m.entity_id
        FROM cube_metric cm
        JOIN metric_definition m ON m.metric_id = cm.metric_id
        WHERE cm.cube_id = ?
        ORDER BY cm.role, cm.metric_id
    """, [cube_id])
    cube["dimensions"] = _q("""
        SELECT cd.dimension_id, d.name, d.dimension_type
        FROM cube_dimension cd
        JOIN dimension d ON d.dimension_id = cd.dimension_id
        WHERE cd.cube_id = ?
    """, [cube_id])
    cube["datasets"] = _q("""
        SELECT cds.dataset_id, cds.rollup_entity_id, ds.table_name, ds.layer
        FROM cube_dataset cds
        JOIN dataset ds ON ds.dataset_id = cds.dataset_id
        WHERE cds.cube_id = ?
    """, [cube_id])
    cube["entity"] = (
        _q1("SELECT * FROM entity WHERE entity_id = ?", [cube["analytical_entity_id"]])
        if cube.get("analytical_entity_id") else None
    )
    return cube


@app.get("/cubes/{cube_id}/metrics", summary="Metrics in a cube")
def get_cube_metrics(cube_id: str) -> list[dict]:
    if not _q1("SELECT 1 FROM analytical_cube WHERE cube_id = ?", [cube_id]):
        raise HTTPException(404, f"Cube '{cube_id}' not found")
    return _q("""
        SELECT cm.metric_id, cm.role, cm.rollup_entity_id, cm.reason,
               m.name, m.derivation_type, m.metric_type, m.additivity,
               m.entity_id, m.display_grain, m.unit
        FROM cube_metric cm
        JOIN metric_definition m ON m.metric_id = cm.metric_id
        WHERE cm.cube_id = ?
        ORDER BY cm.role, cm.metric_id
    """, [cube_id])


@app.get("/cubes/{cube_id}/dimensions", summary="Dimensions in a cube")
def get_cube_dimensions(cube_id: str) -> list[dict]:
    if not _q1("SELECT 1 FROM analytical_cube WHERE cube_id = ?", [cube_id]):
        raise HTTPException(404, f"Cube '{cube_id}' not found")
    return _q("""
        SELECT cd.dimension_id, d.name, d.dimension_type, d.entity_id, d.default_expr
        FROM cube_dimension cd
        JOIN dimension d ON d.dimension_id = cd.dimension_id
        WHERE cd.cube_id = ?
    """, [cube_id])


@app.get("/cubes/{cube_id}/datasets", summary="Datasets in a cube")
def get_cube_datasets(cube_id: str) -> list[dict]:
    if not _q1("SELECT 1 FROM analytical_cube WHERE cube_id = ?", [cube_id]):
        raise HTTPException(404, f"Cube '{cube_id}' not found")
    return _q("""
        SELECT cds.dataset_id, cds.rollup_entity_id, ds.table_name, ds.layer, ds.full_ref
        FROM cube_dataset cds
        JOIN dataset ds ON ds.dataset_id = cds.dataset_id
        WHERE cds.cube_id = ?
    """, [cube_id])


@app.get("/metrics/{metric_id}/cubes", summary="Cubes that contain this metric")
def get_metric_cubes(metric_id: str) -> list[dict]:
    if not _q1("SELECT 1 FROM metric_definition WHERE metric_id = ?", [metric_id]):
        raise HTTPException(404, f"Metric '{metric_id}' not found")
    return _q("""
        SELECT cm.cube_id, cm.role, cm.rollup_entity_id, cm.reason,
               ac.name, ac.cube_type, ac.analytical_entity_id
        FROM cube_metric cm
        JOIN analytical_cube ac ON ac.cube_id = cm.cube_id
        WHERE cm.metric_id = ?
    """, [metric_id])


@app.get("/entities/{entity_id}/cube", summary="The analytical cube anchored at this entity")
def get_entity_cube(entity_id: str) -> dict | None:
    if not _q1("SELECT 1 FROM entity WHERE entity_id = ?", [entity_id]):
        raise HTTPException(404, f"Entity '{entity_id}' not found")
    return _q1("SELECT * FROM analytical_cube WHERE analytical_entity_id = ?", [entity_id])


@app.get("/cube-analysis", summary="Cube cover statistics and cross-cube dependency summary")
def get_cube_analysis() -> dict:
    cubes = _q("SELECT cube_id, cube_type, analytical_entity_id, (SELECT COUNT(*) FROM cube_metric cm WHERE cm.cube_id = analytical_cube.cube_id) AS metric_count FROM analytical_cube ORDER BY cube_id")
    total_metrics = _q1("SELECT COUNT(*) AS n FROM metric_definition WHERE status != 'deprecated'")["n"]
    covered = _q1("SELECT COUNT(DISTINCT metric_id) AS n FROM cube_metric")["n"]
    cross_deps = _q1("""
        SELECT COUNT(*) AS n
        FROM cube_metric cm
        JOIN metric_dependency md ON md.metric_id = cm.metric_id
        JOIN cube_metric cm2 ON cm2.metric_id = md.depends_on_metric_id
        WHERE cm.cube_id <> cm2.cube_id
    """)["n"]
    type_dist = _q("SELECT cube_type, COUNT(*) AS n FROM analytical_cube GROUP BY 1 ORDER BY 1")
    entity_rels = _q1("SELECT COUNT(*) AS n FROM entity_relation")["n"]
    rollup_safe = _q1("SELECT COUNT(*) AS n FROM entity_relation WHERE rollup_safe = true")["n"]
    return {
        "cube_count":             len(cubes),
        "metrics_total":          total_metrics,
        "metrics_covered":        covered,
        "metrics_uncovered":      total_metrics - covered,
        "cross_cube_dependencies":cross_deps,
        "entity_relations":       entity_rels,
        "rollup_safe_relations":  rollup_safe,
        "cube_types":             type_dist,
        "cubes":                  cubes,
    }




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
        "by_kind":       _q("SELECT derivation_type, metric_type, COUNT(*) AS n FROM metric_definition GROUP BY 1, 2 ORDER BY 1, 2"),
        "by_additivity": _q("SELECT additivity, COUNT(*) AS n FROM metric_definition GROUP BY 1 ORDER BY 1"),
        "by_status":     _q("SELECT status, COUNT(*) AS n FROM metric_definition GROUP BY 1 ORDER BY 1"),
        "lineage_by_origin": _q("SELECT origin, COUNT(*) AS n FROM impl_column GROUP BY 1"),
    }


def main() -> None:
    import argparse
    import uvicorn
    p = argparse.ArgumentParser(description="Run the Metric Catalog API.")
    p.add_argument("--db",          default="data/metric_catalog.duckdb")
    p.add_argument("--runtime-db",  default="data/runtime.duckdb",
                   help="Writable DB for quality runs and observations")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    global _CATALOG_PATH, _RUNTIME_PATH
    _CATALOG_PATH = args.db
    _RUNTIME_PATH = args.runtime_db
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
