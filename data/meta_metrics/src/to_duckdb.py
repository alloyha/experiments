
#!/usr/bin/env python3
"""
Transform the Metric Catalog JSON into a normalized DuckDB semantic model.

Usage:
    python metric_catalog_to_duckdb.py metric_catalog_v1.json metric_catalog.duckdb

Outputs:
    - DuckDB database
    - relational schema suitable for ERD tooling
    - optional Graphviz DOT ERD
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower(), flags=re.UNICODE).strip("_")


DDL = """
DROP TABLE IF EXISTS quality_run;
DROP TABLE IF EXISTS quality_contract;
DROP TABLE IF EXISTS impl_join;
DROP TABLE IF EXISTS impl_column;
DROP TABLE IF EXISTS metric_implementation;
DROP TABLE IF EXISTS metric_dimension;
DROP TABLE IF EXISTS metric_dependency;
DROP TABLE IF EXISTS metric_relation;
DROP TABLE IF EXISTS metric_permission;
DROP TABLE IF EXISTS metric_execution;
DROP TABLE IF EXISTS metric_benchmark;
DROP TABLE IF EXISTS metric_usage;
DROP TABLE IF EXISTS metric_change;
DROP TABLE IF EXISTS metric_owner;
DROP TABLE IF EXISTS metric_period;
DROP TABLE IF EXISTS metric_tag;
DROP TABLE IF EXISTS metric_alias;
DROP TABLE IF EXISTS metric_definition;
DROP TABLE IF EXISTS dimension;
DROP TABLE IF EXISTS dataset;
DROP TABLE IF EXISTS entity;

-- First-class entities: grain as an explicit, named object
CREATE TABLE entity (
    entity_id    VARCHAR PRIMARY KEY,
    name         VARCHAR NOT NULL,
    description  VARCHAR,
    pk_column    VARCHAR NOT NULL,
    grain_aliases VARCHAR[]
);

-- Global canonical dimensions with stable IDs
CREATE TABLE dimension (
    dimension_id   VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,
    description    VARCHAR,
    dimension_type VARCHAR NOT NULL DEFAULT 'categorical',
    entity_id      VARCHAR,          -- informational; no FK to allow partial coverage
    default_expr   VARCHAR
);

-- Logical/physical datasets, decoupled from metric semantics
CREATE TABLE dataset (
    dataset_id   VARCHAR PRIMARY KEY,
    name         VARCHAR NOT NULL,
    layer        VARCHAR NOT NULL DEFAULT 'unknown',
    engine       VARCHAR,
    db_catalog   VARCHAR,
    db_schema    VARCHAR,
    table_name   VARCHAR NOT NULL,
    full_ref     VARCHAR NOT NULL,
    warehouse    VARCHAR
);

-- Metric definition: pure business semantics, engine-agnostic
CREATE TABLE metric_definition (
    metric_id               VARCHAR PRIMARY KEY,
    name                    VARCHAR NOT NULL,
    department              VARCHAR NOT NULL,
    description             VARCHAR NOT NULL,
    metric_kind             VARCHAR NOT NULL DEFAULT 'base',
    aggregation             VARCHAR NOT NULL,
    entity_id               VARCHAR REFERENCES entity(entity_id),
    unit                    VARCHAR,
    status                  VARCHAR NOT NULL DEFAULT 'active',
    additivity              VARCHAR NOT NULL DEFAULT 'additive',
    non_additive_dimensions VARCHAR[],
    time_grain              VARCHAR,
    default_period          VARCHAR,
    data_quality            VARCHAR,
    refresh_frequency       VARCHAR,
    superseded_by           VARCHAR,       -- self-ref; integrity validated at application layer
    deprecated_at           DATE,
    deprecation_reason      VARCHAR
);

-- Metric implementation: engine-specific expression, separated from definition
CREATE TABLE metric_implementation (
    impl_id      VARCHAR PRIMARY KEY,
    metric_id    VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    engine       VARCHAR NOT NULL DEFAULT 'pseudocode',
    expression   VARCHAR NOT NULL,
    language     VARCHAR NOT NULL DEFAULT 'pseudocode',
    source_table VARCHAR,
    version      VARCHAR NOT NULL DEFAULT '1.0',
    valid_from   DATE,
    valid_to     DATE,
    is_current   BOOLEAN NOT NULL DEFAULT true
);

-- Column-level lineage with provenance (declared|inferred|generated)
CREATE TABLE impl_column (
    impl_id        VARCHAR NOT NULL REFERENCES metric_implementation(impl_id),
    dataset_id     VARCHAR NOT NULL REFERENCES dataset(dataset_id),
    column_name    VARCHAR NOT NULL,
    role           VARCHAR NOT NULL,
    origin         VARCHAR NOT NULL DEFAULT 'inferred',
    confidence     REAL,
    inference_rule VARCHAR,
    PRIMARY KEY (impl_id, dataset_id, column_name, role)
);

-- Join lineage with provenance
CREATE TABLE impl_join (
    impl_id          VARCHAR NOT NULL REFERENCES metric_implementation(impl_id),
    left_dataset_id  VARCHAR NOT NULL REFERENCES dataset(dataset_id),
    right_dataset_id VARCHAR NOT NULL REFERENCES dataset(dataset_id),
    join_type        VARCHAR NOT NULL DEFAULT 'INNER',
    condition        VARCHAR NOT NULL,
    origin           VARCHAR NOT NULL DEFAULT 'inferred',
    confidence       REAL,
    PRIMARY KEY (impl_id, left_dataset_id, right_dataset_id)
);

-- Computational dependency DAG with provenance
CREATE TABLE metric_dependency (
    metric_id            VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    depends_on_metric_id VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    dependency_type      VARCHAR NOT NULL DEFAULT 'computational',
    origin               VARCHAR NOT NULL DEFAULT 'declared',
    PRIMARY KEY (metric_id, depends_on_metric_id),
    CHECK (metric_id <> depends_on_metric_id)
);

-- Semantic relations (related/alternative/supersedes)
CREATE TABLE metric_relation (
    metric_id         VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    related_metric_id VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    relation_type     VARCHAR NOT NULL DEFAULT 'related',
    PRIMARY KEY (metric_id, related_metric_id, relation_type),
    CHECK (metric_id <> related_metric_id)
);

-- Bridge: metric_definition → global canonical dimension
CREATE TABLE metric_dimension (
    metric_id    VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    dimension_id VARCHAR NOT NULL REFERENCES dimension(dimension_id),
    role         VARCHAR NOT NULL DEFAULT 'grouping',
    required     BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (metric_id, dimension_id, role)
);

-- Quality contract: definition (not observation)
CREATE TABLE quality_contract (
    contract_id VARCHAR PRIMARY KEY,
    metric_id   VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    dimension   VARCHAR NOT NULL,
    rule        VARCHAR NOT NULL,
    threshold   VARCHAR,
    severity    VARCHAR NOT NULL DEFAULT 'warning',
    origin      VARCHAR NOT NULL DEFAULT 'generated',
    UNIQUE (metric_id, dimension, rule)
);

-- Quality run: observation, separated from contract definition
CREATE TABLE quality_run (
    run_id             VARCHAR PRIMARY KEY,
    contract_id        VARCHAR NOT NULL REFERENCES quality_contract(contract_id),
    run_at             TIMESTAMP NOT NULL,
    observed_value     VARCHAR,
    expected_threshold VARCHAR,
    status             VARCHAR NOT NULL,
    execution_context  VARCHAR
);

CREATE TABLE metric_alias (
    metric_id VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    alias     VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, alias)
);

CREATE TABLE metric_tag (
    metric_id VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    tag       VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, tag)
);

CREATE TABLE metric_period (
    metric_id VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    period    VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, period)
);

CREATE TABLE metric_owner (
    metric_id  VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    owner_type VARCHAR NOT NULL DEFAULT 'business',
    team       VARCHAR,
    contact    VARCHAR,
    PRIMARY KEY (metric_id, owner_type)
);

CREATE TABLE metric_change (
    metric_id   VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    change_date DATE,
    change      VARCHAR,
    PRIMARY KEY (metric_id, change_date, change)
);

CREATE TABLE metric_usage (
    metric_id         VARCHAR PRIMARY KEY REFERENCES metric_definition(metric_id),
    when_to_use       VARCHAR,
    example_questions VARCHAR[]
);

CREATE TABLE metric_benchmark (
    metric_id      VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    benchmark_type VARCHAR NOT NULL DEFAULT 'default',
    target         DOUBLE,
    range_low      DOUBLE,
    range_high     DOUBLE,
    population     VARCHAR,
    period         VARCHAR,
    source         VARCHAR,
    valid_from     DATE,
    valid_to       DATE,
    PRIMARY KEY (metric_id, benchmark_type)
);

CREATE TABLE metric_execution (
    metric_id      VARCHAR PRIMARY KEY REFERENCES metric_definition(metric_id),
    endpoint       VARCHAR,
    execution_cost VARCHAR,
    cacheable      BOOLEAN
);

CREATE TABLE metric_permission (
    metric_id  VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    permission VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, permission)
);
"""

def load_catalog(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def populate(con: duckdb.DuckDBPyConnection, catalog: dict) -> None:
    con.execute(DDL)

    # ── entities ──────────────────────────────────────────────────────────────
    for e in catalog.get("entities", []):
        con.execute(
            "INSERT INTO entity VALUES (?, ?, ?, ?, ?)",
            [e["entity_id"], e["name"], e.get("description"),
             e["pk_column"], e.get("grain_aliases", [])],
        )

    # ── canonical dimensions (collected across all metrics) ───────────────────
    seen_dims: dict[str, dict] = {}
    for m in catalog["metrics"]:
        for d in m.get("dimensions", []):
            did = _slug(d["name"])
            if did in seen_dims:
                continue
            jp = d.get("join_path") or ""
            seen_dims[did] = {
                "id":           did,
                "name":         d["name"],
                "dim_type":     "time" if d.get("role") == "temporal" else "categorical",
                "entity_id":    jp.split(".")[0] if "." in jp else None,
                "default_expr": jp.split(".")[-1] if "." in jp else (jp or did),
            }
    for d in seen_dims.values():
        con.execute(
            "INSERT INTO dimension VALUES (?, ?, NULL, ?, ?, ?)",
            [d["id"], d["name"], d["dim_type"], d["entity_id"], d["default_expr"]],
        )

    # ── datasets (pre-scan all lineage) ──────────────────────────────────────
    inserted_datasets: set[str] = set()

    def _ensure_dataset(sid: str, tbl: str | None = None) -> None:
        if sid in inserted_datasets:
            return
        inserted_datasets.add(sid)
        table_name = tbl or sid.split(".")[-1]
        con.execute(
            "INSERT INTO dataset VALUES (?, ?, 'unknown', NULL, NULL, NULL, ?, ?, NULL)",
            [sid, sid, table_name, sid],
        )

    for m in catalog["metrics"]:
        for col in m.get("lineage", {}).get("columns", []):
            _ensure_dataset(col["source"], col.get("table"))
        for j in m.get("lineage", {}).get("joins", []):
            _ensure_dataset(j["left"])
            _ensure_dataset(j["right"])

    # ── metric definitions ────────────────────────────────────────────────────
    for m in catalog["metrics"]:
        mid = m["id"]
        con.execute("""
            INSERT INTO metric_definition VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            mid, m["name"], m["department"], m["description"],
            m.get("metric_kind", "base"), m["aggregation"],
            m.get("entity_id"), m.get("unit"), m["status"],
            m.get("additivity", "additive"),
            m.get("non_additive_dimensions") or [],
            m.get("time_grain"), m.get("default_period"),
            m.get("data_quality"), m.get("refresh_frequency"),
            m.get("superseded_by"), m.get("deprecated_at"),
            m.get("deprecation_reason"),
        ])

        if aliases := [(mid, x) for x in m.get("aliases", [])]:
            con.executemany("INSERT INTO metric_alias VALUES (?, ?)", aliases)
        if tags := [(mid, x) for x in m.get("tags", [])]:
            con.executemany("INSERT INTO metric_tag VALUES (?, ?)", tags)
        if periods := [(mid, p) for p in m.get("supported_periods", [])]:
            con.executemany("INSERT INTO metric_period VALUES (?, ?)", periods)

        owner = m.get("owner", {})
        owners_list = m.get("owners") or ([{"type": "business", **owner}] if owner else [])
        if owner_rows := [(mid, o.get("type", "business"), o.get("team"), o.get("contact"))
                          for o in owners_list]:
            con.executemany("INSERT INTO metric_owner VALUES (?, ?, ?, ?)", owner_rows)

        if changes := [(mid, c.get("date"), c.get("change")) for c in m.get("change_log", [])]:
            con.executemany("INSERT INTO metric_change VALUES (?, ?, ?)", changes)

        usage = m.get("usage_context", {})
        con.execute("INSERT INTO metric_usage VALUES (?, ?, ?)",
                    [mid, usage.get("when_to_use"), usage.get("example_questions", [])])

        if bench := usage.get("benchmarks") or {}:
            if bench.get("type") or bench.get("target"):
                con.execute("""
                    INSERT INTO metric_benchmark VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [mid, bench.get("type", "default"), bench.get("target"),
                      bench.get("range_low"), bench.get("range_high"),
                      bench.get("population"), bench.get("period"),
                      bench.get("source"), bench.get("valid_from"), bench.get("valid_to")])

        access = m.get("access", {})
        if access:
            con.execute("INSERT INTO metric_execution VALUES (?, ?, ?, ?)",
                        [mid, access.get("endpoint"), access.get("execution_cost"),
                         access.get("cacheable")])
            if perm := access.get("requires_permission"):
                con.execute("INSERT INTO metric_permission VALUES (?, ?)", [mid, perm])

        if relations := [(mid, x) for x in usage.get("related_metrics", [])]:
            con.executemany("INSERT INTO metric_relation VALUES (?, ?, 'related')", relations)

        for d in m.get("dimensions", []):
            did = _slug(d["name"])
            if did in seen_dims:
                con.execute("INSERT OR IGNORE INTO metric_dimension VALUES (?, ?, ?, ?)",
                            [mid, did, d.get("role", "grouping"), d.get("required", False)])

    # ── implementations ───────────────────────────────────────────────────────
    for m in catalog["metrics"]:
        mid     = m["id"]
        formula = m.get("formula", {})
        if not formula:
            continue
        version = m.get("version", "1.0")
        impl_id = f"{mid}:v{version}"
        con.execute("""
            INSERT INTO metric_implementation VALUES
            (?, ?, 'pseudocode', ?, ?, ?, ?, NULL, NULL, true)
        """, [impl_id, mid, formula.get("expression"),
              formula.get("language", "pseudocode"),
              formula.get("source_table"), version])

        for col in m.get("lineage", {}).get("columns", []):
            sid = col["source"]
            _ensure_dataset(sid, col.get("table"))
            con.execute("""
                INSERT OR IGNORE INTO impl_column VALUES
                (?, ?, ?, ?, 'inferred', 0.7, 'regex_table_col')
            """, [impl_id, sid, col["column"], col["role"]])

        for j in m.get("lineage", {}).get("joins", []):
            _ensure_dataset(j["left"])
            _ensure_dataset(j["right"])
            con.execute("""
                INSERT OR IGNORE INTO impl_join VALUES
                (?, ?, ?, ?, ?, 'inferred', 0.7)
            """, [impl_id, j["left"], j["right"], j.get("type", "INNER"), j["on"]])

    # ── quality contracts ─────────────────────────────────────────────────────
    for m in catalog["metrics"]:
        mid = m["id"]
        for q in m.get("quality", []):
            cid = f"{mid}:{q['dimension']}:{q['rule']}"
            con.execute(
                "INSERT INTO quality_contract VALUES (?, ?, ?, ?, ?, ?, 'generated')",
                [cid, mid, q["dimension"], q["rule"],
                 q.get("threshold"), q.get("severity", "warning")],
            )

    # ── dependency DAG (second pass — all definitions must exist first) ───────
    all_deps: list = []
    for m in catalog["metrics"]:
        mid = m["id"]
        for d in m.get("dependencies", []):
            all_deps.append((mid, d["depends_on"], d.get("type", "computational"), "declared"))
    if all_deps:
        con.executemany("INSERT INTO metric_dependency VALUES (?, ?, ?, ?)", all_deps)


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW metric_catalog AS
    SELECT
        m.*,
        i.impl_id,
        i.version,
        i.expression    AS formula_expression,
        i.language      AS formula_language,
        i.source_table,
        e.pk_column     AS entity_pk,
        o.owner_team,
        o.owner_contact,
        a.aliases,
        t.tags,
        p.supported_periods,
        u.when_to_use,
        u.example_questions,
        b.target        AS benchmark_target,
        b.range_low     AS benchmark_low,
        b.range_high    AS benchmark_high,
        x.endpoint,
        x.execution_cost,
        x.cacheable
    FROM metric_definition m
    LEFT JOIN metric_implementation i
        ON i.metric_id = m.metric_id AND i.is_current = true
    LEFT JOIN entity e ON e.entity_id = m.entity_id
    LEFT JOIN (
        SELECT metric_id, team AS owner_team, contact AS owner_contact
        FROM metric_owner WHERE owner_type = 'business'
    ) o ON o.metric_id = m.metric_id
    LEFT JOIN (
        SELECT metric_id, list(alias ORDER BY alias) AS aliases
        FROM metric_alias GROUP BY metric_id
    ) a ON a.metric_id = m.metric_id
    LEFT JOIN (
        SELECT metric_id, list(tag ORDER BY tag) AS tags
        FROM metric_tag GROUP BY metric_id
    ) t ON t.metric_id = m.metric_id
    LEFT JOIN (
        SELECT metric_id, list(period ORDER BY period) AS supported_periods
        FROM metric_period GROUP BY metric_id
    ) p ON p.metric_id = m.metric_id
    LEFT JOIN metric_usage u ON u.metric_id = m.metric_id
    LEFT JOIN metric_benchmark b
        ON b.metric_id = m.metric_id AND b.benchmark_type = 'default'
    LEFT JOIN metric_execution x ON x.metric_id = m.metric_id
    """)


def write_mermaid(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    tables = [
        "entity", "dimension", "dataset",
        "metric_definition", "metric_implementation",
        "impl_column", "impl_join",
        "metric_dependency", "metric_relation", "metric_dimension",
        "quality_contract", "quality_run",
        "metric_alias", "metric_tag", "metric_period", "metric_owner",
        "metric_change", "metric_usage", "metric_benchmark",
        "metric_execution", "metric_permission",
    ]

    relationships = [
        ("entity",             "metric_definition",    "defines entity for"),
        ("entity",             "dimension",            "context for"),
        ("metric_definition",  "metric_implementation","implemented by"),
        ("metric_definition",  "metric_dependency",    "depends on"),
        ("metric_definition",  "metric_relation",      "related to"),
        ("metric_definition",  "metric_dimension",     "grouped by"),
        ("metric_definition",  "quality_contract",     "governed by"),
        ("metric_definition",  "metric_alias",         "aliased as"),
        ("metric_definition",  "metric_tag",           "tagged"),
        ("metric_definition",  "metric_period",        "supports period"),
        ("metric_definition",  "metric_owner",         "owned by"),
        ("metric_definition",  "metric_change",        "changelog"),
        ("metric_definition",  "metric_usage",         "usage"),
        ("metric_definition",  "metric_benchmark",     "benchmark"),
        ("metric_definition",  "metric_execution",     "execution"),
        ("metric_definition",  "metric_permission",    "requires"),
        ("dimension",          "metric_dimension",     "used in"),
        ("metric_implementation","impl_column",        "reads column"),
        ("metric_implementation","impl_join",          "joins"),
        ("dataset",            "impl_column",          "sourced from"),
        ("dataset",            "impl_join",            "joined in"),
        ("quality_contract",   "quality_run",          "executed as"),
    ]

    lines = ["```mermaid", "erDiagram"]

    pk_cols = {"metric_definition": "metric_id", "dataset": "dataset_id",
               "entity": "entity_id", "dimension": "dimension_id",
               "metric_implementation": "impl_id", "quality_contract": "contract_id"}
    for table in tables:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        lines.append(f"    {table} {{")
        for col in cols:
            name, dtype = col[0], col[1]
            short_type = dtype.split("(")[0]
            pk = " PK" if pk_cols.get(table) == name else ""
            lines.append(f"        {short_type} {name}{pk}")
        lines.append("    }")

    lines.append("")
    for parent, child, label in relationships:
        lines.append(f'    {parent} ||--o{{ {child} : "{label}"')

    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python metric_catalog_to_duckdb.py "
            "metric_catalog_v1.json [output.duckdb] [output.md]"
        )

    input_path = Path(sys.argv[1])
    db_path  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("metric_catalog.duckdb")
    erd_path = Path(sys.argv[3]) if len(sys.argv) > 3 else db_path.with_suffix(".md")

    catalog = load_catalog(input_path)

    con = duckdb.connect(str(db_path))
    try:
        populate(con, catalog)
        create_views(con)
        write_mermaid(con, erd_path)

        count = con.execute("SELECT COUNT(*) FROM metric_definition").fetchone()[0]
        print(f"Loaded {count} metrics into {db_path}")
        print(f"ERD Mermaid source: {erd_path}")
        print("\nTop-level tables:")
        for row in con.execute("SHOW TABLES").fetchall():
            print(f"  - {row[0]}")
    finally:
        con.close()


if __name__ == "__main__":
    main()

