
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
import sys
from pathlib import Path

import duckdb


DDL = """
DROP TABLE IF EXISTS metric_join;
DROP TABLE IF EXISTS metric_column;
DROP TABLE IF EXISTS metric_dependency;
DROP TABLE IF EXISTS metric_quality;
DROP TABLE IF EXISTS metric_permission;
DROP TABLE IF EXISTS metric_execution;
DROP TABLE IF EXISTS metric_access;
DROP TABLE IF EXISTS metric_relation;
DROP TABLE IF EXISTS metric_benchmark;
DROP TABLE IF EXISTS metric_usage;
DROP TABLE IF EXISTS metric_change;
DROP TABLE IF EXISTS metric_owner;
DROP TABLE IF EXISTS metric_period;
DROP TABLE IF EXISTS metric_dimension;
DROP TABLE IF EXISTS metric_version;
DROP TABLE IF EXISTS metric_formula;
DROP TABLE IF EXISTS metric_tag;
DROP TABLE IF EXISTS metric_alias;
DROP TABLE IF EXISTS metric;
DROP TABLE IF EXISTS data_source;

CREATE TABLE metric (
    metric_id         VARCHAR PRIMARY KEY,
    name              VARCHAR NOT NULL,
    department        VARCHAR NOT NULL,
    description       VARCHAR NOT NULL,
    aggregation       VARCHAR NOT NULL,
    grain             VARCHAR NOT NULL,
    unit              VARCHAR,
    status            VARCHAR NOT NULL,
    refresh_frequency VARCHAR,
    data_quality      VARCHAR,
    default_period    VARCHAR
);

CREATE TABLE metric_alias (
    metric_id VARCHAR NOT NULL REFERENCES metric(metric_id),
    alias     VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, alias)
);

CREATE TABLE metric_tag (
    metric_id VARCHAR NOT NULL REFERENCES metric(metric_id),
    tag       VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, tag)
);

-- NULL valid_to means current/active version.
CREATE TABLE metric_version (
    metric_id    VARCHAR NOT NULL REFERENCES metric(metric_id),
    version      VARCHAR NOT NULL,
    expression   VARCHAR NOT NULL,
    language     VARCHAR NOT NULL,
    source_table VARCHAR,
    valid_from   DATE,
    valid_to     DATE,
    PRIMARY KEY (metric_id, version)
);

CREATE TABLE metric_dimension (
    metric_id VARCHAR NOT NULL REFERENCES metric(metric_id),
    name      VARCHAR NOT NULL,
    role      VARCHAR NOT NULL DEFAULT 'grouping',
    required  BOOLEAN NOT NULL DEFAULT false,
    join_path VARCHAR,
    PRIMARY KEY (metric_id, name)
);

CREATE TABLE metric_period (
    metric_id VARCHAR NOT NULL REFERENCES metric(metric_id),
    period    VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, period)
);

CREATE TABLE metric_owner (
    metric_id  VARCHAR NOT NULL REFERENCES metric(metric_id),
    owner_type VARCHAR NOT NULL DEFAULT 'business',
    team       VARCHAR,
    contact    VARCHAR,
    PRIMARY KEY (metric_id, owner_type)
);

CREATE TABLE metric_change (
    metric_id   VARCHAR NOT NULL REFERENCES metric(metric_id),
    change_date DATE,
    change      VARCHAR,
    PRIMARY KEY (metric_id, change_date, change)
);

CREATE TABLE metric_usage (
    metric_id         VARCHAR PRIMARY KEY REFERENCES metric(metric_id),
    when_to_use       VARCHAR,
    example_questions VARCHAR[]
);

CREATE TABLE metric_benchmark (
    metric_id      VARCHAR NOT NULL REFERENCES metric(metric_id),
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

-- Execution metadata; distinct from access control.
CREATE TABLE metric_execution (
    metric_id      VARCHAR PRIMARY KEY REFERENCES metric(metric_id),
    endpoint       VARCHAR,
    execution_cost VARCHAR,
    cacheable      BOOLEAN
);

CREATE TABLE metric_permission (
    metric_id  VARCHAR NOT NULL REFERENCES metric(metric_id),
    permission VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, permission)
);

-- Semantic relationships (related / alternative / supersedes).
CREATE TABLE metric_relation (
    metric_id         VARCHAR NOT NULL REFERENCES metric(metric_id),
    related_metric_id VARCHAR NOT NULL REFERENCES metric(metric_id),
    relation_type     VARCHAR NOT NULL DEFAULT 'related',
    PRIMARY KEY (metric_id, related_metric_id, relation_type),
    CHECK (metric_id <> related_metric_id)
);

-- Computational dependency DAG (distinct from semantic relations).
CREATE TABLE metric_dependency (
    metric_id             VARCHAR NOT NULL REFERENCES metric(metric_id),
    depends_on_metric_id  VARCHAR NOT NULL REFERENCES metric(metric_id),
    dependency_type       VARCHAR NOT NULL DEFAULT 'computational',
    PRIMARY KEY (metric_id, depends_on_metric_id),
    CHECK (metric_id <> depends_on_metric_id)
);

-- Structured quality contract; supersedes data_quality VARCHAR on metric.
CREATE TABLE metric_quality (
    metric_id VARCHAR NOT NULL REFERENCES metric(metric_id),
    dimension VARCHAR NOT NULL,
    rule      VARCHAR NOT NULL,
    threshold VARCHAR,
    severity  VARCHAR NOT NULL DEFAULT 'warning',
    PRIMARY KEY (metric_id, dimension, rule)
);

-- Physical data lineage: warehouse tables referenced across any metric.
CREATE TABLE data_source (
    source_id  VARCHAR PRIMARY KEY,
    warehouse  VARCHAR,
    db_catalog VARCHAR,
    db_schema  VARCHAR,
    table_name VARCHAR NOT NULL,
    full_ref   VARCHAR NOT NULL
);

-- Columns a metric reads, with their role in the computation.
CREATE TABLE metric_column (
    metric_id   VARCHAR NOT NULL REFERENCES metric(metric_id),
    source_id   VARCHAR NOT NULL REFERENCES data_source(source_id),
    column_name VARCHAR NOT NULL,
    role        VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, source_id, column_name, role)
);

CREATE TABLE metric_join (
    metric_id       VARCHAR NOT NULL REFERENCES metric(metric_id),
    left_source_id  VARCHAR NOT NULL REFERENCES data_source(source_id),
    right_source_id VARCHAR NOT NULL REFERENCES data_source(source_id),
    join_type       VARCHAR NOT NULL DEFAULT 'INNER',
    condition       VARCHAR NOT NULL,
    PRIMARY KEY (metric_id, left_source_id, right_source_id)
);
"""


def load_catalog(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def populate(con: duckdb.DuckDBPyConnection, catalog: dict) -> None:
    con.execute(DDL)

    all_deps: list = []  # collected for second-pass insert after all metric rows exist

    for m in catalog["metrics"]:
        mid = m["id"]

        con.execute("""
            INSERT INTO metric
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            mid, m["name"], m["department"], m["description"],
            m["aggregation"], m["grain"], m.get("unit"),
            m["status"], m.get("refresh_frequency"),
            m.get("data_quality"), m.get("default_period")
        ])

        if aliases := [(mid, x) for x in m.get("aliases", [])]:
            con.executemany("INSERT INTO metric_alias VALUES (?, ?)", aliases)

        if tags := [(mid, x) for x in m.get("tags", [])]:
            con.executemany("INSERT INTO metric_tag VALUES (?, ?)", tags)

        formula = m.get("formula", {})
        if formula:
            con.execute("""
                INSERT INTO metric_version
                VALUES (?, ?, ?, ?, ?, NULL, NULL)
            """, [
                mid,
                m.get("version", "1.0"),
                formula.get("expression"),
                formula.get("language"),
                formula.get("source_table"),
            ])

        if dims := [
            (mid, d["name"], d.get("role", "grouping"), d.get("required", False), d.get("join_path"))
            for d in m.get("dimensions", [])
        ]:
            con.executemany("INSERT INTO metric_dimension VALUES (?, ?, ?, ?, ?)", dims)

        if periods := [(mid, p) for p in m.get("supported_periods", [])]:
            con.executemany("INSERT INTO metric_period VALUES (?, ?)", periods)

        owner = m.get("owner", {})
        owners = m.get("owners") or ([{"type": "business", **owner}] if owner else [])
        if owner_rows := [(mid, o.get("type", "business"), o.get("team"), o.get("contact")) for o in owners]:
            con.executemany("INSERT INTO metric_owner VALUES (?, ?, ?, ?)", owner_rows)

        if changes := [(mid, c.get("date"), c.get("change")) for c in m.get("change_log", [])]:
            con.executemany("INSERT INTO metric_change VALUES (?, ?, ?)", changes)

        usage = m.get("usage_context", {})
        con.execute("""
            INSERT INTO metric_usage
            VALUES (?, ?, ?)
        """, [
            mid,
            usage.get("when_to_use"),
            usage.get("example_questions", [])
        ])

        benchmark = usage.get("benchmarks", {})
        if benchmark:
            con.execute("""
                INSERT INTO metric_benchmark
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                mid,
                benchmark.get("type", "default"),
                benchmark.get("target"),
                benchmark.get("range_low"),
                benchmark.get("range_high"),
                benchmark.get("population"),
                benchmark.get("period"),
                benchmark.get("source"),
                benchmark.get("valid_from"),
                benchmark.get("valid_to"),
            ])

        access = m.get("access", {})
        if access:
            con.execute(
                "INSERT INTO metric_execution VALUES (?, ?, ?, ?)",
                [mid, access.get("endpoint"), access.get("execution_cost"), access.get("cacheable")],
            )
            if perm := access.get("requires_permission"):
                con.execute("INSERT INTO metric_permission VALUES (?, ?)", [mid, perm])

        # The source schema calls these "related metrics or semantic
        # competitors". We preserve that meaning rather than pretending
        # they are dependency edges.
        if relations := [(mid, x) for x in usage.get("related_metrics", [])]:
            con.executemany("INSERT INTO metric_relation VALUES (?, ?, 'related')", relations)

        if deps := [(mid, d["depends_on"], d.get("type", "computational")) for d in m.get("dependencies", [])]:
            all_deps.extend(deps)

        if quality := [(mid, q["dimension"], q["rule"], q.get("threshold"), q.get("severity", "warning")) for q in m.get("quality", [])]:
            con.executemany("INSERT INTO metric_quality VALUES (?, ?, ?, ?, ?)", quality)

        lineage = m.get("lineage", {})
        for col in lineage.get("columns", []):
            sid = col["source"]
            con.execute(
                "INSERT OR IGNORE INTO data_source VALUES (?, ?, ?, ?, ?, ?)",
                [
                    sid,
                    col.get("warehouse"),
                    col.get("catalog"),
                    col.get("schema"),
                    col.get("table") or sid.split(".")[-1],
                    sid,
                ],
            )
        if cols := [
            (mid, c["source"], c["column"], c["role"])
            for c in lineage.get("columns", [])
        ]:
            con.executemany("INSERT INTO metric_column VALUES (?, ?, ?, ?)", cols)

        for j in lineage.get("joins", []):
            for sid in (j["left"], j["right"]):
                con.execute(
                    "INSERT OR IGNORE INTO data_source VALUES (?, NULL, NULL, NULL, ?, ?)",
                    [sid, sid.split(".")[-1], sid],
                )
        if joins := [
            (mid, j["left"], j["right"], j.get("type", "INNER"), j["on"])
            for j in lineage.get("joins", [])
        ]:
            con.executemany("INSERT INTO metric_join VALUES (?, ?, ?, ?, ?)", joins)

    if all_deps:
        con.executemany("INSERT INTO metric_dependency VALUES (?, ?, ?)", all_deps)


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE OR REPLACE VIEW metric_catalog AS
    SELECT
        m.*,
        v.version,
        v.expression AS formula_expression,
        v.language AS formula_language,
        v.source_table,
        o.owner_team,
        o.owner_contact,
        a.aliases,
        t.tags,
        p.supported_periods,
        u.when_to_use,
        u.example_questions,
        b.target AS benchmark_target,
        b.range_low AS benchmark_low,
        b.range_high AS benchmark_high,
        x.endpoint,
        x.execution_cost,
        x.cacheable
    FROM metric m
    LEFT JOIN metric_version v ON v.metric_id = m.metric_id AND v.valid_to IS NULL
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
    LEFT JOIN metric_benchmark b ON b.metric_id = m.metric_id AND b.benchmark_type = 'default'
    LEFT JOIN metric_execution x ON x.metric_id = m.metric_id
    """)


def write_mermaid(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    tables = [
        "metric", "metric_alias", "metric_tag", "metric_version",
        "metric_dimension", "metric_period", "metric_owner",
        "metric_change", "metric_usage", "metric_benchmark",
        "metric_execution", "metric_permission",
        "metric_relation", "metric_dependency", "metric_quality",
        "data_source", "metric_column", "metric_join",
    ]

    # (parent, child, label) — one-to-many FK edges
    relationships = [
        ("metric", "metric_alias",     "contains"),
        ("metric", "metric_tag",       "contains"),
        ("metric", "metric_version",   "versions"),
        ("metric", "metric_dimension", "has"),
        ("metric", "metric_period",    "has"),
        ("metric", "metric_owner",     "owned by"),
        ("metric", "metric_change",    "changelog"),
        ("metric", "metric_usage",     "usage"),
        ("metric", "metric_benchmark", "benchmark"),
        ("metric", "metric_execution", "execution"),
        ("metric", "metric_permission","requires"),
        ("metric", "metric_relation",  "related to"),
        ("metric", "metric_dependency","depends on"),
        ("metric", "metric_quality",   "quality"),
        ("metric", "metric_column",    "lineage"),
        ("metric", "metric_join",      "lineage"),
        ("data_source", "metric_column", "reads from"),
        ("data_source", "metric_join",   "joins"),
    ]

    lines = ["```mermaid", "erDiagram"]

    pk_cols = {"metric": "metric_id", "data_source": "source_id"}
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

        count = con.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
        print(f"Loaded {count} metrics into {db_path}")
        print(f"ERD Mermaid source: {erd_path}")
        print("\nTop-level tables:")
        for row in con.execute("SHOW TABLES").fetchall():
            print(f"  - {row[0]}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
