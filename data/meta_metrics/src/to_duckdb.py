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
from collections import defaultdict
from pathlib import Path

import duckdb


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower(), flags=re.UNICODE).strip("_")


DDL = """
-- Semantic binding layer — dropped first (children of entity/dataset/metric_definition)
DROP TABLE IF EXISTS metric_attribute;
DROP TABLE IF EXISTS attribute_binding;
DROP TABLE IF EXISTS semantic_attribute;
DROP TABLE IF EXISTS cube_dataset;
DROP TABLE IF EXISTS cube_dimension;
DROP TABLE IF EXISTS cube_metric;
DROP TABLE IF EXISTS analytical_cube;
DROP TABLE IF EXISTS entity_relation;
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

-- Cardinality-aware, directional entity relationships.
-- join_expression is kept temporarily for backward compatibility (legacy /
-- pre-binding execution paths). It is NOT canonical: the semantic identity of
-- this relation is (from_entity, from_attribute, to_entity, to_attribute);
-- the physical join is resolved at runtime via attribute_binding
-- (see src/bindings.py: resolve_entity_relation). Do not add new callers that
-- read join_expression directly — resolve through bindings instead.
CREATE TABLE entity_relation (
    relation_id       VARCHAR PRIMARY KEY,
    from_entity_id    VARCHAR NOT NULL REFERENCES entity(entity_id),
    to_entity_id      VARCHAR NOT NULL REFERENCES entity(entity_id),
    relation_type     VARCHAR NOT NULL DEFAULT 'structural',
    cardinality       VARCHAR NOT NULL,  -- one_to_one|one_to_many|many_to_one|many_to_many
    join_expression   VARCHAR,           -- DEPRECATED: legacy physical join; prefer binding resolution
    rollup_safe       BOOLEAN NOT NULL DEFAULT false,
    temporal          BOOLEAN NOT NULL DEFAULT false,
    origin            VARCHAR NOT NULL DEFAULT 'declared',
    confidence        REAL,
    from_attribute_id VARCHAR,          -- semantic_attribute on from_entity (e.g. 'invoice.customer')
    to_attribute_id   VARCHAR,          -- semantic_attribute on to_entity (e.g. 'customer.identifier')
    CHECK (from_entity_id <> to_entity_id)
);

-- ═══════════════════════════════════════════════════════════════════════════
-- SEMANTIC BINDING LAYER
-- Separates semantic concepts (Entity/Attribute/Dimension/Metric) from
-- implementation concepts (Dataset/Column/Expression/Engine). A name like
-- 'customer_id' is never semantic truth by itself — it is at most a bootstrap
-- hint or one physical binding among possibly several for a semantic
-- attribute such as customer.identifier. Provenance is tracked explicitly via
-- (origin, resolution_state, inference_rule) — no numeric confidence score,
-- since we have no calibrated model that would give such a number meaning.
-- ═══════════════════════════════════════════════════════════════════════════

-- First-class attributes owned by entities. This is the canonical semantic
-- vocabulary: 'customer.segment' means something stable regardless of which
-- physical column, table, or warehouse currently implements it.
CREATE TABLE semantic_attribute (
    attribute_id         VARCHAR PRIMARY KEY,       -- e.g. 'customer.identifier'
    entity_id            VARCHAR NOT NULL REFERENCES entity(entity_id),
    name                 VARCHAR NOT NULL,           -- e.g. 'identifier'
    description          VARCHAR,
    semantic_type        VARCHAR NOT NULL,           -- identifier|entity_reference|measure|categorical|temporal|boolean|ordinal|descriptive
    data_type            VARCHAR,
    unit                 VARCHAR,
    references_entity_id VARCHAR REFERENCES entity(entity_id),  -- set when semantic_type = entity_reference
    UNIQUE (entity_id, name)
);

-- Global canonical dimensions with stable IDs
CREATE TABLE dimension (
    dimension_id   VARCHAR PRIMARY KEY,
    name           VARCHAR NOT NULL,
    description    VARCHAR,
    dimension_type VARCHAR NOT NULL DEFAULT 'categorical',
    entity_id      VARCHAR,          -- informational; no FK to allow partial coverage
    default_expr   VARCHAR,          -- DEPRECATED: implementation detail; prefer attribute_id
    attribute_id   VARCHAR           -- semantic_attribute this dimension groups by, when resolvable
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

-- Explicit mapping from a semantic attribute to a physical/logical
-- representation. A semantic attribute may have MULTIPLE bindings (different
-- datasets, engines, environments) — there is no assumption of one universal
-- physical column. An inferred binding is a *candidate*: it must not be
-- treated as canonical just because a heuristic produced it.
CREATE TABLE attribute_binding (
    binding_id       VARCHAR PRIMARY KEY,
    attribute_id     VARCHAR NOT NULL REFERENCES semantic_attribute(attribute_id),
    dataset_id       VARCHAR REFERENCES dataset(dataset_id),
    column_name      VARCHAR,
    expression       VARCHAR,
    engine           VARCHAR,
    binding_role     VARCHAR NOT NULL DEFAULT 'physical',
    origin           VARCHAR NOT NULL DEFAULT 'declared',   -- declared|inferred|imported|generated
    resolution_state VARCHAR NOT NULL DEFAULT 'resolved',   -- resolved|candidate|unresolved|rejected
    inference_rule   VARCHAR,
    valid_from       DATE,
    valid_to         DATE,
    -- When this attribute is one filtered slice of a physical table shared
    -- by several attributes (e.g. opportunity.closed_won__amount lives in
    -- analytics.fct_opportunity WHERE stage='closed_won', alongside
    -- opportunity.open_opportunity__amount in the SAME table with a
    -- different filter) — a WHERE-clause fragment a consumer must apply.
    -- NULL for the common case of an attribute with the table to itself.
    filter_column    VARCHAR,
    filter_value     VARCHAR
);

-- Metric definition: pure business semantics, engine-agnostic
CREATE TABLE metric_definition (
    metric_id               VARCHAR PRIMARY KEY,
    name                    VARCHAR NOT NULL,
    department              VARCHAR NOT NULL,
    description             VARCHAR NOT NULL,
    derivation_type         VARCHAR NOT NULL DEFAULT 'base',   -- base|derived
    metric_type             VARCHAR NOT NULL DEFAULT 'scalar', -- scalar|ratio|cumulative|snapshot|conversion|retention|cohort
    aggregation             VARCHAR NOT NULL,
    entity_id               VARCHAR REFERENCES entity(entity_id),
    display_grain           VARCHAR,       -- presentation label, e.g. "cliente"
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

-- Semantic (not physical) lineage: which semantic attributes a metric uses,
-- and in what role. This is distinct from impl_column, which is physical
-- lineage (dataset/column). A metric's semantic lineage should remain stable
-- even if its physical implementation changes engines or warehouses.
--   Metric -> (metric_attribute) -> SemanticAttribute -> (attribute_binding) -> Dataset/Column
CREATE TABLE metric_attribute (
    metric_id    VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    attribute_id VARCHAR NOT NULL REFERENCES semantic_attribute(attribute_id),
    role         VARCHAR NOT NULL,   -- measure|numerator|denominator|filter|grouping|time
    origin       VARCHAR NOT NULL DEFAULT 'declared',  -- declared|inferred
    -- When two DIFFERENT metrics aggregate the SAME shared attribute but
    -- mean different filtered slices of it (e.g. marketing.mql_volume and
    -- marketing.sql_volume both count lead.identifier, but mean
    -- status='mql' vs status='sql' respectively) — attribute_binding's
    -- filter_column/filter_value can't express this, since that lives on
    -- the ATTRIBUTE (shared by both metrics), not the metric. This is the
    -- per-metric equivalent, checked first and falling back to the
    -- attribute-level filter when unset.
    filter_column VARCHAR,
    filter_value  VARCHAR,
    PRIMARY KEY (metric_id, attribute_id, role)
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

-- Generated analytical cubes (populated by src/cubes.py)
CREATE TABLE analytical_cube (
    cube_id              VARCHAR PRIMARY KEY,
    name                 VARCHAR NOT NULL,
    analytical_entity_id VARCHAR REFERENCES entity(entity_id),
    cube_type            VARCHAR NOT NULL DEFAULT 'process',  -- process|conformed|virtual
    generated            BOOLEAN NOT NULL DEFAULT true,
    explanation          VARCHAR
);

CREATE TABLE cube_metric (
    cube_id          VARCHAR NOT NULL REFERENCES analytical_cube(cube_id),
    metric_id        VARCHAR NOT NULL REFERENCES metric_definition(metric_id),
    role             VARCHAR NOT NULL DEFAULT 'native',  -- native|dependency|composite
    rollup_entity_id VARCHAR,
    reason           VARCHAR,
    PRIMARY KEY (cube_id, metric_id)
);

CREATE TABLE cube_dimension (
    cube_id      VARCHAR NOT NULL REFERENCES analytical_cube(cube_id),
    dimension_id VARCHAR NOT NULL REFERENCES dimension(dimension_id),
    PRIMARY KEY (cube_id, dimension_id)
);

CREATE TABLE cube_dataset (
    cube_id          VARCHAR NOT NULL REFERENCES analytical_cube(cube_id),
    dataset_id       VARCHAR NOT NULL REFERENCES dataset(dataset_id),
    rollup_entity_id VARCHAR,
    PRIMARY KEY (cube_id, dataset_id)
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
            "INSERT INTO dimension VALUES (?, ?, NULL, ?, ?, ?, NULL)",
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
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            mid, m["name"], m["department"], m["description"],
            m.get("derivation_type", "base"), m.get("metric_type", "scalar"), m["aggregation"],
            m.get("entity_id"), m.get("display_grain"), m.get("unit"), m["status"],
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
    declared_pairs: set[tuple[str, str]] = set()
    for m in catalog["metrics"]:
        mid = m["id"]
        for d in m.get("dependencies", []):
            all_deps.append((mid, d["depends_on"], d.get("type", "computational"), "declared"))
            declared_pairs.add((mid, d["depends_on"]))

    # Auto-detected dependencies: tokenize each metric's formula expression
    # and match tokens against OTHER metrics' short id ('churn_rate') or
    # display name ('Churn Rate'), case/underscore-normalized. Only an
    # UNAMBIGUOUS single match is accepted — never guessed among several
    # candidates. This recovers real dependency edges that exist in the
    # catalog but were never manually added to main.py's DEPS dict (24
    # metrics had that curation; most of the other ~186 never got it), so
    # to_dbt.py doesn't have to silently drop every derived metric whose
    # formula happens to reference an existing metric that DEPS forgot.
    #
    # It does NOT invent anything for tokens that don't match any metric at
    # all (e.g. 'churned_customers', 'expected_lifetime_periods') — those
    # are genuine sub-calculations that were never modeled as their own
    # metric anywhere in the catalog. That's a real content gap (someone
    # needs to decide how 'churned_customers' should be counted and declare
    # it as a metric), not a linking bug this pass can or should paper over.
    _SQL_KEYWORDS = {
        "NULLIF", "IF", "CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR",
        "NOT", "COALESCE", "SUM", "AVG", "COUNT", "MIN", "MAX", "CAST", "AS",
    }
    name_to_metrics: dict[str, set[str]] = defaultdict(set)
    for m in catalog["metrics"]:
        mid = m["id"]
        name_to_metrics[mid.split(".", 1)[1].lower()].add(mid)
        norm_name = re.sub(r"[^a-z0-9]+", "_", (m.get("name") or "").lower()).strip("_")
        if norm_name:
            name_to_metrics[norm_name].add(mid)

    inferred_count = 0
    unresolved_tokens: dict[str, set[str]] = defaultdict(set)
    for m in catalog["metrics"]:
        mid = m["id"]
        expr = (m.get("formula", {}) or {}).get("expression") or ""
        if not expr:
            continue
        tokens = {
            t for t in re.findall(r"(?<!\.)\b[A-Za-z][A-Za-z0-9_]*", expr)
            if t.upper() not in _SQL_KEYWORDS
        }
        for tok in tokens:
            candidates = name_to_metrics.get(tok.lower(), set()) - {mid}
            if len(candidates) == 1:
                target = next(iter(candidates))
                if (mid, target) not in declared_pairs:
                    declared_pairs.add((mid, target))
                    all_deps.append((mid, target, "computational", "inferred"))
                    inferred_count += 1
            elif len(candidates) == 0:
                unresolved_tokens[mid].add(tok)
            # len > 1 (ambiguous): deliberately skipped, not guessed

    if all_deps:
        con.executemany("INSERT INTO metric_dependency VALUES (?, ?, ?, ?)", all_deps)

    print(
        f"Dependency extraction: {len(declared_pairs) - inferred_count} declared "
        f"(main.py DEPS) + {inferred_count} auto-inferred (unambiguous formula-token "
        f"match against another metric) = {len(declared_pairs)} total edges.",
    )
    if unresolved_tokens:
        n_metrics = len(unresolved_tokens)
        n_tokens = sum(len(v) for v in unresolved_tokens.values())
        print(
            f"  {n_tokens} formula token(s) across {n_metrics} metric(s) match NO metric "
            "at all — these are sub-calculations never modeled as their own metric "
            "(not a linking bug; someone needs to decide how to count/declare them). "
            "Full list: metric_id -> {unmatched tokens}:",
        )
        for mid in sorted(unresolved_tokens):
            print(f"    {mid}: {sorted(unresolved_tokens[mid])}")


# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC BINDING LAYER — bootstrap migration
#
# Populates semantic_attribute / attribute_binding / metric_attribute from
# data populate() already loaded (entity.pk_column, dataset/impl_column
# physical lineage parsed out of pseudocode formulas, dimension.default_expr).
#
# This is explicitly a MIGRATION, not canonical truth. Every row it writes is
# tagged origin='inferred', resolution_state='candidate' with an
# inference_rule string explaining why — never numeric confidence. A name
# matching a heuristic does not make it correct; it makes it worth reviewing.
# Downstream code (bindings.py, validate.py, cube executability) must treat
# 'candidate' bindings as not-yet-usable for execution.
# ═══════════════════════════════════════════════════════════════════════════

# Naming variants seen in pseudocode formulas that are NOT the entity's own
# name but plausibly refer to it (e.g. 'active_subscription' -> 'subscription').
# This list is a bootstrap heuristic, not a semantic authority — matches it
# produces are still recorded as inferred/candidate, same as an exact match
# recorded as inferred (dataset names here are not verified physical tables).
_ENTITY_ALIAS_PREFIXES: tuple[str, ...] = (
    "active_", "churned_", "new_", "open_", "closed_", "security_",
    "influenced_", "attributed_", "reactivated_",
)

_CATEGORICAL_COLUMN_NAMES: frozenset[str] = frozenset({
    "status", "type", "category", "severity", "segment", "region",
    "channel", "priority", "plan_id", "source", "tier", "stage",
})

# Explicit, human-reviewed aliases for dataset names that don't share a
# substring with their entity (so prefix-stripping can't find them) but are
# unambiguous business-domain equivalences: a status label used as a table
# alias ('closed_won' IS an opportunity in a given state), or an MRR-movement
# type that belongs to the 'movement' entity already in ENTITY_PK. This is
# still a naming-convention judgment call, not a verified physical schema —
# matches from this table are recorded with resolution_state='candidate'
# exactly like every other bootstrap source, just with a distinct
# inference_rule so a reviewer can tell it apart from plain prefix-stripping.
_DATASET_ALIASES: dict[str, str] = {
    "closed_won":            "opportunity",
    "attributed_closed_won": "opportunity",
    "influenced_closed_won": "opportunity",
    "contraction":           "movement",
    "expansion":             "movement",
    "reactivation":          "movement",
    "engagement_survey":     "survey_response",
    "survey":                "survey_response",
    "response":              "survey_response",
    "cash_accounts":         "cash_account",
    "inventory":             "inventory_position",
    "north_star_event":      "event",
}

# Explicit, narrow overrides for entity_reference columns whose naive
# 'strip the _id suffix' guess doesn't name any declared entity, but whose
# true target IS known — e.g. 'owner_id' names the person responsible, and
# sales reps are modeled as 'employee' (same insight already captured for
# the 'vendedor' grain alias above). Deliberately keyed by the exact
# (entity_id, column_name) pair actually observed in the catalog, never a
# blanket "owner_id always means employee" rule — that would be exactly the
# kind of unreviewed guessing this bootstrap layer exists to avoid. Add an
# entry here only once a human has confirmed what a specific column means;
# anything not listed keeps falling through to 'no resolvable target' so
# build_warehouse.py seeds it as an honest NULL instead of guessing wrong.
_REFERENCE_TARGET_OVERRIDES: dict[tuple[str, str], str] = {
    ("opportunity", "owner_id"): "employee",  # a deal's owner is the sales rep who owns it
}


def _normalize_dataset_to_entity(
    dataset_id: str, known_entities: set[str]
) -> tuple[str | None, str | None]:
    """Best-effort match of a formula-derived dataset name to a known entity.
    Returns (entity_id, rule) or (None, None) if nothing matches — unresolved
    cases are left unresolved on purpose rather than guessed."""
    if dataset_id in known_entities:
        return dataset_id, "exact_name_match"
    if dataset_id in _DATASET_ALIASES and _DATASET_ALIASES[dataset_id] in known_entities:
        return _DATASET_ALIASES[dataset_id], f"declared_dataset_alias:{dataset_id}"
    for prefix in _ENTITY_ALIAS_PREFIXES:
        if dataset_id.startswith(prefix):
            candidate = dataset_id[len(prefix):]
            if candidate in known_entities:
                return candidate, f"prefix_strip:{prefix}"
    return None, None


def _infer_semantic_type(
    entity_id: str, column_name: str, role: str, known_entities: frozenset[str] = frozenset()
) -> tuple[str, str | None]:
    """Heuristic semantic_type from a column name + its impl_column role.
    Returns (semantic_type, candidate_referenced_entity_id_or_None)."""
    if column_name == "id":
        return "identifier", None
    if column_name == f"{entity_id}_id":
        # The entity's OWN id, spelled out with the '_id' suffix (e.g. a
        # formula doing 'COUNT(dataset.dataset_id)' just to count rows).
        # Not a reference to another entity, and not a measure either —
        # falling through to the 'measure' default below would create a
        # bogus numeric attribute with the exact same name as this entity's
        # primary key, which is worse than merely wrong: build_warehouse.py
        # builds physical columns by name, so a same-named 'measure'
        # attribute silently overwrites the seeded identifier at insert
        # time. Recognize it explicitly as what it is.
        return "identifier", None
    if column_name.endswith("_id"):
        override = _REFERENCE_TARGET_OVERRIDES.get((entity_id, column_name))
        if override:
            return "entity_reference", override
        return "entity_reference", column_name[:-3]
    if role == "date_key" or column_name.endswith(("_at", "_date")):
        return "temporal", None
    if column_name in _CATEGORICAL_COLUMN_NAMES:
        return "categorical", None
    # A bare column name (no '_id' suffix) that exactly matches another
    # DECLARED entity's own id is a strong, narrow signal it's a reference to
    # that entity too — e.g. a formula's 'deployment.service' column, from
    # 'COUNT(deployment.id WHERE deployment.service=...)', means "which
    # service this deployment belongs to", not a numeric measure named
    # 'service'. This mirrors bindings.py's entity_relation-endpoint override
    # in build_warehouse.py: don't let an accidental default classification
    # (here: "not '_id'-suffixed, so fall through to measure") stand when a
    # much more specific signal — the column IS an entity's name — says
    # otherwise. Still narrow: only fires for an EXACT match against a real,
    # declared entity_id, never a fuzzy/partial one.
    if column_name != entity_id and column_name in known_entities:
        return "entity_reference", column_name
    return "measure", None


def populate_semantic_layer(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Bootstrap semantic_attribute / attribute_binding / metric_attribute
    from already-populated entity / dataset / impl_column / dimension data.
    Idempotent within a single fresh load (tables were just DROP/CREATE'd by
    populate()). Returns a summary dict for reporting."""
    stats = {
        "identifiers": 0, "identifier_bindings": 0,
        "attributes": 0, "attribute_bindings": 0,
        "dimension_links": 0, "metric_attributes": 0,
        "unresolved_datasets": 0,
    }

    entities = con.execute("SELECT entity_id, pk_column FROM entity").fetchall()
    known_entities = {e[0] for e in entities}
    existing_datasets = {r[0] for r in con.execute("SELECT dataset_id FROM dataset").fetchall()}

    def _ensure_entity_dataset(entity_id: str) -> None:
        """attribute_binding.dataset_id has a FK into dataset(dataset_id).
        When we normalize a formula-lineage token (e.g. 'active_subscription')
        or a dimension join_path's entity to its real target entity_id (e.g.
        'subscription'), that entity_id needs its own dataset row to satisfy
        the FK — it won't already have one unless some OTHER lineage row
        happened to use the bare entity name verbatim as a dataset_id."""
        if entity_id in existing_datasets:
            return
        existing_datasets.add(entity_id)
        con.execute(
            "INSERT INTO dataset VALUES (?, ?, 'unknown', NULL, NULL, NULL, ?, ?, NULL)",
            [entity_id, entity_id, entity_id, entity_id],
        )

    # ── 1. Canonical identifier per entity ─────────────────────────────────
    # The ATTRIBUTE is declared/resolved: every non-virtual business entity
    # has an identity by definition — that is not in question. Its physical
    # BINDING (pk_column, from the legacy ENTITY_PK mapping) is only a
    # candidate: it is a naming assumption, never verified against a real
    # physical schema.
    for entity_id, pk_column in entities:
        attr_id = f"{entity_id}.identifier"
        con.execute("""
            INSERT INTO semantic_attribute VALUES (?, ?, 'identifier', ?, 'identifier', NULL, NULL, NULL)
        """, [attr_id, entity_id, f"Canonical identifier of {entity_id}"])
        stats["identifiers"] += 1

        # dataset_id = entity_id: this is NOT a second, separate guess on top
        # of the column-name assumption below — attr_id ('account.identifier')
        # already commits to 'account' as the entity, so recording that same
        # entity as dataset_id adds no new assumption. Leaving it NULL (the
        # previous behavior here) only meant every downstream consumer had to
        # re-derive 'this identifier lives on its own entity's table' by
        # hand — and in practice made review.py's --schema-db check unable to
        # do better than 'ambiguous (N tables have this column)' whenever the
        # bare column name (e.g. 'product_id') also shows up as an FK
        # elsewhere, exactly like the formula-lineage and entity_relation
        # bootstraps above and below this one.
        _ensure_entity_dataset(entity_id)
        con.execute("""
            INSERT INTO attribute_binding VALUES
            (?, ?, ?, ?, NULL, NULL, 'physical', 'inferred', 'candidate', ?, NULL, NULL, NULL, NULL)
        """, [
            f"{attr_id}:bootstrap", attr_id, entity_id, pk_column,
            f"bootstrap_from_entity_pk: column name '{pk_column}' assumed from legacy "
            "ENTITY_PK naming convention; no physical dataset confirmed yet",
        ])
        stats["identifier_bindings"] += 1

    # ── 2. Attributes discovered from formula-derived lineage (impl_column) ─
    # dataset_id here often IS just an entity name (or a naming variant)
    # that appeared inside a pseudocode expression like 'invoice.net_amount'
    # — precisely the naming-as-semantics risk this layer exists to contain.
    # Everything created here is origin='inferred', resolution_state='candidate'.
    #
    # Collision guard: (entity_id, column_name) can be produced by MULTIPLE
    # distinct dataset_ids that are NOT alternate spellings of the same
    # physical thing but genuinely different filtered/typed sources —
    # e.g. 'closed_won.amount' vs 'open_opportunity.amount' vs
    # 'opportunity.amount' are three different subsets of opportunities, not
    # three name-guesses for one column. Collapsing them into a single
    # attribute_id would silently merge different business quantities and
    # make it impossible to ever resolve more than one of the metrics that
    # depend on them correctly. So: only collapse into a bare
    # '{entity}.{column}' attribute when exactly one dataset_id produces
    # that (entity_id, column_name) pair; when more than one does, each
    # keeps its own qualified attribute_id instead of competing for one.
    _dataset_ids_per_col: dict[tuple[str, str], set[str]] = defaultdict(set)
    for dataset_id, column_name, _role in con.execute(
        "SELECT DISTINCT dataset_id, column_name, role FROM impl_column"
    ).fetchall():
        entity_id, _ = _normalize_dataset_to_entity(dataset_id, known_entities)
        if entity_id:
            _dataset_ids_per_col[(entity_id, column_name)].add(dataset_id)

    def _attr_name_for(entity_id: str, dataset_id: str, column_name: str, sem_type: str) -> str:
        if sem_type == "identifier":
            return "identifier"
        # Only qualify the aliased/prefix-derived variants. The variant whose
        # dataset_id equals the entity_id exactly (the "general"/unfiltered
        # source) keeps the plain name — it's unambiguous on its own and
        # reads better than 'opportunity.opportunity__amount'.
        if len(_dataset_ids_per_col[(entity_id, column_name)]) > 1 and dataset_id != entity_id:
            return f"{dataset_id}__{column_name}"
        return column_name

    seen_attrs: set[str] = set()
    unresolved_datasets: set[str] = set()
    for dataset_id, column_name, role in con.execute(
        "SELECT DISTINCT dataset_id, column_name, role FROM impl_column"
    ).fetchall():
        entity_id, rule = _normalize_dataset_to_entity(dataset_id, known_entities)
        if entity_id is None:
            unresolved_datasets.add(dataset_id)
            continue  # left unresolved on purpose — no guessing

        sem_type, ref_hint = _infer_semantic_type(entity_id, column_name, role, known_entities)
        attr_name = _attr_name_for(entity_id, dataset_id, column_name, sem_type)
        attr_id = f"{entity_id}.{attr_name}"

        if sem_type != "identifier" and attr_id not in seen_attrs:
            seen_attrs.add(attr_id)
            references_entity_id = (
                ref_hint if sem_type == "entity_reference" and ref_hint in known_entities else None
            )
            con.execute("""
                INSERT INTO semantic_attribute VALUES (?, ?, ?, NULL, ?, NULL, NULL, ?)
            """, [attr_id, entity_id, attr_name, sem_type, references_entity_id])
            stats["attributes"] += 1
        else:
            seen_attrs.add(attr_id)

        binding_id = f"{attr_id}:from:{dataset_id}"
        if con.execute("SELECT 1 FROM attribute_binding WHERE binding_id = ?",
                        [binding_id]).fetchone():
            continue
        exact = dataset_id == entity_id
        _ensure_entity_dataset(entity_id)
        # Store the ALIAS-NORMALIZED target (entity_id), not the raw
        # formula-lineage token (dataset_id) — the whole point of resolving
        # 'active_subscription'/'security_incident'/'closed_won' through
        # _DATASET_ALIASES / _ENTITY_ALIAS_PREFIXES is that they physically
        # ARE the entity's own table (a slice of it, at most). Persisting the
        # raw token here silently threw that resolution away: every
        # downstream consumer — review.py's --schema-db cross-check,
        # build_warehouse.py, a human reading the table — would have had to
        # re-derive the same alias to recognize 'active_subscription' as
        # 'subscription'. The raw token is not lost, it's kept in
        # binding_id and inference_rule for provenance.
        con.execute("""
            INSERT INTO attribute_binding VALUES
            (?, ?, ?, ?, NULL, NULL, 'physical', 'inferred', 'candidate', ?, NULL, NULL, NULL, NULL)
        """, [
            binding_id, attr_id, entity_id, column_name,
            "bootstrap_from_formula_lineage: dataset name matches entity exactly"
            if exact else
            f"bootstrap_from_formula_lineage: dataset '{dataset_id}' matched entity "
            f"'{entity_id}' via {rule}",
        ])
        stats["attribute_bindings"] += 1
    stats["unresolved_datasets"] = len(unresolved_datasets)

    # ── 2b. Attributes discovered from dimension join_paths ────────────────
    # A dimension's join_path (e.g. 'customer.segment') is itself an
    # 'entity.attribute'-shaped naming-convention string, same risk class as
    # formula lineage — so it goes through the same candidate/inferred path,
    # not treated as declared just because it looks structured.
    dim_rows = con.execute(
        "SELECT dimension_id, default_expr, entity_id FROM dimension"
    ).fetchall()
    for dim_id, default_expr, dim_entity in dim_rows:
        if not default_expr or dim_entity not in known_entities:
            continue
        attr_id = f"{dim_entity}.{default_expr}"
        if attr_id in seen_attrs:
            continue
        seen_attrs.add(attr_id)
        sem_type, ref_hint = _infer_semantic_type(dim_entity, default_expr, "grouping", known_entities)
        if sem_type == "identifier":
            continue  # identifier already exists from step 1
        references_entity_id = (
            ref_hint if sem_type == "entity_reference" and ref_hint in known_entities else None
        )
        con.execute("""
            INSERT INTO semantic_attribute VALUES (?, ?, ?, NULL, ?, NULL, NULL, ?)
        """, [attr_id, dim_entity, default_expr, sem_type, references_entity_id])
        stats["attributes"] += 1
        _ensure_entity_dataset(dim_entity)
        # Physical column name: for entity_reference attributes, every other
        # naming path in this file (formula lineage, entity_relation) assumes
        # the FK column is '<name>_id' — build_warehouse.py builds physical
        # FK columns the same way. A dimension's join_path gives the bare
        # business name ('customer.cohort'), which is right for attr_id/name,
        # but using it unchanged as the PHYSICAL column name was wrong: the
        # real column is 'cohort_id', not 'cohort'. Apply the same suffix
        # convention here instead of leaving it inconsistent with everywhere
        # else references are named.
        physical_column = (
            f"{default_expr}_id"
            if sem_type == "entity_reference" and not default_expr.endswith("_id")
            else default_expr
        )
        con.execute("""
            INSERT INTO attribute_binding VALUES
            (?, ?, ?, ?, NULL, NULL, 'physical', 'inferred', 'candidate', ?, NULL, NULL, NULL, NULL)
        """, [
            f"{attr_id}:from_dimension:{dim_id}", attr_id, dim_entity, physical_column,
            f"bootstrap_from_dimension_join_path: dimension '{dim_id}' declares join_path "
            f"'{dim_entity}.{default_expr}'; entity+column parsed from that string, no "
            "physical dataset confirmed beyond the entity the join_path already commits to"
            + (f"; column suffixed '_id' per this codebase's entity_reference naming "
               f"convention (physical column is '{physical_column}', not bare "
               f"'{default_expr}')" if physical_column != default_expr else ""),
        ])
        stats["attribute_bindings"] += 1

    # ── 3. Link dimensions to semantic attributes where resolvable ─────────
    known_attrs = {r[0] for r in con.execute(
        "SELECT attribute_id FROM semantic_attribute").fetchall()}
    for dim_id, default_expr, dim_entity in dim_rows:
        if not default_expr:
            continue
        # 'id' in a join_path means "this entity's own identifier", which is
        # named '{entity}.identifier' in the semantic layer, never
        # '{entity}.id' — same normalization as everywhere else attribute
        # names are derived from a raw column name.
        attr_name = "identifier" if default_expr == "id" else default_expr
        candidate_attr = None
        if dim_entity and f"{dim_entity}.{attr_name}" in known_attrs:
            candidate_attr = f"{dim_entity}.{attr_name}"
        elif "." in default_expr and default_expr in known_attrs:
            candidate_attr = default_expr
        if candidate_attr:
            con.execute("UPDATE dimension SET attribute_id = ? WHERE dimension_id = ?",
                        [candidate_attr, dim_id])
            stats["dimension_links"] += 1

    # ── 4. Semantic (not physical) lineage per metric ──────────────────────
    role_map = {"numerator": "measure", "date_key": "time", "filter": "filter"}
    for mid, dataset_id, column_name, ic_role in con.execute("""
        SELECT mi.metric_id, ic.dataset_id, ic.column_name, ic.role
        FROM impl_column ic
        JOIN metric_implementation mi ON mi.impl_id = ic.impl_id AND mi.is_current = true
    """).fetchall():
        entity_id, _ = _normalize_dataset_to_entity(dataset_id, known_entities)
        if entity_id is None:
            continue
        sem_type, _ = _infer_semantic_type(entity_id, column_name, ic_role, known_entities)
        attr_name = _attr_name_for(entity_id, dataset_id, column_name, sem_type)
        attr_id = f"{entity_id}.{attr_name}"
        if attr_id not in known_attrs:
            continue  # attribute wasn't resolvable in step 2 — skip, don't guess
        con.execute("INSERT OR IGNORE INTO metric_attribute VALUES (?, ?, ?, 'inferred', NULL, NULL)",
                    [mid, attr_id, role_map.get(ic_role, "measure")])
        stats["metric_attributes"] += 1

    return stats


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
        "entity", "entity_relation", "dimension", "dataset",
        "metric_definition", "metric_implementation",
        "impl_column", "impl_join",
        "metric_dependency", "metric_relation", "metric_dimension",
        "quality_contract", "quality_run",
        "metric_alias", "metric_tag", "metric_period", "metric_owner",
        "metric_change", "metric_usage", "metric_benchmark",
        "metric_execution", "metric_permission",
        "analytical_cube", "cube_metric", "cube_dimension", "cube_dataset",
        # Semantic binding layer — omitted before; this is the core of the
        # semantic/physical separation described in bindings.py and must be
        # visible in the generated ERD.
        "semantic_attribute", "attribute_binding", "metric_attribute",
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
        ("entity",             "entity_relation",      "relates to"),
        ("analytical_cube",    "cube_metric",          "contains"),
        ("analytical_cube",    "cube_dimension",       "sliced by"),
        ("analytical_cube",    "cube_dataset",         "reads from"),
        ("metric_definition",  "cube_metric",          "member of"),
        ("dimension",          "cube_dimension",       "used in cube"),
        ("dataset",            "cube_dataset",         "in cube"),
        # Semantic binding layer
        ("entity",             "semantic_attribute",   "declares attribute"),
        ("semantic_attribute", "attribute_binding",    "bound as"),
        ("dataset",            "attribute_binding",    "backs"),
        ("metric_definition",  "metric_attribute",     "uses attribute"),
        ("semantic_attribute", "metric_attribute",     "used by metric"),
    ]

    lines = ["```mermaid", "erDiagram"]

    pk_cols = {"metric_definition": "metric_id", "dataset": "dataset_id",
               "entity": "entity_id", "entity_relation": "relation_id",
               "dimension": "dimension_id",
               "metric_implementation": "impl_id", "quality_contract": "contract_id",
               "analytical_cube": "cube_id",
               "semantic_attribute": "attribute_id", "attribute_binding": "binding_id"}
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
        sem_stats = populate_semantic_layer(con)
        create_views(con)
        write_mermaid(con, erd_path)

        count = con.execute("SELECT COUNT(*) FROM metric_definition").fetchone()[0]
        print(f"Loaded {count} metrics into {db_path}")
        print(f"ERD Mermaid source: {erd_path}")
        print("\nSemantic binding layer (bootstrap migration):")
        print(f"  - {sem_stats['identifiers']} identifier attributes "
              f"({sem_stats['identifier_bindings']} candidate bindings from ENTITY_PK)")
        print(f"  - {sem_stats['attributes']} attributes discovered from formula lineage "
              f"({sem_stats['attribute_bindings']} candidate bindings)")
        print(f"  - {sem_stats['dimension_links']} dimensions linked to semantic attributes")
        print(f"  - {sem_stats['metric_attributes']} metric semantic-lineage rows")
        print(f"  - {sem_stats['unresolved_datasets']} formula-lineage dataset names left "
              "unresolved (no naming match to a known entity — not guessed)")
        print("\nTop-level tables:")
        for row in con.execute("SHOW TABLES").fetchall():
            print(f"  - {row[0]}")
    finally:
        con.close()


if __name__ == "__main__":
    main()