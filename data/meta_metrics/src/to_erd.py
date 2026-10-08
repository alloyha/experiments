#!/usr/bin/env python3
"""
Render one Mermaid erDiagram per connected component of the catalog's
RESOLVED FK graph, and classify each component's actual shape (star /
snowflake / constellation) instead of assuming every fact table sits at the
center of a clean star.

Sourced entirely from metric_catalog.duckdb's own tables — attribute_binding
(resolution_state='resolved'), dataset.full_ref, semantic_attribute,
metric_definition — never from any physical-warehouse generator's Python
internals. This means the diagram reflects whatever real schema review.py
was last pointed at via --schema-db when bindings were promoted, not one
specific synthetic generator's naming choices. Pass --schema-db here too
(optional) to also pull real SQL column types for the diagram; without it,
columns show their catalog semantic_type instead of a guessed SQL type.

Usage:
    python3 src/generate_erd.py data/metric_catalog.duckdb data/erd [--schema-db data/warehouse.duckdb]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import duckdb


# ── data loading (catalog only, no physical-generator imports) ─────────────

def load_catalog_graph(con: duckdb.DuckDBPyConnection, schema_con=None):
    """Returns (tables, edges):
      tables: dict[full_ref] -> {"pk_column": str | None, "columns": [...]}
      edges: list[(from_table, column_name, to_table)]
    Only resolved bindings are used — an attribute with no resolved binding
    yet simply doesn't appear, exactly like bindings.py's own resolution
    would refuse to guess at it.
    """
    entity_table = {
        r[0]: r[1] for r in con.execute("""
            SELECT sa.entity_id, ds.full_ref
            FROM semantic_attribute sa
            JOIN attribute_binding ab ON ab.attribute_id = sa.attribute_id
                                      AND ab.resolution_state = 'resolved'
            JOIN dataset ds ON ds.dataset_id = ab.dataset_id
            WHERE sa.semantic_type = 'identifier' AND ds.full_ref IS NOT NULL
        """).fetchall()
    }

    real_types: dict[tuple[str, str], str] = {}
    if schema_con is not None:
        real_types = {
            (r[0], r[1]): r[2] for r in schema_con.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns"
            ).fetchall()
        }

    tables: dict[str, dict] = defaultdict(lambda: {"pk_column": None, "columns": []})
    edges: list[tuple[str, str, str]] = []

    rows = con.execute("""
        SELECT ds.full_ref, ab.column_name, sa.semantic_type, sa.references_entity_id,
               ab.filter_column, ab.filter_value
        FROM attribute_binding ab
        JOIN semantic_attribute sa ON sa.attribute_id = ab.attribute_id
        JOIN dataset ds ON ds.dataset_id = ab.dataset_id
        WHERE ab.resolution_state = 'resolved' AND ds.full_ref IS NOT NULL
        ORDER BY ds.full_ref, ab.column_name
    """).fetchall()
    for full_ref, column_name, sem_type, ref_entity, filter_col, filter_val in rows:
        if column_name is None:
            continue
        type_label = real_types.get((full_ref, column_name)) or f"({sem_type})"
        fk_target = None
        if sem_type == "entity_reference" and ref_entity:
            fk_target = entity_table.get(ref_entity)
            if fk_target and fk_target != full_ref:
                edges.append((full_ref, column_name, fk_target))
        if sem_type == "identifier":
            tables[full_ref]["pk_column"] = column_name
        entry = (column_name, type_label, sem_type, fk_target, filter_col, filter_val)
        if entry not in tables[full_ref]["columns"]:
            tables[full_ref]["columns"].append(entry)

    return dict(tables), edges


def _countable_entities(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Entities that are fact-like by catalog semantics (declared measure,
    or grain of a count/count_distinct metric) — not by table-name prefix,
    which a real target schema has no obligation to follow. Excludes
    calendar-grain entities that merely show up as a DAU/WAU/MAU-style
    metric's reporting grain."""
    measure_entities = {
        r[0] for r in con.execute(
            "SELECT DISTINCT entity_id FROM semantic_attribute WHERE semantic_type = 'measure'"
        ).fetchall()
    }
    countable = {
        r[0] for r in con.execute(
            "SELECT DISTINCT entity_id FROM metric_definition "
            "WHERE aggregation IN ('count', 'count_distinct')"
        ).fetchall()
    } - {"day", "week", "month", "period"}
    return measure_entities | countable


def _table_entity(con: duckdb.DuckDBPyConnection, full_ref: str) -> str | None:
    r = con.execute("""
        SELECT sa.entity_id FROM semantic_attribute sa
        JOIN attribute_binding ab ON ab.attribute_id = sa.attribute_id AND ab.resolution_state='resolved'
        JOIN dataset ds ON ds.dataset_id = ab.dataset_id
        WHERE ds.full_ref = ? AND sa.semantic_type = 'identifier' LIMIT 1
    """, [full_ref]).fetchone()
    return r[0] if r else None


# ── graph analysis ───────────────────────────────────────────────────────────

def connected_components(all_tables: set[str], edges: list[tuple[str, str, str]]):
    adj = defaultdict(set)
    for a, _col, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited, components = set(), []
    for t in sorted(all_tables):
        if t in visited or t not in adj:
            continue
        comp, q = set(), deque([t])
        visited.add(t)
        while q:
            cur = q.popleft()
            comp.add(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        components.append(comp)
    isolated = sorted(all_tables - visited)
    return components, isolated


def classify_shape(comp: set[str], edges_in_comp, facts: set[str]) -> tuple[str, str]:
    fact_to_fact = any(a in facts and b in facts for a, _c, b in edges_in_comp)
    if len(facts) > 1 and fact_to_fact:
        return "constellation", (
            f"{len(facts)} tabelas-fato neste componente, com ao menos uma FK "
            "ligando um fato diretamente a outro — não é uma única estrela, é "
            "uma constelação de fatos compartilhando dimensões (e, em parte, "
            "se referenciando entre si)."
        )
    if len(facts) > 1:
        return "constelação (fatos compartilhando dimensões)", (
            f"{len(facts)} tabelas-fato neste componente, todas ligadas apenas "
            "a dimensões compartilhadas, sem FK direta entre fatos."
        )
    if len(facts) == 0:
        return "cluster sem fato (não é estrela nem snowflake)", (
            "Nenhuma tabela-fato neste componente — só dimensões ligadas "
            "entre si, sem um fato no centro."
        )
    fact = next(iter(facts))
    dim_to_dim = any(a != fact and b != fact and a in comp and b in comp for a, _c, b in edges_in_comp)
    if dim_to_dim:
        return "snowflake", (
            f"Uma única tabela-fato ({fact}), mas há dimensão apontando para "
            "outra dimensão dentro do mesmo componente — normalizada em mais "
            "de um nível."
        )
    return "estrela", f"Uma única tabela-fato ({fact}), todo o resto liga direto a ela — estrela clássica."


def render_mermaid(comp: set[str], edges_in_comp, tables: dict, facts: set[str],
                    shape: str, reason: str) -> str:
    lines = ["erDiagram", f"    %% Forma: {shape}"]
    for line in reason.split(". "):
        line = line.strip()
        if line:
            lines.append(f"    %% {line}.")
    lines.append("")

    seen_pairs = set()
    for a, col, b in sorted(edges_in_comp):
        key = (a, b, col)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        lines.append(f'    {a} }}o--|| {b} : "{col}"')

    lines.append("")
    for table_name in sorted(comp):
        t = tables[table_name]
        lines.append(f"    {table_name} {{")
        for column_name, type_label, sem_type, fk_target, filter_col, filter_val in t["columns"]:
            tag = ""
            if column_name == t["pk_column"]:
                tag = "  PK"
            elif fk_target:
                tag = "  FK"
            if filter_col:
                tag += f' "filtro: {filter_col}={filter_val}"'
            lines.append(f"        {type_label} {column_name}{tag}")
        lines.append("    }")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalog_db", nargs="?", default="data/metric_catalog.duckdb")
    p.add_argument("out_dir", nargs="?", default="data/erd")
    p.add_argument("--schema-db", help="Optional: a queryable DB to pull real SQL "
                                        "column types from (information_schema.columns)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.catalog_db, read_only=True)
    schema_con = duckdb.connect(args.schema_db, read_only=True) if args.schema_db else None
    try:
        tables, edges = load_catalog_graph(con, schema_con)
        countable = _countable_entities(con)
        fact_tables = {t for t in tables if (_table_entity(con, t) or "") in countable}

        all_tables = set(tables.keys())
        components, isolated = connected_components(all_tables, edges)

        index = []
        for i, comp in enumerate(sorted(components, key=lambda c: -len(c)), start=1):
            edges_in_comp = [(a, c, b) for a, c, b in edges if a in comp and b in comp]
            facts = comp & fact_tables
            shape, reason = classify_shape(comp, edges_in_comp, facts)
            names = sorted(facts) if facts else sorted(comp - facts)
            title = "_".join(names[:2]) if names else "componente"
            if len(names) > 2:
                title += "_e_outros"
            fname = out_dir / f"schema_{i:02d}_{title}.mermaid"
            fname.write_text(render_mermaid(comp, edges_in_comp, tables, facts, shape, reason), encoding="utf-8")
            index.append({"file": fname.name, "shape": shape, "tables": sorted(comp),
                          "facts": sorted(facts), "reason": reason})
            print(f"[{i}] {shape:35s} {fname.name}  ({len(comp)} tabelas)")

        (out_dir / "_isolated_tables.txt").write_text(
            "Tabelas sem nenhuma FK resolvida (nao pertencem a nenhum "
            "esquema estrela/snowflake ainda):\n\n" + "\n".join(isolated) + "\n",
            encoding="utf-8",
        )
        (out_dir / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(isolated)} tabelas isoladas -> {out_dir / '_isolated_tables.txt'}")
    finally:
        con.close()
        if schema_con is not None:
            schema_con.close()


if __name__ == "__main__":
    main()