#!/usr/bin/env python3
"""
Build a synthetic-but-structurally-realistic physical warehouse from the
semantic layer in metric_catalog.duckdb.

This is NOT a guess echoed back from the candidate bindings (that would be
circular — of course it would "match"). It's an independently-designed
physical schema following normal warehouse conventions (fct_/dim_ prefixes,
one identifier column per table, foreign keys as '<entity>_id'), seeded with
small amounts of consistent synthetic data so FKs actually join. Two known
false-conflicts from the semantic layer are deliberately modeled as ONE
table with a status/type column instead of N separate tables, because
that's what a real warehouse would almost certainly do:
  - opportunity: one fct_opportunity table with a 'stage' column
    (open / closed_won / closed_lost) plus attribution flags, instead of
    separate closed_won / open_opportunity / attributed_closed_won /
    influenced_closed_won tables.
  - movement: one fct_mrr_movement table with a 'movement_type' column
    (expansion / contraction / reactivation), instead of separate tables
    per movement type.
survey_response is deliberately NOT consolidated: customer effort score,
employee engagement, and CSAT are genuinely different survey systems
measuring different things for different subjects, so they get three
separate real tables. This is the general rule, not a survey_response
special case: any entity that is NOT listed in _CONSOLIDATED but has more
than one physical source competing for the same (entity, column) — detected
from the '<source>__<column>' qualified names to_duckdb.py's collision
guard produces — is split into one physical table per source. _CONSOLIDATED
is the deliberate, named exception to that default, not the default itself.

Every table also gets a plain 'id' column that mirrors the table's real
primary key. Several metric formulas in the catalog use '<entity>.id' as
pseudocode shorthand for "this row's own identifier" (e.g.
'COUNT(DISTINCT customer.id WHERE status=...)') rather than the entity's
actual PK column name ('customer_id'). Without this alias, those formulas
have no literal column to resolve against anywhere in the warehouse. This
is purely a convenience alias for running formulas as-written — it is not
a second source of truth; the real PK column is still the primary key.

Usage:
    python3 src/build_warehouse.py data/metric_catalog.duckdb data/warehouse.duckdb
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import duckdb

random.seed(42)  # deterministic sample data across runs

SCHEMA = "analytics"

# Entities consolidated into a single physical table with a discriminator
# column, instead of the naive one-table-per-attribute-source approach.
# Maps: consolidated table -> {qualifier_value: [attribute_name suffixes present]}
_CONSOLIDATED = {
    "opportunity": {
        "table": "fct_opportunity",
        "discriminator_column": "stage",
        "variants": {
            "closed_won":            "closed_won",
            "open_opportunity":      "open",
            "attributed_closed_won": "closed_won",   # same stage, different attribution flag
            "influenced_closed_won": "closed_won",
        },
        "extra_columns": [
            ("marketing_attributed", "BOOLEAN"),
            ("marketing_influenced", "BOOLEAN"),
        ],
    },
    "movement": {
        "table": "fct_mrr_movement",
        "discriminator_column": "movement_type",
        "variants": {
            "expansion":    "expansion",
            "contraction":  "contraction",
            "reactivation": "reactivation",
        },
        "extra_columns": [],
    },
}

_TYPE_MAP = {
    "identifier": "VARCHAR",
    "entity_reference": "VARCHAR",
    "measure": "DOUBLE",
    "categorical": "VARCHAR",
    "temporal": "TIMESTAMP",
    "boolean": "BOOLEAN",
    "ordinal": "INTEGER",
    "descriptive": "VARCHAR",
}

_CATEGORICAL_SAMPLES = {
    "segment": ["enterprise", "mid_market", "smb"],
    "status": ["active", "inactive", "pending"],
    "priority": ["low", "medium", "high", "urgent"],
    "channel": ["web", "mobile", "partner"],
    "severity": ["sev1", "sev2", "sev3", "sev4"],
    "category": ["electronics", "apparel", "home", "other"],
    "region": ["north", "south", "east", "west"],
    "platform": ["web", "ios", "android"],
    "source": ["organic", "paid", "referral"],
}


_UNRESOLVED_FK = "__unresolved__"  # sentinel: FK-shaped column, no resolvable target entity

# Entities whose primary key is a calendar grain rather than a surrogate id.
# Their PK pool gets real, meaningful date-like values instead of
# '<entity>_0001' placeholders (see _pk_pool_for_entity).
_TEMPORAL_GRAIN_ENTITIES = {"day", "week", "month", "period"}


@dataclass
class TableSpec:
    entity_id: str
    table_name: str
    pk_column: str
    # (name, sql_type, semantic_type, fk_target_entity_id | _UNRESOLVED_FK | None)
    columns: list[tuple[str, str, str, str | None]] = field(default_factory=list)
    extra_columns: list[tuple[str, str]] = field(default_factory=list)  # (name, sql_type) — no semantic meaning
    discriminator_column: str | None = None


def _fetch_catalog(con: duckdb.DuckDBPyConnection) -> dict:
    entities = [r[0] for r in con.execute("SELECT entity_id FROM entity ORDER BY 1").fetchall()]

    pk_columns: dict[str, str] = {}
    for entity_id in entities:
        row = con.execute("""
            SELECT ab.column_name FROM attribute_binding ab
            WHERE ab.attribute_id = ? AND ab.column_name IS NOT NULL
            ORDER BY (ab.resolution_state = 'resolved') DESC, ab.binding_id
            LIMIT 1
        """, [f"{entity_id}.identifier"]).fetchone()
        pk_columns[entity_id] = row[0] if row else f"{entity_id}_id"

    attrs = con.execute("""
        SELECT entity_id, name, semantic_type, references_entity_id, attribute_id
        FROM semantic_attribute
        WHERE semantic_type != 'identifier'
        ORDER BY entity_id, name
    """).fetchall()

    # Attributes that an entity_relation actually depends on as a join endpoint.
    # A relation's existence is a stronger, more load-bearing signal that an
    # attribute is a foreign key than semantic_attribute.semantic_type is —
    # the latter can be (and in practice sometimes is) misclassified upstream
    # by inference heuristics. Rather than silently building a physical column
    # that a real, decomposed relation can never join against, treat any
    # relation endpoint as an entity_reference for warehouse-construction
    # purposes regardless of its declared semantic_type.
    relation_attrs = set()
    for row in con.execute(
        "SELECT from_attribute_id FROM entity_relation WHERE from_attribute_id IS NOT NULL "
        "UNION SELECT to_attribute_id FROM entity_relation WHERE to_attribute_id IS NOT NULL"
    ).fetchall():
        relation_attrs.add(row[0])

    # attribute_id -> target entity_id, for FK-shaped columns. Only the
    # from_attribute_id side is mapped: it's the actual foreign-key column on
    # from_entity_id, pointing at to_entity_id. to_attribute_id is normally
    # the target's OWN identifier (e.g. 'customer.identifier'), not a second
    # foreign key, so it is deliberately not mapped here.
    fk_targets: dict[str, str] = {}
    for from_attr, to_entity in con.execute(
        "SELECT from_attribute_id, to_entity_id FROM entity_relation "
        "WHERE from_attribute_id IS NOT NULL"
    ).fetchall():
        fk_targets[from_attr] = to_entity

    # entity_id -> set(dataset_id) is not needed here; what we need is which
    # entities are the grain of a count-style metric (COUNT(x.id ...) /
    # COUNT(DISTINCT x.id ...)) even though they carry no declared 'measure'
    # attribute. Those are transactional/event entities (ticket, incident,
    # delivery, ...) and belong in fct_, not dim_, even without a measure.
    countable_entities = {
        r[0] for r in con.execute(
            "SELECT DISTINCT entity_id FROM metric_definition "
            "WHERE aggregation IN ('count', 'count_distinct')"
        ).fetchall()
    }

    return {
        "entities": entities,
        "pk_columns": pk_columns,
        "attrs": attrs,
        "relation_attrs": relation_attrs,
        "fk_targets": fk_targets,
        "countable_entities": countable_entities,
    }


def build_table_specs(catalog: dict) -> tuple[dict[str, "TableSpec"], dict[str, str], dict[str, dict[str, str]]]:
    """Returns (table_specs keyed by physical table_name, entity_to_table
    keyed by entity_id — the entity's own/default table; for split entities
    this is informational only — and entity_variant_tables, keyed by
    entity_id -> {variant_dataset_id: physical_table_name}, for entities
    with more than one physical source. Callers outside this module (e.g.
    to_dbt.py) use entity_variant_tables together with _CONSOLIDATED to
    resolve a raw dataset_id to the physical table it actually lands in —
    which is NOT always the same as a table literally named after the
    dataset_id; see _CONSOLIDATED for the deliberate exceptions."""
    entities, pk_columns, attrs = catalog["entities"], catalog["pk_columns"], catalog["attrs"]
    relation_attrs = catalog.get("relation_attrs", set())
    fk_targets = catalog.get("fk_targets", {})

    measure_attrs = {
        (e, name) for e, name, t, r, aid in attrs
        if t == "measure" and name != pk_columns.get(e)
    }
    measure_entities = {e for e, _name in measure_attrs}
    # An entity is "fact-like" if it has a declared measure OR is the grain
    # of a count-style metric (COUNT(ticket.id ...)). The latter is common
    # for transactional/event entities (ticket, incident, delivery, hire...)
    # that never carry a 'measure' attribute because they're counted, not
    # summed/averaged — without this they'd be mislabeled dim_ despite being
    # facts.
    # Exception: calendar-grain entities (day/week/month/period) show up as
    # the grain of count_distinct metrics like DAU/WAU/MAU ("count_distinct
    # users, reported by day") — there the countable subject is the *user*,
    # not the day. Being someone else's reporting grain doesn't make an
    # entity fact-like; keep these as what they structurally are: date
    # dimensions.
    countable_entities = catalog.get("countable_entities", set()) - _TEMPORAL_GRAIN_ENTITIES
    fact_entities = measure_entities | countable_entities

    # Detect "split" entities: NOT explicitly consolidated, but with
    # dataset-qualified ('<source>__<column>') attribute names — i.e. two or
    # more genuinely different physical sources compete for the same
    # (entity, column). Per this module's design rule, the default for that
    # shape is N separate real tables, one per source.
    variants_by_entity: dict[str, set[str]] = defaultdict(set)
    for entity_id, name, semantic_type, references_entity_id, attribute_id in attrs:
        if entity_id in _CONSOLIDATED:
            continue
        if "__" in name:
            variant, _rest = name.split("__", 1)
            variants_by_entity[entity_id].add(variant)

    entity_to_table: dict[str, str] = {}
    for entity_id in entities:
        entity_to_table[entity_id] = f"fct_{entity_id}" if entity_id in fact_entities else f"dim_{entity_id}"
    for entity_id, cfg in _CONSOLIDATED.items():
        entity_to_table[entity_id] = cfg["table"]

    # entity_id -> {variant: table_name}, only for split entities.
    entity_variant_tables: dict[str, dict[str, str]] = {}
    for entity_id, variants in variants_by_entity.items():
        entity_variant_tables[entity_id] = {
            variant: f"fct_{entity_id}__{variant}" for variant in sorted(variants)
        }

    specs: dict[str, TableSpec] = {}

    def _get_or_create(table_name: str, entity_id: str) -> TableSpec:
        if table_name not in specs:
            cfg = _CONSOLIDATED.get(entity_id)
            specs[table_name] = TableSpec(
                entity_id=entity_id,
                table_name=table_name,
                pk_column=pk_columns[entity_id],
                extra_columns=list(cfg["extra_columns"]) if cfg else [],
                discriminator_column=cfg["discriminator_column"] if cfg else None,
            )
        return specs[table_name]

    for entity_id in entities:
        if entity_id in entity_variant_tables:
            for table_name in entity_variant_tables[entity_id].values():
                _get_or_create(table_name, entity_id)
        else:
            _get_or_create(entity_to_table[entity_id], entity_id)

    for entity_id, name, semantic_type, references_entity_id, attribute_id in attrs:
        variant: str | None = None
        is_relation_fk = attribute_id in relation_attrs
        is_fk_like = semantic_type == "entity_reference" or is_relation_fk
        if is_fk_like:
            if is_relation_fk and semantic_type != "entity_reference":
                print(
                    f"  note: '{attribute_id}' is declared semantic_type="
                    f"'{semantic_type}' but is used as an entity_relation "
                    "endpoint — building it as a foreign-key column anyway.",
                    file=sys.stderr,
                )
            col_name = f"{name}_id" if not name.endswith("_id") else name
        else:
            col_name = name
            if "__" in name:
                candidate_variant, rest = name.split("__", 1)
                if entity_id in _CONSOLIDATED:
                    # Consolidated: variants share ONE physical column — strip
                    # the qualifier, don't split into a separate table.
                    col_name = rest
                else:
                    col_name, variant = rest, candidate_variant

        # An attribute that shares its name with the entity's OWN primary-key
        # column is not a second, independent data column — building it as
        # one would let it silently overwrite the seeded identifier later
        # (create_and_seed builds the row dict by column name, so the last
        # write wins). This is exactly what happened with the catalog's own
        # 'product.product_id' attribute, which is misclassified as a
        # 'measure' but literally named after product's PK: skip it here
        # rather than let bad upstream classification corrupt every join
        # into this table.
        pk_col_for_entity = pk_columns.get(entity_id)
        if col_name in (pk_col_for_entity, "id"):
            print(
                f"  note: '{attribute_id}' has the same name as {entity_id}'s "
                f"primary key ('{pk_col_for_entity}') — skipping it as a data "
                "column so it cannot overwrite the seeded identifier.",
                file=sys.stderr,
            )
            continue

        # Foreign keys are always VARCHAR (they hold another entity's id), even
        # if the misclassified semantic_type would otherwise map to DOUBLE etc.
        sql_type = "VARCHAR" if is_relation_fk else _TYPE_MAP.get(semantic_type, "VARCHAR")

        # Resolve WHERE an FK-shaped column actually points, explicitly —
        # never by re-guessing from the column's own name at seed time.
        # Priority: (1) an explicit references_entity_id on the attribute,
        # (2) the entity_relation graph (from_attribute_id -> to_entity_id),
        # (3) a same-named registered entity, as a last-resort inference —
        # only accepted if that entity is real. If none of these resolve,
        # the column is left explicitly unresolved: it will be seeded as
        # NULL rather than as a plausible-looking value that can never join
        # (e.g. 'opportunity.owner_id' / 'account.plan_id', which reference
        # an 'owner'/'plan' entity that doesn't exist anywhere in the
        # catalog — better an honest NULL than a fabricated FK).
        fk_target: str | None = None
        if is_fk_like:
            guessed = col_name[:-3] if col_name.endswith("_id") else col_name
            if references_entity_id and references_entity_id in entities:
                fk_target = references_entity_id
            elif attribute_id in fk_targets and fk_targets[attribute_id] in entities:
                fk_target = fk_targets[attribute_id]
            elif guessed in entities:
                fk_target = guessed
            else:
                fk_target = _UNRESOLVED_FK
                print(
                    f"  note: '{attribute_id}' looks like a foreign key "
                    f"(column '{col_name}') but no entity '{guessed}' — or "
                    "any other resolvable target — exists in the catalog. "
                    "Seeding it as NULL instead of a fabricated, unjoinable id.",
                    file=sys.stderr,
                )

        if entity_id in entity_variant_tables and variant is not None:
            # Column belongs only to its own source's table.
            target_tables = [entity_variant_tables[entity_id][variant]]
        elif entity_id in entity_variant_tables:
            # Shared/common column (e.g. an FK) — every split table needs it.
            target_tables = list(entity_variant_tables[entity_id].values())
        else:
            target_tables = [entity_to_table[entity_id]]

        for table_name in target_tables:
            spec = specs[table_name]
            entry = (col_name, sql_type, semantic_type, fk_target)
            if entry not in spec.columns:
                spec.columns.append(entry)

    return specs, entity_to_table, entity_variant_tables


_ENTITY_CATEGORICAL_SAMPLES: dict[tuple[str, str], list[str]] = {
    # Entity-specific overrides, checked before the generic _CATEGORICAL_SAMPLES
    # below — e.g. 'status' means something different per entity (a lead's
    # lifecycle stage vs a generic active/inactive flag), and marketing's
    # mql_volume/sql_volume metrics filter on lead.status specifically
    # expecting 'mql'/'sql' as real values, not the generic fallback.
    ("lead", "status"): ["new", "mql", "sql", "disqualified"],
}


def _sample_value(col_name: str, sql_type: str, row_idx: int, fk_pools: dict[str, list[str]],
                   fk_target: str | None = None, entity_id: str | None = None):
    # FK-shaped columns are resolved explicitly (via fk_target, computed once
    # in build_table_specs) rather than by re-guessing from col_name here.
    if fk_target == _UNRESOLVED_FK:
        return None  # honest NULL — no real target entity exists to join to
    if fk_target is not None:
        return random.choice(fk_pools[fk_target])

    if sql_type == "DOUBLE":
        return round(random.uniform(10, 5000), 2)
    if sql_type == "BOOLEAN":
        return random.choice([True, False])
    if sql_type == "TIMESTAMP":
        return f"2026-{random.randint(1,8):02d}-{random.randint(1,28):02d} 00:00:00"
    if sql_type == "INTEGER":
        return random.randint(1, 10)
    # VARCHAR, not an FK
    base = col_name[:-3] if col_name.endswith("_id") else col_name
    if entity_id and (entity_id, col_name) in _ENTITY_CATEGORICAL_SAMPLES:
        return random.choice(_ENTITY_CATEGORICAL_SAMPLES[(entity_id, col_name)])
    if base in _CATEGORICAL_SAMPLES:
        return random.choice(_CATEGORICAL_SAMPLES[base])
    return f"{base}_{row_idx}"


def _pk_pool_for_entity(entity_id: str, n_rows: int) -> list[str]:
    """Real, meaningful PK values for calendar-grain entities (day/week/
    month/period) instead of the generic '<entity>_0001' surrogate — a date
    dimension whose 'date_day' column holds the string 'day_0001' is not
    structurally realistic, it just looks like an id. Everything else keeps
    the generic surrogate-id pool."""
    if entity_id == "day":
        return [f"2026-01-{i:02d}" for i in range(1, n_rows + 1)]
    if entity_id == "week":
        # Monday-anchored week-start dates, one week apart.
        return [f"2026-{1 + (i - 1) // 4:02d}-{1 + ((i - 1) % 4) * 7:02d}" for i in range(1, n_rows + 1)]
    if entity_id == "month":
        return [f"2026-{i:02d}-01" for i in range(1, n_rows + 1)]
    if entity_id == "period":
        # Sequential quarters starting 2025-Q1, never repeating regardless of
        # n_rows (a PK pool must be unique — 4 fixed quarters would collide
        # for n_rows > 4).
        return [f"{2025 + (i - 1) // 4}-Q{1 + (i - 1) % 4}" for i in range(1, n_rows + 1)]
    return [f"{entity_id}_{i:04d}" for i in range(1, n_rows + 1)]


def create_and_seed(con: duckdb.DuckDBPyConnection, specs: dict[str, TableSpec], n_rows: int = 8) -> None:
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    # Create tables first (order doesn't matter, DuckDB doesn't enforce FK by default here)
    for spec in specs.values():
        col_defs = [f'"{spec.pk_column}" VARCHAR PRIMARY KEY']
        seen = {spec.pk_column}
        # Plain 'id' alias for the entity's own PK. Several metric formulas use
        # '<entity>.id' as pseudocode shorthand for "this row's own identifier"
        # (e.g. 'COUNT(DISTINCT customer.id WHERE ...)') rather than the real PK
        # column name ('customer_id'). This is purely a convenience alias for
        # running formulas as-written — the real PK column above remains the
        # actual primary key. Skipped when pk_column is already literally 'id'.
        if "id" not in seen:
            col_defs.append('"id" VARCHAR')
            seen.add("id")
        for col_name, sql_type, _sem_type, _fk_target in spec.columns:
            if col_name in seen:
                continue
            seen.add(col_name)
            col_defs.append(f'"{col_name}" {sql_type}')
        if spec.discriminator_column and spec.discriminator_column not in seen:
            col_defs.append(f'"{spec.discriminator_column}" VARCHAR')
            seen.add(spec.discriminator_column)
        for col_name, sql_type in spec.extra_columns:
            if col_name in seen:
                continue
            seen.add(col_name)
            col_defs.append(f'"{col_name}" {sql_type}')
        con.execute(f'DROP TABLE IF EXISTS {SCHEMA}."{spec.table_name}"')
        con.execute(f'CREATE TABLE {SCHEMA}."{spec.table_name}" ({", ".join(col_defs)})')

    # Seed PK pools first so FK columns can reference real values
    pk_pools: dict[str, list[str]] = {
        spec.entity_id: _pk_pool_for_entity(spec.entity_id, n_rows)
        for spec in specs.values()
    }
    # Consolidated tables expose their pk pool under each folded entity name too
    for entity_id, cfg in _CONSOLIDATED.items():
        table = cfg["table"]
        owner_spec = specs[table]
        pk_pools[entity_id] = pk_pools[owner_spec.entity_id]

    for spec in specs.values():
        variants = _CONSOLIDATED.get(spec.entity_id, {}).get("variants") if spec.entity_id in _CONSOLIDATED else None
        for i in range(1, n_rows + 1):
            pk_value = pk_pools[spec.entity_id][i - 1]
            row = {spec.pk_column: pk_value}
            if spec.pk_column != "id":
                row["id"] = pk_value
            for col_name, sql_type, sem_type, fk_target in spec.columns:
                # Belt-and-braces: build_table_specs already excludes columns
                # that collide with the PK/id, but never let a data column
                # clobber the identifier we just set, no matter what.
                if col_name in row:
                    continue
                row[col_name] = _sample_value(col_name, sql_type, i, pk_pools, fk_target, spec.entity_id)
            if spec.discriminator_column:
                choices = list(variants.values()) if variants else ["default"]
                row[spec.discriminator_column] = random.choice(choices)
            for col_name, sql_type in spec.extra_columns:
                if col_name in row:
                    continue
                row[col_name] = _sample_value(col_name, sql_type, i, pk_pools, entity_id=spec.entity_id)
            cols = list(row.keys())
            placeholders = ", ".join("?" for _ in cols)
            quoted_cols = ", ".join(f'"{c}"' for c in cols)
            con.execute(
                f'INSERT INTO {SCHEMA}."{spec.table_name}" ({quoted_cols}) VALUES ({placeholders})',
                [row[c] for c in cols],
            )


def main() -> None:
    src_db = sys.argv[1] if len(sys.argv) > 1 else "data/metric_catalog.duckdb"
    out_db = sys.argv[2] if len(sys.argv) > 2 else "data/warehouse.duckdb"

    src = duckdb.connect(src_db, read_only=True)
    catalog = _fetch_catalog(src)
    src.close()

    specs, entity_to_table, _entity_variant_tables = build_table_specs(catalog)

    import os
    if os.path.exists(out_db):
        os.remove(out_db)
    out = duckdb.connect(out_db)
    try:
        create_and_seed(out, specs)
        n_tables = len(specs)
        n_cols = sum(len(s.columns) + 1 + len(s.extra_columns) + (1 if s.discriminator_column else 0)
                     + (0 if s.pk_column == "id" else 1)
                     for s in specs.values())
        print(f"Built {out_db}: {n_tables} table(s), ~{n_cols} column(s), covering "
              f"{len(catalog['entities'])} entities ({len(_CONSOLIDATED)} consolidated).")
        for name, cfg in _CONSOLIDATED.items():
            print(f"  consolidated: {', '.join(cfg['variants'].keys())} -> "
                  f"{SCHEMA}.{cfg['table']} (filter on '{cfg['discriminator_column']}')")
    finally:
        out.close()


if __name__ == "__main__":
    main()