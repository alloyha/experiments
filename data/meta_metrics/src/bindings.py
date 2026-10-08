"""
Binding resolution for the semantic layer.

Resolves semantic concepts (SemanticAttribute, EntityRelation) to their
physical/logical implementation (Dataset, Column, Expression) through
explicit attribute_binding rows — never by re-deriving a physical reference
from a name.

Core invariant: resolution only considers bindings with
resolution_state = 'resolved'. A 'candidate' binding (typically produced by
the bootstrap migration in to_duckdb.py or an inference heuristic) is not
usable for execution until a human — or an explicit, deliberately authorized
rule — promotes it. This means resolution can, and by design sometimes will,
fail on a freshly-migrated catalog: that is the point. Silent guessing is
exactly what this module exists to prevent.

Usage:
    from bindings import resolve_attribute_binding, resolve_entity_identifier, \
        resolve_entity_relation, UnresolvedBindingError, AmbiguousBindingError

    binding = resolve_attribute_binding(con, "customer.identifier")
    join = resolve_entity_relation(con, "invoice→customer",
                                    left_dataset_id="analytics.fct_invoice",
                                    right_dataset_id="analytics.dim_customer")
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb

RESOLVABLE_STATE = "resolved"


# ── Errors ───────────────────────────────────────────────────────────────────

class BindingError(Exception):
    """Base class for binding resolution failures."""


class UnresolvedBindingError(BindingError):
    """Raised when zero valid (resolved) bindings exist for a request.

    Distinguishes two situations in the message: no bindings at all, vs.
    bindings that exist but are still 'candidate' (i.e. inferred and not yet
    reviewed) — the latter is actionable ("promote a candidate") rather than
    "go create a binding from scratch".
    """

    def __init__(self, attribute_id: str, *, candidates: list[str] | None = None,
                 detail: str = ""):
        self.attribute_id = attribute_id
        self.candidates = candidates or []
        msg = f"No resolved binding for '{attribute_id}'."
        if self.candidates:
            msg += (
                f" {len(self.candidates)} candidate binding(s) exist but are "
                f"not resolved: {', '.join(self.candidates)}. "
                "Promote one explicitly if it is correct — do not assume it."
            )
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class AmbiguousBindingError(BindingError):
    """Raised when multiple resolved bindings qualify and no deterministic
    selector (dataset_id / engine) was given to disambiguate."""

    def __init__(self, attribute_id: str, candidates: list["AttributeBinding"]):
        self.attribute_id = attribute_id
        self.candidates = candidates
        options = ", ".join(f"{c.binding_id} (dataset={c.dataset_id}, engine={c.engine})"
                             for c in candidates)
        super().__init__(
            f"Ambiguous binding for '{attribute_id}': {len(candidates)} resolved "
            f"bindings qualify [{options}]. Pass dataset_id and/or engine to "
            "disambiguate — an arbitrary choice is not made automatically."
        )


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class AttributeBinding:
    binding_id: str
    attribute_id: str
    dataset_id: str | None
    column_name: str | None
    expression: str | None
    engine: str | None
    binding_role: str
    origin: str
    resolution_state: str
    inference_rule: str | None
    filter_column: str | None = None
    filter_value: str | None = None

    @property
    def physical_expression(self) -> str:
        """Best available physical reference: explicit expression, or
        dataset.column, or bare column_name if dataset is unknown."""
        if self.expression:
            return self.expression
        if self.dataset_id and self.column_name:
            return f"{self.dataset_id}.{self.column_name}"
        if self.column_name:
            return self.column_name
        raise BindingError(
            f"Binding '{self.binding_id}' has neither expression, dataset+column, "
            "nor a bare column — nothing physical to resolve to."
        )

    @property
    def filter_expression(self) -> str | None:
        """WHERE-clause fragment a consumer must apply, when this attribute
        is one filtered slice of a physical table shared by several
        attributes (e.g. opportunity.closed_won__amount lives in the same
        table as opportunity.open_opportunity__amount, distinguished only by
        `stage`). None for the common case of an attribute owning its table
        outright — callers must not assume every row of the resolved table
        belongs to this attribute unless they check this."""
        if self.filter_column and self.filter_value is not None:
            return f"{self.filter_column} = '{self.filter_value}'"
        return None


@dataclass
class ResolvedJoin:
    relation_id: str
    left_expression: str
    right_expression: str
    join_expression: str


@dataclass
class CubeExecutabilityResult:
    executable: bool
    unresolved_attributes: list[str]
    ambiguous_bindings: list[str]
    missing_relations: list[str]


def _row_to_binding(row: dict) -> AttributeBinding:
    return AttributeBinding(
        binding_id=row["binding_id"],
        attribute_id=row["attribute_id"],
        dataset_id=row["dataset_id"],
        column_name=row["column_name"],
        expression=row["expression"],
        engine=row["engine"],
        binding_role=row["binding_role"],
        origin=row["origin"],
        resolution_state=row["resolution_state"],
        inference_rule=row["inference_rule"],
        filter_column=row.get("filter_column"),
        filter_value=row.get("filter_value"),
    )


def _q(con: duckdb.DuckDBPyConnection, sql: str, params: list) -> list[dict]:
    result = con.execute(sql, params)
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, r)) for r in result.fetchall()]


# ── Core resolution ────────────────────────────────────────────────────────

def resolve_attribute_binding(
    con: duckdb.DuckDBPyConnection,
    attribute_id: str,
    *,
    dataset_id: str | None = None,
    engine: str | None = None,
) -> AttributeBinding:
    """Resolve a semantic attribute to exactly one physical binding.

    Only resolution_state = 'resolved' bindings are considered. Raises
    UnresolvedBindingError if none qualify, AmbiguousBindingError if more
    than one qualifies and dataset_id/engine weren't enough to narrow it
    down. Never guesses.
    """
    if not _q(con, "SELECT 1 FROM semantic_attribute WHERE attribute_id = ?", [attribute_id]):
        raise UnresolvedBindingError(attribute_id, detail="Attribute is not declared at all.")

    where = ["attribute_id = ?", "resolution_state = ?"]
    params: list = [attribute_id, RESOLVABLE_STATE]
    if dataset_id is not None:
        where.append("dataset_id = ?")
        params.append(dataset_id)
    if engine is not None:
        where.append("(engine = ? OR engine IS NULL)")
        params.append(engine)

    rows = _q(con, f"SELECT * FROM attribute_binding WHERE {' AND '.join(where)}", params)

    if not rows:
        candidates = _q(
            con,
            "SELECT binding_id FROM attribute_binding "
            "WHERE attribute_id = ? AND resolution_state = 'candidate'",
            [attribute_id],
        )
        raise UnresolvedBindingError(
            attribute_id, candidates=[c["binding_id"] for c in candidates]
        )

    if len(rows) > 1:
        raise AmbiguousBindingError(attribute_id, [_row_to_binding(r) for r in rows])

    return _row_to_binding(rows[0])


def resolve_entity_identifier(
    con: duckdb.DuckDBPyConnection,
    entity_id: str,
    *,
    dataset_id: str | None = None,
) -> AttributeBinding:
    """Resolve an entity's canonical identifier to a physical binding.

    Entity identity is represented semantically as '{entity_id}.identifier',
    never as a raw column name — ENTITY_PK-style mappings are only ever a
    bootstrap hint feeding a *candidate* binding of this attribute.
    """
    attribute_id = f"{entity_id}.identifier"
    row = _q(
        con,
        "SELECT 1 FROM semantic_attribute WHERE attribute_id = ? AND semantic_type = 'identifier'",
        [attribute_id],
    )
    if not row:
        raise UnresolvedBindingError(
            attribute_id,
            detail=f"Entity '{entity_id}' has no declared identifier attribute.",
        )
    return resolve_attribute_binding(con, attribute_id, dataset_id=dataset_id)


def resolve_entity_relation(
    con: duckdb.DuckDBPyConnection,
    relation_id: str,
    *,
    left_dataset_id: str,
    right_dataset_id: str,
) -> ResolvedJoin:
    """Resolve a semantic entity_relation into a physical join.

    Requires the relation to already carry from_attribute_id / to_attribute_id
    (see cubes.py: _derive_relation_attributes). The legacy join_expression
    column is NOT used here — it is kept only for backward-compatible reads
    elsewhere in the codebase, not as an execution source. If the relation
    hasn't been decomposed into semantic references yet, this raises rather
    than falling back to the raw SQL string.
    """
    rows = _q(
        con,
        "SELECT from_entity_id, to_entity_id, from_attribute_id, to_attribute_id "
        "FROM entity_relation WHERE relation_id = ?",
        [relation_id],
    )
    if not rows:
        raise UnresolvedBindingError(relation_id, detail="No such entity_relation.")
    rel = rows[0]
    from_attr, to_attr = rel["from_attribute_id"], rel["to_attribute_id"]
    if not from_attr or not to_attr:
        raise UnresolvedBindingError(
            relation_id,
            detail=(
                "entity_relation has no from_attribute_id/to_attribute_id — it has not "
                "been decomposed into semantic references. A legacy join_expression may "
                "exist but is not used for resolution."
            ),
        )

    left = resolve_attribute_binding(con, from_attr, dataset_id=left_dataset_id)
    right = resolve_attribute_binding(con, to_attr, dataset_id=right_dataset_id)

    left_expr = _physical_ref(con, left)
    right_expr = _physical_ref(con, right)
    return ResolvedJoin(
        relation_id=relation_id,
        left_expression=left_expr,
        right_expression=right_expr,
        join_expression=f"{left_expr} = {right_expr}",
    )


def _physical_ref(con: duckdb.DuckDBPyConnection, binding: AttributeBinding) -> str:
    """Prefer dataset.table_name.column over dataset_id.column when the
    dataset is registered, since dataset_id is an internal key and
    table_name/full_ref is the actual physical/logical name."""
    if binding.expression:
        return binding.expression
    if binding.dataset_id and binding.column_name:
        ds = _q(con, "SELECT full_ref FROM dataset WHERE dataset_id = ?", [binding.dataset_id])
        table_ref = ds[0]["full_ref"] if ds else binding.dataset_id
        return f"{table_ref}.{binding.column_name}"
    return binding.physical_expression


# ── Review workflow (minimal) ────────────────────────────────────────────────

_ALLOWED_STATES = {"resolved", "candidate", "unresolved", "rejected"}


def promote_binding(con: duckdb.DuckDBPyConnection, binding_id: str,
                     to_state: str = "resolved") -> None:
    """Explicitly promote (or reject) a binding's resolution_state.

    This is the only sanctioned way a 'candidate' binding becomes usable by
    resolve_attribute_binding. There is no automatic promotion and no
    confidence threshold — a human (or an explicitly authorized deterministic
    rule) must call this.
    """
    if to_state not in _ALLOWED_STATES:
        raise ValueError(f"Invalid resolution_state '{to_state}'; must be one of {_ALLOWED_STATES}")
    if not _q(con, "SELECT 1 FROM attribute_binding WHERE binding_id = ?", [binding_id]):
        raise UnresolvedBindingError(binding_id, detail="No such binding_id.")
    con.execute("UPDATE attribute_binding SET resolution_state = ? WHERE binding_id = ?",
                [to_state, binding_id])