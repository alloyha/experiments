"""
Analytical cube cover generation from the metric catalog.

Derives a minimal, semantically safe analytical cube cover from the metric
dependency graph, entity rollup graph, and dimensional compatibility.

Entry points
------------
ensure_entity_relations(con)       — seed entity_relation from ENTITY_RELATIONS
generate_cube_cover(con)           — compute and return CubeCover
persist_cube_cover(con, cover)     — write cover to analytical_cube bridge tables
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import duckdb

# ── Entities that represent time grains, not business objects ────────────────
TIME_GRAIN_ENTITIES: frozenset[str] = frozenset(
    {"month", "period", "day", "week", "row"}
)

# ── Declared entity relationships ─────────────────────────────────────────────
# (relation_id, from_entity, to_entity, cardinality, rollup_safe, temporal, join_expr)
ENTITY_RELATIONS: list[tuple] = [
    # Customer lifecycle
    ("invoice→customer",         "invoice",         "customer",    "many_to_one",  True,  False,
     "invoice.customer_id = customer.customer_id"),
    ("subscription→customer",    "subscription",    "customer",    "many_to_one",  True,  False,
     "subscription.customer_id = customer.customer_id"),
    ("contract→customer",        "contract",        "customer",    "many_to_one",  True,  False,
     "contract.customer_id = customer.customer_id"),
    ("cohort→customer",          "cohort",          "customer",    "many_to_one",  True,  True,
     "cohort.customer_id = customer.customer_id"),
    ("ticket→customer",          "ticket",          "customer",    "many_to_one",  True,  False,
     "ticket.customer_id = customer.customer_id"),
    ("survey_response→customer", "survey_response", "customer",    "many_to_one",  True,  False,
     "survey_response.customer_id = customer.customer_id"),
    # Users are customers (1:1 for B2B; many:1 for B2C — rollup always safe)
    ("user→customer",            "user",            "customer",    "one_to_one",   True,  False,
     "user.customer_id = customer.customer_id"),
    ("session→user",             "session",         "user",        "many_to_one",  True,  False,
     "session.user_id = user.user_id"),
    # Commerce
    ("order→customer",           "order",           "customer",    "many_to_one",  True,  False,
     "order.customer_id = customer.customer_id"),
    ("delivery→order",           "delivery",        "order",       "many_to_one",  True,  False,
     "delivery.order_id = order.order_id"),
    ("cart→user",                "cart",            "user",        "many_to_one",  True,  False,
     "cart.user_id = user.user_id"),
    # Acquisition funnel
    ("opportunity→customer",     "opportunity",     "customer",    "many_to_one",  True,  False,
     "opportunity.customer_id = customer.customer_id"),
    ("lead→opportunity",         "lead",            "opportunity", "many_to_one",  True,  False,
     "lead.opportunity_id = opportunity.opportunity_id"),
    # Campaign attribution is many-to-many: NOT rollup-safe
    ("campaign→customer",        "campaign",        "customer",    "many_to_many", False, False, None),
    # Engineering / Reliability
    ("deployment→service",       "deployment",      "service",     "many_to_one",  True,  False,
     "deployment.service_id = service.service_id"),
    ("incident→service",         "incident",        "service",     "many_to_one",  True,  False,
     "incident.service_id = service.service_id"),
    ("release→service",          "release",         "service",     "many_to_one",  True,  False,
     "release.service_id = service.service_id"),
    ("request→service",          "request",         "service",     "many_to_one",  True,  False,
     "request.service_id = service.service_id"),
    # Data observability
    ("pipeline_run→dataset",     "pipeline_run",    "dataset",     "many_to_one",  True,  False,
     "pipeline_run.dataset_id = dataset.dataset_id"),
    # Supply chain
    ("inventory→sku",            "inventory",       "sku",         "many_to_one",  True,  False,
     "inventory.sku_id = sku.sku_id"),
    ("sku→product",              "sku",             "product",     "many_to_one",  True,  False,
     "sku.product_id = product.product_id"),
    ("purchase_order→product",   "purchase_order",  "product",     "many_to_one",  True,  False,
     "purchase_order.product_id = product.product_id"),
    # Workforce
    ("hire→employee",            "hire",            "employee",    "one_to_one",   True,  False,
     "hire.employee_id = employee.employee_id"),
    ("job→employee",             "job",             "employee",    "many_to_one",  True,  False,
     "job.employee_id = employee.employee_id"),
]

# ── Cube cluster definitions ───────────────────────────────────────────────────
# (cluster_id, display_name, anchor_entity, member_entities, cube_type)
# Derived from entity semantics, not from business domain labels.
CUBE_CLUSTERS: list[tuple[str, str, str, list[str], str]] = [
    (
        "customer_revenue",
        "Customer Revenue",
        "customer",
        ["invoice", "subscription", "contract", "cohort", "customer", "account"],
        "conformed",
    ),
    (
        "acquisition_sales",
        "Acquisition & Sales",
        "opportunity",
        ["lead", "opportunity"],
        "conformed",
    ),
    (
        "product_engagement",
        "Product Engagement",
        "user",
        ["user", "session", "feature"],
        "conformed",
    ),
    (
        "commerce",
        "Commerce",
        "order",
        ["order", "cart", "checkout", "item", "delivery", "offer", "trial"],
        "conformed",
    ),
    (
        "support",
        "Support",
        "ticket",
        ["ticket", "survey_response", "agent"],
        "conformed",
    ),
    (
        "workforce",
        "Workforce",
        "employee",
        ["employee", "hire", "job"],
        "process",
    ),
    (
        "supply_chain_ops",
        "Supply Chain & Operations",
        "product",
        ["product", "sku", "inventory", "purchase_order",
         "operation", "process", "resource", "audit"],
        "conformed",
    ),
    (
        "engineering_data",
        "Engineering & Data",
        "service",
        ["service", "deployment", "incident", "release", "request",
         "pipeline_run", "dataset", "control", "vulnerability"],
        "conformed",
    ),
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class MetricSignature:
    metric_id:               str
    entity_id:               str | None
    dimensions:              frozenset[str]
    dependencies:            frozenset[str]
    dependency_closure:      frozenset[str]
    datasets:                frozenset[str]
    additivity:              str
    non_additive_dimensions: frozenset[str]
    derivation_type:         str
    metric_type:             str
    time_grain:              str | None
    domain:                  str

    @property
    def effective_entity(self) -> str | None:
        """entity_id unless it's a pure time grain — those are not business objects."""
        return None if self.entity_id in TIME_GRAIN_ENTITIES else self.entity_id


@dataclass
class CompatibilityResult:
    compatible:        bool
    common_entity:     str | None
    rollup_hops_left:  int
    rollup_hops_right: int
    reasons:           list[str]
    penalties:         list[str]


@dataclass
class CubeCost:
    cube_count:            int = 0
    cross_cube_dependency: int = 0
    duplicate_metric:      int = 0
    rollup_hop:            int = 0
    unsafe_join:           int = 0

    # (cube_count, cross_dep, dup_metric, rollup_hop, unsafe_join)
    _W: tuple = (10, 3, 2, 1, 1_000_000)

    def score(self) -> float:
        return (
            self._W[0] * self.cube_count
            + self._W[1] * self.cross_cube_dependency
            + self._W[2] * self.duplicate_metric
            + self._W[3] * self.rollup_hop
            + self._W[4] * self.unsafe_join
        )

    def as_dict(self) -> dict:
        return {
            "cube_count":            self.cube_count,
            "cross_cube_dependency": self.cross_cube_dependency,
            "duplicate_metric":      self.duplicate_metric,
            "rollup_hop":            self.rollup_hop,
            "unsafe_join":           self.unsafe_join,
            "score":                 self.score(),
        }


@dataclass
class CubeCandidate:
    cube_id:         str
    name:            str
    entity_id:       str | None
    cube_type:       str
    metrics:         list[MetricSignature] = field(default_factory=list)
    dimensions:      set[str]             = field(default_factory=set)
    datasets:        set[str]             = field(default_factory=set)
    rollup_entities: dict[str, str]       = field(default_factory=dict)
    metric_roles:    dict[str, str]       = field(default_factory=dict)
    metric_reasons:  dict[str, str]       = field(default_factory=dict)
    reasons:         list[str]            = field(default_factory=list)
    rejected:        list[str]            = field(default_factory=list)

    def add_metric(
        self,
        sig: MetricSignature,
        role: str = "native",
        rollup_via: str | None = None,
        reason: str = "",
    ) -> None:
        if sig.metric_id in self.metric_ids:
            return
        self.metrics.append(sig)
        self.dimensions.update(sig.dimensions)
        self.datasets.update(sig.datasets)
        self.metric_roles[sig.metric_id] = role
        if rollup_via:
            self.rollup_entities[sig.metric_id] = rollup_via
        if reason:
            self.metric_reasons[sig.metric_id] = reason

    @property
    def metric_ids(self) -> set[str]:
        return {m.metric_id for m in self.metrics}


@dataclass
class CubeCover:
    cubes:             list[CubeCandidate]
    metrics_covered:   set[str]
    metrics_uncovered: set[str]
    cost:              CubeCost

    def summary(self) -> dict:
        return {
            "cube_count":        len(self.cubes),
            "metrics_covered":   len(self.metrics_covered),
            "metrics_uncovered": len(self.metrics_uncovered),
            "cost":              self.cost.as_dict(),
            "cubes": [
                {
                    "cube_id":      c.cube_id,
                    "name":         c.name,
                    "entity_id":    c.entity_id,
                    "type":         c.cube_type,
                    "metric_count": len(c.metrics),
                    "reasons":      c.reasons,
                }
                for c in self.cubes
            ],
        }


# ── Entity graph ──────────────────────────────────────────────────────────────

class EntityGraph:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.edges: dict[str, list[dict]] = defaultdict(list)
        for row in con.execute("""
            SELECT relation_id, from_entity_id, to_entity_id,
                   cardinality, rollup_safe, temporal
            FROM entity_relation
        """).fetchall():
            self.edges[row[1]].append({
                "relation_id":  row[0],
                "to_entity_id": row[2],
                "cardinality":  row[3],
                "rollup_safe":  row[4],
                "temporal":     row[5],
            })

    def can_rollup(self, from_entity: str, to_entity: str) -> bool:
        return self.safe_rollup_path(from_entity, to_entity) is not None

    def safe_rollup_path(
        self, from_entity: str, to_entity: str
    ) -> list[str] | None:
        """Shortest path following only rollup_safe=True edges."""
        if from_entity == to_entity:
            return [from_entity]
        visited: set[str] = {from_entity}
        queue: deque[list[str]] = deque([[from_entity]])
        while queue:
            path = queue.popleft()
            for edge in self.edges.get(path[-1], []):
                if not edge["rollup_safe"]:
                    continue
                nxt = edge["to_entity_id"]
                if nxt == to_entity:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def common_rollup_entity(self, entities: list[str]) -> str | None:
        """Closest entity all given entities can safely roll up to."""
        if not entities:
            return None
        if len(entities) == 1:
            return entities[0]
        sets = [self._reachable_safe(e) for e in entities]
        common = sets[0].intersection(*sets[1:])
        if not common:
            return None
        best, best_dist = None, float("inf")
        for candidate in common:
            dist = sum(
                len(self.safe_rollup_path(e, candidate) or []) - 1
                for e in entities
            )
            if dist < best_dist:
                best_dist = dist
                best = candidate
        return best

    def _reachable_safe(self, entity: str) -> set[str]:
        visited: set[str] = {entity}
        queue: deque[str] = deque([entity])
        while queue:
            cur = queue.popleft()
            for edge in self.edges.get(cur, []):
                if edge["rollup_safe"] and edge["to_entity_id"] not in visited:
                    visited.add(edge["to_entity_id"])
                    queue.append(edge["to_entity_id"])
        return visited

    def has_unsafe_relation(self, from_entity: str, to_entity: str) -> bool:
        return any(
            e["to_entity_id"] == to_entity and not e["rollup_safe"]
            for e in self.edges.get(from_entity, [])
        )


# ── Metric graph ──────────────────────────────────────────────────────────────

class MetricGraph:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.edges: dict[str, list[str]] = defaultdict(list)
        for row in con.execute(
            "SELECT metric_id, depends_on_metric_id FROM metric_dependency"
        ).fetchall():
            self.edges[row[0]].append(row[1])

    def dependency_closure(self, metric_id: str) -> frozenset[str]:
        visited: set[str] = set()
        queue: deque[str] = deque(self.edges.get(metric_id, []))
        while queue:
            dep = queue.popleft()
            if dep not in visited:
                visited.add(dep)
                queue.extend(self.edges.get(dep, []))
        return frozenset(visited)


# ── Load signatures from DuckDB ───────────────────────────────────────────────

def load_metric_signatures(
    con: duckdb.DuckDBPyConnection,
    metric_graph: MetricGraph,
) -> dict[str, MetricSignature]:
    sigs: dict[str, MetricSignature] = {}

    metrics = {
        r[0]: r
        for r in con.execute("""
            SELECT metric_id, entity_id, additivity, derivation_type,
                   metric_type, time_grain, non_additive_dimensions
            FROM metric_definition
            WHERE status != 'deprecated'
        """).fetchall()
    }

    dims_by: dict[str, set[str]] = defaultdict(set)
    for mid, did in con.execute(
        "SELECT metric_id, dimension_id FROM metric_dimension"
    ).fetchall():
        dims_by[mid].add(did)

    datasets_by: dict[str, set[str]] = defaultdict(set)
    for mid, dsid in con.execute("""
        SELECT mi.metric_id, ic.dataset_id
        FROM impl_column ic
        JOIN metric_implementation mi ON mi.impl_id = ic.impl_id AND mi.is_current = true
    """).fetchall():
        datasets_by[mid].add(dsid)

    deps_by: dict[str, set[str]] = defaultdict(set)
    for mid, dep in con.execute(
        "SELECT metric_id, depends_on_metric_id FROM metric_dependency"
    ).fetchall():
        deps_by[mid].add(dep)

    for mid, row in metrics.items():
        nad = row[6] or []
        sigs[mid] = MetricSignature(
            metric_id=mid,
            entity_id=row[1],
            dimensions=frozenset(dims_by[mid]),
            dependencies=frozenset(deps_by[mid]),
            dependency_closure=metric_graph.dependency_closure(mid),
            datasets=frozenset(datasets_by[mid]),
            additivity=row[2] or "additive",
            non_additive_dimensions=frozenset(nad),
            derivation_type=row[3] or "base",
            metric_type=row[4] or "scalar",
            time_grain=row[5],
            domain=mid.split(".")[0],
        )
    return sigs


# ── Compatibility check ───────────────────────────────────────────────────────

def metrics_compatible(
    left: MetricSignature,
    right: MetricSignature,
    entity_graph: EntityGraph,
) -> CompatibilityResult:
    reasons: list[str] = []
    penalties: list[str] = []
    le, re = left.effective_entity, right.effective_entity

    if le == re:
        label = f"'{le}'" if le else "virtual/time-grain"
        return CompatibilityResult(
            compatible=True, common_entity=le,
            rollup_hops_left=0, rollup_hops_right=0,
            reasons=[f"Same entity: {label}"],
            penalties=["virtual_only"] if not le else [],
        )

    if not le or not re:
        no_entity = left.metric_id if not le else right.metric_id
        return CompatibilityResult(
            compatible=False, common_entity=None,
            rollup_hops_left=0, rollup_hops_right=0,
            reasons=[f"'{no_entity}' has no business entity"],
            penalties=["no_entity"],
        )

    # Explicit many-to-many → reject fact-to-fact join
    if entity_graph.has_unsafe_relation(le, re) or entity_graph.has_unsafe_relation(re, le):
        reasons.append(f"'{le}' ↔ '{re}' is many-to-many (unsafe join)")
        return CompatibilityResult(
            compatible=False, common_entity=None,
            rollup_hops_left=0, rollup_hops_right=0,
            reasons=reasons, penalties=["unsafe_join"],
        )

    common = entity_graph.common_rollup_entity([le, re])
    if not common:
        reasons.append(f"No rollup-safe path from '{le}' or '{re}' to a common entity")
        return CompatibilityResult(
            compatible=False, common_entity=None,
            rollup_hops_left=0, rollup_hops_right=0,
            reasons=reasons, penalties=["no_common_rollup"],
        )

    lpath = entity_graph.safe_rollup_path(le, common) or [le]
    rpath = entity_graph.safe_rollup_path(re, common) or [re]
    hl, hr = len(lpath) - 1, len(rpath) - 1

    reasons += [
        f"'{le}' → '{common}' in {hl} hop(s): {' → '.join(lpath)}",
        f"'{re}' → '{common}' in {hr} hop(s): {' → '.join(rpath)}",
    ]
    if hl > 0 and hr > 0:
        # Both are lower-grain; must aggregate independently to common entity first
        reasons.append(
            f"Fact-to-fact: '{le}' and '{re}' each pre-aggregate to '{common}' before combining"
        )
    if max(hl, hr) > 2:
        penalties.append(f"excessive_rollup_distance (max_hops={max(hl, hr)})")

    return CompatibilityResult(
        compatible=True, common_entity=common,
        rollup_hops_left=hl, rollup_hops_right=hr,
        reasons=reasons, penalties=penalties,
    )


# ── Cube cover algorithm ──────────────────────────────────────────────────────

def generate_cube_cover(con: duckdb.DuckDBPyConnection) -> CubeCover:
    """
    Greedy cube cover.

    Phase 1: Assign each metric to a cluster by effective_entity.
    Phase 2: Assign derived metrics to the cube that contains most of their deps.
    Phase 3: Remaining metrics → virtual cube.
    Phase 4: Count cross-cube dependencies and compute cost.
    """
    entity_graph = EntityGraph(con)
    metric_graph = MetricGraph(con)
    sigs = load_metric_signatures(con, metric_graph)
    if not sigs:
        return CubeCover(cubes=[], metrics_covered=set(), metrics_uncovered=set(),
                         cost=CubeCost())

    assigned: dict[str, str] = {}
    cubes: dict[str, CubeCandidate] = {}

    # Phase 1: entity-cluster assignment
    for cid, display_name, anchor, members, ctype in CUBE_CLUSTERS:
        cube = CubeCandidate(
            cube_id=f"cube_{cid}",
            name=display_name,
            entity_id=anchor,
            cube_type=ctype,
            reasons=[f"Anchor entity: {anchor}; member entities: {', '.join(members)}"],
        )
        for mid, sig in sigs.items():
            if mid in assigned:
                continue
            eff = sig.effective_entity
            if eff not in members:
                continue
            rollup_via: str | None = None
            reason = f"entity_id='{eff}' is in cluster '{cid}'"
            if eff != anchor:
                path = entity_graph.safe_rollup_path(eff, anchor)
                if path:
                    rollup_via = anchor
                    reason += f"; rolls up {' → '.join(path)}"
                else:
                    reason += f" (no rollup path to anchor '{anchor}')"
            cube.add_metric(sig, role="native", rollup_via=rollup_via, reason=reason)
            assigned[mid] = cube.cube_id

        if cube.metrics:
            cubes[cube.cube_id] = cube

    # Phase 2: derived metrics follow their deps
    for mid, sig in sigs.items():
        if mid in assigned:
            continue
        if sig.derivation_type == "derived" and sig.dependencies:
            dep_counts: dict[str, int] = defaultdict(int)
            for dep_id in sig.dependency_closure:
                if dep_id in assigned:
                    dep_counts[assigned[dep_id]] += 1
            if dep_counts:
                best_cid = max(dep_counts, key=dep_counts.get)
                reason = (
                    f"derived; {dep_counts[best_cid]}/{len(sig.dependency_closure)} "
                    f"deps in '{best_cid}'"
                )
                cubes[best_cid].add_metric(sig, role="dependency", reason=reason)
                assigned[mid] = best_cid

    # Phase 3: everything else → virtual cube
    virtual = CubeCandidate(
        cube_id="cube_virtual",
        name="Virtual / Cross-Domain",
        entity_id=None,
        cube_type="virtual",
        reasons=["Metrics without a single business entity or with cross-domain deps"],
    )
    for mid, sig in sigs.items():
        if mid not in assigned:
            ent = sig.entity_id
            reason = (
                f"No cluster match — entity_id='{ent}', "
                f"derivation_type='{sig.derivation_type}'"
            )
            virtual.add_metric(sig, role="composite", reason=reason)
            assigned[mid] = virtual.cube_id

    if virtual.metrics:
        cubes[virtual.cube_id] = virtual

    # Phase 4: scoring
    cross_cube_deps = sum(
        1
        for mid, cid in assigned.items()
        for dep_id in sigs[mid].dependencies
        if dep_id in assigned and assigned[dep_id] != cid
    )
    rollup_hops = sum(
        len(entity_graph.safe_rollup_path(sig.effective_entity, cube.entity_id) or []) - 1
        for cube in cubes.values()
        if cube.entity_id
        for sig in cube.metrics
        if sig.effective_entity
        and sig.effective_entity != cube.entity_id
        and entity_graph.can_rollup(sig.effective_entity, cube.entity_id)
    )
    covered = set(assigned)
    all_ids = set(sigs)

    return CubeCover(
        cubes=list(cubes.values()),
        metrics_covered=covered,
        metrics_uncovered=all_ids - covered,
        cost=CubeCost(
            cube_count=len(cubes),
            cross_cube_dependency=cross_cube_deps,
            rollup_hop=rollup_hops,
        ),
    )


def persist_cube_cover(con: duckdb.DuckDBPyConnection, cover: CubeCover) -> None:
    """Write cube cover to DB. Idempotent — clears previous cover first."""
    con.execute("DELETE FROM cube_dataset")
    con.execute("DELETE FROM cube_dimension")
    con.execute("DELETE FROM cube_metric")
    con.execute("DELETE FROM analytical_cube")

    for cube in cover.cubes:
        entities_in_cube = ", ".join(sorted({
            s.entity_id for s in cube.metrics if s.entity_id
        } - TIME_GRAIN_ENTITIES)) or "none"
        explanation = (
            f"type={cube.cube_type} | anchor={cube.entity_id or 'none'} | "
            f"entities=[{entities_in_cube}] | "
            + " | ".join(cube.reasons)
        )
        con.execute(
            "INSERT INTO analytical_cube VALUES (?, ?, ?, ?, true, ?)",
            [cube.cube_id, cube.name, cube.entity_id, cube.cube_type, explanation[:2000]],
        )
        for sig in cube.metrics:
            con.execute(
                "INSERT INTO cube_metric VALUES (?, ?, ?, ?, ?)",
                [
                    cube.cube_id, sig.metric_id,
                    cube.metric_roles.get(sig.metric_id, "native"),
                    cube.rollup_entities.get(sig.metric_id),
                    (cube.metric_reasons.get(sig.metric_id, ""))[:500],
                ],
            )
        for dim_id in cube.dimensions:
            if con.execute("SELECT 1 FROM dimension WHERE dimension_id = ?", [dim_id]).fetchone():
                con.execute("INSERT OR IGNORE INTO cube_dimension VALUES (?, ?)",
                            [cube.cube_id, dim_id])
        for ds_id in cube.datasets:
            if con.execute("SELECT 1 FROM dataset WHERE dataset_id = ?", [ds_id]).fetchone():
                con.execute("INSERT OR IGNORE INTO cube_dataset VALUES (?, ?, ?)",
                            [cube.cube_id, ds_id, cube.entity_id])


def ensure_entity_relations(con: duckdb.DuckDBPyConnection) -> int:
    """Seed entity_relation from ENTITY_RELATIONS. Skips missing entities. Returns rows inserted."""
    existing_entities = {
        r[0] for r in con.execute("SELECT entity_id FROM entity").fetchall()
    }
    existing_relations = {
        r[0] for r in con.execute("SELECT relation_id FROM entity_relation").fetchall()
    }
    inserted = 0
    for rel_id, from_e, to_e, card, rollup, temporal, join_expr in ENTITY_RELATIONS:
        if rel_id in existing_relations:
            continue
        if from_e not in existing_entities or to_e not in existing_entities:
            continue
        con.execute("""
            INSERT INTO entity_relation
            VALUES (?, ?, ?, 'structural', ?, ?, ?, ?, 'declared', 1.0)
        """, [rel_id, from_e, to_e, card, join_expr, rollup, temporal])
        inserted += 1
    return inserted
