"""
Scenario / invariant tests for the pharmacology experiment.

These tests validate domain behaviour rather than textual output.

They exercise the separation:

    Duwhal
        -> candidate discovery / structural proximity

    Solver
        -> burden
        -> adverse effects
        -> feasibility
        -> objective J_p(M)
        -> exact optimum


Run
===

    uv add --dev pytest

    pytest -q test_scenarios.py

or:

    python -m pytest -q test_scenarios.py


Important
=========

All Demo* medications and their numerical parameters are synthetic.
These tests validate software/model semantics, not clinical claims.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import duckdb
import pytest

from explore import (
    build_duwhal_interactions,
    candidate_coverage,
    generate_candidates,
    load_adverse_support,
    load_therapeutic_support,
)

from solver import (
    ClinicalModel,
    RegimenSolver,
    resolve_pathology,
)


# ============================================================
# CONFIGURATION
# ============================================================


DB_PATH = (
    Path(__file__).resolve().parent
    / "patologias.duckdb"
)

EPSILON = 1e-12


# Presentation IDs from seed.sql
TYLENOL_500 = 1
TYLENOL_SYRUP = 2
ADVIL = 3
CLARITIN = 4

DEMO_ANTITUSSIVE = 5
DEMO_ANTIEMETIC_A = 6
DEMO_ANTIEMETIC_B = 7
DEMO_PHOTOPHOBIA = 8
DEMO_DECONGESTANT_A = 9
DEMO_DECONGESTANT_B = 10
DEMO_GASTRIC = 11
DEMO_FLU_MULTI = 12


# Symptom IDs from seed.sql
FEVER = 1
HEADACHE = 2
CONGESTION = 3
CORYZA = 4
COUGH = 5
SNEEZING = 6
PHOTOPHOBIA = 7
NAUSEA = 8
STOMACH_PAIN = 9
DROWSINESS = 10
ALLERGIC_REACTION = 11


# ============================================================
# HELPERS
# ============================================================


def build_domain(
    pathology: str | int,
):
    """
    Open a fresh read-only connection and construct model + solver.

    Caller must close the returned connection.
    """

    if not DB_PATH.exists():
        raise RuntimeError(
            f"Database not found: {DB_PATH}\n"
            "Run first:\n"
            "    python3 ingest.py"
        )

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    pathology_id = resolve_pathology(
        con,
        str(pathology),
    )

    model = ClinicalModel(
        con,
        pathology_id,
    )

    solver = RegimenSolver(
        model,
    )

    return (
        con,
        model,
        solver,
    )


def candidate_row(
    candidates,
    presentation_id: int,
):
    """
    Return one Duwhal candidate row by presentation ID.
    """

    match = candidates[
        candidates["presentation_id"]
        == presentation_id
    ]

    assert not match.empty, (
        f"Presentation {presentation_id} "
        f"was not returned by Duwhal.\n\n"
        f"Candidates:\n{candidates}"
    )

    return match.iloc[0]


def generated_candidates(
    con,
    base_evaluation,
    *,
    depth: int,
    top: int = 100,
):
    """
    Generate Duwhal candidates from the unresolved support
    of a solver evaluation.
    """

    interactions = (
        build_duwhal_interactions(
            con
        )
    )

    unresolved = (
        base_evaluation
        .symptom_space
        .unresolved_support
    )

    return generate_candidates(
        interactions,
        unresolved,
        top=top,
        depth=depth,
    )


def all_regimens(
    presentation_ids,
):
    """
    Enumerate the complete powerset of presentation IDs.
    """

    ids = tuple(
        sorted(
            presentation_ids
        )
    )

    for size in range(
        len(ids) + 1
    ):
        yield from combinations(
            ids,
            size,
        )


def exact_global_optimum(
    solver: RegimenSolver,
    presentation_ids,
):
    """
    Exhaustively compute:

        argmin J_p(M)

    over feasible regimens.

    We intentionally avoid relying on report ordering or
    textual output from solver.py.
    """

    best = None

    for regimen in all_regimens(
        presentation_ids
    ):
        evaluation = solver.evaluate(
            regimen
        )

        if not (
            evaluation
            .feasibility
            .feasible
        ):
            continue

        if (
            best is None
            or evaluation.objective
            < best.objective
            - EPSILON
        ):
            best = evaluation

    assert best is not None

    return best


def adverse_risk_for_symptom(
    evaluation,
    symptom_id: int,
):
    """
    Find aggregated adverse-risk state for one symptom.
    """

    matches = [
        risk
        for risk
        in evaluation
        .feasibility
        .adverse_risks
        if risk.symptom_id
        == symptom_id
    ]

    assert matches, (
        f"No adverse-risk state found "
        f"for symptom_id={symptom_id}"
    )

    assert len(matches) == 1

    return matches[0]


# ============================================================
# 1. DOMINATION
# ============================================================


def test_demo_antiemetic_b_dominates_a():
    """
    Synthetic ground truth:

    DemoAntiemetic B dominates DemoAntiemetic A.

    Both target nausea, but B has:

        - stronger therapeutic effect;
        - lower complexity;
        - no seeded ADR;

    and therefore must obtain a lower objective when evaluated
    alone for migraine.
    """

    con, model, solver = build_domain(
        "Enxaqueca"
    )

    try:
        therapeutic = (
            load_therapeutic_support(
                con
            )
        )

        adverse = (
            load_adverse_support(
                con
            )
        )

        # Same therapeutic target.
        assert therapeutic[
            DEMO_ANTIEMETIC_A
        ] == frozenset(
            {NAUSEA}
        )

        assert therapeutic[
            DEMO_ANTIEMETIC_B
        ] == frozenset(
            {NAUSEA}
        )

        # B is cheaper in complexity.
        assert (
            model
            .presentation_by_id[
                DEMO_ANTIEMETIC_B
            ]
            .complexity_cost
            <
            model
            .presentation_by_id[
                DEMO_ANTIEMETIC_A
            ]
            .complexity_cost
        )

        # A generates drowsiness; B has no seeded ADR.
        assert (
            DROWSINESS
            in adverse[
                DEMO_ANTIEMETIC_A
            ]
        )

        assert (
            DEMO_ANTIEMETIC_B
            not in adverse
            or not adverse[
                DEMO_ANTIEMETIC_B
            ]
        )

        eval_a = solver.evaluate(
            (DEMO_ANTIEMETIC_A,)
        )

        eval_b = solver.evaluate(
            (DEMO_ANTIEMETIC_B,)
        )

        assert (
            eval_a
            .feasibility
            .feasible
        )

        assert (
            eval_b
            .feasibility
            .feasible
        )

        assert (
            eval_b.objective
            <
            eval_a.objective
            - EPSILON
        )

        # B should leave fewer unresolved symptoms
        # under the current synthetic seed.
        assert (
            len(
                eval_b
                .symptom_space
                .unresolved_support
            )
            <=
            len(
                eval_a
                .symptom_space
                .unresolved_support
            )
        )

    finally:
        con.close()


# ============================================================
# 2. STRUCTURALLY ATTRACTIVE BUT INFEASIBLE
# ============================================================


def test_claritin_is_structurally_attractive_but_infeasible():
    """
    Claritin should be found by Duwhal from untreated flu because
    it directly covers congestion + coryza.

    Nevertheless the clinical feasibility layer must reject it
    because of the synthetic allergic-reaction threshold.
    """

    con, model, solver = build_domain(
        "Gripe"
    )

    try:
        base = solver.evaluate(
            ()
        )

        candidates = generated_candidates(
            con,
            base,
            depth=1,
        )

        row = candidate_row(
            candidates,
            CLARITIN,
        )

        therapeutic = (
            load_therapeutic_support(
                con
            )
        )

        covered, coverage = (
            candidate_coverage(
                CLARITIN,
                base
                .symptom_space
                .unresolved_support,
                therapeutic,
            )
        )

        assert covered == frozenset(
            {
                CONGESTION,
                CORYZA,
            }
        )

        assert coverage == pytest.approx(
            0.50
        )

        assert float(
            row["score"]
        ) > 0

        claritin = solver.evaluate(
            (CLARITIN,)
        )

        assert not (
            claritin
            .feasibility
            .feasible
        )

        violations = " ".join(
            claritin
            .feasibility
            .violations
        )

        assert (
            "Reação alérgica"
            in violations
        )

    finally:
        con.close()


# ============================================================
# 3. LOWER J WITHOUT INCREASING CONTROLLED SET
# ============================================================


def test_demo_flu_multi_can_reduce_j_without_increasing_c():
    """
    DemoFlu Multi deliberately demonstrates that:

        ΔJ < 0

    does not imply:

        Δ|C| > 0.

    It reduces total residual burden enough to improve J while
    its effects remain below the contextual control thresholds.
    """

    con, model, solver = build_domain(
        "Gripe"
    )

    try:
        base = solver.evaluate(
            ()
        )

        candidate = solver.evaluate(
            (DEMO_FLU_MULTI,)
        )

        assert (
            candidate
            .feasibility
            .feasible
        )

        delta_j = (
            candidate.objective
            - base.objective
        )

        assert delta_j < -EPSILON

        assert (
            len(
                candidate
                .symptom_space
                .controlled_support
            )
            ==
            len(
                base
                .symptom_space
                .controlled_support
            )
        )

        # Current synthetic seed gives zero controlled symptoms
        # in both states.
        assert not (
            candidate
            .symptom_space
            .controlled_support
        )

    finally:
        con.close()


# ============================================================
# 4. INDIRECT STRUCTURAL CANDIDATE
# ============================================================


def test_tylenol_syrup_is_indirect_for_migraine_depth_2():
    """
    Tylenol syrup does not directly cover any unresolved migraine
    symptom, but Duwhal reaches it at depth 2 through structural
    proximity.

    Expected semantic distinction:

        Duwhal proximity > 0
        direct coverage  = 0
        ΔJ               > 0
    """

    con, model, solver = build_domain(
        "Enxaqueca"
    )

    try:
        base = solver.evaluate(
            ()
        )

        candidates = generated_candidates(
            con,
            base,
            depth=2,
        )

        row = candidate_row(
            candidates,
            TYLENOL_SYRUP,
        )

        assert int(
            row["hops"]
        ) >= 2

        assert float(
            row["score"]
        ) > 0

        therapeutic = (
            load_therapeutic_support(
                con
            )
        )

        covered, coverage = (
            candidate_coverage(
                TYLENOL_SYRUP,
                base
                .symptom_space
                .unresolved_support,
                therapeutic,
            )
        )

        assert covered == frozenset()

        assert coverage == pytest.approx(
            0.0
        )

        candidate = solver.evaluate(
            (TYLENOL_SYRUP,)
        )

        delta_j = (
            candidate.objective
            - base.objective
        )

        assert delta_j > EPSILON

    finally:
        con.close()


# ============================================================
# 5. EMERGENT AGGREGATED ADR RISK
# ============================================================


def test_presentations_5_and_6_are_jointly_infeasible():
    """
    Both presentations are individually admissible for
    drowsiness:

        P5 = 0.12 < 0.18
        P6 = 0.10 < 0.18

    but together:

        P = 1 - (1 - .12)(1 - .10)
          = 0.208

    and therefore:

        0.208 > 0.18.

    This validates that feasibility uses aggregated occurrence
    probability rather than individual-event burden.
    """

    con, model, solver = build_domain(
        "Gripe"
    )

    try:
        only_5 = solver.evaluate(
            (DEMO_ANTITUSSIVE,)
        )

        only_6 = solver.evaluate(
            (DEMO_ANTIEMETIC_A,)
        )

        together = solver.evaluate(
            (
                DEMO_ANTITUSSIVE,
                DEMO_ANTIEMETIC_A,
            )
        )

        assert (
            only_5
            .feasibility
            .feasible
        )

        assert (
            only_6
            .feasibility
            .feasible
        )

        assert not (
            together
            .feasibility
            .feasible
        )

        risk = adverse_risk_for_symptom(
            together,
            DROWSINESS,
        )

        expected_probability = (
            1
            - (1 - 0.12)
            * (1 - 0.10)
        )

        assert (
            risk.aggregated_probability
            ==
            pytest.approx(
                expected_probability
            )
        )

        assert (
            risk.aggregated_probability
            ==
            pytest.approx(
                0.208
            )
        )

        assert (
            risk.max_admissible_risk
            ==
            pytest.approx(
                0.18
            )
        )

        assert (
            risk.aggregated_probability
            >
            risk.max_admissible_risk
        )

        assert not risk.admissible

    finally:
        con.close()


# ============================================================
# 6. INDIRECT != DIRECT COVERAGE
# ============================================================


def test_indirect_recommendations_are_not_direct_coverage():
    """
    A multi-hop Duwhal recommendation must not be promoted to
    direct therapeutic coverage merely because graph proximity
    exists.

    We use migraine because Tylenol syrup is a known synthetic
    counterexample:

        headache
            -> another presentation
            -> Tylenol syrup

    while Tylenol syrup itself has no direct therapeutic edge to
    any unresolved migraine symptom.
    """

    con, model, solver = build_domain(
        "Enxaqueca"
    )

    try:
        base = solver.evaluate(
            ()
        )

        candidates = generated_candidates(
            con,
            base,
            depth=2,
        )

        therapeutic = (
            load_therapeutic_support(
                con
            )
        )

        unresolved = (
            base
            .symptom_space
            .unresolved_support
        )

        indirect_count = 0

        for row in candidates.to_dict(
            orient="records"
        ):
            presentation_id = int(
                row["presentation_id"]
            )

            covered, coverage = (
                candidate_coverage(
                    presentation_id,
                    unresolved,
                    therapeutic,
                )
            )

            # Classification comes from domain semantics,
            # not graph depth alone.
            direct = bool(
                covered
            )

            if not direct:
                indirect_count += 1

                assert coverage == pytest.approx(
                    0.0
                )

                # The candidate may still have a positive
                # Duwhal score due to multi-hop structure.
                assert (
                    float(
                        row["score"]
                    )
                    > 0
                )

        # Ensure this test did not pass vacuously.
        assert indirect_count >= 1

        # Explicit known counterexample.
        syrup = candidate_row(
            candidates,
            TYLENOL_SYRUP,
        )

        covered, coverage = (
            candidate_coverage(
                TYLENOL_SYRUP,
                unresolved,
                therapeutic,
            )
        )

        assert not covered

        assert coverage == pytest.approx(
            0.0
        )

        assert int(
            syrup["hops"]
        ) >= 2

    finally:
        con.close()


# ============================================================
# 7. EXACT OPTIMUM RECOVERABLE FROM DUWHAL CANDIDATES
# ============================================================


@pytest.mark.parametrize(
    "pathology,depth",
    [
        ("Gripe", 2),
        ("Enxaqueca", 2),
    ],
)
def test_exact_optimum_is_recoverable_from_duwhal_candidates(
    pathology,
    depth,
):
    """
    Oracle-recall test.

    For a simple scenario starting from the empty regimen:

        K = Duwhal candidates

    and:

        M* = exact global feasible optimum

    require:

        M* ⊆ K.

    This does NOT yet prove that a guided search algorithm will
    discover M*.

    It proves the weaker and necessary condition that Duwhal did
    not prune away any presentation required by the exact optimum.
    """

    con, model, solver = build_domain(
        pathology
    )

    try:
        base = solver.evaluate(
            ()
        )

        candidates = generated_candidates(
            con,
            base,
            depth=depth,
            top=100,
        )

        candidate_ids = frozenset(
            int(value)
            for value
            in candidates[
                "presentation_id"
            ].tolist()
        )

        optimum = exact_global_optimum(
            solver,
            model.presentation_by_id.keys(),
        )

        optimum_ids = frozenset(
            optimum.regimen
        )

        missing = (
            optimum_ids
            - candidate_ids
        )

        assert not missing, (
            f"Exact optimum for {pathology} "
            f"cannot be recovered from the initial "
            f"Duwhal candidate set.\n\n"
            f"Exact optimum regimen: "
            f"{sorted(optimum_ids)}\n"
            f"Exact optimum J: "
            f"{optimum.objective:.6f}\n"
            f"Duwhal candidates: "
            f"{sorted(candidate_ids)}\n"
            f"Missing optimum presentations: "
            f"{sorted(missing)}"
        )

    finally:
        con.close()
