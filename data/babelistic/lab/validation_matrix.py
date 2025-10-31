"""
Complete Validation Matrix for Spatial Query Framework

This exhaustively enumerates all 144 possible combinations and validates them.

pingu@falcon:~/github/experiments/data/babelistic$ python3 lab/validation_matrix.py 
╔═══════════════════════════════════════════════════════════════╗
║          COMPLETE VALIDATION MATRIX SUMMARY                   ║
╠═══════════════════════════════════════════════════════════════╣
║ Total combinations:       144                           ║
║ ✅ Valid & Supported:      26                           ║
║ ⚠️  Valid but Undefined:   50                           ║
║ ❓ Questionable:           32                           ║
║ ❌ Invalid (Semantic):      9                           ║
║ ❌ Invalid (Type Error):   27                           ║
╚═══════════════════════════════════════════════════════════════╝

Coverage: 18.1% fully supported
Research needed: 34.7% undefined + 22.2% questionable

======================================================================
EXAMPLE VALIDATIONS:
======================================================================

🔹 Uncertain point × Known region, CONTAINMENT:
   Status: ✅ Valid & Supported
   Reason: P(X ∈ R) - Core framework formula
   Method: framework_crisp

🔹 Known point × Known point, CONTAINMENT:
   Status: ❌ Invalid (Semantic)
   Reason: A point cannot contain another point (zero volume)

🔹 Fuzzy point × Known region, CONTAINMENT:
   Status: ❓ Questionable
   Reason: Fuzzy points unclear
   Research: Phase A

🔹 Known region × Known region, CONTAINMENT:
   Status: ✅ Valid & Supported
   Reason: P(uniform point from R₁ in R₂) - requires semantic clarification
   Research: Phase B: Currently interprets as uniform sampling. Could also mean R₁⊆R₂ (bool)


======================================================================
Exporting to DataFrame...

DataFrame shape: (144, 10)

Status distribution:
status
⚠️ Valid but Undefined    50
❓ Questionable            32
❌ Invalid (Type Error)    27
✅ Valid & Supported       26
❌ Invalid (Semantic)       9
Name: count, dtype: int64
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd


class ValidationStatus(Enum):
    """Status of a query combination"""
    VALID_SUPPORTED = "✅ Valid & Supported"
    VALID_UNDEFINED = "⚠️ Valid but Undefined"
    QUESTIONABLE = "❓ Questionable"
    INVALID_SEMANTIC = "❌ Invalid (Semantic)"
    INVALID_TYPE = "❌ Invalid (Type Error)"


class ComputationMethod(Enum):
    """Computation methods (from original ontology)"""
    FRAMEWORK_CRISP = "framework_crisp"
    FRAMEWORK_FUZZY = "framework_fuzzy"
    FRAMEWORK_BUFFERED = "framework_buffered"
    MONTE_CARLO = "monte_carlo"
    GEOMETRIC_TEST = "geometric_test"
    GEOMETRIC_INTERSECTION = "geometric_intersection"
    DISTANCE_COMPUTATION = "distance_computation"
    FUZZY_MEMBERSHIP = "fuzzy_membership"
    UNSUPPORTED = "unsupported"


class EpistemicType(Enum):
    """Epistemic states"""
    KNOWN = "known"
    UNCERTAIN = "uncertain"
    FUZZY = "fuzzy"


class QueryType(Enum):
    """Query types"""
    CONTAINMENT = "containment"
    PROXIMITY = "proximity"
    DISTANCE = "distance"
    OVERLAP = "overlap"


@dataclass
class ValidationResult:
    """Result of validating a query combination"""
    status: ValidationStatus
    reason: str
    computation_method: Optional[ComputationMethod] = None
    result_type: Optional[str] = None  # "bool", "float", "distribution"
    research_notes: Optional[str] = None


class CompleteValidationMatrix:
    """
    Exhaustive validation of all 144 possible query combinations.
    
    This is the AUTHORITATIVE source for determining if a query is valid.
    """
    
    def __init__(self):
        self.matrix = self._build_complete_matrix()
    
    def _build_complete_matrix(self) -> Dict[Tuple, ValidationResult]:
        """Build the complete 144-entry validation matrix"""
        matrix = {}
        
        # ================================================================
        # POINT SUBJECT × POINT TARGET (36 combinations)
        # ================================================================
        
        # -------------------- CONTAINMENT --------------------
        # Point cannot contain point (semantic error)
        for s_epi in EpistemicType:
            for t_epi in EpistemicType:
                matrix[("point", s_epi, "point", t_epi, QueryType.CONTAINMENT)] = ValidationResult(
                    status=ValidationStatus.INVALID_SEMANTIC,
                    reason="A point cannot contain another point (zero volume)",
                    research_notes="Containment requires target to have positive measure"
                )
        
        # -------------------- PROXIMITY --------------------
        # Point-to-point proximity is valid for all epistemic combinations
        
        # Known × Known
        matrix[("point", EpistemicType.KNOWN, "point", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="Deterministic distance comparison",
            computation_method=ComputationMethod.DISTANCE_COMPUTATION,
            result_type="bool"
        )
        
        # Uncertain × Known
        matrix[("point", EpistemicType.UNCERTAIN, "point", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(d(X, p) ≤ δ) where X ~ distribution",
            computation_method=ComputationMethod.FRAMEWORK_BUFFERED,
            result_type="float"
        )
        
        # Uncertain × Uncertain
        matrix[("point", EpistemicType.UNCERTAIN, "point", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(d(X, Y) ≤ δ) requires convolution of distributions",
            research_notes="Need to compute distribution of d(X,Y). Can be done via Monte Carlo or analytical convolution if distributions allow."
        )
        
        # Known × Uncertain (symmetric)
        matrix[("point", EpistemicType.KNOWN, "point", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="Symmetric to uncertain × known",
            computation_method=ComputationMethod.FRAMEWORK_BUFFERED,
            result_type="float"
        )
        
        # Fuzzy point cases
        for t_epi in EpistemicType:
            matrix[("point", EpistemicType.FUZZY, "point", t_epi, QueryType.PROXIMITY)] = ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                reason="Fuzzy points have unclear semantics",
                research_notes="Phase A research: Are fuzzy points valid?"
            )
        
        for s_epi in [EpistemicType.KNOWN, EpistemicType.UNCERTAIN]:
            matrix[("point", s_epi, "point", EpistemicType.FUZZY, QueryType.PROXIMITY)] = ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                reason="Fuzzy points have unclear semantics",
                research_notes="Phase A research: Are fuzzy points valid?"
            )
        
        # -------------------- DISTANCE --------------------
        # Similar to proximity but returns distance value/distribution
        
        matrix[("point", EpistemicType.KNOWN, "point", EpistemicType.KNOWN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="Deterministic distance computation",
            computation_method=ComputationMethod.DISTANCE_COMPUTATION,
            result_type="float"
        )
        
        matrix[("point", EpistemicType.UNCERTAIN, "point", EpistemicType.KNOWN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Distribution of d(X, p) or E[d(X, p)]",
            research_notes="Can return expected distance or full distribution. Need to define semantics."
        )
        
        matrix[("point", EpistemicType.UNCERTAIN, "point", EpistemicType.UNCERTAIN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Distribution of d(X, Y) via convolution",
            research_notes="Computationally complex but theoretically valid"
        )
        
        matrix[("point", EpistemicType.KNOWN, "point", EpistemicType.UNCERTAIN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Symmetric to uncertain × known",
            research_notes="Same as above"
        )
        
        # Fuzzy cases
        for s_epi in EpistemicType:
            for t_epi in EpistemicType:
                if s_epi == EpistemicType.FUZZY or t_epi == EpistemicType.FUZZY:
                    if ("point", s_epi, "point", t_epi, QueryType.DISTANCE) not in matrix:
                        matrix[("point", s_epi, "point", t_epi, QueryType.DISTANCE)] = ValidationResult(
                            status=ValidationStatus.QUESTIONABLE,
                            reason="Fuzzy points unclear",
                            research_notes="Phase A"
                        )
        
        # -------------------- OVERLAP --------------------
        # Points don't overlap (type error)
        for s_epi in EpistemicType:
            for t_epi in EpistemicType:
                matrix[("point", s_epi, "point", t_epi, QueryType.OVERLAP)] = ValidationResult(
                    status=ValidationStatus.INVALID_TYPE,
                    reason="OVERLAP requires regions (points have no area)",
                    research_notes="Points can only coincide, not overlap"
                )
        
        # ================================================================
        # POINT SUBJECT × REGION TARGET (36 combinations)
        # ================================================================
        
        # -------------------- CONTAINMENT --------------------
        
        # Known × Known
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="Point-in-polygon test (deterministic)",
            computation_method=ComputationMethod.GEOMETRIC_TEST,
            result_type="bool"
        )
        
        # Uncertain × Known (CORE GEOFENCE CASE)
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(X ∈ R) - Core framework formula",
            computation_method=ComputationMethod.FRAMEWORK_CRISP,
            result_type="float",
            research_notes="THE primary use case for the framework"
        )
        
        # Known × Fuzzy
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="Fuzzy membership degree μ(p)",
            computation_method=ComputationMethod.FUZZY_MEMBERSHIP,
            result_type="float"
        )
        
        # Uncertain × Fuzzy
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="∫ p(x)·μ(x) dx - Fuzzy framework formula",
            computation_method=ComputationMethod.FRAMEWORK_FUZZY,
            result_type="float"
        )
        
        # Known × Uncertain
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(p ∈ R̃) over region ensemble",
            computation_method=ComputationMethod.MONTE_CARLO,
            result_type="float"
        )
        
        # Uncertain × Uncertain
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(X ∈ R̃) - Double uncertainty",
            computation_method=ComputationMethod.MONTE_CARLO,
            result_type="float"
        )
        
        # Fuzzy point cases
        for t_epi in EpistemicType:
            matrix[("point", EpistemicType.FUZZY, "region", t_epi, QueryType.CONTAINMENT)] = ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                reason="Fuzzy points unclear",
                research_notes="Phase A"
            )
        
        # -------------------- PROXIMITY --------------------
        
        # Known × Known
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="d(p, ∂R) ≤ δ (deterministic)",
            computation_method=ComputationMethod.DISTANCE_COMPUTATION,
            result_type="bool"
        )
        
        # Uncertain × Known (GEOFENCE WITH BUFFER)
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(d(X, ∂R) ≤ δ) - Buffered geofence",
            computation_method=ComputationMethod.FRAMEWORK_BUFFERED,
            result_type="float",
            research_notes="Common use case: proximity alerts"
        )
        
        # Known × Fuzzy
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Distance to fuzzy boundary unclear",
            research_notes="Phase B: How to define d(p, fuzzy_boundary)? Multiple interpretations possible."
        )
        
        # Uncertain × Fuzzy
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Proximity to fuzzy region boundary",
            research_notes="Phase B: Requires fuzzy buffer definition. Could use μ_buffered(x) = max_{y:d(x,y)≤δ} μ(y)"
        )
        
        # Known × Uncertain
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(d(p, ∂R̃) ≤ δ) over region ensemble",
            research_notes="Phase B: Requires distance to uncertain boundary"
        )
        
        # Uncertain × Uncertain
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(d(X, ∂R̃) ≤ δ) - Double uncertainty",
            research_notes="Phase D Priority 1: Practically important (uncertain person near uncertain hazard)"
        )
        
        # Fuzzy point cases
        for t_epi in EpistemicType:
            matrix[("point", EpistemicType.FUZZY, "region", t_epi, QueryType.PROXIMITY)] = ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                reason="Fuzzy points unclear",
                research_notes="Phase A"
            )
        
        # -------------------- DISTANCE --------------------
        
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="d(p, ∂R) deterministic",
            computation_method=ComputationMethod.DISTANCE_COMPUTATION,
            result_type="float"
        )
        
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="E[d(X, ∂R)] or distribution of distances",
            research_notes="Can compute, need to define return semantics"
        )
        
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.UNCERTAIN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="E[d(p, ∂R̃)] over region ensemble",
            research_notes="Phase D Priority 3"
        )
        
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Distribution of d(X, ∂R̃)",
            research_notes="Complex convolution"
        )
        
        matrix[("point", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Distance to fuzzy boundary",
            research_notes="Phase B: Multiple possible definitions"
        )
        
        matrix[("point", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Expected distance to fuzzy boundary",
            research_notes="Phase B"
        )
        
        # Fuzzy points
        for t_epi in EpistemicType:
            matrix[("point", EpistemicType.FUZZY, "region", t_epi, QueryType.DISTANCE)] = ValidationResult(
                status=ValidationStatus.QUESTIONABLE,
                reason="Fuzzy points unclear",
                research_notes="Phase A"
            )
        
        # -------------------- OVERLAP --------------------
        # Point-region overlap is type error
        for s_epi in EpistemicType:
            for t_epi in EpistemicType:
                matrix[("point", s_epi, "region", t_epi, QueryType.OVERLAP)] = ValidationResult(
                    status=ValidationStatus.INVALID_TYPE,
                    reason="Points don't overlap regions (contained or not)",
                    research_notes="Use CONTAINMENT instead"
                )
        
        # ================================================================
        # REGION SUBJECT × POINT TARGET (36 combinations)
        # ================================================================
        # These are mostly symmetric to point × region
        
        # -------------------- CONTAINMENT --------------------
        # Semantically: Does the region contain the point? (same as point in region)
        
        matrix[("region", EpistemicType.KNOWN, "point", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="Same as point in region (symmetric)",
            computation_method=ComputationMethod.GEOMETRIC_TEST,
            result_type="bool"
        )
        
        matrix[("region", EpistemicType.KNOWN, "point", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(X ∈ R) symmetric to uncertain point × known region",
            computation_method=ComputationMethod.FRAMEWORK_CRISP,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "point", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(p ∈ R̃) over region ensemble",
            computation_method=ComputationMethod.MONTE_CARLO,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "point", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(X ∈ R̃) symmetric",
            computation_method=ComputationMethod.MONTE_CARLO,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.FUZZY, "point", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="μ(p) symmetric",
            computation_method=ComputationMethod.FUZZY_MEMBERSHIP,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.FUZZY, "point", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="∫ p(x)·μ(x) dx symmetric",
            computation_method=ComputationMethod.FRAMEWORK_FUZZY,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.KNOWN, "point", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.QUESTIONABLE,
            reason="Fuzzy points unclear",
            research_notes="Phase A"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "point", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.QUESTIONABLE,
            reason="Fuzzy points unclear",
            research_notes="Phase A"
        )
        
        matrix[("region", EpistemicType.FUZZY, "point", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.QUESTIONABLE,
            reason="Fuzzy points unclear",
            research_notes="Phase A"
        )
        
        # -------------------- PROXIMITY, DISTANCE, OVERLAP --------------------
        # Similar patterns - add systematically
        
        for query_type in [QueryType.PROXIMITY, QueryType.DISTANCE]:
            for s_epi in [EpistemicType.KNOWN, EpistemicType.UNCERTAIN]:
                for t_epi in [EpistemicType.KNOWN, EpistemicType.UNCERTAIN]:
                    # Mirror the point × region cases
                    matrix[("region", s_epi, "point", t_epi, query_type)] = ValidationResult(
                        status=ValidationStatus.VALID_UNDEFINED,
                        reason="Symmetric to point × region (subject/target reversed)",
                        research_notes="Same semantics as reversed case"
                    )
            
            # Fuzzy cases
            for s_epi in EpistemicType:
                for t_epi in EpistemicType:
                    if s_epi == EpistemicType.FUZZY or t_epi == EpistemicType.FUZZY:
                        if ("region", s_epi, "point", t_epi, query_type) not in matrix:
                            matrix[("region", s_epi, "point", t_epi, query_type)] = ValidationResult(
                                status=ValidationStatus.QUESTIONABLE,
                                reason="Fuzzy points or fuzzy semantics unclear",
                                research_notes="Phase A or Phase B"
                            )
        
        # OVERLAP: type error
        for s_epi in EpistemicType:
            for t_epi in EpistemicType:
                matrix[("region", s_epi, "point", t_epi, QueryType.OVERLAP)] = ValidationResult(
                    status=ValidationStatus.INVALID_TYPE,
                    reason="Regions don't overlap with points",
                    research_notes="Use CONTAINMENT"
                )
        
        # ================================================================
        # REGION SUBJECT × REGION TARGET (36 combinations) ⭐ COMPLEX
        # ================================================================
        
        # -------------------- CONTAINMENT --------------------
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(uniform point from R₁ in R₂) - requires semantic clarification",
            computation_method=ComputationMethod.FRAMEWORK_CRISP,
            result_type="float",
            research_notes="Phase B: Currently interprets as uniform sampling. Could also mean R₁⊆R₂ (bool)"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Multiple interpretations: P(R̃₁⊆R₂)? P(R̃₁∩R₂≠∅)? P(uniform from R̃₁ in R₂)?",
            research_notes="Phase B CRITICAL: Need to clarify containment semantics for region×region"
        )
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(uniform from R₁ in R̃₂)?",
            research_notes="Phase B"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Containment semantics unclear with double uncertainty",
            research_notes="Phase B + Phase D"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="∫∫ μ(x)·𝟙_R(x) dx (fuzzy overlap measure)",
            computation_method=ComputationMethod.FRAMEWORK_FUZZY,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Containment in fuzzy target unclear",
            research_notes="Phase B: Multiple interpretations possible"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="∫∫ μ₁(x)·μ₂(x) dx (fuzzy intersection)",
            computation_method=ComputationMethod.FRAMEWORK_FUZZY,
            result_type="float"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Uncertain region with fuzzy target",
            research_notes="Phase D Priority 2"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Fuzzy region with uncertain target",
            research_notes="Phase D Priority 2"
        )
        
        # -------------------- OVERLAP --------------------
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="R₁ ∩ R₂ ≠ ∅? (deterministic geometric test)",
            computation_method=ComputationMethod.GEOMETRIC_INTERSECTION,
            result_type="bool"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="P(R̃₁ ∩ R₂ ≠ ∅)",
            computation_method=ComputationMethod.MONTE_CARLO,
            result_type="float",
            research_notes="Documented in original matrix"
        )
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.UNCERTAIN, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(R₁ ∩ R̃₂ ≠ ∅) symmetric to above",
            research_notes="Should be same as uncertain × known"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(R̃₁ ∩ R̃₂ ≠ ∅) - double Monte Carlo",
            research_notes="Phase D Priority 2: Computationally expensive but valid"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.KNOWN, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Fuzzy overlap requires threshold or returns continuous degree",
            research_notes="Phase B: How to define overlap for fuzzy regions?"
        )
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Symmetric to above",
            research_notes="Phase B"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.FUZZY, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Fuzzy-fuzzy overlap (degree of intersection)",
            research_notes="Phase D Priority 2: Fuzzy set theory provides guidance"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Mixed epistemic states",
            research_notes="Phase D Priority 3"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.UNCERTAIN, QueryType.OVERLAP)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Symmetric to above",
            research_notes="Phase D Priority 3"
        )
        
        # -------------------- PROXIMITY --------------------
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="d(R₁, R₂) ≤ δ or buffer test",
            computation_method=ComputationMethod.GEOMETRIC_INTERSECTION,
            result_type="bool",
            research_notes="Phase B: Need to specify distance semantics (boundary, Hausdorff, etc.)"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(d(R̃₁, R₂) ≤ δ)",
            research_notes="Phase B + Phase D Priority 1"
        )
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Symmetric to above",
            research_notes="Phase B + Phase D"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="P(d(R̃₁, R̃₂) ≤ δ) - complex",
            research_notes="Phase D Priority 3"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Proximity with fuzzy boundaries unclear",
            research_notes="Phase D Priority 2"
        )
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Symmetric to above",
            research_notes="Phase D Priority 2"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.FUZZY, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Fuzzy proximity (fuzzy distance?)",
            research_notes="Phase D Priority 2: Fuzzy topology literature"
        )
        
        matrix[("region", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Mixed epistemic",
            research_notes="Phase D Priority 3"
        )
        
        matrix[("region", EpistemicType.FUZZY, "region", EpistemicType.UNCERTAIN, QueryType.PROXIMITY)] = ValidationResult(
            status=ValidationStatus.VALID_UNDEFINED,
            reason="Symmetric",
            research_notes="Phase D Priority 3"
        )
        
        # -------------------- DISTANCE --------------------
        
        matrix[("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.DISTANCE)] = ValidationResult(
            status=ValidationStatus.VALID_SUPPORTED,
            reason="d(R₁, R₂) deterministic",
            computation_method=ComputationMethod.DISTANCE_COMPUTATION,
            result_type="float",
            research_notes="Phase B CRITICAL: Define distance semantics (boundary-boundary, Hausdorff, centroid, etc.)"
        )
        
        # All other region×region distance combinations
        for s_epi in [EpistemicType.UNCERTAIN, EpistemicType.FUZZY]:
            for t_epi in EpistemicType:
                if (s_epi, t_epi) != (EpistemicType.KNOWN, EpistemicType.KNOWN):
                    matrix[("region", s_epi, "region", t_epi, QueryType.DISTANCE)] = ValidationResult(
                        status=ValidationStatus.VALID_UNDEFINED,
                        reason="Distance with uncertain/fuzzy regions",
                        research_notes="Phase D: Extends from known×known case"
                    )
        
        for t_epi in [EpistemicType.UNCERTAIN, EpistemicType.FUZZY]:
            if ("region", EpistemicType.KNOWN, "region", t_epi, QueryType.DISTANCE) not in matrix:
                matrix[("region", EpistemicType.KNOWN, "region", t_epi, QueryType.DISTANCE)] = ValidationResult(
                    status=ValidationStatus.VALID_UNDEFINED,
                    reason="Distance with uncertain/fuzzy target",
                    research_notes="Phase D"
                )
        
        return matrix
    
    def validate(self, subject_extent: str, subject_epistemic: EpistemicType,
                 target_extent: str, target_epistemic: EpistemicType,
                 query_type: QueryType) -> ValidationResult:
        """
        Validate a specific query combination.
        
        Returns ValidationResult with status and details.
        """
        key = (subject_extent, subject_epistemic, target_extent, target_epistemic, query_type)
        
        result = self.matrix.get(key)
        
        if result is None:
            return ValidationResult(
                status=ValidationStatus.INVALID_SEMANTIC,
                reason=f"Combination not in validation matrix: {key}",
                research_notes="Should not happen if matrix is complete"
            )
        
        return result
    
    def get_statistics(self) -> dict:
        """Get statistics about the validation matrix"""
        from collections import Counter
        
        status_counts = Counter(r.status for r in self.matrix.values())
        
        return {
            "total_combinations": len(self.matrix),
            "valid_supported": status_counts[ValidationStatus.VALID_SUPPORTED],
            "valid_undefined": status_counts[ValidationStatus.VALID_UNDEFINED],
            "questionable": status_counts[ValidationStatus.QUESTIONABLE],
            "invalid_semantic": status_counts[ValidationStatus.INVALID_SEMANTIC],
            "invalid_type": status_counts[ValidationStatus.INVALID_TYPE]
        }
    
    def export_to_dataframe(self):
        """Export matrix to pandas DataFrame for analysis"""
        rows = []
        for (s_ext, s_epi, t_ext, t_epi, q_type), result in self.matrix.items():
            rows.append({
                "subject_extent": s_ext,
                "subject_epistemic": s_epi.value,
                "target_extent": t_ext,
                "target_epistemic": t_epi.value,
                "query_type": q_type.value,
                "status": result.status.value,
                "reason": result.reason,
                "computation_method": result.computation_method.value if result.computation_method else None,
                "result_type": result.result_type,
                "research_notes": result.research_notes
            })
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print a summary of the validation matrix"""
        stats = self.get_statistics()
        
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║          COMPLETE VALIDATION MATRIX SUMMARY                   ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        print(f"║ Total combinations:       {stats['total_combinations']:3d}                           ║")
        print(f"║ ✅ Valid & Supported:     {stats['valid_supported']:3d}                           ║")
        print(f"║ ⚠️  Valid but Undefined:  {stats['valid_undefined']:3d}                           ║")
        print(f"║ ❓ Questionable:          {stats['questionable']:3d}                           ║")
        print(f"║ ❌ Invalid (Semantic):    {stats['invalid_semantic']:3d}                           ║")
        print(f"║ ❌ Invalid (Type Error):  {stats['invalid_type']:3d}                           ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        print(f"Coverage: {stats['valid_supported']/stats['total_combinations']*100:.1f}% fully supported")
        print(f"Research needed: {stats['valid_undefined']/stats['total_combinations']*100:.1f}% undefined + {stats['questionable']/stats['total_combinations']*100:.1f}% questionable")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Create the complete validation matrix
    validator = CompleteValidationMatrix()
    
    # Print summary
    validator.print_summary()
    
    # Test some specific cases
    print("\n" + "="*70)
    print("EXAMPLE VALIDATIONS:")
    print("="*70 + "\n")
    
    # Classic geofence
    result = validator.validate(
        "point", EpistemicType.UNCERTAIN,
        "region", EpistemicType.KNOWN,
        QueryType.CONTAINMENT
    )
    print(f"🔹 Uncertain point × Known region, CONTAINMENT:")
    print(f"   Status: {result.status.value}")
    print(f"   Reason: {result.reason}")
    if result.computation_method:
        print(f"   Method: {result.computation_method.value}")
    print()
    
    # Invalid case
    result = validator.validate(
        "point", EpistemicType.KNOWN,
        "point", EpistemicType.KNOWN,
        QueryType.CONTAINMENT
    )
    print(f"🔹 Known point × Known point, CONTAINMENT:")
    print(f"   Status: {result.status.value}")
    print(f"   Reason: {result.reason}")
    print()
    
    # Questionable case
    result = validator.validate(
        "point", EpistemicType.FUZZY,
        "region", EpistemicType.KNOWN,
        QueryType.CONTAINMENT
    )
    print(f"🔹 Fuzzy point × Known region, CONTAINMENT:")
    print(f"   Status: {result.status.value}")
    print(f"   Reason: {result.reason}")
    if result.research_notes:
        print(f"   Research: {result.research_notes}")
    print()
    
    # Region × Region ambiguity
    result = validator.validate(
        "region", EpistemicType.KNOWN,
        "region", EpistemicType.KNOWN,
        QueryType.CONTAINMENT
    )
    print(f"🔹 Known region × Known region, CONTAINMENT:")
    print(f"   Status: {result.status.value}")
    print(f"   Reason: {result.reason}")
    if result.research_notes:
        print(f"   Research: {result.research_notes}")
    print()
    
    # Export to DataFrame for analysis
    print("\n" + "="*70)
    print("Exporting to DataFrame...")
    df = validator.export_to_dataframe()
    
    print(f"\nDataFrame shape: {df.shape}")
    print("\nStatus distribution:")
    print(df['status'].value_counts())