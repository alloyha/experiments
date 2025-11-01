"""
FIXED: Mature Implementation Gap Analysis for 98 Valid Combinations

After refinements:
- NO fuzzy points (eliminated 27 combinations)
- NO semantic errors (eliminated 36 combinations)
- Split CONTAINMENT into 4 query types (added specificity)

Result: 98 valid combinations, ALL with clear semantics
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import pandas as pd


class ImplementationStatus(Enum):
    """Implementation status categories"""
    IMPLEMENTED = auto()          # ✅ Working in codebase
    STRAIGHTFORWARD = auto()      # 🟢 < 50 LOC, clear algorithm
    MODERATE = auto()             # 🟡 50-200 LOC, some complexity
    COMPLEX = auto()              # 🔴 > 200 LOC, significant work
    RESEARCH_NEEDED = auto()      # 🔵 Needs algorithm research
    BLOCKED = auto()              # ⚫ Waiting on dependencies


@dataclass
class ImplementationGap:
    """Detailed information about an implementation gap"""
    # Identity
    subject_extent: str
    subject_epistemic: str
    target_extent: str
    target_epistemic: str
    query_type: str
    
    # Status
    status: ImplementationStatus
    
    # Implementation details
    current_support: str
    algorithm_sketch: str
    estimated_loc: int
    
    # Dependencies and notes
    dependencies: List[str] = field(default_factory=list)
    computational_strategy: str = "unknown"
    result_type: str = "float"  # float, bool, dict
    complexity_class: str = "?"  # O(1), O(n), O(n²), etc.
    research_notes: str = ""
    
    def __str__(self):
        return (f"({self.subject_extent}, {self.subject_epistemic}) × "
                f"({self.target_extent}, {self.target_epistemic}), "
                f"{self.query_type}")


class GapAnalyzer:
    """Comprehensive gap analysis for all 98 combinations"""
    
    def __init__(self):
        self.gaps = self._analyze_all_gaps()
        self._validate_count()
    
    def _validate_count(self):
        """Ensure we have exactly 98 combinations"""
        if len(self.gaps) != 98:
            print(f"⚠️  WARNING: Expected 98 combinations, got {len(self.gaps)}")
            print(f"   Missing {98 - len(self.gaps)} combinations!")
    
    def _analyze_all_gaps(self) -> List[ImplementationGap]:
        """Analyze all 98 combinations"""
        gaps = []
        
        # MEMBERSHIP queries (21 combinations)
        gaps.extend(self._analyze_membership())
        
        # SUBSET queries (9 combinations)
        gaps.extend(self._analyze_subset())
        
        # INTERSECTION queries (9 combinations)
        gaps.extend(self._analyze_intersection())
        
        # OVERLAP_FRACTION queries (9 combinations)
        gaps.extend(self._analyze_overlap_fraction())
        
        # PROXIMITY queries (25 combinations)
        gaps.extend(self._analyze_proximity())
        
        # DISTANCE queries (25 combinations)
        gaps.extend(self._analyze_distance())
        
        return gaps
    
    # ========================================================================
    # MEMBERSHIP (21 combinations)
    # ========================================================================
    
    def _analyze_membership(self) -> List[ImplementationGap]:
        gaps = []
        
        # ─────────────────────────────────────────────────────────────────
        # Point × Region (6 combinations)
        # ─────────────────────────────────────────────────────────────────
        
        gaps.append(ImplementationGap(
            subject_extent="point", subject_epistemic="known",
            target_extent="region", target_epistemic="known",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="Region.indicator(point) → bool",
            algorithm_sketch="Point-in-polygon test (existing)",
            estimated_loc=0,
            computational_strategy="analytical",
            result_type="bool"
        ))
        
        gaps.append(ImplementationGap(
            subject_extent="point", subject_epistemic="uncertain",
            target_extent="region", target_epistemic="known",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="ProbabilityEstimator.compute() - CORE FRAMEWORK",
            algorithm_sketch="∫ p(x)·I_R(x) dx via mollified indicator",
            estimated_loc=0,
            computational_strategy="framework"
        ))
        
        gaps.append(ImplementationGap(
            subject_extent="point", subject_epistemic="known",
            target_extent="region", target_epistemic="fuzzy",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="FuzzyRegion.membership(point)",
            algorithm_sketch="Evaluate μ(p)",
            estimated_loc=0,
            computational_strategy="analytical"
        ))
        
        gaps.append(ImplementationGap(
            subject_extent="point", subject_epistemic="uncertain",
            target_extent="region", target_epistemic="fuzzy",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="Framework with fuzzy membership",
            algorithm_sketch="∫ p(x)·μ(x) dx",
            estimated_loc=0,
            computational_strategy="framework"
        ))
        
        gaps.append(ImplementationGap(
            subject_extent="point", subject_epistemic="known",
            target_extent="region", target_epistemic="uncertain",
            query_type="membership",
            status=ImplementationStatus.STRAIGHTFORWARD,
            current_support="Missing",
            algorithm_sketch="Monte Carlo over region ensemble",
            estimated_loc=15,
            computational_strategy="monte_carlo"
        ))
        
        gaps.append(ImplementationGap(
            subject_extent="point", subject_epistemic="uncertain",
            target_extent="region", target_epistemic="uncertain",
            query_type="membership",
            status=ImplementationStatus.STRAIGHTFORWARD,
            current_support="Missing",
            algorithm_sketch="Double Monte Carlo",
            estimated_loc=25,
            computational_strategy="monte_carlo"
        ))
        
        # ─────────────────────────────────────────────────────────────────
        # Region × Point (6 combinations) - Symmetric
        # ─────────────────────────────────────────────────────────────────
        
        # known × known
        gaps.append(ImplementationGap(
            subject_extent="region", subject_epistemic="known",
            target_extent="point", target_epistemic="known",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="Symmetric to point×region",
            algorithm_sketch="region.indicator(point)",
            estimated_loc=0,
            computational_strategy="analytical"
        ))
        
        # known × uncertain
        gaps.append(ImplementationGap(
            subject_extent="region", subject_epistemic="known",
            target_extent="point", target_epistemic="uncertain",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="Symmetric - framework",
            algorithm_sketch="Same as uncertain point × known region",
            estimated_loc=0,
            computational_strategy="framework"
        ))
        
        # uncertain × known
        gaps.append(ImplementationGap(
            subject_extent="region", subject_epistemic="uncertain",
            target_extent="point", target_epistemic="known",
            query_type="membership",
            status=ImplementationStatus.STRAIGHTFORWARD,
            current_support="Missing",
            algorithm_sketch="P(point ∈ R̃)",
            estimated_loc=10,
            computational_strategy="monte_carlo"
        ))
        
        # uncertain × uncertain
        gaps.append(ImplementationGap(
            subject_extent="region", subject_epistemic="uncertain",
            target_extent="point", target_epistemic="uncertain",
            query_type="membership",
            status=ImplementationStatus.STRAIGHTFORWARD,
            current_support="Missing",
            algorithm_sketch="Double Monte Carlo (symmetric)",
            estimated_loc=10,
            computational_strategy="monte_carlo"
        ))
        
        # fuzzy × known
        gaps.append(ImplementationGap(
            subject_extent="region", subject_epistemic="fuzzy",
            target_extent="point", target_epistemic="known",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="Symmetric - fuzzy membership",
            algorithm_sketch="μ(point)",
            estimated_loc=0,
            computational_strategy="analytical"
        ))
        
        # fuzzy × uncertain
        gaps.append(ImplementationGap(
            subject_extent="region", subject_epistemic="fuzzy",
            target_extent="point", target_epistemic="uncertain",
            query_type="membership",
            status=ImplementationStatus.IMPLEMENTED,
            current_support="Symmetric - framework",
            algorithm_sketch="∫ p(x)·μ(x) dx",
            estimated_loc=0,
            computational_strategy="framework"
        ))
        
        # ─────────────────────────────────────────────────────────────────
        # Region × Region (9 combinations)
        # ─────────────────────────────────────────────────────────────────
        
        # All 9 combinations: 3 subject epistemic × 3 target epistemic
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                # known × known: straightforward but needs RegionDistribution
                if s_epi == "known" and t_epi == "known":
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="membership",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="RegionDistribution exists, needs wiring",
                        algorithm_sketch="Uniform sampling + framework",
                        estimated_loc=40,
                        computational_strategy="framework"
                    ))
                
                # Fuzzy cases
                elif "fuzzy" in [s_epi, t_epi]:
                    if s_epi == "fuzzy" and t_epi == "fuzzy":
                        gaps.append(ImplementationGap(
                            subject_extent="region", subject_epistemic=s_epi,
                            target_extent="region", target_epistemic=t_epi,
                            query_type="membership",
                            status=ImplementationStatus.STRAIGHTFORWARD,
                            current_support="Missing",
                            algorithm_sketch="∫∫ μ₁(x)·μ₂(x) dx (fuzzy intersection)",
                            estimated_loc=25,
                            computational_strategy="framework"
                        ))
                    else:
                        gaps.append(ImplementationGap(
                            subject_extent="region", subject_epistemic=s_epi,
                            target_extent="region", target_epistemic=t_epi,
                            query_type="membership",
                            status=ImplementationStatus.MODERATE,
                            current_support="Missing",
                            algorithm_sketch="Fuzzy/crisp mix",
                            estimated_loc=60,
                            computational_strategy="framework/hybrid"
                        ))
                
                # Uncertain cases
                else:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="membership",
                        status=ImplementationStatus.MODERATE,
                        current_support="Missing",
                        algorithm_sketch="Monte Carlo over ensembles",
                        estimated_loc=50,
                        computational_strategy="monte_carlo/hybrid"
                    ))
        
        return gaps
    
    # ========================================================================
    # SUBSET (9 combinations - all region × region)
    # ========================================================================
    
    def _analyze_subset(self) -> List[ImplementationGap]:
        gaps = []
        
        # All 9: 3 subject epistemic × 3 target epistemic
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                # known × known: straightforward
                if s_epi == "known" and t_epi == "known":
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="subset",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="Check I_s ≤ I_t everywhere",
                        estimated_loc=25,
                        computational_strategy="analytical"
                    ))
                
                # uncertain × known: straightforward
                elif s_epi == "uncertain" and t_epi == "known":
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="subset",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="P(R̃ ⊆ target)",
                        estimated_loc=15,
                        computational_strategy="monte_carlo"
                    ))
                
                # Fuzzy cases: research needed
                elif "fuzzy" in [s_epi, t_epi]:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="subset",
                        status=ImplementationStatus.RESEARCH_NEEDED,
                        current_support="Missing",
                        algorithm_sketch="Fuzzy subset degree - multiple definitions",
                        estimated_loc=40,
                        computational_strategy="research",
                        research_notes="Zadeh vs normalized definitions"
                    ))
                
                # Other uncertain cases
                else:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="subset",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="Monte Carlo variant",
                        estimated_loc=20,
                        computational_strategy="monte_carlo"
                    ))
        
        return gaps
    
    # ========================================================================
    # INTERSECTION (9 combinations - all region × region)
    # ========================================================================
    
    def _analyze_intersection(self) -> List[ImplementationGap]:
        gaps = []
        
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                # known × known: straightforward
                if s_epi == "known" and t_epi == "known":
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="intersection",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="Check if R₁ ∩ R₂ ≠ ∅",
                        estimated_loc=15,
                        computational_strategy="analytical"
                    ))
                
                # Fuzzy cases: moderate complexity
                elif "fuzzy" in [s_epi, t_epi]:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="intersection",
                        status=ImplementationStatus.MODERATE,
                        current_support="Missing",
                        algorithm_sketch="Fuzzy intersection (needs threshold or degree)",
                        estimated_loc=40,
                        computational_strategy="monte_carlo/framework"
                    ))
                
                # Uncertain cases: straightforward Monte Carlo
                else:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="intersection",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="P(R̃₁ ∩ R̃₂ ≠ ∅)",
                        estimated_loc=20,
                        computational_strategy="monte_carlo"
                    ))
        
        return gaps
    
    # ========================================================================
    # OVERLAP_FRACTION (9 combinations - all region × region)
    # ========================================================================
    
    def _analyze_overlap_fraction(self) -> List[ImplementationGap]:
        gaps = []
        
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                # known × known: straightforward
                if s_epi == "known" and t_epi == "known":
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="overlap_fraction",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="|R₁∩R₂| / |R₁|",
                        estimated_loc=25,
                        computational_strategy="analytical"
                    ))
                
                # Fuzzy cases: moderate
                elif "fuzzy" in [s_epi, t_epi]:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="overlap_fraction",
                        status=ImplementationStatus.MODERATE,
                        current_support="Missing",
                        algorithm_sketch="Fuzzy overlap fraction",
                        estimated_loc=45,
                        computational_strategy="monte_carlo/framework"
                    ))
                
                # Uncertain cases: straightforward
                else:
                    gaps.append(ImplementationGap(
                        subject_extent="region", subject_epistemic=s_epi,
                        target_extent="region", target_epistemic=t_epi,
                        query_type="overlap_fraction",
                        status=ImplementationStatus.STRAIGHTFORWARD,
                        current_support="Missing",
                        algorithm_sketch="E[|R̃∩target|/|R̃|]",
                        estimated_loc=20,
                        computational_strategy="monte_carlo"
                    ))
        
        return gaps
    
    # ========================================================================
    # PROXIMITY (25 combinations)
    # ========================================================================
    
    def _analyze_proximity(self) -> List[ImplementationGap]:
        gaps = []
        
        # ─────────────────────────────────────────────────────────────────
        # Point × Point (4 combinations: 2×2)
        # ─────────────────────────────────────────────────────────────────
        
        for s_epi in ["known", "uncertain"]:
            for t_epi in ["known", "uncertain"]:
                
                if s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                    support = "metric.distance(p1, p2) <= threshold"
                elif s_epi == "uncertain" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                    support = "Framework with DiskRegion"
                elif s_epi == "known" and t_epi == "uncertain":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                    support = "Symmetric to above"
                else:  # uncertain × uncertain
                    status = ImplementationStatus.MODERATE
                    loc = 30
                    support = "Missing"
                
                gaps.append(ImplementationGap(
                    subject_extent="point", subject_epistemic=s_epi,
                    target_extent="point", target_epistemic=t_epi,
                    query_type="proximity",
                    status=status,
                    current_support=support,
                    algorithm_sketch="P(d(X,Y) ≤ δ)" if "uncertain" in [s_epi, t_epi] else "d(p1,p2) ≤ δ",
                    estimated_loc=loc,
                    computational_strategy="monte_carlo" if loc > 0 else "analytical/framework"
                ))
        
        # ─────────────────────────────────────────────────────────────────
        # Point × Region (6 combinations: 2×3)
        # ─────────────────────────────────────────────────────────────────
        
        for s_epi in ["known", "uncertain"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                if s_epi == "uncertain" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                    support = "BufferedPolygonRegion + framework (CORE GEOFENCE)"
                    strat = "framework"
                elif s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                    support = "Distance to boundary"
                    strat = "analytical"
                elif s_epi == "uncertain" and t_epi == "uncertain":
                    status = ImplementationStatus.MODERATE
                    loc = 80
                    support = "Missing - PRIORITY 1"
                    strat = "hybrid"
                elif t_epi == "fuzzy":
                    status = ImplementationStatus.RESEARCH_NEEDED
                    loc = 70
                    support = "Missing - fuzzy boundary distance"
                    strat = "research"
                else:
                    status = ImplementationStatus.STRAIGHTFORWARD
                    loc = 40
                    support = "Missing"
                    strat = "monte_carlo"
                
                gaps.append(ImplementationGap(
                    subject_extent="point", subject_epistemic=s_epi,
                    target_extent="region", target_epistemic=t_epi,
                    query_type="proximity",
                    status=status,
                    current_support=support,
                    algorithm_sketch="P(d(X, ∂R) ≤ δ)",
                    estimated_loc=loc,
                    computational_strategy=strat
                ))
        
        # ─────────────────────────────────────────────────────────────────
        # Region × Point (6 combinations: 3×2) - Mostly symmetric
        # ─────────────────────────────────────────────────────────────────
        
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain"]:
                
                # Similar to point × region (symmetric)
                if s_epi == "known" and t_epi in ["known", "uncertain"]:
                    status = ImplementationStatus.STRAIGHTFORWARD
                    loc = 10
                elif s_epi == "fuzzy":
                    status = ImplementationStatus.RESEARCH_NEEDED
                    loc = 70
                else:
                    status = ImplementationStatus.MODERATE
                    loc = 50
                
                gaps.append(ImplementationGap(
                    subject_extent="region", subject_epistemic=s_epi,
                    target_extent="point", target_epistemic=t_epi,
                    query_type="proximity",
                    status=status,
                    current_support="Symmetric to point×region",
                    algorithm_sketch="d(R, point) ≤ δ",
                    estimated_loc=loc,
                    computational_strategy="monte_carlo" if loc > 20 else "analytical"
                ))
        
        # ─────────────────────────────────────────────────────────────────
        # Region × Region (9 combinations: 3×3)
        # ─────────────────────────────────────────────────────────────────
        
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                if s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.MODERATE
                    loc = 100
                    support = "Needs RegionDistanceSemantics"
                    strat = "analytical"
                elif "fuzzy" in [s_epi, t_epi]:
                    status = ImplementationStatus.RESEARCH_NEEDED
                    loc = 80
                    support = "Fuzzy proximity unclear"
                    strat = "research"
                else:
                    status = ImplementationStatus.MODERATE
                    loc = 80
                    support = "Missing"
                    strat = "monte_carlo/hybrid"
                
                gaps.append(ImplementationGap(
                    subject_extent="region", subject_epistemic=s_epi,
                    target_extent="region", target_epistemic=t_epi,
                    query_type="proximity",
                    status=status,
                    current_support=support,
                    algorithm_sketch="d(R₁, R₂) ≤ δ with specified semantics",
                    estimated_loc=loc,
                    computational_strategy=strat
                ))
        
        return gaps
    
    # ========================================================================
    # DISTANCE (25 combinations - similar structure to PROXIMITY)
    # ========================================================================
    
    def _analyze_distance(self) -> List[ImplementationGap]:
        gaps = []
        
        # Point × Point (4)
        for s_epi in ["known", "uncertain"]:
            for t_epi in ["known", "uncertain"]:
                
                if s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                else:
                    status = ImplementationStatus.STRAIGHTFORWARD
                    loc = 30
                
                gaps.append(ImplementationGap(
                    subject_extent="point", subject_epistemic=s_epi,
                    target_extent="point", target_epistemic=t_epi,
                    query_type="distance",
                    status=status,
                    current_support="Implemented" if loc == 0 else "Missing",
                    algorithm_sketch="d(p1,p2)" if loc == 0 else "E[d(X,Y)] or distribution",
                    estimated_loc=loc,
                    computational_strategy="analytical" if loc == 0 else "monte_carlo"
                ))
        
        # Point × Region (6)
        for s_epi in ["known", "uncertain"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                if s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                elif t_epi == "fuzzy":
                    status = ImplementationStatus.RESEARCH_NEEDED
                    loc = 70
                else:
                    status = ImplementationStatus.STRAIGHTFORWARD
                    loc = 30
                
                gaps.append(ImplementationGap(
                    subject_extent="point", subject_epistemic=s_epi,
                    target_extent="region", target_epistemic=t_epi,
                    query_type="distance",
                    status=status,
                    current_support="Partial" if loc == 0 else "Missing",
                    algorithm_sketch="d(point, ∂region)",
                    estimated_loc=loc,
                    computational_strategy="analytical" if loc == 0 else "monte_carlo"
                ))
        
        # Region × Point (6) - symmetric
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain"]:
                
                if s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.IMPLEMENTED
                    loc = 0
                elif s_epi == "fuzzy":
                    status = ImplementationStatus.RESEARCH_NEEDED
                    loc = 70
                else:
                    status = ImplementationStatus.STRAIGHTFORWARD
                    loc = 30
                
                gaps.append(ImplementationGap(
                    subject_extent="region", subject_epistemic=s_epi,
                    target_extent="point", target_epistemic=t_epi,
                    query_type="distance",
                    status=status,
                    current_support="Symmetric",
                    algorithm_sketch="d(∂region, point)",
                    estimated_loc=loc,
                    computational_strategy="analytical" if loc == 0 else "monte_carlo"
                ))
        
        # Region × Region (9)
        for s_epi in ["known", "uncertain", "fuzzy"]:
            for t_epi in ["known", "uncertain", "fuzzy"]:
                
                if s_epi == "known" and t_epi == "known":
                    status = ImplementationStatus.MODERATE
                    loc = 100
                elif "fuzzy" in [s_epi, t_epi]:
                    status = ImplementationStatus.RESEARCH_NEEDED
                    loc = 80
                else:
                    status = ImplementationStatus.MODERATE
                    loc = 70
                
                gaps.append(ImplementationGap(
                    subject_extent="region", subject_epistemic=s_epi,
                    target_extent="region", target_epistemic=t_epi,
                    query_type="distance",
                    status=status,
                    current_support="Needs semantics" if s_epi == "known" and t_epi == "known" else "Missing",
                    algorithm_sketch="d(R₁, R₂) with specified semantics",
                    estimated_loc=loc,
                    computational_strategy="analytical" if s_epi == "known" and t_epi == "known" else "monte_carlo"
                ))
        
        return gaps
    
    # ========================================================================
    # Analysis and Reporting
    # ========================================================================
    
    def get_statistics(self) -> Dict:
        """Compute statistics across all gaps"""
        stats = {
            'total': len(self.gaps),
            'by_status': {},
            'by_query_type': {},
            'total_estimated_loc': 0,
            'by_computational_strategy': {}
        }
        
        for gap in self.gaps:
            # Count by status
            status = gap.status.name
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by query type
            stats['by_query_type'][gap.query_type] = stats['by_query_type'].get(gap.query_type, 0) + 1
            
            # Sum LOC
            if gap.status != ImplementationStatus.IMPLEMENTED:
                stats['total_estimated_loc'] += gap.estimated_loc
            
            # Count by strategy
            strat = gap.computational_strategy
            stats['by_computational_strategy'][strat] = stats['by_computational_strategy'].get(strat, 0) + 1
        
        return stats
    
    def print_summary(self):
        """Print executive summary"""
        stats = self.get_statistics()
        
        print("="*80)
        print("MATURE IMPLEMENTATION GAP ANALYSIS (98 Valid Combinations)")
        print("="*80)
        print()
        print(f"Total combinations: {stats['total']}")
        
        if stats['total'] != 98:
            print(f"⚠️  WARNING: Expected 98, got {stats['total']} (missing {98 - stats['total']})!")
        
        print()
        
        print("Status Breakdown:")
        print("-"*80)
        for status_name, count in sorted(stats['by_status'].items()):
            pct = count / stats['total'] * 100
            emoji = {
                'IMPLEMENTED': '✅',
                'STRAIGHTFORWARD': '🟢',
                'MODERATE': '🟡',
                'COMPLEX': '🔴',
                'RESEARCH_NEEDED': '🔵',
                'BLOCKED': '⚫'
            }.get(status_name, '•')
            print(f"  {emoji} {status_name:20s}: {count:3d} ({pct:5.1f}%)")
        
        print()
        print(f"Estimated LOC to implement: {stats['total_estimated_loc']:,}")
        print()
        
        print("By Query Type:")
        print("-"*80)
        for q_type, count in sorted(stats['by_query_type'].items()):
            expected = {
                'membership': 21,
                'subset': 9,
                'intersection': 9,
                'overlap_fraction': 9,
                'proximity': 25,
                'distance': 25
            }.get(q_type, 0)
            
            warning = "" if count == expected else f" ⚠️  (expected {expected})"
            print(f"  {q_type:20s}: {count:3d}{warning}")
        
        print()
        print("By Computational Strategy:")
        print("-"*80)
        for strat, count in sorted(stats['by_computational_strategy'].items()):
            print(f"  {strat:20s}: {count:3d}")
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export for detailed analysis"""
        return pd.DataFrame([
            {
                'combination': str(gap),
                'status': gap.status.name,
                'query_type': gap.query_type,
                'subject': f"{gap.subject_extent},{gap.subject_epistemic}",
                'target': f"{gap.target_extent},{gap.target_epistemic}",
                'estimated_loc': gap.estimated_loc,
                'strategy': gap.computational_strategy,
                'current_support': gap.current_support
            }
            for gap in self.gaps
        ])
    
    def print_detailed_breakdown(self):
        """Print detailed breakdown by query type"""
        print("\n" + "="*80)
        print("DETAILED BREAKDOWN BY QUERY TYPE")
        print("="*80)
        
        by_query = {}
        for gap in self.gaps:
            if gap.query_type not in by_query:
                by_query[gap.query_type] = []
            by_query[gap.query_type].append(gap)
        
        for q_type in ['membership', 'subset', 'intersection', 'overlap_fraction', 'proximity', 'distance']:
            gaps = by_query.get(q_type, [])
            print(f"\n{q_type.upper()} ({len(gaps)} combinations)")
            print("-"*80)
            
            # Group by status
            by_status = {}
            for gap in gaps:
                status = gap.status.name
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(gap)
            
            for status in ['IMPLEMENTED', 'STRAIGHTFORWARD', 'MODERATE', 'COMPLEX', 'RESEARCH_NEEDED']:
                status_gaps = by_status.get(status, [])
                if status_gaps:
                    emoji = {
                        'IMPLEMENTED': '✅',
                        'STRAIGHTFORWARD': '🟢',
                        'MODERATE': '🟡',
                        'COMPLEX': '🔴',
                        'RESEARCH_NEEDED': '🔵'
                    }[status]
                    print(f"\n  {emoji} {status} ({len(status_gaps)}):")
                    for gap in status_gaps[:5]:  # Show first 5
                        print(f"    • {gap}")
                    if len(status_gaps) > 5:
                        print(f"    ... and {len(status_gaps) - 5} more")


if __name__ == "__main__":
    # Create the complete validation matrix
    analyzer = GapAnalyzer()
    
    # Print summary
    analyzer.print_summary()
    
    # Print detailed breakdown
    analyzer.print_detailed_breakdown()
    
    print("\n" + "="*80)
    print("PRIORITY IMPLEMENTATION ROADMAP")
    print("="*80)
    print()
    print("Phase 1: Quick Wins (Straightforward, < 50 LOC each)")
    print("  • Point in uncertain region")
    print("  • Region × region SUBSET/INTERSECTION/OVERLAP_FRACTION (known×known)")
    print("  • Uncertain distance statistics")
    print("  Estimated: ~200 LOC total")
    print()
    print("Phase 2: Core Extensions (Moderate, 50-200 LOC)")
    print("  • RegionDistribution integration (known×known membership)")
    print("  • Uncertain × uncertain proximity")
    print("  • Uncertain point × uncertain region proximity (PRIORITY!)")
    print("  • Region distance semantics system")
    print("  Estimated: ~400 LOC total")
    print()
    print("Phase 3: Research Cases")
    print("  • Fuzzy subset semantics")
    print("  • Fuzzy proximity/distance definitions")
    print("  Estimated: ~200 LOC after research")
    print()
    print("="*80)
    stats = analyzer.get_statistics()
    print(f"Total estimated: ~{stats['total_estimated_loc']} LOC for full coverage")
    implemented = stats['by_status'].get('IMPLEMENTED', 0)
    print(f"Current coverage: ~{implemented/stats['total']*100:.0f}% ({implemented}/{stats['total']})")
    
    straightforward = stats['by_status'].get('STRAIGHTFORWARD', 0)
    moderate = stats['by_status'].get('MODERATE', 0)
    after_phase_1_2 = implemented + straightforward + moderate
    print(f"After Phase 1+2: ~{after_phase_1_2/stats['total']*100:.0f}% ({after_phase_1_2}/{stats['total']})")
    print("="*80)
    
    # Validation check
    print("\n" + "="*80)
    print("VALIDATION CHECK")
    print("="*80)
    
    expected_counts = {
        'membership': 21,
        'subset': 9,
        'intersection': 9,
        'overlap_fraction': 9,
        'proximity': 25,
        'distance': 25
    }
    
    actual_counts = {}
    for gap in analyzer.gaps:
        actual_counts[gap.query_type] = actual_counts.get(gap.query_type, 0) + 1
    
    all_correct = True
    for q_type, expected in expected_counts.items():
        actual = actual_counts.get(q_type, 0)
        status = "✅" if actual == expected else "❌"
        print(f"{status} {q_type:20s}: {actual:3d} / {expected:3d}")
        if actual != expected:
            all_correct = False
    
    print()
    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())
    
    if total_actual == total_expected and all_correct:
        print(f"✅ VALIDATION PASSED: All {total_expected} combinations accounted for!")
    else:
        print(f"❌ VALIDATION FAILED: Expected {total_expected}, got {total_actual}")
        if total_actual < total_expected:
            print(f"   Missing {total_expected - total_actual} combinations")
        else:
            print(f"   {total_actual - total_expected} extra combinations")
    
    print("="*80)