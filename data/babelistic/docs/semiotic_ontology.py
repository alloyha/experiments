"""
Complete Ontology for Spatial Probability Framework

CORE INSIGHT: Separate physical entities from epistemic states

Physical Layer: WHAT exists (Subject, Target)
Epistemic Layer: HOW CERTAIN we are (Query, Membership)
Metric Layer: HOW we measure (MetricSpace, distance_threshold)

This gives us a complete, exhaustive interaction space.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Union, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np


# ============================================================================
# LAYER 0: SPATIAL FOUNDATION
# ============================================================================

class MetricSpace(ABC):
    """Defines distance measurement"""
    @abstractmethod
    def distance(self, x, y) -> float:
        pass


# ============================================================================
# LAYER 1: PHYSICAL ENTITIES (Ontic - "what exists")
# ============================================================================

class PhysicalEntity(ABC):
    """
    Base class for physical spatial entities.
    
    These represent WHAT exists in space, independent of our knowledge:
    - A person (subject)
    - A building (target)
    - A zone (target)
    
    Key: Physical entities are NOT inherently uncertain.
          Uncertainty comes from our OBSERVATION of them (epistemic layer).
    """
    
    @abstractmethod
    def spatial_extent(self) -> str:
        """Is this a point or a region?"""
        pass


class PointEntity(PhysicalEntity):
    """
    A physical entity occupying a single point in space.
    
    Examples:
    - A person
    - A vehicle
    - A drone
    - A sensor
    
    Note: The entity IS a point, but we may be UNCERTAIN about where.
    """
    
    def spatial_extent(self):
        return "point"
    
    def __repr__(self):
        return f"PointEntity()"


class RegionEntity(PhysicalEntity):
    """
    A physical entity occupying a region of space.
    
    Examples:
    - A building (polygon)
    - A forest (irregular region)
    - A country (complex polygon)
    - A geofence zone (circle or polygon)
    
    Note: The entity IS a region, but boundaries may be UNCERTAIN.
    """
    
    def __init__(self, geometry_type: str):
        """
        Parameters
        ----------
        geometry_type : str
            'disk', 'polygon', 'ellipse', 'implicit', 'multi', etc.
        """
        self.geometry_type = geometry_type
    
    def spatial_extent(self):
        return "region"
    
    def __repr__(self):
        return f"RegionEntity(type={self.geometry_type})"


# ============================================================================
# LAYER 2: EPISTEMIC STATES (How certain are we?)
# ============================================================================

class EpistemicType(Enum):
    """
    Classification of uncertainty/knowledge about spatial entities.
    
    This captures HOW CERTAIN we are about where something is or
    what its boundaries are.
    """
    KNOWN = "known"              # Perfect knowledge (deterministic)
    UNCERTAIN = "uncertain"       # Probabilistic knowledge (distribution)
    FUZZY = "fuzzy"              # Graded membership (fuzzy set)
    UNKNOWN = "unknown"          # Complete ignorance


class EpistemicState(ABC):
    """
    Represents our state of knowledge about a physical entity's location/extent.
    
    This is the KEY abstraction that separates:
    - WHAT exists (PhysicalEntity)
    - WHAT WE KNOW about it (EpistemicState)
    """
    
    @abstractmethod
    def epistemic_type(self) -> EpistemicType:
        pass


# ─────────────────────────────────────────────────────────────────────────
# Epistemic States for POINT ENTITIES
# ─────────────────────────────────────────────────────────────────────────

class KnownPoint(EpistemicState):
    """
    We know EXACTLY where the point entity is.
    
    Example: "The tower is at exactly (37.7749, -122.4194)"
    """
    def __init__(self, location: np.ndarray):
        self.location = location
    
    def epistemic_type(self):
        return EpistemicType.KNOWN


class UncertainPoint(EpistemicState):
    """
    We are UNCERTAIN about where the point entity is.
    
    Example: "GPS says person is at (37.7749, -122.4194) ± 10m"
    
    Represented by a probability distribution over possible locations.
    """
    def __init__(self, distribution: 'UncertaintyDistribution'):
        self.distribution = distribution
    
    def epistemic_type(self):
        return EpistemicType.UNCERTAIN


class FuzzyPoint(EpistemicState):
    """
    The point's location has FUZZY semantics.
    
    Example: "The person is 'near' the building"
            (not a probability, but a degree of truth)
    
    This is rare for points but included for completeness.
    """
    def __init__(self, fuzzy_membership):
        self.membership = fuzzy_membership
    
    def epistemic_type(self):
        return EpistemicType.FUZZY


# ─────────────────────────────────────────────────────────────────────────
# Epistemic States for REGION ENTITIES
# ─────────────────────────────────────────────────────────────────────────

class KnownRegion(EpistemicState):
    """
    We know EXACTLY what the region's boundaries are.
    
    Example: "Building footprint from cadastral data"
    
    This is a CRISP set: indicator ∈ {0, 1}
    """
    def __init__(self, region: 'Region'):
        self.region = region
    
    def epistemic_type(self):
        return EpistemicType.KNOWN


class UncertainRegion(EpistemicState):
    """
    We are UNCERTAIN about the region's boundaries.
    
    Example: "Forest extent estimated from satellite imagery"
            "Flood zone forecast (probabilistic boundary)"
    
    Represented by a probability distribution over possible region shapes.
    
    Note: This is DIFFERENT from fuzzy! Each possible region is crisp,
          but we don't know WHICH crisp region is the true one.
    """
    def __init__(self, region_ensemble: list):
        """
        Parameters
        ----------
        region_ensemble : list of (Region, probability) tuples
            Multiple possible regions with their probabilities
        """
        self.ensemble = region_ensemble  # [(region1, p1), (region2, p2), ...]
    
    def epistemic_type(self):
        return EpistemicType.UNCERTAIN


class FuzzyRegion(EpistemicState):
    """
    The region has FUZZY boundaries (graded membership).
    
    Example: "The forest" - gradual transition from forest to non-forest
            "Urban area" - no sharp boundary, gradually becomes suburban
            "Influence zone" - membership decreases with distance
    
    This is a FUZZY set: membership ∈ [0, 1]
    
    Key difference from UncertainRegion:
    - Fuzzy: The boundary IS gradual (ontological vagueness)
    - Uncertain: The boundary is sharp, but we don't know where (epistemic uncertainty)
    """
    def __init__(self, membership_function):
        """
        Parameters
        ----------
        membership_function : callable
            Function x → [0, 1] giving degree of membership
        """
        self.membership = membership_function
    
    def epistemic_type(self):
        return EpistemicType.FUZZY


# ============================================================================
# LAYER 3: ROLES (Subject vs Target)
# ============================================================================

@dataclass
class Subject:
    """
    The entity we're asking about (the "subject" of the query).
    
    Examples:
    - A person with GPS uncertainty
    - A vehicle with known location
    - A drone swarm (region entity) with uncertain extent
    
    Composition: PhysicalEntity + EpistemicState
    """
    entity: PhysicalEntity
    state: EpistemicState
    
    def __repr__(self):
        return f"Subject({self.entity}, {self.state.epistemic_type().value})"


@dataclass
class Target:
    """
    The reference entity we're checking against (the "target").
    
    Examples:
    - A building with known footprint
    - A geofence zone (known region)
    - A border with fuzzy boundaries
    
    Composition: PhysicalEntity + EpistemicState
    """
    entity: PhysicalEntity
    state: EpistemicState
    
    def __repr__(self):
        return f"Target({self.entity}, {self.state.epistemic_type().value})"


# ============================================================================
# LAYER 4: QUERY SPECIFICATION
# ============================================================================

class QueryType(Enum):
    """Types of spatial queries we can ask"""
    CONTAINMENT = "containment"        # Is subject IN target?
    PROXIMITY = "proximity"            # Is subject NEAR target?
    DISTANCE = "distance"              # How far is subject FROM target?
    OVERLAP = "overlap"                # Do regions overlap?


@dataclass
class Query:
    """
    A complete spatial query specification.
    
    This brings together:
    - WHO: subject (what we're asking about)
    - WHERE: target (what we're checking against)
    - WHAT: query type (containment, proximity, etc.)
    - HOW: metric space and parameters
    
    Examples
    --------
    Q1: "Is person (GPS ±10m) inside building (known)?"
        Subject: PointEntity + UncertainPoint
        Target: RegionEntity + KnownRegion
        Query type: CONTAINMENT
    
    Q2: "Is person (GPS ±10m) within 50m of building (known)?"
        Subject: PointEntity + UncertainPoint
        Target: RegionEntity + KnownRegion
        Query type: PROXIMITY (threshold=50m)
    
    Q3: "Does parking lot (known) overlap with building buffer (known + 50m)?"
        Subject: RegionEntity + KnownRegion
        Target: RegionEntity + KnownRegion
        Query type: PROXIMITY (threshold=50m)
    
    Q4: "Is person (GPS ±10m) in forest (fuzzy boundaries)?"
        Subject: PointEntity + UncertainPoint
        Target: RegionEntity + FuzzyRegion
        Query type: CONTAINMENT
    """
    subject: Subject
    target: Target
    query_type: QueryType
    metric_space: MetricSpace
    
    # Query parameters
    distance_threshold: Optional[float] = None  # For PROXIMITY queries
    
    def __repr__(self):
        return (f"Query(\n"
                f"  subject={self.subject},\n"
                f"  target={self.target},\n"
                f"  type={self.query_type.value},\n"
                f"  threshold={self.distance_threshold}\n"
                f")")


# ============================================================================
# LAYER 5: EXHAUSTIVE INTERACTION CLASSIFICATION
# ============================================================================

@dataclass
class InteractionSignature:
    """
    Signature uniquely identifying an interaction type.
    
    This is the KEY to exhaustive classification:
    Every valid query maps to exactly one interaction signature.
    """
    subject_extent: str        # "point" or "region"
    subject_epistemic: EpistemicType
    target_extent: str         # "point" or "region"
    target_epistemic: EpistemicType
    query_type: QueryType
    
    def __hash__(self):
        return hash((
            self.subject_extent,
            self.subject_epistemic,
            self.target_extent,
            self.target_epistemic,
            self.query_type
        ))


class ComputationMethod(Enum):
    """Methods for computing query results"""
    
    # Probabilistic methods (return probability ∈ [0, 1])
    FRAMEWORK_CRISP = "framework_crisp"          # Core: P(X ∈ R)
    FRAMEWORK_FUZZY = "framework_fuzzy"          # Fuzzy: P(X ∈ μ)
    FRAMEWORK_BUFFERED = "framework_buffered"    # Proximity: P(d(X,R) ≤ δ)
    MONTE_CARLO = "monte_carlo"                  # Uncertain boundaries
    
    # Deterministic methods (return bool or float)
    GEOMETRIC_TEST = "geometric_test"            # Point in region?
    GEOMETRIC_INTERSECTION = "geometric_intersection"  # Regions overlap?
    DISTANCE_COMPUTATION = "distance_computation"     # Exact distance
    
    # Fuzzy methods (return degree ∈ [0, 1])
    FUZZY_MEMBERSHIP = "fuzzy_membership"        # μ(x)
    
    # Invalid
    UNSUPPORTED = "unsupported"


# ============================================================================
# THE EXHAUSTIVE INTERACTION MATRIX
# ============================================================================

class InteractionClassifier:
    """
    Classifies ALL possible interactions and maps them to computation methods.
    
    This is the HEART of the ontology: it exhaustively enumerates what
    computations are valid and how to perform them.
    """
    
    def __init__(self):
        """Build the exhaustive interaction matrix"""
        self.matrix = self._build_interaction_matrix()
    
    def _build_interaction_matrix(self):
        """
        Exhaustively enumerate all valid interaction signatures.
        
        Format: InteractionSignature → (ComputationMethod, result_type, description)
        """
        matrix = {}
        
        # ═════════════════════════════════════════════════════════════════
        # POINT SUBJECT INTERACTIONS
        # ═════════════════════════════════════════════════════════════════
        
        # ─────────────────────────────────────────────────────────────────
        # Known Point Subject
        # ─────────────────────────────────────────────────────────────────
        
        # Known point vs Known region - CONTAINMENT
        matrix[InteractionSignature("point", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = (
            ComputationMethod.GEOMETRIC_TEST,
            bool,
            "Is point inside region? (deterministic)"
        )
        
        # Known point vs Fuzzy region - CONTAINMENT
        matrix[InteractionSignature("point", EpistemicType.KNOWN, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = (
            ComputationMethod.FUZZY_MEMBERSHIP,
            float,  # Returns membership degree
            "What is membership degree μ(point)?"
        )
        
        # Known point vs Known region - PROXIMITY
        matrix[InteractionSignature("point", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = (
            ComputationMethod.DISTANCE_COMPUTATION,
            bool,  # distance <= threshold
            "Is point within threshold of region boundary? (deterministic)"
        )
        
        # ─────────────────────────────────────────────────────────────────
        # Uncertain Point Subject (THE MAIN CASE for geofencing!)
        # ─────────────────────────────────────────────────────────────────
        
        # Uncertain point vs Known region - CONTAINMENT
        matrix[InteractionSignature("point", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = (
            ComputationMethod.FRAMEWORK_CRISP,
            float,  # Probability
            "P(uncertain point ∈ known region) - Core framework formula"
        )
        
        # Uncertain point vs Known region - PROXIMITY (GEOFENCE!)
        matrix[InteractionSignature("point", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = (
            ComputationMethod.FRAMEWORK_BUFFERED,
            float,  # Probability
            "P(uncertain point within threshold of region) - Geofence case"
        )
        
        # Uncertain point vs Fuzzy region - CONTAINMENT
        matrix[InteractionSignature("point", EpistemicType.UNCERTAIN, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = (
            ComputationMethod.FRAMEWORK_FUZZY,
            float,  # Probability
            "P(uncertain point has fuzzy membership) - ∫ p(x)·μ(x) dx"
        )
        
        # Uncertain point vs Uncertain region - CONTAINMENT
        matrix[InteractionSignature("point", EpistemicType.UNCERTAIN, "region", EpistemicType.UNCERTAIN, QueryType.CONTAINMENT)] = (
            ComputationMethod.MONTE_CARLO,
            float,  # Probability
            "P(uncertain point ∈ uncertain region) - Monte Carlo over both"
        )
        
        # ═════════════════════════════════════════════════════════════════
        # REGION SUBJECT INTERACTIONS
        # ═════════════════════════════════════════════════════════════════
        
        # ─────────────────────────────────────────────────────────────────
        # Known Region Subject
        # ─────────────────────────────────────────────────────────────────
        
        # Known region vs Known region - OVERLAP
        matrix[InteractionSignature("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.OVERLAP)] = (
            ComputationMethod.GEOMETRIC_INTERSECTION,
            bool,
            "Do two known regions overlap? (deterministic)"
        )
        
        # Known region vs Known region - PROXIMITY
        matrix[InteractionSignature("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.PROXIMITY)] = (
            ComputationMethod.GEOMETRIC_INTERSECTION,
            bool,
            "Are regions within threshold? (deterministic - test buffered region)"
        )
        
        # Known region vs Known region - CONTAINMENT (interpreted as uniform)
        matrix[InteractionSignature("region", EpistemicType.KNOWN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = (
            ComputationMethod.FRAMEWORK_CRISP,
            float,  # Probability
            "P(random point from subject region ∈ target region) - Uniform interpretation"
        )
        
        # ─────────────────────────────────────────────────────────────────
        # Uncertain Region Subject
        # ─────────────────────────────────────────────────────────────────
        
        # Uncertain region vs Known region - CONTAINMENT
        matrix[InteractionSignature("region", EpistemicType.UNCERTAIN, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = (
            ComputationMethod.MONTE_CARLO,
            float,  # Probability
            "P(uncertain region overlaps with known region)"
        )
        
        # ─────────────────────────────────────────────────────────────────
        # Fuzzy Region Subject
        # ─────────────────────────────────────────────────────────────────
        
        # Fuzzy region vs Known region - CONTAINMENT
        matrix[InteractionSignature("region", EpistemicType.FUZZY, "region", EpistemicType.KNOWN, QueryType.CONTAINMENT)] = (
            ComputationMethod.FRAMEWORK_FUZZY,
            float,  # Fuzzy degree
            "Fuzzy overlap between fuzzy subject and crisp target"
        )
        
        # Fuzzy region vs Fuzzy region - CONTAINMENT
        matrix[InteractionSignature("region", EpistemicType.FUZZY, "region", EpistemicType.FUZZY, QueryType.CONTAINMENT)] = (
            ComputationMethod.FRAMEWORK_FUZZY,
            float,  # Fuzzy degree
            "Fuzzy overlap between two fuzzy regions"
        )
        
        return matrix
    
    def classify(self, query: Query) -> Tuple[ComputationMethod, type, str]:
        """
        Classify a query and determine how to compute it.
        
        Returns
        -------
        computation_method : ComputationMethod
            Which method to use
        result_type : type
            Type of result (bool, float, etc.)
        description : str
            Human-readable description
        """
        signature = InteractionSignature(
            subject_extent=query.subject.entity.spatial_extent(),
            subject_epistemic=query.subject.state.epistemic_type(),
            target_extent=query.target.entity.spatial_extent(),
            target_epistemic=query.target.state.epistemic_type(),
            query_type=query.query_type
        )
        
        result = self.matrix.get(signature)
        
        if result is None:
            return (
                ComputationMethod.UNSUPPORTED,
                None,
                f"Unsupported interaction: {signature}"
            )
        
        return result
    
    def get_all_supported_interactions(self):
        """Return all supported interaction types"""
        return list(self.matrix.keys())
    
    def print_interaction_matrix(self):
        """Print the full interaction matrix in human-readable form"""
        print("╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║              EXHAUSTIVE SPATIAL INTERACTION MATRIX                        ║")
        print("╠═══════════════════════════════════════════════════════════════════════════╣")
        print("║                                                                           ║")
        
        for signature, (method, result_type, description) in sorted(
            self.matrix.items(),
            key=lambda x: (x[0].subject_extent, x[0].subject_epistemic.value, x[0].query_type.value)
        ):
            print(f"║ Subject: {signature.subject_extent:6} ({signature.subject_epistemic.value:10})")
            print(f"║ Target:  {signature.target_extent:6} ({signature.target_epistemic.value:10})")
            print(f"║ Query:   {signature.query_type.value:20}")
            print(f"║ → Method: {method.value:25} → {result_type.__name__ if result_type else 'N/A':5}")
            print(f"║   {description}")
            print("║" + "─" * 75)
        
        print("╚═══════════════════════════════════════════════════════════════════════════╝")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_point_to_region_geofence():
    """Classic geofence: uncertain point vs known region with proximity"""
    
    # Physical entities
    person = PointEntity()
    building = RegionEntity(geometry_type="polygon")
    
    # Epistemic states
    person_location = UncertainPoint(
        GaussianDistribution(mean=np.array([37.7749, -122.4194]), cov=np.eye(2)*0.0001)
    )
    building_footprint = KnownRegion(
        PolygonRegion(building_vertices)
    )
    
    # Compose into Subject and Target
    subject = Subject(entity=person, state=person_location)
    target = Target(entity=building, state=building_footprint)
    
    # Create query
    query = Query(
        subject=subject,
        target=target,
        query_type=QueryType.PROXIMITY,
        metric_space=GeoSpace(),
        distance_threshold=50.0
    )
    
    # Classify
    classifier = InteractionClassifier()
    method, result_type, description = classifier.classify(query)
    
    print(f"Query: {description}")
    print(f"Method: {method.value}")
    print(f"Result type: {result_type.__name__}")
    # → Method: framework_buffered
    # → Result type: float (probability)


def example_region_to_region_semantic_clarity():
    """Region-to-region with CLEAR semantics"""
    
    parking = RegionEntity(geometry_type="polygon")
    building = RegionEntity(geometry_type="polygon")
    
    # CASE 1: Do they overlap? (deterministic)
    query1 = Query(
        subject=Subject(parking, KnownRegion(parking_polygon)),
        target=Target(building, KnownRegion(building_polygon)),
        query_type=QueryType.OVERLAP,
        metric_space=GeoSpace()
    )
    # → Method: geometric_intersection, Result: bool
    
    # CASE 2: P(random point from parking is in building)? (probabilistic)
    query2 = Query(
        subject=Subject(parking, KnownRegion(parking_polygon)),
        target=Target(building, KnownRegion(building_polygon)),
        query_type=QueryType.CONTAINMENT,  # Different query type!
        metric_space=GeoSpace()
    )
    # → Method: framework_crisp, Result: float (probability)
    
    # CASE 3: Are they within 50m? (deterministic with buffered region)
    query3 = Query(
        subject=Subject(parking, KnownRegion(parking_polygon)),
        target=Target(building, KnownRegion(building_polygon)),
        query_type=QueryType.PROXIMITY,
        metric_space=GeoSpace(),
        distance_threshold=50.0
    )
    # → Method: geometric_intersection (test parking ∩ buffer(building, 50m))
    # → Result: bool


def example_fuzzy_boundaries():
    """Fuzzy region support"""
    
    person = PointEntity()
    forest = RegionEntity(geometry_type="implicit")
    
    # Forest has gradual boundary (fuzzy)
    def forest_membership(x):
        # Membership decreases with distance from forest center
        center = np.array([37.5, -122.5])
        dist = np.linalg.norm(x - center, axis=-1)
        return np.exp(-dist**2 / 1000)  # Gaussian falloff
    
    query = Query(
        subject=Subject(person, UncertainPoint(GaussianDistribution(...))),
        target=Target(forest, FuzzyRegion(forest_membership)),
        query_type=QueryType.CONTAINMENT,
        metric_space=GeoSpace()
    )
    
    # → Method: framework_fuzzy
    # → Computes: ∫ p(x) · μ_forest(x) dx


if __name__ == "__main__":
    classifier = InteractionClassifier()
    classifier.print_interaction_matrix()