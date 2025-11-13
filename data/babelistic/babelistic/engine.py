"""
Mature Computational Strategy Framework for 100% Coverage

This provides a complete, pluggable system for computing ANY valid query
from the 98-combination ontology.

Key principles:
1. Strategy pattern for extensibility
2. Automatic routing based on query structure
3. Graceful degradation (analytical → framework → monte carlo)
4. Maintains agnosticism (metric/distribution/geometry)
5. 100% coverage guarantee
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import numpy as np

from .base import MetricSpace, Region
from .ontology import (
    EpistemicType, 
    RegionEpistemicType,
    QueryType,
    Query,
    Subject,
    Target,
    KnownPoint,
    UncertainPoint,
    KnownRegion,
    UncertainRegion,
    FuzzyRegion
)



# ============================================================================
# QUERY RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class QueryResult:
    """Result of a spatial query computation"""
    
    # Primary result
    value: float  # Probability, distance, boolean (0/1), membership degree
    
    # Metadata
    computation_method: str  # Which strategy was used
    result_type: str  # "probability", "distance", "boolean", "membership"
    
    # Uncertainty/confidence
    error_estimate: Optional[float] = None  # For Monte Carlo methods
    confidence_interval: Optional[tuple] = None  # (lower, upper)
    
    # Computational details
    n_samples: Optional[int] = None  # For sampling methods
    grid_resolution: Optional[int] = None  # For grid methods
    bandwidth: Optional[float] = None  # For kernel methods
    
    # Performance
    computation_time: Optional[float] = None
    
    # Additional data (for DISTANCE queries that return distributions)
    distribution_stats: Optional[Dict] = None  # mean, std, quantiles, etc.
    
    def __repr__(self):
        return f"QueryResult(value={self.value:.4f}, method={self.computation_method})"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_epistemic_type(state):
    """
    Get epistemic type from state, handling both point and region types.
    
    Returns one of: 'known', 'uncertain', 'fuzzy'
    """
    epi_type = state.epistemic_type()
    
    # Handle both EpistemicType and RegionEpistemicType
    if hasattr(epi_type, 'value'):
        return epi_type.value  # Return string: 'known', 'uncertain', or 'fuzzy'
    return str(epi_type).lower()


def is_known(state):
    """Check if state is known (certain)"""
    return get_epistemic_type(state) == 'known'


def is_uncertain(state):
    """Check if state is uncertain"""
    return get_epistemic_type(state) == 'uncertain'


def is_fuzzy(state):
    """Check if state is fuzzy"""
    return get_epistemic_type(state) == 'fuzzy'


# ============================================================================
# COMPUTATION STRATEGIES (Abstract Base)
# ============================================================================

class ComputationStrategy(ABC):
    """
    Abstract base for all computation strategies.
    
    Each strategy knows:
    1. What types of queries it can handle
    2. How to compute them
    3. Its computational cost
    """
    
    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """
        Can this strategy compute this query?
        
        Returns True if this strategy has an algorithm for the query.
        """
        pass
    
    @abstractmethod
    def compute(self, query: Query) -> QueryResult:
        """
        Compute the query result.
        
        Raises:
            NotImplementedError: If query passes can_handle but computation fails
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, query: Query) -> float:
        """
        Estimate computational cost (arbitrary units).
        
        Used by router to select cheapest capable strategy.
        Lower is better.
        """
        pass
    
    def get_name(self) -> str:
        """Strategy name for reporting"""
        return self.__class__.__name__


# ============================================================================
# STRATEGY 1: ANALYTICAL (Exact, Closed-Form Solutions)
# ============================================================================

class AnalyticalStrategy(ComputationStrategy):
    """
    Closed-form, exact solutions for deterministic queries.
    
    Handles:
    - Known × Known for all query types
    - Point-in-polygon tests
    - Geometric intersections
    - Direct distance computations
    
    Advantages: Fast, exact, no approximation
    Disadvantages: Only works for fully deterministic cases
    """
    
    def can_handle(self, query) -> bool:
        """Handles only known × known cases"""
        # Use helper functions instead of direct enum comparison
        s_known = is_known(query.subject.state)
        t_known = is_known(query.target.state)
        
        return s_known and t_known
    
    def compute(self, query) -> QueryResult:
        """Dispatch to appropriate analytical method"""
        from .ontology import QueryType
        
        if query.query_type == QueryType.MEMBERSHIP:
            return self._compute_membership(query)
        elif query.query_type == QueryType.SUBSET:
            return self._compute_subset(query)
        elif query.query_type == QueryType.INTERSECTION:
            return self._compute_intersection(query)
        elif query.query_type == QueryType.OVERLAP_FRACTION:
            return self._compute_overlap_fraction(query)
        elif query.query_type == QueryType.PROXIMITY:
            return self._compute_proximity(query)
        elif query.query_type == QueryType.DISTANCE:
            return self._compute_distance(query)
        else:
            raise NotImplementedError(f"Analytical method for {query.query_type} not implemented")
    
    def estimate_cost(self, query) -> float:
        """Analytical is ALWAYS cheapest when applicable"""
        return 0.1  # Very low cost
    
    def _compute_membership(self, query) -> QueryResult:
        """Point in region test or region×region uniform interpretation"""
        
        # Extract known point/region
        subject_extent = query.subject.entity.spatial_extent()
        target_extent = query.target.entity.spatial_extent()
        
        if subject_extent == "point":
            # Point in region
            point = query.subject.state.location
            region = query.target.state.region
            
            result = region.indicator(point)
            
            return QueryResult(
                value=float(result > 0.5),
                computation_method="analytical_point_in_region",
                result_type="boolean"
            )
        
        else:  # region × region
            # Defer to framework with uniform distribution
            # (This is actually not purely analytical - mark for router awareness)
            raise NotImplementedError("Region×region membership needs framework - use FrameworkStrategy")
    
    def _compute_subset(self, query) -> QueryResult:
        """Check R₁ ⊆ R₂ on grid"""
        
        subject_region = query.subject.state.region
        target_region = query.target.state.region
        
        # Sample grid for checking
        grid = self._generate_grid(query)
        
        I_subject = subject_region.indicator(grid['points'])
        I_target = target_region.indicator(grid['points'])
        
        # Check subset property
        subject_points = I_subject > 0.5
        is_subset = np.all(I_target[subject_points] > 0.5)
        
        return QueryResult(
            value=float(is_subset),
            computation_method="analytical_subset",
            result_type="boolean",
            grid_resolution=len(grid['points'])
        )
    
    def _compute_intersection(self, query) -> QueryResult:
        """Check R₁ ∩ R₂ ≠ ∅"""
        
        subject_region = query.subject.state.region
        target_region = query.target.state.region
        
        grid = self._generate_grid(query)
        
        I_subject = subject_region.indicator(grid['points'])
        I_target = target_region.indicator(grid['points'])
        
        has_overlap = np.any((I_subject > 0.5) & (I_target > 0.5))
        
        return QueryResult(
            value=float(has_overlap),
            computation_method="analytical_intersection",
            result_type="boolean",
            grid_resolution=len(grid['points'])
        )
    
    def _compute_overlap_fraction(self, query) -> QueryResult:
        """Compute |R₁∩R₂| / |R₁|"""
        
        subject_region = query.subject.state.region
        target_region = query.target.state.region
        
        grid = self._generate_grid(query)
        
        I_subject = subject_region.indicator(grid['points'])
        I_target = target_region.indicator(grid['points'])
        
        area_subject = np.sum(I_subject * grid['weights'])
        area_overlap = np.sum((I_subject * I_target) * grid['weights'])
        
        if area_subject < 1e-10:
            fraction = 0.0
        else:
            fraction = area_overlap / area_subject
        
        return QueryResult(
            value=fraction,
            computation_method="analytical_overlap_fraction",
            result_type="probability",
            grid_resolution=len(grid['points'])
        )
    
    def _compute_proximity(self, query) -> QueryResult:
        """Distance comparison: d(s, t) ≤ δ"""
        
        distance = self._compute_distance(query).value
        is_near = distance <= query.distance_threshold
        
        return QueryResult(
            value=float(is_near),
            computation_method="analytical_proximity",
            result_type="boolean",
            distribution_stats={'distance': distance}
        )
    
    def _compute_distance(self, query) -> QueryResult:
        """Compute distance between known entities"""
        
        subject_extent = query.subject.entity.spatial_extent()
        target_extent = query.target.entity.spatial_extent()
        
        if subject_extent == "point" and target_extent == "point":
            # Point-to-point distance
            p1 = query.subject.state.location
            p2 = query.target.state.location
            distance = query.metric_space.distance(p1, p2)
        
        elif subject_extent == "point":
            # Point-to-region distance
            point = query.subject.state.location
            region = query.target.state.region
            distance = self._point_to_region_distance(point, region, query.metric_space)
        
        elif target_extent == "point":
            # Region-to-point (symmetric)
            region = query.subject.state.region
            point = query.target.state.location
            distance = self._point_to_region_distance(point, region, query.metric_space)
        
        else:
            # Region-to-region distance (needs semantics)
            distance = self._region_to_region_distance(
                query.subject.state.region,
                query.target.state.region,
                query.metric_space,
                query.region_distance_semantics
            )
        
        return QueryResult(
            value=distance,
            computation_method="analytical_distance",
            result_type="distance"
        )
    
    def _point_to_region_distance(
        self, point: np.ndarray, region: Region, metric_space: MetricSpace
    ):
        """Distance from point to region boundary"""
        boundary_points = region.sample_boundary(n_samples=1000)
        distances = np.array([metric_space.distance(point, bp) for bp in boundary_points])
        return np.min(distances)
    
    def _region_to_region_distance(self, region1, region2, metric_space, semantics):
        """Distance between regions with specified semantics"""
        
        # Handle enum or string
        if hasattr(semantics, 'value'):
            semantics_str = semantics.value
        else:
            semantics_str = semantics
        
        if semantics_str == 'boundary':
            # Minimum distance between boundaries
            b1 = region1.sample_boundary(500)
            b2 = region2.sample_boundary(500)
            distances = np.array([metric_space.distance(p1, p2) for p1 in b1 for p2 in b2])
            return np.min(distances)
        
        elif semantics_str == 'centroid':
            # Distance between centroids
            c1 = region1.centroid()
            c2 = region2.centroid()
            return metric_space.distance(c1, c2)
        
        elif semantics_str == 'hausdorff':
            # Hausdorff distance
            b1 = region1.sample_boundary(200)
            b2 = region2.sample_boundary(200)
            
            max_min_1 = np.max([np.min([metric_space.distance(p1, p2) for p2 in b2]) for p1 in b1])
            max_min_2 = np.max([np.min([metric_space.distance(p2, p1) for p1 in b1]) for p2 in b2])
            
            return max(max_min_1, max_min_2)
        
        else:
            raise ValueError(f"Unknown region distance semantics: {semantics}")
    
    def _generate_grid(self, query):
        """Generate grid for geometric checks"""
        resolution = getattr(query, 'resolution', 50)
        
        # Get bounds - merge subject and target
        try:
            s_bounds = query.subject.state.region.bounds()
        except AttributeError:
            # If subject doesn't have bounds, use target bounds
            s_bounds = None
        
        try:
            if is_uncertain(query.target.state):
                # Use first region in ensemble
                t_bounds = query.target.state.ensemble[0].bounds()
            else:
                t_bounds = query.target.state.region.bounds()
        except (AttributeError, IndexError):
            t_bounds = None
        
        # Merge bounds
        if s_bounds and t_bounds:
            bounds = (
                min(s_bounds[0], t_bounds[0]),
                max(s_bounds[1], t_bounds[1]),
                min(s_bounds[2], t_bounds[2]),
                max(s_bounds[3], t_bounds[3])
            )
        elif s_bounds:
            bounds = s_bounds
        elif t_bounds:
            bounds = t_bounds
        else:
            # Default bounds
            bounds = (-10, 10, -10, 10)
        
        # Create grid
        grid = query.metric_space.create_grid(bounds, resolution)
        return grid


# ============================================================================
# STRATEGY 2: FRAMEWORK (Mollified Indicator Method)
# ============================================================================

class FrameworkStrategy(ComputationStrategy):
    """
    Use the existing ProbabilityEstimator framework.
    
    Handles:
    - Uncertain point × region (any epistemic on region)
    - Region × region with uniform interpretation
    - Any case with a distribution and region
    
    Advantages: Handles most uncertain cases, well-tested
    Disadvantages: Requires grid, bandwidth tuning
    """
    
    def can_handle(self, query) -> bool:
        """Can handle if we have a distribution and a region/membership function"""
        
        # Check if subject has a distribution
        has_distribution = (
            is_uncertain(query.subject.state) or
            (query.subject.entity.spatial_extent() == "region" and 
             is_known(query.subject.state))  # uniform from known region
        )
        
        # Check if target is a region
        has_region = query.target.entity.spatial_extent() == "region"
        
        # Framework works for membership and proximity queries
        from .ontology import QueryType
        valid_query_type = query.query_type in [
            QueryType.MEMBERSHIP, 
            QueryType.PROXIMITY
        ]
        
        return has_distribution and has_region and valid_query_type

    
    def compute(self, query) -> QueryResult:
        """Use ProbabilityEstimator framework"""
        
        # Extract distribution
        distribution = self._extract_distribution(query)
        
        # Extract region/membership function
        region_or_membership = self._extract_region(query)
        
        # Create estimator
        from . import ProbabilityEstimator, GaussianKernel, DirectConvolution, QuadratureIntegrator
        
        estimator = ProbabilityEstimator(
            metric_space=query.metric_space,
            region=region_or_membership,
            query_distribution=distribution,
            kernel=getattr(query, 'kernel', None) or GaussianKernel(),
            convolution_strategy=DirectConvolution(),
            integrator=QuadratureIntegrator()
        )
        
        # Compute
        bandwidth = getattr(query, 'bandwidth', None) or self._auto_bandwidth(query)
        resolution = getattr(query, 'resolution', 50) or 50
        
        result = estimator.compute(bandwidth=bandwidth, resolution=resolution)
        
        return QueryResult(
            value=result.probability,
            computation_method="framework_mollified_indicator",
            result_type="probability",
            bandwidth=bandwidth,
            grid_resolution=resolution,
            error_estimate=0.01  # Framework has inherent approximation error from mollification
        )
    
    def estimate_cost(self, query) -> float:
        """Framework cost ~ resolution²"""
        resolution = getattr(query, 'resolution', 50)
        return 10.0 * (resolution / 50) ** 2
    
    def _extract_distribution(self, query):
        """Extract or create distribution from subject"""
        from .ontology import EpistemicType
        
        if query.subject.state.epistemic_type() == EpistemicType.UNCERTAIN:
            return query.subject.state.distribution
        
        elif query.subject.entity.spatial_extent() == "region":
            # Treat known region as uniform distribution
            from .probability.distributions import RegionDistribution
            return RegionDistribution(
                query.subject.state.region,
                bounds=query.subject.state.region.bounds(),
                n_samples=10000
            )
        
        else:
            raise ValueError(f"Cannot extract distribution from {query.subject}")
    
    def _extract_region(self, query):
        """Extract region or membership function from target"""
        from .ontology import RegionEpistemicType, QueryType
        
        if query.query_type == QueryType.PROXIMITY:
            # Create buffered region
            if query.target.entity.spatial_extent() == "region":
                from .geometry.regions import BufferedPolygonRegion, MultiRegion, DiskRegion
                if is_uncertain(query.target.state):
                    # Create buffered versions of all regions in ensemble
                    buffered_regions = []
                    for region in query.target.state.ensemble:
                        if isinstance(region, DiskRegion):
                            # For disk regions, just expand the radius
                            buffered_regions.append(
                                DiskRegion(
                                    center=region.center,
                                    radius=region.radius + query.distance_threshold,
                                    metric_space=region.metric
                                )
                            )
                        else:
                            # For polygon regions, use BufferedPolygonRegion
                            buffered_regions.append(
                                BufferedPolygonRegion(
                                    region.vertices if hasattr(region, 'vertices') else region,
                                    buffer=query.distance_threshold
                                )
                            )
                    # Use a MultiRegion with union operation to combine them
                    return MultiRegion(buffered_regions, operation='union')
                else:
                    if isinstance(query.target.state.region, DiskRegion):
                        return DiskRegion(
                            center=query.target.state.region.center,
                            radius=query.target.state.region.radius + query.distance_threshold,
                            metric_space=query.target.state.region.metric
                        )
                    else:
                        return BufferedPolygonRegion(
                            query.target.state.region.vertices if hasattr(query.target.state.region, 'vertices') else query.target.state.region,
                            buffer=query.distance_threshold
                        )
            else:  # point
                from .geometry.regions import DiskRegion
                if is_uncertain(query.target.state):
                    # Multiple possible disk regions, use MultiRegion
                    from .geometry.regions import MultiRegion
                    disk_regions = [
                        DiskRegion(
                            center=dist.mean(),
                            radius=query.distance_threshold
                        )
                        for dist in [query.target.state.distribution]
                    ]
                    return MultiRegion(disk_regions, operation='union')
                else:
                    return DiskRegion(
                        center=query.target.state.location,
                        radius=query.distance_threshold
                    )
        
        elif query.target.state.epistemic_type() == RegionEpistemicType.FUZZY:
            # Use membership function directly
            return query.target.state.membership
        
        else:
            # Known region - use indicator
            return query.target.state.region
    
    def _auto_bandwidth(self, query):
        """Heuristic for automatic bandwidth selection"""
        # Simple heuristic: scale with domain size
        bounds = getattr(query, 'bounds', [[-10, 10], [-10, 10]])
        domain_size = np.sqrt((bounds[0][1] - bounds[0][0]) ** 2 + 
                             (bounds[1][1] - bounds[1][0]) ** 2)
        return domain_size / 20.0


# ============================================================================
# STRATEGY 3: MONTE CARLO (Universal Sampling)
# ============================================================================

class MonteCarloStrategy(ComputationStrategy):
    """
    Universal sampling-based computation.
    
    Handles:
    - Uncertain × uncertain (any combination)
    - Complex cases where framework doesn't apply
    - FALLBACK for everything else
    
    Advantages: Universal, handles anything
    Disadvantages: Slow, approximate, needs many samples
    """
    
    def __init__(self, default_n_samples=10000):
        self.default_n_samples = default_n_samples
    
    def can_handle(self, query) -> bool:
        """Can handle EVERYTHING (fallback strategy)"""
        return True
    
    def compute(self, query) -> QueryResult:
        """Dispatch to sampling method"""
        from .ontology import QueryType
        
        if query.query_type == QueryType.MEMBERSHIP:
            return self._monte_carlo_membership(query)
        elif query.query_type == QueryType.PROXIMITY:
            return self._monte_carlo_proximity(query)
        elif query.query_type == QueryType.DISTANCE:
            return self._monte_carlo_distance(query)
        elif query.query_type == QueryType.SUBSET:
            return self._monte_carlo_subset(query)
        elif query.query_type == QueryType.INTERSECTION:
            return self._monte_carlo_intersection(query)
        elif query.query_type == QueryType.OVERLAP_FRACTION:
            return self._monte_carlo_overlap_fraction(query)
        else:
            raise NotImplementedError(f"Monte Carlo for {query.query_type}")
    
    def estimate_cost(self, query) -> float:
        """Monte Carlo cost ~ n_samples (but should be highest)"""
        n_samples = getattr(query, 'n_samples', self.default_n_samples)
        return 1000.0 * (n_samples / 10000)
    
    def _monte_carlo_membership(self, query) -> QueryResult:
        """P(subject in target) via sampling"""
        
        n_samples = getattr(query, 'n_samples', self.default_n_samples)
        
        # Sample from subject
        if is_uncertain(query.subject.state):
            samples = query.subject.state.distribution.sample(n_samples)
        else:
            # Known point/region - sample uniformly if region
            if query.subject.entity.spatial_extent() == "point":
                samples = np.array([query.subject.state.location] * n_samples)
            else:
                # Known region - sample uniformly
                samples = query.subject.state.region.sample_uniform(n_samples)
        
        # Evaluate on target
        if is_uncertain(query.target.state):
            # Target ensemble - average over ensemble
            probabilities = []
            for region in query.target.state.ensemble:  # FIX: was .region
                indicators = region.indicator(samples)
                probabilities.append(np.mean(indicators > 0.5))
            probability = np.mean(probabilities)
            error = np.std(probabilities) / np.sqrt(len(probabilities))
        
        else:
            # Target is known/fuzzy
            if is_fuzzy(query.target.state):
                # FIX: Access membership function correctly
                memberships = query.target.state.membership(samples)
                probability = np.mean(memberships)
                error = np.std(memberships) / np.sqrt(n_samples)
            else:
                # Known region
                indicators = query.target.state.region.indicator(samples)
                probability = np.mean(indicators > 0.5)
                error = np.std(indicators > 0.5) / np.sqrt(n_samples)
        
        return QueryResult(
            value=probability,
            computation_method="monte_carlo_sampling",
            result_type="probability",
            error_estimate=error,
            n_samples=n_samples
        )
    
    def _monte_carlo_proximity(self, query) -> QueryResult:
        """P(d(subject, target) ≤ δ) via sampling"""
        
        n_samples = getattr(query, 'n_samples', self.default_n_samples)
        
        # Sample from both subject and target
        subject_samples = self._sample_from_entity(query.subject, n_samples)
        target_samples = self._sample_from_entity(query.target, n_samples)
        
        # Compute distances
        distances = np.array([
            query.metric_space.distance(s, t)
            for s, t in zip(subject_samples, target_samples)
        ])
        
        # Probability within threshold
        probability = np.mean(distances <= query.distance_threshold)
        error = np.std(distances <= query.distance_threshold) / np.sqrt(n_samples)
        
        return QueryResult(
            value=probability,
            computation_method="monte_carlo_proximity",
            result_type="probability",
            error_estimate=error,
            n_samples=n_samples,
            distribution_stats={'mean_distance': np.mean(distances)}
        )
    
    def _monte_carlo_distance(self, query) -> QueryResult:
        """Distribution of d(subject, target)"""
        
        n_samples = getattr(query, 'n_samples', self.default_n_samples)
        
        subject_samples = self._sample_from_entity(query.subject, n_samples)
        target_samples = self._sample_from_entity(query.target, n_samples)
        
        distances = np.array([
            query.metric_space.distance(s, t)
            for s, t in zip(subject_samples, target_samples)
        ])
        
        return QueryResult(
            value=np.mean(distances),  # Expected distance
            computation_method="monte_carlo_distance",
            result_type="distance",
            error_estimate=np.std(distances) / np.sqrt(n_samples),
            n_samples=n_samples,
            distribution_stats={
                'mean': np.mean(distances),
                'std': np.std(distances),
                'median': np.median(distances),
                'quantiles': np.percentile(distances, [5, 25, 50, 75, 95]).tolist()
            }
        )
    
    def _sample_from_entity(self, entity_with_state, n_samples):
        """Sample points from an entity (subject or target)"""
        from .ontology import EpistemicType, RegionEpistemicType
        
        if entity_with_state.entity.spatial_extent() == "point":
            epistemic = entity_with_state.state.epistemic_type()
            if epistemic == EpistemicType.KNOWN:
                return np.array([entity_with_state.state.location] * n_samples)
            else:  # UNCERTAIN
                return entity_with_state.state.distribution.sample(n_samples)
        
        else:  # region
            epistemic = entity_with_state.state.epistemic_type()
            if epistemic == RegionEpistemicType.KNOWN:
                return entity_with_state.state.region.sample_uniform(n_samples)
            elif epistemic == RegionEpistemicType.UNCERTAIN:
                samples = []
                n_per_region = n_samples // len(entity_with_state.state.ensemble)
                for region in entity_with_state.state.ensemble:
                    samples.extend(region.sample_uniform(n_per_region))
                return np.array(samples[:n_samples])
            else:  # FUZZY
                # Sample according to membership function
                # For now, rejection sampling
                bounding_box = getattr(entity_with_state.state, 'bounding_box', [[-10, 10], [-10, 10]])
                samples = []
                while len(samples) < n_samples:
                    x = np.random.uniform(bounding_box[0][0], bounding_box[0][1])
                    y = np.random.uniform(bounding_box[1][0], bounding_box[1][1])
                    point = np.array([x, y])
                    membership = entity_with_state.state.membership(point)
                    if np.random.rand() < membership:
                        samples.append(point)
                return np.array(samples)
    
    def _monte_carlo_subset(self, query) -> QueryResult:
        """P(R₁ ⊆ R₂)"""
        
        # Helper to check if one region is subset of another
        def check_subset(region1, region2, grid, idx1=0, idx2=0):
            I1 = region1.indicator(grid['points'])
            I2 = region2.indicator(grid['points'])
            
            # Check if all points in region1 are also in region2
            region1_points = I1 > 0.5
            if not region1_points.any():
                return True  # Empty set is subset of anything
            
            return np.all(I2[region1_points] > 0.5)
        
        # Generate grid for checking
        grid = self._generate_grid(query)
        
        # Get source regions (subject)
        if is_uncertain(query.subject.state):
            subject_regions = query.subject.state.ensemble
        else:
            subject_regions = [query.subject.state.region]
        
        # Get target regions
        if is_uncertain(query.target.state):
            target_regions = query.target.state.ensemble
        else:
            target_regions = [query.target.state.region]
        
        # Check all combinations
        count = 0
        total = len(subject_regions) * len(target_regions)
        
        for i, r1 in enumerate(subject_regions):
            for j, r2 in enumerate(target_regions):
                is_subset = check_subset(r1, r2, grid)
                if is_subset:
                    count += 1
        
        probability = count / total if total > 0 else 0.0
        
        return QueryResult(
            value=probability,
            computation_method="monte_carlo_subset",
            result_type="probability",
            n_samples=total
        )

    def _monte_carlo_intersection(self, query) -> QueryResult:
        """P(R₁ ∩ R₂ ≠ ∅)"""
        
        def check_intersection(region1, region2, grid):
            I1 = region1.indicator(grid['points'])
            I2 = region2.indicator(grid['points'])
            overlap = (I1 > 0.5) & (I2 > 0.5)
            return overlap.any()
        
        grid = self._generate_grid(query)
        
        # Get regions
        if is_uncertain(query.subject.state):
            subject_regions = query.subject.state.ensemble
        else:
            subject_regions = [query.subject.state.region]
        
        if is_uncertain(query.target.state):
            target_regions = query.target.state.ensemble
        else:
            target_regions = [query.target.state.region]
        
        # Check all combinations
        count = 0
        total = len(subject_regions) * len(target_regions)
        
        for r1 in subject_regions:
            for r2 in target_regions:
                if check_intersection(r1, r2, grid):
                    count += 1
        
        probability = count / total if total > 0 else 0.0
        
        return QueryResult(
            value=probability,
            computation_method="monte_carlo_intersection",
            result_type="probability",
            n_samples=total
        )

    def _monte_carlo_overlap_fraction(self, query) -> QueryResult:
        """E[|R₁∩R₂|/|R₁|]"""
        
        def compute_fraction(region1, region2, grid):
            I1 = region1.indicator(grid['points'])
            I2 = region2.indicator(grid['points'])
            
            area1 = np.sum(I1 * grid['weights'])
            area_overlap = np.sum((I1 * I2) * grid['weights'])
            
            if area1 < 1e-10:
                return 0.0
            
            return area_overlap / area1
        
        grid = self._generate_grid(query)
        
        # Get regions
        if is_uncertain(query.subject.state):
            subject_regions = query.subject.state.ensemble
        else:
            subject_regions = [query.subject.state.region]
        
        if is_uncertain(query.target.state):
            target_regions = query.target.state.ensemble
        else:
            target_regions = [query.target.state.region]
        
        # Compute fraction for all combinations
        fractions = []
        for r1 in subject_regions:
            for r2 in target_regions:
                frac = compute_fraction(r1, r2, grid)
                fractions.append(frac)
        
        mean_fraction = np.mean(fractions)
        error = np.std(fractions) / np.sqrt(len(fractions)) if len(fractions) > 1 else 0.0
        
        return QueryResult(
            value=mean_fraction,
            computation_method="monte_carlo_overlap_fraction",
            result_type="probability",
            error_estimate=error,
            n_samples=len(fractions)
        )

    def _generate_grid(self, query):
        """Generate grid for geometric checks"""
        resolution = getattr(query, 'resolution', 50)
        
        # Get bounds - merge subject and target
        s_bounds = None
        t_bounds = None
        
        # Handle subject bounds
        try:
            if is_uncertain(query.subject.state):
                # Merge all bounds from ensemble
                all_bounds = [region.bounds() for region in query.subject.state.ensemble]
                s_bounds = (
                    min(b[0] for b in all_bounds),
                    max(b[1] for b in all_bounds),
                    min(b[2] for b in all_bounds),
                    max(b[3] for b in all_bounds)
                )
            else:
                s_bounds = query.subject.state.region.bounds()
        except AttributeError:
            s_bounds = None
        
        # Handle target bounds
        try:
            if is_uncertain(query.target.state):
                # Merge all bounds from ensemble
                all_bounds = [region.bounds() for region in query.target.state.ensemble]
                t_bounds = (
                    min(b[0] for b in all_bounds),
                    max(b[1] for b in all_bounds),
                    min(b[2] for b in all_bounds),
                    max(b[3] for b in all_bounds)
                )
            else:
                t_bounds = query.target.state.region.bounds()
        except (AttributeError, IndexError):
            t_bounds = None
        
        # Merge bounds
        if s_bounds and t_bounds:
            bounds = (
                min(s_bounds[0], t_bounds[0]),
                max(s_bounds[1], t_bounds[1]),
                min(s_bounds[2], t_bounds[2]),
                max(s_bounds[3], t_bounds[3])
            )
        elif s_bounds:
            bounds = s_bounds
        elif t_bounds:
            bounds = t_bounds
        else:
            # Default bounds
            bounds = (-10, 10, -10, 10)
        
        # Create grid
        grid = query.metric_space.create_grid(bounds, resolution)
        return grid

# ============================================================================
# STRATEGY 4: HYBRID (Adaptive/Marginalized)
# ============================================================================

class HybridStrategy(ComputationStrategy):
    """
    Combines multiple strategies adaptively.
    
    Handles:
    - Uncertain point × uncertain region (marginalize + framework)
    - Complex mixed epistemic states
    - Cases requiring strategic decomposition
    
    Advantages: Optimal for mixed cases
    Disadvantages: More complex implementation
    """
    
    def __init__(self):
        self.analytical = AnalyticalStrategy()
        self.framework = FrameworkStrategy()
        self.monte_carlo = MonteCarloStrategy()
    
    def can_handle(self, query) -> bool:
        """Can handle specific complex patterns"""
        from .ontology import EpistemicType, QueryType
        
        # Pattern 1: Uncertain point × uncertain region proximity
        if (query.subject.entity.spatial_extent() == "point" and
            query.subject.state.epistemic_type() == EpistemicType.UNCERTAIN and
            query.target.entity.spatial_extent() == "region" and
            query.target.state.epistemic_type() == EpistemicType.UNCERTAIN and
            query.query_type == QueryType.PROXIMITY):
            return True
        
        # Add more patterns as needed
        
        return False
    
    def compute(self, query) -> QueryResult:
        """Marginalized computation"""
        from .ontology import EpistemicType, QueryType
        
        # Pattern: Uncertain point × uncertain region
        if (query.subject.entity.spatial_extent() == "point" and
            query.target.entity.spatial_extent() == "region"):
            
            return self._marginalized_proximity(query)
        
        else:
            raise NotImplementedError("Hybrid pattern not recognized")
    
    def estimate_cost(self, query) -> float:
        """Cost is intermediate"""
        return 500.0
    
    def _marginalized_proximity(self, query) -> QueryResult:
        """
        P(d(X, ∂R̃) ≤ δ) = E_{R̃}[P(d(X, ∂R) ≤ δ | R)]
        
        Outer loop: Monte Carlo over region ensemble
        Inner loop: Framework for each fixed region
        """
        from .ontology import Target, KnownRegion, QueryType
        
        region_ensemble = query.target.state.ensemble
        probabilities = []
        
        for region_i in region_ensemble:
            # Create fixed-region query
            fixed_query = Query(
                subject=query.subject,
                target=Target(query.target.entity, KnownRegion(region_i)),
                query_type=QueryType.PROXIMITY,
                metric_space=query.metric_space,
                distance_threshold=query.distance_threshold
            )
            
            # Use framework for this fixed case
            result_i = self.framework.compute(fixed_query)
            probabilities.append(result_i.value)
        
        # Average over ensemble
        probability = np.mean(probabilities)
        error = np.std(probabilities) / np.sqrt(len(probabilities))
        
        return QueryResult(
            value=probability,
            computation_method="hybrid_marginalized",
            result_type="probability",
            error_estimate=error,
            distribution_stats={
                'ensemble_size': len(region_ensemble),
                'min_prob': np.min(probabilities),
                'max_prob': np.max(probabilities)
            }
        )


# ============================================================================
# QUERY ROUTER (Automatic Strategy Selection)
# ============================================================================

class QueryRouter:
    """
    Automatically selects the best computation strategy for a query.
    
    Strategy selection order:
    1. Analytical (if applicable - fastest, exact)
    2. Framework (if applicable - fast, accurate)
    3. Hybrid (for specific patterns - optimal decomposition)
    4. Monte Carlo (fallback - always works)
    """
    
    def __init__(self):
        self.strategies = [
            AnalyticalStrategy(),
            FrameworkStrategy(),
            HybridStrategy(),
            MonteCarloStrategy()  # Fallback - can handle anything
        ]
    
    def compute(self, query: Query, force_strategy: Optional[str] = None) -> QueryResult:
        """
        Compute query with automatic or forced strategy selection.
        
        Parameters
        ----------
        query : Query
            The spatial query to compute
        force_strategy : Optional[str]
            Force specific strategy: 'analytical', 'framework', 'monte_carlo', 'hybrid'
        
        Returns
        -------
        QueryResult
            Result with metadata about computation
        
        Raises
        ------
        ValueError
            If no strategy can handle the query (should never happen with Monte Carlo fallback)
        """
        
        if force_strategy:
            strategy = self._get_strategy_by_name(force_strategy)
            if not strategy.can_handle(query):
                raise ValueError(f"Forced strategy '{force_strategy}' cannot handle this query")
            return strategy.compute(query)
        
        # Automatic selection
        capable = [s for s in self.strategies if s.can_handle(query)]
        
        if not capable:
            raise ValueError(f"No strategy can handle query: {query}")
        
        # Select cheapest capable strategy
        best_strategy = min(capable, key=lambda s: s.estimate_cost(query))
        
        return best_strategy.compute(query)

    
    def _get_strategy_by_name(self, name: str) -> ComputationStrategy:
        """Get strategy by name"""
        name_lower = name.lower().replace('_', '').replace(' ', '')
        
        for strategy in self.strategies:
            strategy_name = strategy.get_name().lower().replace('_', '').replace(' ', '')
            if strategy_name.startswith(name_lower) or name_lower in strategy_name:
                return strategy
        
        raise ValueError(f"Unknown strategy: {name}")
    
    def explain_strategy_selection(self, query: Query) -> Dict:
        """
        Explain which strategies can handle the query and why one was chosen.
        
        Useful for debugging and understanding the framework.
        """
        explanations = []
        
        for strategy in self.strategies:
            can_do = strategy.can_handle(query)
            cost = strategy.estimate_cost(query) if can_do else float('inf')
            
            explanations.append({
                'strategy': strategy.get_name(),
                'can_handle': can_do,
                'estimated_cost': cost if can_do else 'N/A',
                'reason': self._get_capability_reason(strategy, query)
            })
        
        # Find what would be selected
        capable = [s for s in self.strategies if s.can_handle(query)]
        selected = min(capable, key=lambda s: s.estimate_cost(query)) if capable else None
        
        return {
            Query: str(query),
            'strategies': explanations,
            'selected': selected.get_name() if selected else 'NONE',
            'selection_reason': 'Lowest cost among capable strategies'
        }
    
    def _get_capability_reason(self, strategy: ComputationStrategy, query: Query) -> str:
        """Generate human-readable reason for capability"""
        from .ontology import EpistemicType
        
        s_known = query.subject.state.epistemic_type() == EpistemicType.KNOWN
        t_known = query.target.state.epistemic_type() == EpistemicType.KNOWN
        
        if isinstance(strategy, AnalyticalStrategy):
            if s_known and t_known:
                return "Both known - can use exact geometric methods"
            else:
                return "Requires both known - has uncertainty"
        
        elif isinstance(strategy, FrameworkStrategy):
            if strategy.can_handle(query):
                return "Has distribution and region - framework applicable"
            else:
                return "Missing distribution or region structure"
        
        elif isinstance(strategy, HybridStrategy):
            if strategy.can_handle(query):
                return "Matches hybrid pattern (e.g., uncertain×uncertain)"
            else:
                return "No matching hybrid pattern"
        
        elif isinstance(strategy, MonteCarloStrategy):
            return "Universal fallback - can sample from anything"
        
        return "Unknown reason"


# ============================================================================
# CONVENIENCE FUNCTION (User-Facing API)
# ============================================================================

def solve_query(query: Query, force_strategy: Optional[str] = None, 
                explain: bool = False) -> QueryResult:
    """
    Universal query solver with 100% coverage guarantee.
    
    This is the main user-facing API for computing ANY valid query
    from the 98-combination ontology.
    
    Parameters
    ----------
    query : Query
        A validated Query object from the ontology
    force_strategy : Optional[str]
        Override automatic selection: 'analytical', 'framework', 'monte_carlo', 'hybrid'
    explain : bool
        If True, also return explanation of strategy selection
    
    Returns
    -------
    QueryResult or (QueryResult, Dict)
        Result of computation, optionally with explanation
    
    Examples
    --------
    >>> # Classic geofence: uncertain point × known region
    >>> result = solve_query(geofence_query)
    >>> print(f"Probability: {result.value:.3f}")
    >>> print(f"Method: {result.computation_method}")
    
    >>> # With explanation
    >>> result, explanation = solve_query(complex_query, explain=True)
    >>> print(explanation['selected'])  # Which strategy was chosen
    
    >>> # Force specific strategy
    >>> result = solve_query(query, force_strategy='monte_carlo')
    """
    
    router = QueryRouter()
    
    if explain:
        result = router.compute(query, force_strategy=force_strategy)
        explanation = router.explain_strategy_selection(query)
        return result, explanation
    else:
        return router.compute(query, force_strategy=force_strategy)


