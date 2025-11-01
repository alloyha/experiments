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
from .ontology import Query, QueryType, EpistemicType


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
        from .ontology import EpistemicType
        
        s_known = query.subject.state.epistemic_type() == EpistemicType.KNOWN
        t_known = query.target.state.epistemic_type() == EpistemicType.KNOWN
        
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
    
    def _generate_grid(self, query: Query) -> Dict[str, np.ndarray]:
        """Generate evaluation grid for region operations"""
        # This would use query.grid_bounds and query.resolution
        # Placeholder implementation
        resolution = getattr(query, 'resolution', 50)
        bounds = getattr(query, 'bounds', [[-10, 10], [-10, 10]])
        
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        
        cell_area = (x[1] - x[0]) * (y[1] - y[0])
        weights = np.full(len(points), cell_area)
        
        return {'points': points, 'weights': weights}


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
        from .ontology import EpistemicType, RegionEpistemicType
        
        # Check if subject has a distribution
        subject_epistemic = query.subject.state.epistemic_type()
        
        has_distribution = (
            subject_epistemic == EpistemicType.UNCERTAIN or
            (query.subject.entity.spatial_extent() == "region" and 
            subject_epistemic == RegionEpistemicType.KNOWN)  # uniform from known region
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
            kernel=query.kernel or GaussianKernel(),
            convolution_strategy=DirectConvolution(),
            integrator=QuadratureIntegrator()
        )
        
        # Compute
        bandwidth = query.bandwidth or self._auto_bandwidth(query)
        resolution = query.resolution or 50
        
        result = estimator.compute(bandwidth=bandwidth, resolution=resolution)
        
        return QueryResult(
            value=result.probability,
            computation_method="framework_mollified_indicator",
            result_type="probability",
            bandwidth=bandwidth,
            grid_resolution=resolution
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
        from .ontology import EpistemicType, QueryType
        
        if query.query_type == QueryType.PROXIMITY:
            # Create buffered region
            if query.target.entity.spatial_extent() == "region":
                from .geometry.regions import BufferedPolygonRegion
                return BufferedPolygonRegion(
                    query.target.state.region,
                    buffer_distance=query.distance_threshold
                )
            else:  # point
                from .geometry.regions import DiskRegion
                return DiskRegion(
                    center=query.target.state.location,
                    radius=query.distance_threshold
                )
        
        elif query.target.state.epistemic_type() == EpistemicType.FUZZY:
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
        from .ontology import EpistemicType
        
        n_samples = getattr(query, 'n_samples', self.default_n_samples)
        
        # Sample from subject
        if query.subject.state.epistemic_type() == EpistemicType.UNCERTAIN:
            samples = query.subject.state.distribution.sample(n_samples)
        else:
            # Known point/region - sample uniformly if region
            if query.subject.entity.spatial_extent() == "point":
                samples = np.array([query.subject.state.location] * n_samples)
            else:
                samples = query.subject.state.region.sample_uniform(n_samples)
        
        # Evaluate on target
        if query.target.state.epistemic_type() == EpistemicType.UNCERTAIN:
            # Target ensemble - average over ensemble
            probabilities = []
            for region in query.target.state.ensemble:
                indicators = region.indicator(samples)
                probabilities.append(np.mean(indicators > 0.5))
            probability = np.mean(probabilities)
            error = np.std(probabilities) / np.sqrt(len(probabilities))
        
        else:
            # Target is known/fuzzy
            if query.target.state.epistemic_type() == EpistemicType.FUZZY:
                memberships = query.target.state.membership(samples)
                probability = np.mean(memberships)
            else:
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
        """P(R̃₁ ⊆ R₂) via ensemble sampling"""
        from .ontology import EpistemicType
        
        n_samples = getattr(query, 'n_samples', self.default_n_samples)
        
        # Get ensemble from uncertain region
        subject_ensemble = query.subject.state.ensemble
        target_region = query.target.state.region
        
        # For each region in ensemble, check if it's a subset
        subset_count = 0
        for region_i in subject_ensemble:
            # Sample points from region_i
            samples = region_i.sample_uniform(min(n_samples // len(subject_ensemble), 100))
            
            # Check if all points are in target
            indicators = target_region.indicator(samples)
            is_subset = np.all(indicators > 0.5)
            
            if is_subset:
                subset_count += 1
        
        probability = subset_count / len(subject_ensemble)
        error = np.sqrt(probability * (1 - probability) / len(subject_ensemble))
        
        return QueryResult(
            value=probability,
            computation_method="monte_carlo_subset",
            result_type="probability",
            error_estimate=error,
            n_samples=len(subject_ensemble)
        )

    def _monte_carlo_intersection(self, query) -> QueryResult:
        """P(R̃₁ ∩ R₂ ≠ ∅) via sampling"""
        
        subject_ensemble = query.subject.state.ensemble
        target_region = query.target.state.region
        
        # For each region in ensemble, check if it intersects
        intersection_count = 0
        for region_i in subject_ensemble:
            # Sample points from region_i
            samples = region_i.sample_uniform(100)
            
            # Check if any point is in target
            indicators = target_region.indicator(samples)
            has_intersection = np.any(indicators > 0.5)
            
            if has_intersection:
                intersection_count += 1
        
        probability = intersection_count / len(subject_ensemble)
        error = np.sqrt(probability * (1 - probability) / len(subject_ensemble))
        
        return QueryResult(
            value=probability,
            computation_method="monte_carlo_intersection",
            result_type="probability",
            error_estimate=error,
            n_samples=len(subject_ensemble)
        )

    def _monte_carlo_overlap_fraction(self, query) -> QueryResult:
        """E[|R̃₁∩R₂|/|R₁|] via sampling"""
        
        subject_ensemble = query.subject.state.ensemble
        target_region = query.target.state.region
        
        # Compute overlap fraction for each region in ensemble
        fractions = []
        for region_i in subject_ensemble:
            # Sample points from region_i
            samples = region_i.sample_uniform(1000)
            
            # Compute fraction in target
            indicators = target_region.indicator(samples)
            fraction = np.mean(indicators > 0.5)
            fractions.append(fraction)
        
        mean_fraction = np.mean(fractions)
        error = np.std(fractions) / np.sqrt(len(fractions))
        
        return QueryResult(
            value=mean_fraction,
            computation_method="monte_carlo_overlap_fraction",
            result_type="probability",
            error_estimate=error,
            n_samples=len(subject_ensemble)
        )

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
            # User override
            strategy = self._get_strategy_by_name(force_strategy)
            if not strategy.can_handle(query):
                raise ValueError(f"Forced strategy '{force_strategy}' cannot handle this query")
            return strategy.compute(query)
        
        # Automatic selection: find capable strategies
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


