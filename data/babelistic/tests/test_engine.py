"""
TDD Implementation Plan - Phase 1: Foundation (70% Coverage)

Test-Driven Development approach:
1. Write failing test
2. Implement minimal code to pass
3. Refactor
4. Repeat

Phase 1 targets: 18% → 70% coverage (~50 cases)
Focus: Straightforward + Moderate cases
Defer: Research cases (fuzzy semantics)

Estimated: 1,100 LOC implementation + 800 LOC tests = 1,900 LOC total
Timeline: 2-3 weeks for one developer
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================================
# PHASE 1.1: Region Infrastructure (Required for everything)
# ============================================================================

class TestRegionInfrastructure:
    """
    Test suite for basic region methods needed by all strategies.
    
    New methods to add:
    - Region.sample_uniform(n) → np.ndarray
    - Region.area() → float  
    - Region.centroid() → np.ndarray
    """
    
    def test_disk_region_sample_uniform(self):
        """Test uniform sampling from disk region"""
        from babelistic.geometry.regions import DiskRegion
        
        center = np.array([0.0, 0.0])
        radius = 1.0
        disk = DiskRegion(center, radius)
        
        # Sample points
        samples = disk.sample_uniform(n_samples=1000)
        
        # Assertions
        assert samples.shape == (1000, 2)
        
        # All points should be inside disk
        distances = np.linalg.norm(samples - center, axis=1)
        assert np.all(distances <= radius)
        
        # Should be reasonably uniform (not all at boundary)
        # Check that some points are in inner half
        inner_points = np.sum(distances < radius / 2)
        assert inner_points > 200  # Expect ~250 if uniform
        
    def test_polygon_region_sample_uniform(self):
        """Test uniform sampling from polygon"""
        from babelistic.geometry.regions import PolygonRegion
        
        # Unit square
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon = PolygonRegion(vertices)
        
        samples = polygon.sample_uniform(n_samples=1000)
        
        assert samples.shape == (1000, 2)
        
        # All points inside polygon
        indicators = polygon.indicator(samples)
        assert np.all(indicators > 0.5)
        
        # Reasonably uniform distribution
        mean_pos = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean_pos, [0.5, 0.5], atol=0.1)
    
    def test_region_area(self):
        """Test area computation"""
        from babelistic.geometry.regions import DiskRegion, PolygonRegion
        
        # Circle area = πr²
        disk = DiskRegion(np.array([0, 0]), radius=1.0)
        area = disk.area()
        np.testing.assert_allclose(area, np.pi, rtol=0.05)
        
        # Unit square area = 1
        square = PolygonRegion(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        area = square.area()
        np.testing.assert_allclose(area, 1.0, rtol=0.05)
    
    def test_region_centroid(self):
        """Test centroid computation"""
        from babelistic.geometry.regions import DiskRegion, PolygonRegion
        
        # Disk centroid = center
        disk = DiskRegion(np.array([3, 4]), radius=2.0)
        centroid = disk.centroid()
        np.testing.assert_allclose(centroid, [3, 4], atol=0.1)
        
        # Square centroid = (0.5, 0.5)
        square = PolygonRegion(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        centroid = square.centroid()
        np.testing.assert_allclose(centroid, [0.5, 0.5], atol=0.1)


# ============================================================================
# PHASE 1.2: AnalyticalStrategy - Geometric Operations
# ============================================================================

class TestAnalyticalStrategy:
    """Test suite for analytical (exact) computations on known entities"""
    
    def test_subset_known_regions(self):
        """Test: Is R₁ ⊆ R₂?"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import AnalyticalStrategy, Query
        from babelistic.ontology import QueryType
        
        # Small disk inside large disk
        inner = DiskRegion(np.array([0, 0]), radius=1.0)
        outer = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query(inner, outer, QueryType.SUBSET)
        strategy = AnalyticalStrategy()
        
        result = strategy.compute(query)
        
        assert result.value == 1.0  # True (inner ⊆ outer)
        assert result.result_type == "boolean"
        
        # Test reverse (outer ⊄ inner)
        query_reverse = self._make_query(outer, inner, QueryType.SUBSET)
        result = strategy.compute(query_reverse)
        assert result.value == 0.0  # False
    
    def test_intersection_known_regions(self):
        """Test: R₁ ∩ R₂ ≠ ∅?"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import AnalyticalStrategy
        from babelistic.ontology import QueryType
        
        # Overlapping disks
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([1, 0]), radius=1.0)
        
        query = self._make_query(disk1, disk2, QueryType.INTERSECTION)
        strategy = AnalyticalStrategy()
        
        result = strategy.compute(query)
        assert result.value == 1.0  # True (they overlap)
        
        # Non-overlapping disks
        disk3 = DiskRegion(np.array([5, 0]), radius=1.0)
        query_disjoint = self._make_query(disk1, disk3, QueryType.INTERSECTION)
        result = strategy.compute(query_disjoint)
        assert result.value == 0.0  # False (disjoint)
    
    def test_overlap_fraction(self):
        """Test: |R₁∩R₂| / |R₁|"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import AnalyticalStrategy
        from babelistic.ontology import QueryType
        
        # Two identical circles → 100% overlap
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([0, 0]), radius=1.0)
        
        query = self._make_query(disk1, disk2, QueryType.OVERLAP_FRACTION)
        strategy = AnalyticalStrategy()
        
        result = strategy.compute(query)
        np.testing.assert_allclose(result.value, 1.0, rtol=0.05)
        
        # Circle entirely inside larger circle → 100%
        inner = DiskRegion(np.array([0, 0]), radius=1.0)
        outer = DiskRegion(np.array([0, 0]), radius=2.0)
        query = self._make_query(inner, outer, QueryType.OVERLAP_FRACTION)
        result = strategy.compute(query)
        np.testing.assert_allclose(result.value, 1.0, rtol=0.05)
        
        # Partially overlapping → ~50%
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([1, 0]), radius=1.0)
        query = self._make_query(disk1, disk2, QueryType.OVERLAP_FRACTION)
        result = strategy.compute(query)
        assert 0.3 < result.value < 0.7  # Rough overlap
    
    def test_region_distance_boundary_semantics(self):
        """Test: d(R₁, R₂) with boundary semantics"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import AnalyticalStrategy
        from babelistic.ontology import QueryType, RegionDistanceSemantics
        
        # Two circles, gap of 1.0 between boundaries
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([3, 0]), radius=1.0)  # centers 3 apart, radii 1 each
        
        query = self._make_query(disk1, disk2, QueryType.DISTANCE)
        query.region_distance_semantics = RegionDistanceSemantics.BOUNDARY_TO_BOUNDARY
        
        strategy = AnalyticalStrategy()
        result = strategy.compute(query)
        
        # Distance = 3 - 1 - 1 = 1.0
        np.testing.assert_allclose(result.value, 1.0, rtol=0.1)
    
    def test_region_distance_centroid_semantics(self):
        """Test: d(R₁, R₂) with centroid semantics"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import AnalyticalStrategy
        from babelistic.ontology import QueryType, RegionDistanceSemantics
        
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([3, 4]), radius=1.0)
        
        query = self._make_query(disk1, disk2, QueryType.DISTANCE)
        query.region_distance_semantics = RegionDistanceSemantics.CENTROID_TO_CENTROID
        
        strategy = AnalyticalStrategy()
        result = strategy.compute(query)
        
        # Distance = sqrt(3² + 4²) = 5.0
        np.testing.assert_allclose(result.value, 5.0, rtol=0.01)
    
    def _make_query(self, subject_region, target_region, query_type):
        """Helper to create test queries"""
        from babelistic.ontology import Query, Subject, Target, KnownRegion, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(RegionEntity("test"), KnownRegion(subject_region)),
            target=Target(RegionEntity("test"), KnownRegion(target_region)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )


# ============================================================================
# PHASE 1.3: MonteCarloStrategy - Uncertain Entities
# ============================================================================

class TestMonteCarloStrategy:
    """Test suite for Monte Carlo sampling methods"""
    
    def test_uncertain_point_known_region_membership(self):
        """Test: P(X ∈ R) via Monte Carlo (should match framework)"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.probability.distributions import GaussianDistribution
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        # Gaussian centered at origin, unit variance
        dist = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2))
        region = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query_uncertain_point(dist, region, QueryType.MEMBERSHIP)
        query.n_samples = 10000
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # Should be high probability (within 2σ)
        assert 0.8 < result.value < 1.0
        assert result.error_estimate is not None
        assert result.error_estimate < 0.05  # Reasonable error
    
    def test_uncertain_uncertain_point_proximity(self):
        """Test: P(d(X, Y) ≤ δ) for two uncertain points"""
        from babelistic.probability.distributions import GaussianDistribution
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        # Two Gaussians close together
        dist1 = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.1)
        dist2 = GaussianDistribution(mean=np.array([1, 0]), cov=np.eye(2) * 0.1)
        
        query = self._make_query_two_uncertain_points(dist1, dist2, QueryType.PROXIMITY)
        query.distance_threshold = 2.0
        query.n_samples = 5000
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # Should be high probability (means are 1.0 apart, threshold 2.0)
        assert 0.7 < result.value < 1.0
        assert result.computation_method == "monte_carlo_proximity"
    
    def test_known_point_uncertain_region_membership(self):
        """Test: P(p ∈ R̃) over region ensemble"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        point = np.array([0, 0])
        
        # Ensemble of regions: some contain point, some don't
        ensemble = [
            DiskRegion(np.array([0, 0]), radius=2.0),  # Contains
            DiskRegion(np.array([0, 0]), radius=1.5),  # Contains
            DiskRegion(np.array([5, 0]), radius=1.0),  # Doesn't contain
            DiskRegion(np.array([0, 0]), radius=0.5),  # Doesn't contain
        ]
        
        query = self._make_query_point_uncertain_region(point, ensemble, QueryType.MEMBERSHIP)
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # 2 out of 4 regions contain the point
        np.testing.assert_allclose(result.value, 0.5, rtol=0.01)
    
    def test_uncertain_region_subset(self):
        """Test: P(R̃₁ ⊆ R₂)"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        # Target: large circle
        target = DiskRegion(np.array([0, 0]), radius=5.0)
        
        # Ensemble: some circles inside, some outside
        ensemble = [
            DiskRegion(np.array([0, 0]), radius=2.0),  # Inside
            DiskRegion(np.array([1, 0]), radius=2.0),  # Inside
            DiskRegion(np.array([0, 0]), radius=6.0),  # Outside (too big)
            DiskRegion(np.array([10, 0]), radius=1.0), # Outside (far away)
        ]
        
        query = self._make_query_uncertain_region_known(ensemble, target, QueryType.SUBSET)
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # 2 out of 4 are subsets
        np.testing.assert_allclose(result.value, 0.5, rtol=0.01)
    
    def test_uncertain_region_intersection(self):
        """Test: P(R̃₁ ∩ R₂ ≠ ∅)"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        target = DiskRegion(np.array([0, 0]), radius=2.0)
        
        # Ensemble: some overlap, some don't
        ensemble = [
            DiskRegion(np.array([0, 0]), radius=1.0),  # Overlaps
            DiskRegion(np.array([1, 0]), radius=1.0),  # Overlaps
            DiskRegion(np.array([10, 0]), radius=1.0), # No overlap
            DiskRegion(np.array([5, 5]), radius=1.0),  # No overlap
        ]
        
        query = self._make_query_uncertain_region_known(ensemble, target, QueryType.INTERSECTION)
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # 2 out of 4 intersect
        np.testing.assert_allclose(result.value, 0.5, rtol=0.01)
    
    def test_uncertain_uncertain_region_overlap_fraction(self):
        """Test: E[|R̃₁∩R̃₂|/|R̃₁|]"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        # Two ensembles of circles
        ensemble1 = [
            DiskRegion(np.array([0, 0]), radius=1.0),
            DiskRegion(np.array([0.5, 0]), radius=1.0),
        ]
        
        ensemble2 = [
            DiskRegion(np.array([0, 0]), radius=1.0),
            DiskRegion(np.array([1, 0]), radius=1.0),
        ]
        
        query = self._make_query_two_uncertain_regions(ensemble1, ensemble2, 
                                                        QueryType.OVERLAP_FRACTION)
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # Should be some positive overlap fraction
        assert 0.0 < result.value < 1.0
        assert result.result_type == "probability"
    
    def test_distance_distribution(self):
        """Test: Distribution of d(X, Y) for uncertain points"""
        from babelistic.probability.distributions import GaussianDistribution
        from babelistic.engine import MonteCarloStrategy
        from babelistic.ontology import QueryType
        
        # Two Gaussians
        dist1 = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2))
        dist2 = GaussianDistribution(mean=np.array([3, 4]), cov=np.eye(2))
        
        query = self._make_query_two_uncertain_points(dist1, dist2, QueryType.DISTANCE)
        query.n_samples = 5000
        
        strategy = MonteCarloStrategy()
        result = strategy.compute(query)
        
        # Mean distance should be around 5.0 (3² + 4²)
        assert 4.0 < result.value < 6.0
        
        # Should have distribution stats
        assert result.distribution_stats is not None
        assert 'mean' in result.distribution_stats
        assert 'std' in result.distribution_stats
        assert 'quantiles' in result.distribution_stats
    
    # Helper methods
    def _make_query_uncertain_point(self, dist, region, query_type):
        """Helper for uncertain point × known region queries"""
        from babelistic.ontology import Query, Subject, Target, UncertainPoint, KnownRegion
        from babelistic.ontology import PointEntity, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(PointEntity(), UncertainPoint(dist)),
            target=Target(RegionEntity("test"), KnownRegion(region)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )
    
    def _make_query_two_uncertain_points(self, dist1, dist2, query_type):
        """Helper for uncertain point × uncertain point queries"""
        from babelistic.ontology import Query, QueryType, Subject, Target, UncertainPoint, PointEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        query = Query(
            subject=Subject(PointEntity(), UncertainPoint(dist1)),
            target=Target(PointEntity(), UncertainPoint(dist2)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )
        
        # Always set these for tests
        if query_type == QueryType.PROXIMITY:
            query.distance_threshold = 2.0  # Add this
        
        query.bounds = [[-10, 10], [-10, 10]]  # Add this
        query.resolution = 50
        query.n_samples = 1000
        
        return query
        
    def _make_query_point_uncertain_region(self, point, ensemble, query_type):
        """Helper for known point × uncertain region"""
        from babelistic.ontology import Query, Subject, Target, KnownPoint, UncertainRegion
        from babelistic.ontology import PointEntity, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(PointEntity(), KnownPoint(point)),
            target=Target(RegionEntity("test"), UncertainRegion(ensemble)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )
    
    def _make_query_uncertain_region_known(self, ensemble, region, query_type):
        """Helper for uncertain region × known region"""
        from babelistic.ontology import Query, Subject, Target, UncertainRegion, KnownRegion
        from babelistic.ontology import RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(RegionEntity("test"), UncertainRegion(ensemble)),
            target=Target(RegionEntity("test"), KnownRegion(region)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )
    
    def _make_query_two_uncertain_regions(self, ensemble1, ensemble2, query_type):
        """Helper for uncertain region × uncertain region"""
        from babelistic.ontology import Query, Subject, Target, UncertainRegion, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(RegionEntity("test"), UncertainRegion(ensemble1)),
            target=Target(RegionEntity("test"), UncertainRegion(ensemble2)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )


# ============================================================================
# PHASE 1.4: FrameworkStrategy - RegionDistribution Integration
# ============================================================================

class TestFrameworkStrategy:
    """Test integration of RegionDistribution with framework"""
    
    def test_region_region_membership_uniform_sampling(self):
        """Test: P(uniform point from R₁ in R₂)"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import FrameworkStrategy
        from babelistic.ontology import QueryType
        
        # Small circle inside large circle
        inner = DiskRegion(np.array([0, 0]), radius=1.0)
        outer = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query_region_region(inner, outer, QueryType.MEMBERSHIP)
        query.bandwidth = 0.2
        query.resolution = 64
        
        strategy = FrameworkStrategy()
        
        # Should be able to handle this
        assert strategy.can_handle(query)
        
        result = strategy.compute(query)
        
        # All points from inner should be in outer
        np.testing.assert_allclose(result.value, 1.0, rtol=0.1)
    
    def test_region_region_partial_overlap(self):
        """Test: P(uniform from R₁ in R₂) with partial overlap"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import FrameworkStrategy
        from babelistic.ontology import QueryType
        
        # Two circles partially overlapping
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([1, 0]), radius=1.0)
        
        query = self._make_query_region_region(disk1, disk2, QueryType.MEMBERSHIP)
        query.bandwidth = 0.2
        query.resolution = 64
        
        strategy = FrameworkStrategy()
        result = strategy.compute(query)
        
        # Should be partial overlap (~40-60%)
        assert 0.3 < result.value < 0.7
    
    def _make_query_region_region(self, region1, region2, query_type):
        """Helper for region × region queries"""
        from babelistic.ontology import Query, Subject, Target, KnownRegion, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(RegionEntity("test"), KnownRegion(region1)),
            target=Target(RegionEntity("test"), KnownRegion(region2)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )


# ============================================================================
# PHASE 1.5: QueryRouter - Automatic Strategy Selection
# ============================================================================

class TestQueryRouter:
    """Test automatic strategy selection and fallback"""
    
    def test_router_selects_analytical_for_known_known(self):
        """Router should prefer analytical for known×known"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import QueryRouter
        from babelistic.ontology import QueryType
        
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query_known_regions(disk1, disk2, QueryType.SUBSET)
        
        router = QueryRouter()
        result = router.compute(query)
        
        # Should use analytical
        assert "analytical" in result.computation_method.lower()
    
    def test_router_selects_framework_for_uncertain_point(self):
        """Router should prefer framework for uncertain point × known region"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.probability.distributions import GaussianDistribution
        from babelistic.engine import QueryRouter
        from babelistic.ontology import QueryType
        
        dist = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2))
        region = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query_uncertain_point_known_region(dist, region, 
                                                               QueryType.MEMBERSHIP)
        query.bandwidth = 0.3
        query.resolution = 32
        
        router = QueryRouter()
        result = router.compute(query)
        
        # Should use framework
        assert "framework" in result.computation_method.lower()
    
    def test_router_fallback_to_monte_carlo(self):
        """Router should fall back to Monte Carlo for complex cases"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import QueryRouter
        from babelistic.ontology import QueryType
        
        # Uncertain region × uncertain region (no analytical or framework)
        ensemble1 = [DiskRegion(np.array([0, 0]), radius=1.0)]
        ensemble2 = [DiskRegion(np.array([1, 0]), radius=1.0)]
        
        query = self._make_query_two_uncertain_regions(ensemble1, ensemble2,
                                                        QueryType.INTERSECTION)
        query.n_samples = 100  # Small for test speed
        
        router = QueryRouter()
        result = router.compute(query)
        
        # Should use Monte Carlo
        assert "monte_carlo" in result.computation_method.lower()
    
    def test_force_strategy_override(self):
        """Test user can override strategy selection"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import QueryRouter
        from babelistic.ontology import QueryType
        
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query_known_regions(disk1, disk2, QueryType.SUBSET)
        
        router = QueryRouter()
        
        # Force Monte Carlo even though analytical is available
        result = router.compute(query, force_strategy='monte_carlo')
        
        assert "monte_carlo" in result.computation_method.lower()
    
    def test_explain_strategy_selection(self):
        """Test strategy selection explanation"""
        from babelistic.geometry.regions import DiskRegion
        from babelistic.engine import QueryRouter
        from babelistic.ontology import QueryType
        
        disk1 = DiskRegion(np.array([0, 0]), radius=1.0)
        disk2 = DiskRegion(np.array([0, 0]), radius=2.0)
        
        query = self._make_query_known_regions(disk1, disk2, QueryType.SUBSET)
        
        router = QueryRouter()
        explanation = router.explain_strategy_selection(query)
        
        assert 'selected' in explanation
        assert 'strategies' in explanation
        assert explanation['selected'] == 'AnalyticalStrategy'
    
    # Helpers
    def _make_query_known_regions(self, region1, region2, query_type):
        from babelistic.ontology import Query, Subject, Target, KnownRegion, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(RegionEntity("test"), KnownRegion(region1)),
            target=Target(RegionEntity("test"), KnownRegion(region2)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )
    
    def _make_query_uncertain_point_known_region(self, dist, region, query_type):
        from babelistic.ontology import Query, Subject, Target, UncertainPoint, KnownRegion
        from babelistic.ontology import PointEntity, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(PointEntity(), UncertainPoint(dist)),
            target=Target(RegionEntity("test"), KnownRegion(region)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )
    
    def _make_query_two_uncertain_regions(self, ensemble1, ensemble2, query_type):
        from babelistic.ontology import Query, Subject, Target, UncertainRegion, RegionEntity
        from babelistic.geometry.metric_spaces import EuclideanSpace
        
        return Query(
            subject=Subject(RegionEntity("test"), UncertainRegion(ensemble1)),
            target=Target(RegionEntity("test"), UncertainRegion(ensemble2)),
            query_type=query_type,
            metric_space=EuclideanSpace()
        )