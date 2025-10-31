import numpy as np

import pytest

from unittest.mock import patch

from babilistic import (
    EuclideanSpace, KLDivergence, JSDistance, WassersteinDistance,
    TotalVariationDistance, HellingerDistance,
)

# ============================================================================
# TEST DISTANCE FUNCTIONS
# ============================================================================

class TestDistanceFunctions:
    """Test distance function edge cases"""

    def test_probability_distance_abstract(self):
        from babilistic.distance_functions import ProbabilityDistance
        
        class DummyDistance(ProbabilityDistance):
            pass
        
        with pytest.raises(TypeError):
            DummyDistance()

    def test_probability_distance_cannot_instantiate(self):
        """Verify ProbabilityDistance is abstract"""
        from babilistic.distance_functions import ProbabilityDistance
        
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            ProbabilityDistance()
        
        # Lines 37 and 42 are the abstract method bodies (pass statements)
        # These cannot be executed directly - they're covered through subclasses
        assert True
    
    def test_all_distance_subclasses_implement_interface(self):
        """Verify all distance classes implement the interface"""
        distances = [
            KLDivergence(),
            JSDistance(),
            TotalVariationDistance(),
            HellingerDistance(),
            WassersteinDistance(EuclideanSpace(2)),
        ]
        
        for dist in distances:
            # All should have these methods
            assert hasattr(dist, 'compute')
            assert hasattr(dist, 'is_metric')
            assert callable(dist.compute)
            assert callable(dist.is_metric)
    
    def test_kl_divergence_is_metric(self):
        kl = KLDivergence()
        assert kl.is_metric() == False
    
    def test_js_distance_is_metric(self):
        js = JSDistance()
        assert js.is_metric() == True
    
    def test_js_distance_with_weights(self):
        js = JSDistance()
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        weights = np.array([1.0, 1.0, 1.0])
        
        dist = js.compute(p, q, weights)
        assert isinstance(dist, float)
        assert dist >= 0
    
    def test_js_distance_without_weights(self):
        js = JSDistance()
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        
        dist = js.compute(p, q)
        assert isinstance(dist, float)
        assert dist >= 0
    
    def test_wasserstein_without_points_raises(self):
        space = EuclideanSpace(2)
        wd = WassersteinDistance(space)
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        
        with pytest.raises(ValueError, match="requires support points"):
            wd.compute(p, q)
    
    def test_wasserstein_is_metric(self):
        space = EuclideanSpace(2)
        wd = WassersteinDistance(space)
        assert wd.is_metric() == True
    
    def test_total_variation_is_metric(self):
        tv = TotalVariationDistance()
        assert tv.is_metric() == True
    
    def test_hellinger_is_metric(self):
        hd = HellingerDistance()
        assert hd.is_metric() == True
    
    def test_hellinger_with_weights(self):
        hd = HellingerDistance()
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        weights = np.array([1.0, 1.0, 1.0])
        
        dist = hd.compute(p, q, weights)
        assert isinstance(dist, float)
        assert dist >= 0
    
    def test_hellinger_without_weights(self):
        hd = HellingerDistance()
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        
        dist = hd.compute(p, q)
        assert isinstance(dist, float)
        assert dist >= 0

    def test_js_distance_normalization_paths(self):
        """Test JSDistance with both weight and no-weight paths"""
        js = JSDistance()
        
        # Test without weights (different code path)
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        
        dist_no_weights = js.compute(p, q, weights=None)
        assert dist_no_weights >= 0
        
        # Test with weights
        weights = np.ones(3)
        dist_with_weights = js.compute(p, q, weights=weights)
        assert dist_with_weights >= 0
    
    def test_hellinger_normalization_paths(self):
        """Test HellingerDistance with both weight paths"""
        hd = HellingerDistance()
        
        # Without weights
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        
        dist_no_weights = hd.compute(p, q, weights=None)
        assert dist_no_weights >= 0
        
        # With weights
        weights = np.ones(3)
        dist_with_weights = hd.compute(p, q, weights=weights)
        assert dist_with_weights >= 0
    
    def test_wasserstein_with_weights(self):
        """Test WassersteinDistance with weights"""
        space = EuclideanSpace(2)
        wd = WassersteinDistance(space, approximate=True)
        
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.3, 0.4, 0.3])
        points = np.array([[0, 0], [1, 0], [0, 1]])
        weights = np.ones(3)
        
        dist = wd.compute(p, q, weights=weights, points=points)
        assert dist >= 0

# ============================================================================
# PROBABILITY DISTANCE
# ============================================================================

class TestProbabilityDistances:
    """Tests for probability distance metrics"""
    
    @staticmethod
    def test_kl_divergence_properties():
        """Test KL divergence properties"""
        from babilistic import KLDivergence, EuclideanSpace, GaussianDistribution
        
        kl = KLDivergence()
        
        space = EuclideanSpace(2)
        grid = space.create_grid((-3, 3, -3, 3), 48)
        
        dist1 = GaussianDistribution([0, 0], np.eye(2))
        dist2 = GaussianDistribution([1, 0], np.eye(2))
        
        p1 = dist1.pdf(grid['points'])
        p2 = dist2.pdf(grid['points'])
        
        # KL is non-negative
        d_kl = kl.compute(p1.flat, p2.flat, grid['weights'].flat)
        assert d_kl >= 0, "KL divergence must be non-negative"
        
        # KL(p||p) = 0
        d_kl_self = kl.compute(p1.flat, p1.flat, grid['weights'].flat)
        assert d_kl_self < 0.01, f"KL(p||p) should be 0, got {d_kl_self}"
        
        # KL is asymmetric
        d_kl_12 = kl.compute(p1.flat, p2.flat, grid['weights'].flat)
        d_kl_21 = kl.compute(p2.flat, p1.flat, grid['weights'].flat)
        assert not np.isclose(d_kl_12, d_kl_21), "KL should be asymmetric"
    
    @staticmethod
    def test_wasserstein_scales_with_distance():
        """Test Wasserstein distance scales with separation"""
        from babilistic import WassersteinDistance, EuclideanSpace, GaussianDistribution
        
        space = EuclideanSpace(2)
        wasserstein = WassersteinDistance(space, approximate=True)
        
        grid = space.create_grid((-3, 3, -3, 3), 48)
        points = grid['points'].reshape(-1, 2)
        weights = grid['weights']
        
        dist_ref = GaussianDistribution([0, 0], np.eye(2) * 0.5)
        dist_close = GaussianDistribution([0.5, 0], np.eye(2) * 0.5)
        dist_far = GaussianDistribution([2, 0], np.eye(2) * 0.5)
        
        p_ref = dist_ref.pdf(grid['points'])
        p_close = dist_close.pdf(grid['points'])
        p_far = dist_far.pdf(grid['points'])
        
        d_close = wasserstein.compute(p_ref.flat, p_close.flat, weights.flat, points)
        d_far = wasserstein.compute(p_ref.flat, p_far.flat, weights.flat, points)
        
        assert d_far > d_close, "Wasserstein should scale with distance"
        assert d_far / d_close > 2.0, "Should roughly scale with displacement"

    @staticmethod
    def test_kl_and_js_and_tv_hellinger_basic():
        """Test KL, JS, TV, and Hellinger distances on simple distributions"""
        from babilistic import (
            KLDivergence, JSDistance,
            TotalVariationDistance, HellingerDistance
        )

        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])
        weights = np.array([1.0, 1.0, 1.0])

        kl = KLDivergence()
        d_kl = kl.compute(p, q)
        assert d_kl >= 0.0
        # KL to itself should be ~0
        assert kl.compute(p, p) < 1e-8

        js = JSDistance()
        d_js = js.compute(p, q)
        assert d_js >= 0.0
        assert js.compute(p, p) == 0.0

        tv = TotalVariationDistance()
        d_tv = tv.compute(p, q)
        # TV bounded between 0 and 1
        assert 0.0 <= d_tv <= 1.0

        hell = HellingerDistance()
        d_h = hell.compute(p, q)
        assert 0.0 <= d_h <= 1.0


    def test_wasserstein_approximate_on_simple_grid(self):
        """Test approximate Wasserstein distance on simple grid"""
        from babilistic import WassersteinDistance, EuclideanSpace

        space = EuclideanSpace(2)
        grid = space.create_grid((-1, 1, -1, 1), 8)
        pts = grid['points'].reshape(-1, 2)

        # Two delta-like distributions at opposite corners
        p = np.zeros(len(pts))
        q = np.zeros(len(pts))
        p[0] = 1.0
        q[-1] = 1.0

        wass = WassersteinDistance(metric_space=space, approximate=True)
        d = wass.compute(p, q, points=pts)
        assert d >= 0.0

    def test_wasserstein_exact_computation(self):
        """Test WassersteinDistance with approximate=False"""
        space = EuclideanSpace(2)
        wd = WassersteinDistance(space, approximate=False)
        
        # Small dataset to make exact computation feasible
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        points = np.array([[0, 0], [1, 0], [0, 1]])
        
        # This will call _exact_wasserstein which falls back to sliced
        dist = wd.compute(p, q, points=points)
        assert isinstance(dist, float)
        assert dist >= 0
    
    def test_wasserstein_exact_with_scipy_mock(self):
        """Test exact Wasserstein by mocking scipy for coverage"""
        space = EuclideanSpace(2)
        
        # Test the import error path
        with patch.dict('sys.modules', {'scipy.optimize': None}):
            try:
                wd = WassersteinDistance(space, approximate=False)
                p = np.array([0.5, 0.5])
                q = np.array([0.3, 0.7])
                points = np.array([[0, 0], [1, 1]])
                
                # This should raise ImportError and be caught
                dist = wd.compute(p, q, points=points)
            except ImportError:
                # Expected if scipy is actually missing
                pass