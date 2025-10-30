import numpy as np

# ============================================================================
# PROBABILITY DISTANCE TESTS
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