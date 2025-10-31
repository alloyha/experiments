import numpy as np

# ============================================================================
# MATHEMATICAL VALIDATION TESTS
# ============================================================================

class TestMathematicalProperties:
    """Validate mathematical properties and theorems"""
    
    @staticmethod
    def test_metric_axioms():
        """Test metric space axioms"""
        from babilistic import EuclideanSpace, ManhattanSpace
        
        for space in [EuclideanSpace(2), ManhattanSpace(2)]:
            p1 = np.array([0, 0])
            p2 = np.array([1, 1])
            p3 = np.array([2, 0])
            
            # 1. Non-negativity
            assert space.distance(p1, p2) >= 0
            
            # 2. Identity: d(x,x) = 0
            assert np.isclose(space.distance(p1, p1), 0)
            
            # 3. Symmetry: d(x,y) = d(y,x)
            assert np.isclose(space.distance(p1, p2), space.distance(p2, p1))
            
            # 4. Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
            d_13 = space.distance(p1, p3)
            d_12 = space.distance(p1, p2)
            d_23 = space.distance(p2, p3)
            assert d_13 <= d_12 + d_23 + 1e-10  # Small tolerance
    
    @staticmethod
    def test_probability_bounds():
        """Test probabilities are in [0, 1]"""
        from babilistic import (
            ProbabilityEstimator, EuclideanSpace, DiskRegion,
            GaussianDistribution, GaussianKernel, FFTConvolution,
            QuadratureIntegrator
        )
        
        # Generate random scenarios
        np.random.seed(42)
        
        for _ in range(10):
            center = np.random.randn(2) * 10
            radius = np.random.rand() * 5 + 1
            query_mean = center + np.random.randn(2) * 2
            
            estimator = ProbabilityEstimator(
                EuclideanSpace(2),
                DiskRegion(center, radius),
                GaussianDistribution(query_mean, np.eye(2)),
                GaussianKernel(),
                FFTConvolution(),
                QuadratureIntegrator()
            )
            
            result = estimator.compute(bandwidth=0.3, resolution=48)
            assert 0 <= result.probability <= 1, \
                f"Probability out of bounds: {result.probability}"
    
    @staticmethod
    def test_bandwidth_convergence():
        """Test w_field → indicator as bandwidth → 0"""
        from babilistic import (
            EuclideanSpace, DiskRegion, GaussianKernel, FFTConvolution
        )
        
        space = EuclideanSpace(2)
        region = DiskRegion([0, 0], 1.0)
        kernel = GaussianKernel()
        conv = FFTConvolution()
        
        grid = space.create_grid((-2, 2, -2, 2), 128)
        indicator = region.indicator(grid['points'])
        
        bandwidths = [0.5, 0.2, 0.1, 0.05]
        errors = []
        
        for h in bandwidths:
            w_field = conv.convolve(indicator, kernel, h, grid, space)
            
            # Compare in interior (avoid boundary)
            interior_mask = np.linalg.norm(grid['points'], axis=-1) < 0.7
            error = np.abs(w_field[interior_mask] - indicator[interior_mask]).mean()
            errors.append(error)
        
        # Error should decrease
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i+1] - 0.01, \
                "Error should decrease as bandwidth decreases"
    
    @staticmethod
    def test_distribution_normalization():
        """Test that distributions integrate to 1"""
        from babilistic import (
            GaussianDistribution, StudentTDistribution, EuclideanSpace,
            QuadratureIntegrator
        )
        
        space = EuclideanSpace(2)
        grid = space.create_grid((-4, 4, -4, 4), 64)
        integrator = QuadratureIntegrator()
        
        distributions = [
            GaussianDistribution([0, 0], np.eye(2)),
            StudentTDistribution([0, 0], np.eye(2), df=5)
        ]
        
        for dist in distributions:
            pdf = dist.pdf(grid['points'])
            integral = integrator.integrate(pdf, grid['weights'])
            
            assert np.isclose(integral, 1.0, atol=0.1), \
                f"Distribution doesn't integrate to 1: {integral}"
