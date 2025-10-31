import numpy as np

import pytest

from babelistic import ProbabilityEstimator

from babelistic import (
    GaussianKernel, EpanechnikovKernel, TriangularKernel,
    QuarticKernel, UniformKernel, MaternKernel
)

from babelistic import (
    MonteCarloIntegrator,
    QuadratureIntegrator,
)

from babelistic import (
    EuclideanSpace,
    ManhattanSpace,
    GeoSpace,
)

from babelistic import (
    DiskRegion,
)

from babelistic import (
    GaussianDistribution,
)

from babelistic import (
    DirectConvolution,
)


# ============================================================================
# UNIT TESTS: KERNELS
# ============================================================================

class TestKernels:
    """Unit tests for all Kernel implementations"""
    
    @staticmethod
    def test_kernel_normalization():
        """Verify all kernels are properly normalized"""
        
        
        kernels = [
            ("Gaussian", GaussianKernel()),
            ("Epanechnikov", EpanechnikovKernel()),
            ("Triangular", TriangularKernel()),
            ("Quartic", QuarticKernel()),
            ("Uniform", UniformKernel()),
            ("Matern(2.5)", MaternKernel(nu=2.5))
        ]
        
        bandwidth = 1.0
        
        for name, kernel in kernels:
            # Value at origin should be maximum
            val_at_zero = kernel.evaluate(np.array([0.0]), bandwidth)[0]
            val_away = kernel.evaluate(np.array([0.5]), bandwidth)[0]
            
            assert val_at_zero >= val_away, \
                f"{name}: kernel should be maximum at origin"
            
            # Compact support kernels should be zero beyond bandwidth
            if kernel.is_compact():
                support = kernel.support_radius(bandwidth)
                val_beyond = kernel.evaluate(np.array([support * 1.1]), bandwidth)[0]
                assert np.isclose(val_beyond, 0.0), \
                    f"{name}: compact kernel not zero beyond support"
    
    @staticmethod
    def test_kernel_monotonicity():
        """Test that kernels decay monotonically"""
        from babelistic import GaussianKernel, MaternKernel
        
        distances = np.linspace(0, 3, 100)
        
        for kernel in [GaussianKernel(), MaternKernel(nu=2.5)]:
            vals = kernel.evaluate(distances, bandwidth=1.0)
            
            # Check monotonic decrease
            diffs = np.diff(vals)
            assert np.all(diffs <= 0), "Kernel should be monotonically decreasing"
    
    @staticmethod
    def test_matern_special_cases():
        """Test Matérn kernel special cases"""
        from babelistic import MaternKernel
        
        distances = np.array([0.0, 0.5, 1.0, 2.0])
        
        # Test different nu values
        for nu in [0.5, 1.5, 2.5]:
            kernel = MaternKernel(nu=nu)
            vals = kernel.evaluate(distances, bandwidth=1.0)
            
            assert np.isclose(vals[0], 1.0), f"Matérn({nu}) should be 1 at origin"
            assert vals[-1] < vals[0], f"Matérn({nu}) should decay"
    
    @staticmethod
    def test_kernel_properties():
        """Test mathematical properties of kernels"""
        from babelistic import EpanechnikovKernel, QuarticKernel
        
        # Epanechnikov: (1 - t²) for |t| ≤ 1
        epan = EpanechnikovKernel()
        assert np.isclose(epan.evaluate(np.array([0.0]), 1.0)[0], 1.0)
        assert np.isclose(epan.evaluate(np.array([0.5]), 1.0)[0], 0.75)
        assert np.isclose(epan.evaluate(np.array([1.0]), 1.0)[0], 0.0)
        
        # Quartic: (15/16)(1 - t²)² for |t| ≤ 1
        quartic = QuarticKernel()
        assert np.isclose(quartic.evaluate(np.array([0.0]), 1.0)[0], 15.0/16.0)
    
    def test_matern_kernel_invalid_nu(self):
        with pytest.raises(ValueError, match="nu must be positive"):
            MaternKernel(nu=-1.0)
    
    def test_matern_kernel_nu_05(self):
        kernel = MaternKernel(nu=0.5)
        distances = np.array([0, 0.5, 1.0, 2.0])
        vals = kernel.evaluate(distances, bandwidth=1.0)
        assert len(vals) == 4
        assert np.all(vals >= 0)
    
    def test_matern_kernel_nu_15(self):
        kernel = MaternKernel(nu=1.5)
        distances = np.array([0, 0.5, 1.0, 2.0])
        vals = kernel.evaluate(distances, bandwidth=1.0)
        assert len(vals) == 4
        assert np.all(vals >= 0)
    
    def test_matern_kernel_nu_25(self):
        kernel = MaternKernel(nu=2.5)
        distances = np.array([0, 0.5, 1.0, 2.0])
        vals = kernel.evaluate(distances, bandwidth=1.0)
        assert len(vals) == 4
        assert np.all(vals >= 0)
    
    def test_matern_kernel_general_nu(self):
        kernel = MaternKernel(nu=3.5)
        distances = np.array([0, 0.5, 1.0, 2.0])
        vals = kernel.evaluate(distances, bandwidth=1.0)
        assert len(vals) == 4
        assert np.all(vals >= 0)
    
    def test_matern_support_radius(self):
        kernel = MaternKernel(nu=2.5)
        radius = kernel.support_radius(1.0)
        assert radius == 3.0

    def test_matern_kernel_zero_distance(self):
        """Test MaternKernel with distance = 0 (edge case)"""
        kernel = MaternKernel(nu=3.5)  # General nu
        
        # Distance array with zeros
        distances = np.array([0.0, 1e-11, 0.5, 1.0])
        
        vals = kernel.evaluate(distances, bandwidth=1.0)
        
        # At distance=0, kernel should be 1.0
        assert np.isclose(vals[0], 1.0)
        assert np.isclose(vals[1], 1.0)  # Very small distance


def test_gaussian_kernel_evaluate_and_support():
    from babelistic import GaussianKernel

    k = GaussianKernel()
    d = np.array([0.0, 1.0, 3.0])
    vals = k.evaluate(d, bandwidth=1.0)

    assert np.isclose(vals[0], 1.0)
    assert vals[1] < vals[0]
    # 3-sigma cutoff -> support_radius should be 3.0
    assert np.isclose(k.support_radius(1.0), 3.0)


def test_epanechnikov_kernel_support_and_boundary():
    from babelistic import EpanechnikovKernel

    k = EpanechnikovKernel()
    d = np.array([0.0, 1.0, 1.5])
    vals = k.evaluate(d, bandwidth=1.0)

    # At d=0 -> max value
    assert vals[0] > 0.0
    # At boundary d=1.0 -> value == 0
    assert np.isclose(vals[1], 0.0)
    # Outside support -> zero
    assert vals[2] == 0.0
    assert np.isclose(k.support_radius(1.0), 1.0)

class TestCompleteIntegration:
    """Additional integration tests for full coverage"""
    
    def test_all_kernels(self):
        """Test all kernel types in pipeline"""
        kernels = [
            GaussianKernel(),
            EpanechnikovKernel(),
            UniformKernel(),
            TriangularKernel(),
            QuarticKernel(),
            MaternKernel(nu=2.5)
        ]
        
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.2)
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        for kernel in kernels:
            estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
            result = estimator.compute(bandwidth=0.3, resolution=32)
            assert 0 <= result.probability <= 1
    
    def test_all_metric_spaces(self):
        """Test all metric space types"""
        spaces = [
            (EuclideanSpace(2), np.array([0, 0]), np.array([1, 1])),
            (ManhattanSpace(2), np.array([0, 0]), np.array([1, 1])),
            (GeoSpace(radius=6371.0), np.array([37.7749, -122.4194]), np.array([37.7750, -122.4195])),
        ]
        
        for space, center, query_mean in spaces:
            region = DiskRegion(center=center, radius=1.0, metric_space=space)
            query_dist = GaussianDistribution(mean=query_mean, cov=np.eye(2) * 0.1)
            kernel = GaussianKernel()
            conv = DirectConvolution()
            integrator = QuadratureIntegrator()
            
            estimator = ProbabilityEstimator(space, region, query_dist, kernel, conv, integrator)
            result = estimator.compute(bandwidth=0.5, resolution=32)
            assert 0 <= result.probability <= 1
    
    def test_all_integrators(self):
        """Test all integrator types"""
        integrators = [
            QuadratureIntegrator(),
            MonteCarloIntegrator(n_samples=1000)
        ]
        
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.2)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        
        for integrator in integrators:
            estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
            result = estimator.compute(bandwidth=0.3, resolution=32)
            assert 0 <= result.probability <= 1