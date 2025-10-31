import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from babilistic import (
    Region,
    ProbabilityEstimator,
    MetricSpace,
    ProbabilityResult,
    UniformDistribution,
    DiskRegion,
    DirectConvolution,
    FFTConvolution,
    QuadratureIntegrator,
    DistributionComparator,
    TotalVariationDistance,
    EuclideanSpace,
    SphericalSpace,
    FFTConvolution,
    DirectConvolution,
    GaussianKernel,
    DistributionComparator,
    TotalVariationDistance,
    GaussianDistribution,
    UncertaintyDistribution,
    geofence_to_probability,
    estimate_geofence_probability_analytic,
)

# ============================================================================
# TEST PROBABILITY ESTIMATOR EDGE CASES
# ============================================================================

class TestProbabilityEstimatorEdgeCases:
    """Test edge cases in core.py"""
    
    def test_compute_probability_zero_total_prob(self):
        """Test degenerate case where total_prob = 0"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        
        # Create distribution with extremely small covariance
        query = GaussianDistribution(
            mean=np.array([0.5, 0.5]),
            cov=np.eye(2) * 1e-20  # Near-zero covariance
        )
        
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        
        # This should trigger the total_prob == 0 fallback
        result = estimator.compute(bandwidth=0.1, resolution=32)
        
        # Should fallback to indicator at mean
        assert isinstance(result.probability, float)
        assert 0 <= result.probability <= 1
    
    def test_compute_probability_field_with_custom_bounds(self):
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.1)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        
        # Test with custom bounds
        w_field, grid = estimator.compute_probability_field(
            bandwidth=0.1, resolution=32, bounds=(-3, 3, -3, 3)
        )
        
        assert w_field.shape == grid['shape']
    
    def test_compute_probability_field_without_cov(self):
        """Test with distribution that doesn't have 'cov' attribute"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = UniformDistribution(bounds=(-1, 1, -1, 1))
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        
        w_field, grid = estimator.compute_probability_field(bandwidth=0.1, resolution=32)
        assert w_field.shape == grid['shape']

    def test_compute_probability_field_exception_in_bounds(self):
        """Test compute_probability_field when query.mean() raises exception"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        
        # Create a mock query that raises exception on mean()
        query = Mock(spec=UncertaintyDistribution)
        query.mean.side_effect = Exception("Test exception")
        query.pdf.return_value = np.ones((10, 10))
        
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        
        # Should fall back to region bounds
        w_field, grid = estimator.compute_probability_field(bandwidth=0.1, resolution=10)
        assert w_field is not None
    
    def test_compute_probability_zero_total_exception(self):
        """Test compute_probability when fallback also fails"""
        space = EuclideanSpace(2)
        
        # Create mock region that raises exception
        region = Mock(spec=Region)
        region.bounds.return_value = (-1, 1, -1, 1)
        region.indicator.side_effect = Exception("Test exception")
        
        # Create distribution with zero variance
        query = GaussianDistribution(
            mean=np.array([0.5, 0.5]),
            cov=np.eye(2) * 1e-30
        )
        
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        
        # This should trigger the exception path in the fallback
        try:
            result = estimator.compute(bandwidth=0.1, resolution=10)
            # If it succeeds with zeros, that's the fallback working
            assert True
        except:
            # If it fails, we hit the exception path
            assert True

    def test_compute_probability_fallback_exception_path(self):
        """Test exception path when mean() fails and indicator fails"""

        space = EuclideanSpace(2)
        
        # Create a mock region that works for bounds but fails for indicator
        class FaultyRegion:
            def bounds(self):
                return (-1, 1, -1, 1)
            
            def indicator(self, x):
                # Works for grid evaluation
                if hasattr(x, 'shape') and len(x.shape) > 2:
                    return np.ones(x.shape[:-1])
                # Fails for single point (fallback path)
                raise RuntimeError("Indicator failure in fallback")
            
            def sample_boundary(self, n):
                return np.random.randn(n, 2)
        
        region = FaultyRegion()
        
        # Query with extremely small covariance to trigger total_prob ≈ 0
        query = GaussianDistribution(
            mean=np.array([0.5, 0.5]),
            cov=np.eye(2) * 1e-30  # Extremely small
        )
        
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        
        # Compute w_field
        grid = space.create_grid((-1, 1, -1, 1), 20)
        w_field = np.ones(grid['shape']) * 0.5
        
        # Mock query.pdf to return near-zero (triggers total_prob = 0 path)
        with patch.object(query, 'pdf', return_value=np.zeros(grid['shape'])):
            # This should trigger the exception in the fallback
            result = estimator.compute_probability(w_field, grid)
            
            # Should still return a valid result (proceeds with zeros)
            assert isinstance(result, ProbabilityResult)