import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

# Import all modules
from babilistic import (
    # Metric Spaces
    EuclideanSpace, ManhattanSpace, SphericalSpace, GeoSpace,
    # Regions
    DiskRegion, PolygonRegion, EllipseRegion, ImplicitRegion,
    BufferedPolygonRegion, MultiRegion,
    # Distributions
    GaussianDistribution, UniformDistribution, MixtureDistribution,
    StudentTDistribution, LogNormalDistribution, EmpiricalDistribution,
    # Kernels
    GaussianKernel, EpanechnikovKernel, UniformKernel,
    TriangularKernel, QuarticKernel, MaternKernel,
    # Convolution
    DirectConvolution, SparseConvolution, FFTConvolution,
    # Integrators
    QuadratureIntegrator, MonteCarloIntegrator,
    # Distance functions
    KLDivergence, JSDistance, WassersteinDistance,
    TotalVariationDistance, HellingerDistance,
    # Core
    ProbabilityEstimator, ProbabilityResult,
    # Comparators
    DistributionComparator,
    # Geofence
    GeofenceAdapter, GeofenceMetricSpace, GeofenceRegion,
    geofence_to_probability, estimate_geofence_probability_analytic,
)



# ============================================================================
# TEST CONVOLUTION STRATEGIES
# ============================================================================

class TestConvolutionStrategies:
    """Test SparseConvolution and FFTConvolution edge cases"""
    
    def test_sparse_convolution(self):
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        kernel = EpanechnikovKernel()  # Compact support
        
        grid = space.create_grid((-2, 2, -2, 2), 50)
        indicator = region.indicator(grid['points'])
        
        conv = SparseConvolution()
        w_field = conv.convolve(indicator, kernel, 0.5, grid, space)
        
        assert w_field.shape == grid['shape']
        assert np.all(w_field >= 0)
        assert np.all(w_field <= 1.0)
    
    def test_fft_convolution_non_euclidean_raises(self):
        space = ManhattanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0, metric_space=space)
        kernel = GaussianKernel()
        
        grid = space.create_grid((-2, 2, -2, 2), 50)
        indicator = region.indicator(grid['points'])
        
        conv = FFTConvolution()
        with pytest.raises(ValueError, match="FFT convolution requires Euclidean"):
            conv.convolve(indicator, kernel, 0.5, grid, space)
    
    def test_fft_convolution_euclidean(self):
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        kernel = GaussianKernel()
        
        grid = space.create_grid((-2, 2, -2, 2), 50)
        indicator = region.indicator(grid['points'])
        
        conv = FFTConvolution()
        w_field = conv.convolve(indicator, kernel, 0.5, grid, space)
        
        assert w_field.shape == grid['shape']
        assert np.all(w_field >= 0)
        assert np.all(w_field <= 1.0)
    
    def test_fft_convolution_normalization(self):
        """Test FFTConvolution normalization path"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        kernel = GaussianKernel()
        
        grid = space.create_grid((-2, 2, -2, 2), 60)
        indicator = region.indicator(grid['points'])
        
        conv = FFTConvolution()
        w_field = conv.convolve(indicator, kernel, 0.3, grid, space)
        
        # Check clipping is applied
        assert np.all(w_field >= 0.0)
        assert np.all(w_field <= 1.0)
    
    def test_sparse_convolution_no_support_points(self):
        """Test SparseConvolution when no points are within support"""
        space = EuclideanSpace(2)
        
        # Create a region far from query area
        region = DiskRegion(center=np.array([0, 0]), radius=0.1)
        
        # Use very small bandwidth
        kernel = EpanechnikovKernel()  # Compact support
        
        # Grid far from region
        grid = space.create_grid((10, 12, 10, 12), 20)
        indicator = region.indicator(grid['points'])
        
        conv = SparseConvolution()
        w_field = conv.convolve(indicator, kernel, 0.05, grid, space)
        
        # All values should be zero or very close
        assert np.all(w_field >= 0)
    
    def test_sparse_convolution_all_points_outside_support(self):
        """Test SparseConvolution when all points are outside kernel support"""
        space = EuclideanSpace(2)
        
        # Create a tiny region at origin
        region = DiskRegion(center=np.array([0, 0]), radius=0.01)
        
        # Create grid far from region
        grid = space.create_grid((10, 11, 10, 11), 30)
        indicator = region.indicator(grid['points'])
        
        # Use kernel with very small compact support
        kernel = EpanechnikovKernel()
        
        conv = SparseConvolution()
        w_field = conv.convolve(indicator, kernel, bandwidth=0.01, grid=grid, metric_space=space)
        
        # All values should be zero since no region points are within support
        assert np.all(w_field >= 0)
        assert np.all(w_field <= 1.0)
    
    def test_sparse_convolution_no_support_coverage(self):
        """Test SparseConvolution when query points have no region neighbors"""
        space = EuclideanSpace(2)
        
        # Create region at (0,0)
        region = DiskRegion(center=np.array([0, 0]), radius=0.1)
        
        # Create grid very far from region at (100, 100)
        grid = space.create_grid((100, 101, 100, 101), 50)
        indicator = region.indicator(grid['points'])
        
        # Use compact kernel with tiny bandwidth
        kernel = EpanechnikovKernel()  # Compact support
        
        conv = SparseConvolution()
        w_field = conv.convolve(indicator, kernel, bandwidth=0.05, grid=grid, metric_space=space)
        
        # Should be all zeros - the continue statement prevents unnecessary computation
        assert np.allclose(w_field, 0.0)