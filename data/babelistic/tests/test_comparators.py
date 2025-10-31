import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

# Import all modules
from babelistic import (
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
# TEST DISTRIBUTION COMPARATOR
# ============================================================================

class TestDistributionComparator:
    """Test DistributionComparator class"""
    
    def test_init(self):
        distance = KLDivergence()
        comparator = DistributionComparator(distance)
        assert comparator.distance == distance
    
    def test_compare_w_fields(self):
        comparator = DistributionComparator(TotalVariationDistance())
        w1 = np.array([[0.5, 0.6], [0.7, 0.8]])
        w2 = np.array([[0.4, 0.5], [0.6, 0.7]])
        grid = {
            'points': np.zeros((2, 2, 2)),
            'weights': np.ones((2, 2))
        }
        
        result = comparator.compare_w_fields(w1, w2, grid)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compare_query_distributions(self):
        comparator = DistributionComparator(TotalVariationDistance())
        dist1 = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2))
        dist2 = GaussianDistribution(mean=np.array([1, 1]), cov=np.eye(2))
        
        space = EuclideanSpace(2)
        grid = space.create_grid((0, 2, 0, 2), 10)
        
        result = comparator.compare_query_distributions(dist1, dist2, grid)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_compare_query_distributions_wasserstein(self):
        space = EuclideanSpace(2)
        comparator = DistributionComparator(WassersteinDistance(space))
        dist1 = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2))
        dist2 = GaussianDistribution(mean=np.array([0.5, 0.5]), cov=np.eye(2))
        
        grid = space.create_grid((0, 2, 0, 2), 10)
        
        result = comparator.compare_query_distributions(dist1, dist2, grid)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_convergence_analysis(self):
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.1)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        comparator = DistributionComparator(TotalVariationDistance())
        
        result = comparator.convergence_analysis(
            estimator, bandwidth=0.1, resolutions=[32, 64]
        )
        
        assert 'resolutions' in result
        assert 'probabilities' in result
        assert 'convergence_distances' in result
        assert 'converged' in result
    
    def test_sensitivity_analysis(self):
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.1)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        comparator = DistributionComparator(TotalVariationDistance())
        
        result = comparator.sensitivity_analysis(
            estimator, bandwidths=[0.05, 0.1, 0.15], resolution=32
        )
        
        assert 'bandwidths' in result
        assert 'probabilities' in result
        assert 'distance_matrix' in result
        assert 'max_variation' in result
        assert 'mean_variation' in result