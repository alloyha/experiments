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
# TEST ABSTRACT BASE CLASSES (base.py - 0% coverage)
# ============================================================================

class TestAbstractBases:
    """Test that abstract methods raise NotImplementedError"""
    
    def test_metric_space_abstract(self):
        from babelistic.base import MetricSpace
        
        class DummyMetricSpace(MetricSpace):
            pass
        
        with pytest.raises(TypeError):
            DummyMetricSpace()
    
    def test_region_abstract(self):
        from babelistic.base import Region
        
        class DummyRegion(Region):
            pass
        
        with pytest.raises(TypeError):
            DummyRegion()
    
    def test_uncertainty_distribution_abstract(self):
        from babelistic.base import UncertaintyDistribution
        
        class DummyDistribution(UncertaintyDistribution):
            pass
        
        with pytest.raises(TypeError):
            DummyDistribution()
    
    def test_kernel_abstract(self):
        from babelistic.base import Kernel
        
        class DummyKernel(Kernel):
            pass
        
        with pytest.raises(TypeError):
            DummyKernel()
    
    def test_convolution_strategy_abstract(self):
        from babelistic.base import ConvolutionStrategy
        
        class DummyConvolution(ConvolutionStrategy):
            pass
        
        with pytest.raises(TypeError):
            DummyConvolution()
    
    def test_integrator_abstract(self):
        from babelistic.base import Integrator
        
        class DummyIntegrator(Integrator):
            pass
        
        with pytest.raises(TypeError):
            DummyIntegrator()
    
    def test_abstract_classes_are_abstract(self):
        """Verify abstract classes cannot be instantiated"""
        from babelistic.base import (
            MetricSpace, Region, UncertaintyDistribution,
            Kernel, ConvolutionStrategy, Integrator
        )
        
        abstract_classes = [
            MetricSpace, Region, UncertaintyDistribution,
            Kernel, ConvolutionStrategy, Integrator
        ]
        
        for cls in abstract_classes:
            with pytest.raises(TypeError):
                cls()
    
    def test_abstract_distance_function(self):
        """Test ProbabilityDistance abstract class"""
        from babelistic.distance_functions import ProbabilityDistance
        
        with pytest.raises(TypeError):
            ProbabilityDistance()
