from .base import (
    MetricSpace,
    DistanceFunction,
    Region,
    UncertaintyDistribution,
    Kernel,
    ConvolutionStrategy,
    Integrator,
)

from .geometry.metric_spaces import (
    EuclideanSpace,
    ManhattanSpace,
    SphericalSpace,
    GeoSpace,
)

from .geometry.distance_functions import (
    KLDivergence,
    JSDistance,
    TotalVariationDistance,
    HellingerDistance,
    WassersteinDistance,
)

from .geometry.regions import (
    DiskRegion,
    PolygonRegion,
    EllipseRegion,
    ImplicitRegion,
    BufferedPolygonRegion,
    MultiRegion,
)

from .probability.distributions import (
    GaussianDistribution,
    UniformDistribution,
    StudentTDistribution,
    LogNormalDistribution,
    MixtureDistribution,
    EmpiricalDistribution,
)

from .probability.kernels import (
    GaussianKernel,
    EpanechnikovKernel,
    UniformKernel,
    QuarticKernel,
    TriangularKernel,
    MaternKernel,
)

from .probability.comparators import (
    DistributionComparator,
)

from .computation.convolution_strategies import (
    DirectConvolution,
    SparseConvolution,
    FFTConvolution,
)

from .computation.integrators import (
    QuadratureIntegrator,
    MonteCarloIntegrator,
)

# Expose high-level API from main
from .estimators import (
    ProbabilityEstimator,
    ProbabilityResult,
)

from .integrations.geofence import (
    GeofenceAdapter,
    GeofenceMetricSpace,
    GeofenceRegion,
    geofence_to_probability,
    estimate_geofence_probability_analytic,
)

__all__ = [
    # spaces
    "MetricSpace",
    "EuclideanSpace",
    "GeoSpace",
    "ManhattanSpace",
    "SphericalSpace",
    "GeoSpace",
    # distances
    "DistanceFunction",
    "KLDivergence",
    "JSDistance",
    "TotalVariationDistance",
    "HellingerDistance",
    "WassersteinDistance",
    # regions
    "Region",
    "DiskRegion",
    "PolygonRegion",
    "EllipseRegion",
    "ImplicitRegion",
    "BufferedPolygonRegion",
    "MultiRegion",
    # distributions
    "UncertaintyDistribution",
    "GaussianDistribution",
    "UniformDistribution",
    "StudentTDistribution",
    "LogNormalDistribution",
    "MixtureDistribution",
    "EmpiricalDistribution",
    # kernels
    "Kernel",
    "GaussianKernel",
    "EpanechnikovKernel",
    "UniformKernel",
    "QuarticKernel",
    "TriangularKernel",
    "MaternKernel",
    # convolution & integrators
    "ConvolutionStrategy",
    "DirectConvolution",
    "SparseConvolution",
    "FFTConvolution",
    "Integrator",
    "QuadratureIntegrator",
    "MonteCarloIntegrator",
    # main APIs
    "ProbabilityEstimator",
    "ProbabilityResult",
    "DistributionComparator",
    "GeofenceAdapter",
    "GeofenceMetricSpace",
    "GeofenceRegion",
    "geofence_to_probability",
    "estimate_geofence_probability_analytic",
]
