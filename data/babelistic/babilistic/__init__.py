from .metric_spaces import (
    MetricSpace,
    EuclideanSpace,
    ManhattanSpace,
    SphericalSpace,
    GeoSpace,
)
from .distance_functions import (
    ProbabilityDistance,
    KLDivergence,
    JSDistance,
    TotalVariationDistance,
    HellingerDistance,
    WassersteinDistance,
)
from .regions import (
    Region,
    DiskRegion,
    PolygonRegion,
    EllipseRegion,
    ImplicitRegion,
    BufferedPolygonRegion,
    MultiRegion,
)
from .distributions import (
    UncertaintyDistribution,
    GaussianDistribution,
    UniformDistribution,
    StudentTDistribution,
    LogNormalDistribution,
    MixtureDistribution,
    EmpiricalDistribution,
)
from .kernels import (
    Kernel,
    GaussianKernel,
    EpanechnikovKernel,
    UniformKernel,
    QuarticKernel,
    TriangularKernel,
    MaternKernel,
)
from .convolution_strategies import (
    ConvolutionStrategy,
    DirectConvolution,
    SparseConvolution,
    FFTConvolution,
)
from .integrators import (
    Integrator,
    QuadratureIntegrator,
    MonteCarloIntegrator,
)

# Expose high-level API from main
from .core import (
    ProbabilityEstimator,
    ProbabilityResult,
)

from .comparators import (
    DistributionComparator,
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
    "ProbabilityDistance",
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
