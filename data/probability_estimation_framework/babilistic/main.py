"""
Agnostic Kernel-Based Probability Estimation Framework

Computes P(X ∈ R) for uncertain query points and regions
on arbitrary metric spaces with pluggable components.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .metric_spaces import (
    MetricSpace, 
    EuclideanSpace, 
    ManhattanSpace, 
    SphericalSpace,
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
    MonteCarloIntegrator
)

# ============================================================================
# MAIN FRAMEWORK
# ============================================================================

@dataclass
class ProbabilityResult:
    """Result container"""
    probability: float
    error_estimate: float
    w_field: np.ndarray
    grid: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class ProbabilityEstimator:
    """
    Framework for computing P(X ∈ R) with arbitrary components
    """
    
    def __init__(self,
                 metric_space: MetricSpace,
                 region: Region,
                 query_distribution: UncertaintyDistribution,
                 kernel: Kernel,
                 convolution_strategy: ConvolutionStrategy,
                 integrator: Integrator):
        
        self.space = metric_space
        self.region = region
        self.query = query_distribution
        self.kernel = kernel
        self.conv_strategy = convolution_strategy
        self.integrator = integrator
    
    def compute_probability_field(self, 
                                  bandwidth: float,
                                  resolution: int = 128,
                                  bounds: Optional[Tuple] = None) -> np.ndarray:
        """
        Compute mollified indicator w(x) = (I_R * K_h)(x)
        """
        # Create grid
        if bounds is None:
            # Default bounds should cover both the region and the bulk of the
            # query distribution. Using only the region bounds can produce
            # misleading normalized probabilities when the query mass lies
            # far from the region (the grid would exclude most of the PDF).
            region_bounds = self.region.bounds()
            try:
                q_mean = np.asarray(self.query.mean())
                q_cov = getattr(self.query, 'cov', None)
                if q_cov is not None:
                    q_std = np.sqrt(np.abs(np.diag(q_cov)))
                    margin = 4.0 * q_std
                    query_bounds = (
                        q_mean[0] - margin[0], q_mean[0] + margin[0],
                        q_mean[1] - margin[1], q_mean[1] + margin[1]
                    )
                    # Merge region and query bounds
                    bounds = (
                        min(region_bounds[0], query_bounds[0]),
                        max(region_bounds[1], query_bounds[1]),
                        min(region_bounds[2], query_bounds[2]),
                        max(region_bounds[3], query_bounds[3]),
                    )
                else:
                    bounds = region_bounds
            except Exception:
                bounds = region_bounds
        
        grid = self.space.create_grid(bounds, resolution)
        
        # Evaluate region indicator
        points = grid['points']
        indicator = self.region.indicator(points)
        
        # Convolve
        w_field = self.conv_strategy.convolve(
            indicator, self.kernel, bandwidth, grid, self.space
        )
        
        return w_field, grid
    
    def compute_probability(self,
                           w_field: np.ndarray,
                           grid: Dict[str, np.ndarray]) -> ProbabilityResult:
        """
        Compute P(X ∈ R) = ∫ p_X(x) · w(x) dx
        """
        # Evaluate query distribution on grid
        points = grid['points']
        p_X = self.query.pdf(points)
        
        # Compute total probability mass inside the grid (for diagnostics).
        total_prob = self.integrator.integrate(p_X, grid['weights'])
        if total_prob == 0:
            # Degenerate / numerically zero total probability (e.g., near-zero covariance)
            # Fall back to deterministic approximation: treat the query as concentrated at its mean
            # and return indicator(mean) as the probability. This avoids numerical underflow
            # for extremely small covariances where the PDF is effectively a delta.
            try:
                mean_pt = self.query.mean()
                prob = float(self.region.indicator(np.asarray(mean_pt)))
                return ProbabilityResult(
                    probability=prob,
                    error_estimate=0.0,
                    w_field=np.zeros_like(points[..., 0]),
                    grid=grid,
                    metadata={
                        'space': type(self.space).__name__,
                        'region': type(self.region).__name__,
                        'kernel': type(self.kernel).__name__,
                        'convolution': type(self.conv_strategy).__name__,
                        'query_mean': mean_pt
                    }
                )
            except Exception:
                # If we cannot compute a deterministic fallback, proceed with zeros
                pass
        
        # Compute integrand
        integrand = p_X * w_field

        # Integrate numerator (unnormalized) and estimate error
        probability = self.integrator.integrate(integrand, grid['weights'])
        error = self.integrator.estimate_error(integrand, grid['weights'])

        # Normalize by total probability mass of the query on the grid to
        # obtain a true probability in [0, 1]. This protects against
        # mismatches of units (degrees vs radians) or coarse grids where
        # the PDF does not integrate to 1 numerically.
        if total_prob > 0:
            probability = probability / total_prob
            error = error / total_prob
        
        return ProbabilityResult(
            probability=probability,
            error_estimate=error,
            w_field=w_field,
            grid=grid,
            metadata={
                'space': type(self.space).__name__,
                'region': type(self.region).__name__,
                'kernel': type(self.kernel).__name__,
                'convolution': type(self.conv_strategy).__name__,
                'query_mean': self.query.mean()
            }
        )
    
    def compute(self,
                bandwidth: float,
                resolution: int = 128,
                bounds: Optional[Tuple] = None) -> ProbabilityResult:
        """One-shot computation"""
        w_field, grid = self.compute_probability_field(bandwidth, resolution, bounds)
        return self.compute_probability(w_field, grid)




# ============================================================================
# GEOFENCE INTEGRATION MODULE
# ============================================================================

class GeofenceAdapter:
    """
    Adapter to bridge geofence distance metrics into agnostic framework.
    
    Converts geofence-style distance calculations (with DistanceMetric protocol)
    into framework-compatible MetricSpace implementations.
    """
    
    def __init__(self, distance_metric_func):
        """
        Parameters
        ----------
        distance_metric_func : callable
            Function with signature (lat1, lon1, lat2, lon2) -> distance
            Supports both scalar and array inputs for lat1/lon1
        """
        self.distance_func = distance_metric_func
    
    def to_metric_space(self) -> MetricSpace:
        """Convert to agnostic MetricSpace"""
        return GeofenceMetricSpace(self.distance_func)


class GeofenceMetricSpace(MetricSpace):
    """
    MetricSpace wrapper for geofence distance functions.
    
    This allows any geofence DistanceMetric (Haversine, Manhattan, OSM routing, etc.)
    to be used seamlessly in the agnostic probability framework.
    """
    
    def __init__(self, distance_func):
        """
        Parameters
        ----------
        distance_func : callable
            Function (lat1, lon1, lat2, lon2) -> distance_in_meters
            Must support array broadcasting for lat1/lon1
        """
        self.distance_func = distance_func
        self._earth_radius = 6371000.0  # meters
    
    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute distance in meters.
        
        Parameters
        ----------
        x1, x2 : np.ndarray
            Points in [lat, lon] format (degrees)
        """
        # Internal convention: x1/x2 are in radians. User-provided distance_func
        # typically expects degrees, so convert to degrees when calling it.
        def _to_deg(a):
            return np.degrees(a)

        if x1.ndim == 1 and x2.ndim == 1:
            # Single point to single point
            return self.distance_func(_to_deg(x1[0]), _to_deg(x1[1]), _to_deg(x2[0]), _to_deg(x2[1]))
        elif x1.ndim == 2 and x2.ndim == 1:
            # Array of points to single point (for Monte Carlo)
            return self.distance_func(_to_deg(x1[:, 0]), _to_deg(x1[:, 1]), _to_deg(x2[0]), _to_deg(x2[1]))
        # Note: 3D grid inputs are not supported by the geofence adapter wrapper
        # and should raise a ValueError to keep behavior explicit for callers.
        elif x1.ndim == 3 and x2.ndim == 1:
            raise ValueError(f"Unsupported shape combination: x1.shape={x1.shape}, x2.shape={x2.shape}")
        else:
            raise ValueError(f"Unsupported shape combination: x1.shape={x1.shape}, x2.shape={x2.shape}")
    
    def area_element(self, x: np.ndarray) -> np.ndarray:
        """
        Compute local area element at lat/lon position.
        Uses spherical geometry: dA = R² cos(lat) dlon dlat
        """
        # Internal coordinates are radians; x[...,0] is latitude in radians
        lat = x[..., 0]
        return self._earth_radius**2 * np.cos(lat)
        
    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:
        """
        Create lon/lat grid (compatible with geofence conventions).
        
        Parameters
        ----------
        bounds : tuple
            (lat_min, lat_max, lon_min, lon_max) in degrees
        resolution : int
            Number of points along latitude direction
        """
        # Bounds are provided in degrees; convert to radians internally
        lat_min, lat_max, lon_min, lon_max = bounds
        lat_min_rad = np.radians(lat_min)
        lat_max_rad = np.radians(lat_max)
        lon_min_rad = np.radians(lon_min)
        lon_max_rad = np.radians(lon_max)

        # Adaptive longitude resolution based on latitude (use midpoint)
        lat_mid = 0.5 * (lat_min_rad + lat_max_rad)
        lon_range = lon_max_rad - lon_min_rad
        lat_range = lat_max_rad - lat_min_rad

        lon_res = int(resolution * (lon_range / max(lat_range * np.cos(lat_mid), 1e-6)))
        lon_res = max(lon_res, 60)

        lat = np.linspace(lat_min_rad, lat_max_rad, resolution)
        lon = np.linspace(lon_min_rad, lon_max_rad, lon_res)
        LAT, LON = np.meshgrid(lat, lon, indexing='ij')

        dlon = lon[1] - lon[0]
        dlat = lat[1] - lat[0]

        # Store as [lat(rad), lon(rad)] for internal computations
        points = np.stack([LAT, LON], axis=-1)
        weights = self.area_element(points) * dlon * dlat
        
        return {
            'points': points,
            'weights': weights,
            'LAT': LAT, 'LON': LON,
            'lat': lat, 'lon': lon,
            'shape': LAT.shape
        }


class GeofenceRegion(Region):
    """
    Region defined by geofence reference point and distance threshold.
    
    This creates a region where all points within distance d0 of a
    reference location (with uncertainty) are considered "inside".
    """
    
    def __init__(self, 
                 reference_lat: float,
                 reference_lon: float,
                 reference_uncertainty: float,
                 distance_threshold: float,
                 distance_metric):
        """
        Parameters
        ----------
        reference_lat, reference_lon : float
            Reference point coordinates (degrees)
        reference_uncertainty : float
            Uncertainty in reference location (meters, isotropic)
        distance_threshold : float
            Maximum distance for "inside" region (meters)
        distance_metric : callable
            Distance function (lat1, lon1, lat2, lon2) -> distance_m
        """
        # Store reference coordinates internally in radians
        self.ref_lat = np.radians(reference_lat)
        self.ref_lon = np.radians(reference_lon)
        self.ref_uncertainty = reference_uncertainty
        self.d0 = distance_threshold
        self.distance_metric = distance_metric
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """
        Membership function: 1 if within distance threshold, 0 otherwise.
        
        For deterministic reference: indicator is sharp.
        For uncertain reference: we use expected indicator (simplified).
        """
        # Accept either degrees (external API) or radians (internal grids).
        x_arr = np.asarray(x)
        # Heuristic: if coordinates are large (> 2π) treat as degrees and convert
        def _to_radians_if_degrees(a):
            a = np.asarray(a)
            if a.ndim == 0:
                return np.radians(a) if abs(a) > 2 * np.pi else a
            if np.abs(a).max() > 2 * np.pi:
                return np.radians(a)
            return a

        x_rad = _to_radians_if_degrees(x_arr)
        # Compute distances from all query points to reference
        original_shape = x_rad.shape[:-1]
        
        if x_rad.ndim == 1:
            # Single point
            lat_deg = np.degrees(x_rad[0])
            lon_deg = np.degrees(x_rad[1])
            ref_lat_deg = np.degrees(self.ref_lat)
            ref_lon_deg = np.degrees(self.ref_lon)
            dist = self.distance_metric(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg)
            return float(dist <= self.d0)
        elif x_rad.ndim == 2:
            # Array of points
            lat_deg = np.degrees(x_rad[:, 0])
            lon_deg = np.degrees(x_rad[:, 1])
            ref_lat_deg = np.degrees(self.ref_lat)
            ref_lon_deg = np.degrees(self.ref_lon)
            dist = self.distance_metric(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg)
            return (dist <= self.d0).astype(float)
        else:
            # Grid of points
            x_flat = x_rad.reshape(-1, x_rad.shape[-1])
            lat_deg = np.degrees(x_flat[:, 0])
            lon_deg = np.degrees(x_flat[:, 1])
            ref_lat_deg = np.degrees(self.ref_lat)
            ref_lon_deg = np.degrees(self.ref_lon)
            dist = self.distance_metric(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg)
            return (dist <= self.d0).astype(float).reshape(original_shape)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """
        Sample points on the boundary (circle at distance d0).
        
        Uses small-angle approximation for lat/lon circle.
        """
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Convert distance to degrees (approximate)
        # At latitude φ: Δlat ≈ distance / 111320 m/deg
        #                Δlon ≈ distance / (111320 * cos(φ)) m/deg
        # Return boundary samples in degrees for external consumption.
        ref_lat_deg = np.degrees(self.ref_lat)
        ref_lon_deg = np.degrees(self.ref_lon)

        lat_scale = 111320.0  # meters per degree latitude
        lon_scale = lat_scale * np.cos(np.radians(ref_lat_deg))

        dlat_deg = self.d0 / lat_scale
        dlon_deg = self.d0 / lon_scale

        # Sample circle (degrees)
        lats = ref_lat_deg + dlat_deg * np.sin(theta)
        lons = ref_lon_deg + dlon_deg * np.cos(theta)

        return np.column_stack([lats, lons])
        
    def bounds(self) -> Tuple:
        """
        Return bounding box for geofence region.
        
        Accounts for reference uncertainty and distance threshold.
        """
        # Total extent = d0 + 3*sigma (3-sigma coverage)
        total_extent_m = self.d0 + 3 * self.ref_uncertainty

        # Convert to degrees for bounding box output
        ref_lat_deg = np.degrees(self.ref_lat)
        ref_lon_deg = np.degrees(self.ref_lon)

        lat_scale = 111320.0
        lon_scale = lat_scale * np.cos(np.radians(ref_lat_deg))

        margin_lat = total_extent_m / lat_scale
        margin_lon = total_extent_m / lon_scale

        return (
            ref_lat_deg - margin_lat,
            ref_lat_deg + margin_lat,
            ref_lon_deg - margin_lon,
            ref_lon_deg + margin_lon
        )


def geofence_to_probability(
    subject_lat: float,
    subject_lon: float,
    subject_uncertainty: Union[float, np.ndarray],
    reference_lat: float,
    reference_lon: float,
    reference_uncertainty: Union[float, np.ndarray],
    distance_threshold: float,
    distance_metric,
    bandwidth: float = None,
    resolution: int = 128,
    kernel: Optional[Kernel] = None,
    integrator: Optional[Integrator] = None
) -> ProbabilityResult:
    """
    Compute P(subject within distance_threshold of reference) using agnostic framework.
    
    This is the main integration function that converts geofence problems into
    the agnostic probability framework format.
    
    Parameters
    ----------
    subject_lat, subject_lon : float
        Subject point coordinates (degrees)
    subject_uncertainty : float or array
        Uncertainty in subject location (meters)
        Can be scalar (isotropic) or 2x2 covariance matrix
    reference_lat, reference_lon : float
        Reference point coordinates (degrees)
    reference_uncertainty : float or array
        Uncertainty in reference location (meters)
    distance_threshold : float
        Distance threshold d0 (meters)
    distance_metric : callable
        Distance function (lat1, lon1, lat2, lon2) -> distance_m
    bandwidth : float, optional
        Kernel bandwidth (default: auto-select based on uncertainties)
    resolution : int
        Grid resolution for computation
    kernel : Kernel, optional
        Smoothing kernel (default: GaussianKernel)
    integrator : Integrator, optional
        Integration method (default: QuadratureIntegrator)
    
    Returns
    -------
    ProbabilityResult
        Complete probability computation result
    
    Examples
    --------
    >>> from geofence_module import HaversineDistance
    >>> 
    >>> # Simple isotropic case
    >>> result = geofence_to_probability(
    ...     subject_lat=37.7749, subject_lon=-122.4194, subject_uncertainty=10.0,
    ...     reference_lat=37.7750, reference_lon=-122.4195, reference_uncertainty=5.0,
    ...     distance_threshold=50.0,
    ...     distance_metric=HaversineDistance()
    ... )
    >>> print(f"P(within 50m) = {result.probability:.4f}")
    
    >>> # With custom kernel and Manhattan distance
    >>> from geofence_module import ManhattanDistance
    >>> result = geofence_to_probability(
    ...     subject_lat=40.7580, subject_lon=-73.9855, subject_uncertainty=20.0,
    ...     reference_lat=40.7590, reference_lon=-73.9865, reference_uncertainty=15.0,
    ...     distance_threshold=200.0,
    ...     distance_metric=ManhattanDistance(),
    ...     kernel=EpanechnikovKernel(),
    ...     bandwidth=30.0
    ... )
    """
    
    # Convert distance metric to MetricSpace
    adapter = GeofenceAdapter(distance_metric)
    metric_space = adapter.to_metric_space()
    
    # Create region (geofence around reference with threshold)
    region = GeofenceRegion(
        reference_lat=reference_lat,
        reference_lon=reference_lon,
        reference_uncertainty=reference_uncertainty if np.isscalar(reference_uncertainty) else np.sqrt(np.trace(reference_uncertainty)),
        distance_threshold=distance_threshold,
        distance_metric=distance_metric
    )
    
    # Create query distribution (subject with uncertainty) using radians internally
    # Convert uncertainties from meters -> degrees -> radians
    lat_scale = 111320.0
    lon_scale = lat_scale * np.cos(np.radians(subject_lat))

    if np.isscalar(subject_uncertainty):
        # Isotropic uncertainty in meters -> convert to degrees then to radians
        var_lat_deg = (subject_uncertainty / lat_scale) ** 2
        var_lon_deg = (subject_uncertainty / lon_scale) ** 2
        # convert variance from deg^2 to rad^2
        deg2rad = (np.pi / 180.0) ** 2
        subject_cov_rad = np.diag([var_lat_deg * deg2rad, var_lon_deg * deg2rad])
    else:
        # Anisotropic covariance in meters -> convert to degrees then to radians
        subject_cov_m = np.asarray(subject_uncertainty)
        scale_matrix = np.diag([1/lat_scale, 1/lon_scale])
        subject_cov_deg = scale_matrix @ subject_cov_m @ scale_matrix.T
        deg2rad = (np.pi / 180.0) ** 2
        subject_cov_rad = subject_cov_deg * deg2rad

    # Create GaussianDistribution with mean/cov in radians
    query = GaussianDistribution(
        mean=np.radians(np.array([subject_lat, subject_lon])),
        cov=subject_cov_rad
    )
    
    # Auto-select bandwidth if not provided
    if bandwidth is None:
        # Use larger of subject/reference uncertainty
        if np.isscalar(subject_uncertainty):
            subject_std = subject_uncertainty
        else:
            subject_std = np.sqrt(np.trace(subject_uncertainty))
        
        if np.isscalar(reference_uncertainty):
            reference_std = reference_uncertainty
        else:
            reference_std = np.sqrt(np.trace(reference_uncertainty))
        
        # Bandwidth ~ 20% of total uncertainty
        total_std = np.sqrt(subject_std**2 + reference_std**2)
        bandwidth = 0.2 * total_std
    
    # bandwidth is provided in meters (internal distance metric returns meters)
    bandwidth_m = bandwidth
    
    # Set defaults
    if kernel is None:
        kernel = GaussianKernel()
    if integrator is None:
        integrator = QuadratureIntegrator()
    
    # Choose convolution strategy based on distance metric
    # Use DirectConvolution for non-Euclidean metrics (OSM routing, etc.)
    # Use FFTConvolution for geometric metrics on uniform grids
    if hasattr(distance_metric, '__class__') and 'OSM' in distance_metric.__class__.__name__:
        conv_strategy = DirectConvolution()
    else:
        conv_strategy = DirectConvolution()  # Safe default for lat/lon
    
    # Create estimator
    estimator = ProbabilityEstimator(
        metric_space=metric_space,
        region=region,
        query_distribution=query,
        kernel=kernel,
        convolution_strategy=conv_strategy,
        integrator=integrator
    )
    
    # Compute probability
    # Pass bandwidth in meters to estimator (metric returns meters)
    result = estimator.compute(bandwidth=bandwidth_m, resolution=resolution)
    
    # Add geofence-specific metadata
    result.metadata.update({
        'geofence_mode': True,
        'distance_metric': distance_metric.__class__.__name__ if hasattr(distance_metric, '__class__') else 'custom',
        'distance_threshold_m': distance_threshold,
        'subject_uncertainty_m': subject_uncertainty if np.isscalar(subject_uncertainty) else 'anisotropic',
        'reference_uncertainty_m': reference_uncertainty if np.isscalar(reference_uncertainty) else 'anisotropic',
        'bandwidth_m': bandwidth
    })

    # Re-normalize using discrete integrals to be robust to coarse grids vs
    # very narrow PDFs. Compute denom = ∫ p(x) dA and numer = ∫ p(x) w(x) dA,
    # then set probability = numer/denom.
    try:
        grid = result.grid
        points = grid['points']
        p_X = query.pdf(points)
        denom = integrator.integrate(p_X, grid['weights'])
        if denom > 0:
            numer = integrator.integrate(p_X * result.w_field, grid['weights'])
            result.probability = numer / denom
            result.error_estimate = integrator.estimate_error(p_X * result.w_field, grid['weights']) / max(denom, 1e-30)
    except Exception:
        pass

    return result


# ============================================================================
# GEOFENCE-SPECIFIC UTILITIES
# ============================================================================

def estimate_geofence_probability_analytic(
    subject_uncertainty: float,
    reference_uncertainty: float,
    mean_separation: float,
    distance_threshold: float
) -> float:
    """
    Analytic approximation for isotropic geofence probability (Rice distribution).
    
    Valid only for:
    - Isotropic uncertainties (circular)
    - Euclidean-like metrics (Haversine, Euclidean tangent)
    - Small distances relative to Earth radius
    
    Parameters
    ----------
    subject_uncertainty : float
        Standard deviation of subject location (meters)
    reference_uncertainty : float
        Standard deviation of reference location (meters)
    mean_separation : float
        Distance between mean positions (meters)
    distance_threshold : float
        Distance threshold d0 (meters)
    
    Returns
    -------
    probability : float
        P(distance ≤ d0)
    
    Examples
    --------
    >>> # Quick analytic estimate
    >>> p = estimate_geofence_probability_analytic(
    ...     subject_uncertainty=10.0,
    ...     reference_uncertainty=5.0,
    ...     mean_separation=30.0,
    ...     distance_threshold=50.0
    ... )
    >>> print(f"P ≈ {p:.4f}")
    """
    from scipy.stats import rice
    
    sigma_total = np.sqrt(subject_uncertainty**2 + reference_uncertainty**2)
    
    if sigma_total == 0:
        return 1.0 if mean_separation <= distance_threshold else 0.0
    
    # Rice distribution parameters
    b = mean_separation / sigma_total  # non-centrality
    x = distance_threshold / sigma_total  # threshold
    
    return float(rice.cdf(x, b))

