import time
import inspect
from dataclasses import dataclass, field
from typing import (
    Union, 
    Tuple, 
    Optional, 
    Literal, 
    Dict, 
    List, 
    Any, 
    Callable, 
    Protocol,
)

import numpy as np
from scipy.stats import rice, beta

# Earth radius in meters (mean radius for spherical approximation)
R_EARTH = 6371000.0


# ============================================================================
# DISTANCE METRICS
# ============================================================================

class DistanceMetric(Protocol):
    """
    Protocol for distance functions on lat/lon points.
    
    All distance metrics must support:
    - Scalar inputs: distance(lat1, lon1, lat2, lon2) -> float
    - Array inputs: distance(lat1_array, lon1_array, lat2, lon2) -> array
      (for Monte Carlo sampling efficiency)
    """
    def __call__(self, 
                 lat1: Union[float, np.ndarray], 
                 lon1: Union[float, np.ndarray],
                 lat2: float, 
                 lon2: float) -> Union[float, np.ndarray]:
        """Compute distance in meters"""
        ...


class HaversineDistance:
    """
    Great circle distance on perfect sphere (default).
    
    Mathematical definition:
        Uses haversine formula for great circle arc length.
        Treats Earth as sphere with radius R_EARTH.
    
    Properties:
        Accuracy: ±0.5% vs true WGS84 ellipsoid
        Speed: ~10 flops (very fast)
        Symmetric: Yes
        Triangle inequality: Yes
    
    Use cases:
        - General purpose geospatial calculations
        - Works globally (equator to poles)
        - Default for most applications
    
    Limitations:
        - Ignores Earth's oblateness (0.3% error)
        - Not suitable for survey-grade precision
    
    Examples:
        >>> metric = HaversineDistance()
        >>> d = metric(37.7749, -122.4194, 37.7750, -122.4195)
        >>> print(f"Distance: {d:.2f} meters")
    """
    def __call__(self, lat1, lon1, lat2, lon2):
        return haversine_distance_m(lat1, lon1, lat2, lon2)


class GreatCircleDistance:
    """
    Alias for HaversineDistance (more intuitive name).
    
    Great circle = shortest path on sphere surface.
    """
    def __init__(self):
        self._haversine = HaversineDistance()
    
    def __call__(self, lat1, lon1, lat2, lon2):
        return self._haversine(lat1, lon1, lat2, lon2)


class VincentyDistance:
    """
    Geodesic distance on WGS84 ellipsoid (high precision).
    
    Mathematical definition:
        Uses Vincenty's formulae for geodesics on oblate ellipsoid.
        Accounts for Earth's actual shape (flattening ~1/298.257).
    
    Properties:
        Accuracy: ±0.5mm (industry standard)
        Speed: ~100 flops (iterative, 10x slower than haversine)
        Symmetric: Yes
        Triangle inequality: Yes
    
    Use cases:
        - Survey-grade GPS applications
        - Long distances where 0.3% matters
        - Polar regions (haversine breaks down)
        - Legal/cadastral boundaries
    
    Limitations:
        - Requires geopy library (optional dependency)
        - Slower than haversine
        - Can fail for antipodal points (opposite sides of Earth)
    
    Dependencies:
        pip install geopy
    
    Examples:
        >>> metric = VincentyDistance()
        >>> d = metric(0.0, 0.0, 0.0, 180.0)  # Half Earth circumference
    """
    def __init__(self, cache_size: int = 10000):
        try:
            from geopy.distance import geodesic
            self._geodesic = geodesic
        except ImportError:
            raise ImportError(
                "VincentyDistance requires geopy. Install: pip install geopy"
            )
        
        # Cache for performance
        from functools import lru_cache
        self._compute_cached = lru_cache(maxsize=cache_size)(self._compute_single)
    
    def _compute_single(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute single distance (cached)"""
        return self._geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def __call__(self, lat1, lon1, lat2, lon2):
        if np.isscalar(lat1):
            # Round to 6 decimals (~10cm) for cache hits
            return self._compute_cached(
                round(float(lat1), 6), round(float(lon1), 6),
                round(float(lat2), 6), round(float(lon2), 6)
            )
        else:
            # Vectorized for arrays
            return np.array([
                self._compute_cached(
                    round(float(la1), 6), round(float(lo1), 6),
                    round(float(lat2), 6), round(float(lon2), 6)
                )
                for la1, lo1 in zip(lat1, lon1)
            ])


class LpDistance:
    """
    Generalized Lp (Minkowski) distance in local tangent plane.
    
    Mathematical definition:
        d = (|Δlat_m|^p + |Δlon_m|^p)^(1/p)
        where Δlat_m, Δlon_m are differences in meters using local scale.
    
    Special cases:
        p=1: Manhattan distance (L1, city-block)
        p=2: Euclidean distance (L2, straight line in tangent plane)
        p=∞: Chebyshev distance (L∞, max of components)
    
    Properties:
        Accuracy: Good for small distances (<10km)
        Speed: ~20 flops (fast)
        Symmetric: Yes
        Triangle inequality: Yes (for p ≥ 1)
    
    Use cases:
        - Urban navigation (Manhattan for grid streets)
        - Rectangular geofences (Chebyshev)
        - Custom norms (p=3, p=1.5, etc.)
    
    Limitations:
        - Tangent plane approximation (breaks down for large distances)
        - Not suitable for polar regions
        - Ignores Earth's curvature
    
    Args:
        p: Norm parameter (1 ≤ p ≤ ∞)
    
    Examples:
        >>> metric = LpDistance(p=1)  # Manhattan
        >>> metric = LpDistance(p=2)  # Euclidean
        >>> metric = LpDistance(p=float('inf'))  # Chebyshev
    """
    def __init__(self, p: float = 2.0):
        if p < 1:
            raise ValueError(f"p must be ≥ 1, got {p}")
        self.p = p
    
    def __call__(self, lat1, lon1, lat2, lon2):
        # Convert to meters using local scale
        lat_diff_m = np.abs(lat1 - lat2) * 111132.954
        
        # Longitude scale depends on latitude (use midpoint or average)
        if np.isscalar(lat1):
            lat_mid = (lat1 + lat2) / 2.0
        else:
            lat_mid = lat1  # For arrays, use sample latitude
        
        lon_scale = 111132.954 * np.cos(np.deg2rad(lat_mid))
        lon_diff_m = np.abs(lon1 - lon2) * lon_scale
        
        if np.isinf(self.p):
            # L∞ norm (Chebyshev)
            return np.maximum(lat_diff_m, lon_diff_m)
        else:
            # Lp norm
            return (lat_diff_m**self.p + lon_diff_m**self.p)**(1.0/self.p)


class ManhattanDistance(LpDistance):
    """
    L1 / city-block / taxicab distance.
    
    Mathematical definition:
        d = |Δlat_m| + |Δlon_m|
    
    Properties:
        Always ≥ great circle distance (upper bound)
    
    Use cases:
        - Urban navigation on grid streets (Manhattan, Barcelona)
        - Situations where diagonal movement impossible
        - Conservative distance estimates
    
    Examples:
        >>> metric = ManhattanDistance()
        >>> # Distance is sum of north-south + east-west
    """
    def __init__(self):
        super().__init__(p=1.0)


class EuclideanTangentDistance(LpDistance):
    """
    L2 / Euclidean distance in tangent plane.
    
    Mathematical definition:
        d = sqrt(Δlat_m² + Δlon_m²)
    
    Properties:
        Nearly identical to haversine for distances <10km
        Simpler calculation (no trig)
    
    Use cases:
        - Very local scenarios (indoor, warehouse)
        - When flat Earth assumption acceptable
        - Performance-critical local calculations
    
    Examples:
        >>> metric = EuclideanTangentDistance()
        >>> # Treats Earth as flat locally
    """
    def __init__(self):
        super().__init__(p=2.0)


class ChebyshevDistance(LpDistance):
    """
    L∞ / max / rectangular distance.
    
    Mathematical definition:
        d = max(|Δlat_m|, |Δlon_m|)
    
    Properties:
        Defines square/rectangular regions
        Distance = largest component
    
    Use cases:
        - Rectangular geofences
        - Bounding box checks
        - When constraint is "both lat AND lon within tolerance"
    
    Examples:
        >>> metric = ChebyshevDistance()
        >>> # d0=100 means ±100m in BOTH directions (square fence)
    """
    def __init__(self):
        super().__init__(p=float('inf'))


class WeightedDistance:
    """
    Distance scaled by a cost/pricing/difficulty function.
    
    Mathematical definition:
        d_weighted = d_geometric × cost_factor(location)
    
    The cost function can represent:
        - Terrain difficulty (elevation, obstacles)
        - Traffic patterns (congestion zones)
        - Delivery pricing zones
        - Risk assessment (danger areas)
        - Travel time (instead of distance)
    
    Properties:
        Preserves properties of base metric if cost > 0
        Can break triangle inequality if cost varies spatially
    
    Use cases:
        - Delivery cost optimization
        - Hiking difficulty estimation
        - Drone battery consumption
        - Accessibility planning
    
    Args:
        base_metric: Underlying geometric distance
        cost_function: Callable(lat, lon) -> float multiplier (>0)
    
    Examples:
        >>> def delivery_cost(lat, lon):
        ...     zone = get_zone(lat, lon)
        ...     return {'downtown': 1.0, 'suburbs': 1.5}[zone]
        >>> 
        >>> metric = WeightedDistance(HaversineDistance(), delivery_cost)
        >>> # Now d0=5000 means "5000 cost-units", not meters
    """
    def __init__(self, 
                 base_metric: DistanceMetric,
                 cost_function: Callable[[float, float], float]):
        self.base_metric = base_metric
        self.cost_function = cost_function
    
    def __call__(self, lat1, lon1, lat2, lon2):
        base_dist = self.base_metric(lat1, lon1, lat2, lon2)
        
        # Apply cost at midpoint (or could integrate along path for more accuracy)
        if np.isscalar(lat1):
            lat_mid = (lat1 + lat2) / 2.0
            lon_mid = (lon1 + lon2) / 2.0
            cost = self.cost_function(lat_mid, lon_mid)
        else:
            # Vectorized for arrays
            lat_mid = (lat1 + lat2) / 2.0
            lon_mid = (lon1 + lon2) / 2.0
            cost = np.array([
                self.cost_function(float(la), float(lo)) 
                for la, lo in zip(lat_mid, lon_mid)
            ])
        
        return base_dist * cost


class CustomDistance:
    """
    User-defined distance function wrapper.
    
    Allows arbitrary distance calculations to be plugged into the framework.
    
    Properties:
        Depends entirely on user implementation
    
    Use cases:
        - Road network distances (via routing API)
        - Graph distances
        - Custom business logic
    
    Args:
        distance_func: Callable with signature:
                      (lat1, lon1, lat2, lon2) -> distance_in_meters
                      Must support both scalar and array inputs for lat1/lon1
    
    Examples:
        >>> def road_distance(lat1, lon1, lat2, lon2):
        ...     # Call routing API
        ...     route = get_route((lat1, lon1), (lat2, lon2))
        ...     return route.distance_meters
        >>> 
        >>> metric = CustomDistance(road_distance)
        >>> params = MethodParams(distance_metric=metric)
    """
    def __init__(self, distance_func: Callable):
        self.distance_func = distance_func
        
        # Validate signature
        sig = inspect.signature(distance_func)
        if len(sig.parameters) != 4:
            raise ValueError(
                f"distance_func must take 4 parameters (lat1, lon1, lat2, lon2), "
                f"got {len(sig.parameters)}"
            )
    
    def __call__(self, lat1, lon1, lat2, lon2):
        return self.distance_func(lat1, lon1, lat2, lon2)


# Convenience factory functions
def lp_distance(p: float) -> LpDistance:
    """
    Factory function for creating Lp distances.
    
    Examples:
        >>> metric = lp_distance(1)      # Manhattan
        >>> metric = lp_distance(2)      # Euclidean
        >>> metric = lp_distance(3)      # L3 norm
        >>> metric = lp_distance(np.inf) # Chebyshev
    """
    return LpDistance(p)


def weighted_haversine(cost_func: Callable[[float, float], float]) -> WeightedDistance:
    """
    Create a cost-weighted haversine distance.
    
    Args:
        cost_func: Function(lat, lon) -> multiplier
        
    Example:
        >>> def urban_congestion(lat, lon):
        ...     if in_downtown(lat, lon):
        ...         return 2.0  # 2x "farther" due to traffic
        ...     return 1.0
        >>> 
        >>> metric = weighted_haversine(urban_congestion)
    """
    return WeightedDistance(HaversineDistance(), cost_func)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class GeoPoint:
    """Represents a geodetic point (lat, lon)."""
    lat: float
    lon: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lon)
    
    def __repr__(self) -> str:
        return f"GeoPoint(lat={self.lat:.6f}, lon={self.lon:.6f})"


@dataclass
class Covariance2D:
    """Wrapper for 2×2 ENU covariance matrix."""
    _matrix: np.ndarray = field(init=False)
    
    def __init__(self, data: Union[float, np.ndarray, 'Covariance2D']):
        if isinstance(data, Covariance2D):
            self._matrix = data._matrix.copy()
        else:
            self._matrix = self._process_input(data)
    
    @staticmethod
    def _process_input(inp: Union[float, np.ndarray]) -> np.ndarray:
        """Convert input to 2x2 covariance matrix."""
        if np.isscalar(inp):
            sigma = float(inp)
            if sigma < 0:
                raise ValueError(f"Sigma must be non-negative, got {sigma}")
            return np.eye(2) * (sigma ** 2)
        
        arr = np.asarray(inp, dtype=float)
        if arr.shape == (2,):
            # Diagonal variances
            return np.diag(arr)
        elif arr.shape == (2, 2):
            return arr.copy()
        else:
            raise ValueError(f"Invalid covariance shape: {arr.shape}")
    
    @classmethod
    def from_input(cls, inp: Any) -> 'Covariance2D':
        """Create Covariance2D from various input types."""
        return cls(inp)
    
    def as_matrix(self) -> np.ndarray:
        """Return the 2x2 covariance matrix."""
        return self._matrix.copy()
    
    def is_isotropic(self, rtol: float = 1e-6) -> bool:
        """Check if covariance is isotropic (circular)."""
        eigs = np.linalg.eigvalsh(self._matrix)
        return np.allclose(eigs[0], eigs[1], rtol=rtol)
    
    def max_std(self) -> float:
        """Return maximum standard deviation."""
        eigs = np.clip(np.linalg.eigvalsh(self._matrix), a_min=0.0, a_max=None)
        return float(np.sqrt(eigs.max()))
    
    def cond(self) -> float:
        """Return condition number."""
        try:
            return float(np.linalg.cond(self._matrix))
        except Exception:
            return float('inf')
    
    def __repr__(self) -> str:
        if self.is_isotropic():
            std = self.max_std()
            return f"Covariance2D(isotropic, σ={std:.3f}m)"
        else:
            return f"Covariance2D(anisotropic, max_σ={self.max_std():.3f}m)"


@dataclass
class Subject:
    """Subject point with uncertainty."""
    mu: GeoPoint
    Sigma: Covariance2D
    id: Optional[str] = None
    
    def __repr__(self) -> str:
        id_str = f", id='{self.id}'" if self.id else ""
        return f"Subject({self.mu}, {self.Sigma}{id_str})"


@dataclass
class Reference:
    """Reference point with uncertainty."""
    mu: GeoPoint
    Sigma: Covariance2D
    id: Optional[str] = None
    
    def __repr__(self) -> str:
        id_str = f", id='{self.id}'" if self.id else ""
        return f"Reference({self.mu}, {self.Sigma}{id_str})"


@dataclass
class MethodParams:
    """Parameters for computation methods."""
    mode: Literal['auto', 'analytic', 'mc_ecef', 'mc_tangent'] = 'auto'
    prob_threshold: float = 0.95
    n_mc: int = 200_000
    batch_size: int = 100_000
    conservative_decision: bool = True
    random_state: Optional[int] = None
    use_antithetic: bool = True
    cp_alpha: float = 0.05
    distance_metric: Optional[DistanceMetric] = None  # NEW in v4
    
    def get_distance_metric(self) -> DistanceMetric:
        """Get the distance metric, defaulting to haversine."""
        return self.distance_metric if self.distance_metric is not None else HaversineDistance()
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'prob_threshold': self.prob_threshold,
            'n_mc': self.n_mc,
            'batch_size': self.batch_size,
            'conservative_decision': self.conservative_decision,
            'random_state': self.random_state,
            'use_antithetic': self.use_antithetic,
            'cp_alpha': self.cp_alpha,
            'distance_metric': type(self.distance_metric).__name__ if self.distance_metric else 'HaversineDistance'
        }


@dataclass
class Scenario:
    """Structured scenario for profiling."""
    name: str
    subject: Subject
    reference: Reference
    d0: float
    method_params: MethodParams = field(default_factory=MethodParams)
    expected_behavior: str = ""
    
    # For analytics compatibility
    sigma_P_scalar: Optional[float] = None
    sigma_Q_scalar: Optional[float] = None


@dataclass
class ProbabilityResult:
    """Result from probability computation."""
    fulfilled: bool
    probability: float
    method: str
    mc_stderr: Optional[float] = None
    n_samples: Optional[int] = None
    cp_lower: Optional[float] = None
    cp_upper: Optional[float] = None
    sigma_cond: Optional[float] = None
    max_std_m: Optional[float] = None
    delta_m: Optional[float] = None
    decision_by: Optional[str] = None


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def latlon_to_ecef(lat_deg: float, lon_deg: float, R: float = R_EARTH) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)

    x = R * np.cos(phi) * np.cos(lam)
    y = R * np.cos(phi) * np.sin(lam)
    z = R * np.sin(phi)

    return np.array([x, y, z], dtype=float)


def geodetic_to_ecef_jacobian(lat_deg: float, lon_deg: float, R: float = R_EARTH) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)

    dxdphi = -R * np.sin(phi) * np.cos(lam)
    dydphi = -R * np.sin(phi) * np.sin(lam)
    dzdphi = R * np.cos(phi)

    dxdlam = -R * np.cos(phi) * np.sin(lam)
    dydlam = R * np.cos(phi) * np.cos(lam)
    dzdlam = 0.0

    return np.array([[dxdphi, dxdlam],
                     [dydphi, dydlam],
                     [dzdphi, dzdlam]], dtype=float)


def haversine_distance_m(lat1: Union[float, np.ndarray],
                         lon1: Union[float, np.ndarray],
                         lat2: float,
                         lon2: float,
                         R: float = R_EARTH) -> Union[float, np.ndarray]:
    phi1 = np.deg2rad(lat1)
    lam1 = np.deg2rad(lon1)
    phi2 = np.deg2rad(lat2)
    lam2 = np.deg2rad(lon2)

    dphi = phi2 - phi1
    dlam = lam2 - lam1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def enu_to_radians_scale(lat_deg: float) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)
    meters_per_deg_lat = 111132.954
    meters_per_deg_lon = 111132.954 * np.cos(lat_rad)

    return np.diag([1.0 / meters_per_deg_lat, 1.0 / meters_per_deg_lon])


def _ensure_psd(mat: np.ndarray) -> np.ndarray:
    """Force symmetric PSD by clipping small negative eigenvalues to zero."""
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    return (eigvecs * eigvals_clipped) @ eigvecs.T


def _add_jitter(mat: np.ndarray, rel: float = 1e-12) -> np.ndarray:
    trace = np.trace(mat)
    jitter = rel * max(trace, 1.0)
    return mat + np.eye(mat.shape[0]) * jitter


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Return the (lower, upper) Clopper-Pearson (1-alpha) interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lower), float(upper)


def _choose_method_heuristic(Sigma_X_m2: np.ndarray, delta_m: float,
                             max_std_thresh: float = 200.0, delta_thresh: float = 5000.0,
                             cond_thresh: float = 20.0) -> Tuple[str, Dict]:
    """Conservative heuristic to decide if tangent-plane MC is acceptable."""
    eigs = np.linalg.eigvalsh(Sigma_X_m2)
    eigs = np.clip(eigs, a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs.max()))
    
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')

    is_local = (max_std < max_std_thresh) and (delta_m < delta_thresh) and (cond < cond_thresh)

    reason = f"max_std={max_std:.3f}m, delta={delta_m:.1f}m, cond={cond:.3f}"
    chosen = 'mc_tangent' if is_local else 'mc_ecef'
    return chosen, {'max_std_m': max_std, 'cond': cond, 'delta_m': delta_m, 'reason': reason}


# ============================================================================
# MAIN API FUNCTION
# ============================================================================

def check_geo_prob(
    subject: Union[Subject, Tuple[float, float], GeoPoint],
    reference: Union[Reference, Tuple[float, float], GeoPoint],
    d0_meters: float,
    method_params: Optional[MethodParams] = None
) -> ProbabilityResult:
    """
    Compute probability that distance between two uncertain points is ≤ d0_meters.
    
    Args:
        subject: Subject object or (lat, lon) tuple for first point
        reference: Reference object or (lat, lon) tuple for second point  
        d0_meters: Distance threshold in meters
        method_params: MethodParams object with computation settings including distance_metric
        
    Distance Metrics (set via method_params.distance_metric):
        - HaversineDistance() [default]: Great circle on sphere
        - VincentyDistance(): High-precision WGS84 ellipsoid (requires geopy)
        - ManhattanDistance(): L1 / city-block distance
        - ChebyshevDistance(): L∞ / rectangular distance
        - EuclideanTangentDistance(): L2 in tangent plane
        - lp_distance(p): Generalized Lp norm
        - WeightedDistance(base, cost_func): Cost-weighted distance
        - CustomDistance(func): User-defined distance function
        
    Examples:
        # Default (haversine)
        >>> result = check_geo_prob(subject, reference, d0=100)
        
        # Urban navigation (Manhattan)
        >>> params = MethodParams(distance_metric=ManhattanDistance())
        >>> result = check_geo_prob(subject, reference, d0=100, params)
        
        # Rectangular geofence (Chebyshev)
        >>> params = MethodParams(distance_metric=ChebyshevDistance())
        >>> result = check_geo_prob(subject, reference, d0=100, params)
        
        # Custom weighted distance
        >>> def delivery_cost(lat, lon):
        ...     return 1.5 if in_suburbs(lat, lon) else 1.0
        >>> metric = weighted_haversine(delivery_cost)
        >>> params = MethodParams(distance_metric=metric)
        >>> result = check_geo_prob(subject, reference, d0=5000, params)
    """
    
    if method_params is None:
        method_params = MethodParams()
    
    # Get distance metric from params (defaults to haversine)
    distance_metric = method_params.get_distance_metric()
    
    # Convert to Subject/Reference if needed
    if not isinstance(subject, Subject):
        if isinstance(subject, GeoPoint):
            subject = Subject(subject, Covariance2D(0.0))
        elif isinstance(subject, (tuple, list)) and len(subject) == 3:
            lat, lon, sigma = subject
            subject = Subject(GeoPoint(lat, lon), Covariance2D(sigma))
        elif isinstance(subject, (tuple, list)) and len(subject) == 2:
            subject = Subject(GeoPoint(*subject), Covariance2D(0.0))
        else:
            subject = Subject(GeoPoint(*subject), Covariance2D(0.0))
    
    if not isinstance(reference, Reference):
        if isinstance(reference, GeoPoint):
            reference = Reference(reference, Covariance2D(0.0))
        elif isinstance(reference, (tuple, list)) and len(reference) == 3:
            lat, lon, sigma = reference
            reference = Reference(GeoPoint(lat, lon), Covariance2D(sigma))
        elif isinstance(reference, (tuple, list)) and len(reference) == 2:
            reference = Reference(GeoPoint(*reference), Covariance2D(0.0))
        else:
            reference = Reference(GeoPoint(*reference), Covariance2D(0.0))
    
    # Validate
    if d0_meters < 0:
        raise ValueError("d0_meters must be non-negative")
    if not 0 <= method_params.prob_threshold <= 1:
        raise ValueError("prob_threshold must be in [0,1]")
    if method_params.n_mc < 100:
        raise ValueError("n_mc must be >= 100")

    # Extract coordinates and uncertainties
    mu_P = np.array(subject.mu.to_tuple())
    mu_Q = np.array(reference.mu.to_tuple())
    Sigma_P_mat = _ensure_psd(subject.Sigma.as_matrix())
    Sigma_Q_mat = _ensure_psd(reference.Sigma.as_matrix())
    Sigma_X_m2 = Sigma_P_mat + Sigma_Q_mat

    # Deterministic case
    if np.allclose(Sigma_X_m2, 0.0, atol=1e-14):
        delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        prob = 1.0 if delta <= d0_meters else 0.0
        return ProbabilityResult(
            fulfilled=(prob >= method_params.prob_threshold),
            probability=float(prob),
            method='deterministic',
            mc_stderr=0.0,
            n_samples=0,
            cp_lower=float(prob),
            cp_upper=float(prob),
            sigma_cond=None,
            max_std_m=0.0,
            delta_m=float(delta),
            decision_by='point_estimate'
        )

    # Check if analytic method is available
    is_isotropic = subject.Sigma.is_isotropic() and reference.Sigma.is_isotropic()

    if method_params.mode == 'auto':
        delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        chosen, diag = _choose_method_heuristic(Sigma_X_m2, float(delta))
        if is_isotropic:
            method_to_use = 'analytic'
        else:
            method_to_use = chosen
        mode = method_to_use
    else:
        mode = method_params.mode

    if mode == 'analytic':
        if not is_isotropic:
            raise ValueError("analytic mode requires both uncertainties to be isotropic")
        
        sigma_P = subject.Sigma.max_std()
        sigma_Q = reference.Sigma.max_std()
        return _compute_analytic_rice(mu_P, mu_Q, sigma_P, sigma_Q, d0_meters, 
                                      method_params.prob_threshold, distance_metric)

    if mode == 'mc_ecef':
        return _compute_mc_ecef(
            mu_P, mu_Q, Sigma_P_mat, Sigma_Q_mat, d0_meters,
            method_params.prob_threshold, method_params.n_mc, method_params.batch_size, 
            method_params.random_state, method_params.conservative_decision,
            method_params.use_antithetic, method_params.cp_alpha, distance_metric
        )

    if mode == 'mc_tangent':
        delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        return _compute_mc_tangent(
            mu_P, mu_Q, Sigma_X_m2, d0_meters,
            method_params.prob_threshold, method_params.n_mc, method_params.batch_size, 
            method_params.random_state, method_params.conservative_decision, float(delta),
            method_params.use_antithetic, method_params.cp_alpha, distance_metric
        )

    raise ValueError(f"Unknown mode: {mode}")


def _compute_analytic_rice(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    sigma_P: float,
    sigma_Q: float,
    d0_meters: float,
    prob_threshold: float,
    distance_metric: DistanceMetric = None
) -> ProbabilityResult:
    if distance_metric is None:
        distance_metric = HaversineDistance()
    
    sigma_X = np.sqrt(sigma_P ** 2 + sigma_Q ** 2)
    delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])

    if sigma_X == 0.0:
        prob = 1.0 if delta <= d0_meters else 0.0
    else:
        b = delta / sigma_X
        x = d0_meters / sigma_X
        prob = float(rice.cdf(x, b))

    fulfilled = prob >= prob_threshold
    return ProbabilityResult(
        fulfilled=fulfilled,
        probability=prob,
        method='analytic_rice',
        mc_stderr=None,
        n_samples=None,
        cp_lower=None,
        cp_upper=None,
        sigma_cond=None,
        max_std_m=float(sigma_X),
        delta_m=float(delta),
        decision_by='point_estimate'
    )


def _compute_mc_ecef(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    Sigma_P_mat: np.ndarray,
    Sigma_Q_mat: np.ndarray,
    d0_meters: float,
    prob_threshold: float,
    n_mc: int,
    batch_size: int,
    random_state: Optional[int],
    conservative_decision: bool = True,
    use_antithetic: bool = True,
    cp_alpha: float = 0.05,
    distance_metric: DistanceMetric = None
) -> ProbabilityResult:
    if distance_metric is None:
        distance_metric = HaversineDistance()
    
    rng = np.random.default_rng(random_state)

    mu_P_ecef = latlon_to_ecef(mu_P[0], mu_P[1])
    mu_Q_ecef = latlon_to_ecef(mu_Q[0], mu_Q[1])

    Jp = geodetic_to_ecef_jacobian(mu_P[0], mu_P[1])
    Jq = geodetic_to_ecef_jacobian(mu_Q[0], mu_Q[1])

    phi_p = np.deg2rad(mu_P[0])
    phi_q = np.deg2rad(mu_Q[0])
    A_p = np.array([[0.0, 1.0 / R_EARTH],
                    [1.0 / (R_EARTH * np.cos(phi_p)), 0.0]], dtype=float)
    A_q = np.array([[0.0, 1.0 / R_EARTH],
                    [1.0 / (R_EARTH * np.cos(phi_q)), 0.0]], dtype=float)

    Sigma_P_phi_lambda = _ensure_psd(A_p @ Sigma_P_mat @ A_p.T)
    Sigma_Q_phi_lambda = _ensure_psd(A_q @ Sigma_Q_mat @ A_q.T)

    Sigma_P_ecef = _ensure_psd(Jp @ Sigma_P_phi_lambda @ Jp.T)
    Sigma_Q_ecef = _ensure_psd(Jq @ Sigma_Q_phi_lambda @ Jq.T)

    mu_X_ecef = mu_P_ecef - mu_Q_ecef
    Sigma_X_ecef = _ensure_psd(Sigma_P_ecef + Sigma_Q_ecef)

    # Diagnostics
    Sigma_X_m2 = Sigma_P_mat + Sigma_Q_mat
    eigs_m2 = np.clip(np.linalg.eigvalsh(Sigma_X_m2), a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs_m2.max()))
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')
    delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])

    # Cholesky for sampling
    Sigma_for_chol = Sigma_X_ecef.copy()
    try:
        L = np.linalg.cholesky(Sigma_for_chol)
    except np.linalg.LinAlgError:
        Sigma_for_chol = _add_jitter(Sigma_for_chol, rel=1e-10)
        L = np.linalg.cholesky(Sigma_for_chol)

    count_inside = 0
    total = 0
    n_remaining = n_mc

    while n_remaining > 0:
        n_batch = min(batch_size, n_remaining)
        if use_antithetic:
            if n_batch % 2 == 1 and n_batch > 1:
                n_batch -= 1

            half = max(1, n_batch // 2)
            z = rng.standard_normal((3, half))
            z_pair = np.concatenate([z, -z], axis=1)
            samples_X_ecef = (L @ z_pair).T + mu_X_ecef
            
            if (min(batch_size, n_remaining) % 2 == 1) and (n_remaining >= 1):
                z_last = rng.standard_normal((3, 1))
                samples_last = (L @ z_last).T + mu_X_ecef
                samples_X_ecef = np.vstack([samples_X_ecef, samples_last])
        else:
            z = rng.standard_normal((3, n_batch))
            samples_X_ecef = (L @ z).T + mu_X_ecef

        P_ecef = mu_Q_ecef + samples_X_ecef

        x, y, zc = P_ecef[:, 0], P_ecef[:, 1], P_ecef[:, 2]
        r = np.sqrt(x * x + y * y + zc * zc)
        lat_samples = np.rad2deg(np.arcsin(np.clip(zc / r, -1.0, 1.0)))
        lon_samples = np.rad2deg(np.arctan2(y, x))

        dists = distance_metric(lat_samples, lon_samples, mu_Q[0], mu_Q[1])
        count_inside += int(np.count_nonzero(dists <= d0_meters))
        total += samples_X_ecef.shape[0]
        n_remaining -= samples_X_ecef.shape[0]

    prob = count_inside / total
    mc_stderr = float(np.sqrt(prob * (1 - prob) / total)) if total > 0 else None
    cp_l, cp_u = clopper_pearson(count_inside, total, alpha=cp_alpha)

    if conservative_decision:
        fulfilled = (cp_l >= prob_threshold)
        decision_by = 'cp_lower'
    else:
        fulfilled = (prob >= prob_threshold)
        decision_by = 'point_estimate'

    return ProbabilityResult(
        fulfilled=fulfilled,
        probability=float(prob),
        method='mc_ecef',
        mc_stderr=mc_stderr,
        n_samples=int(total),
        cp_lower=float(cp_l),
        cp_upper=float(cp_u),
        sigma_cond=float(cond),
        max_std_m=float(max_std),
        delta_m=float(delta),
        decision_by=decision_by
    )


def _compute_mc_tangent(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    Sigma_X_m2: np.ndarray,
    d0_meters: float,
    prob_threshold: float,
    n_mc: int,
    batch_size: int,
    random_state: Optional[int],
    conservative_decision: bool,
    delta: float,
    use_antithetic: bool = True,
    cp_alpha: float = 0.05,
    distance_metric: DistanceMetric = None
) -> ProbabilityResult:
    if distance_metric is None:
        distance_metric = HaversineDistance()
    
    rng = np.random.default_rng(random_state)

    lat_center = (mu_P[0] + mu_Q[0]) / 2.0
    scale_to_deg = enu_to_radians_scale(lat_center)
    Sigma_X_deg2 = _ensure_psd(scale_to_deg @ Sigma_X_m2 @ scale_to_deg.T)

    # Diagnostics
    eigs = np.clip(np.linalg.eigvalsh(Sigma_X_m2), a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs.max()))
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')

    # Cholesky
    Sigma_for_chol = Sigma_X_deg2.copy()
    try:
        L2 = np.linalg.cholesky(Sigma_for_chol)
    except np.linalg.LinAlgError:
        Sigma_for_chol = _add_jitter(Sigma_for_chol, rel=1e-10)
        L2 = np.linalg.cholesky(Sigma_for_chol)

    mu_X_deg = np.array(mu_P) - np.array(mu_Q)

    count_inside = 0
    total = 0
    n_remaining = n_mc

    while n_remaining > 0:
        n_batch = min(batch_size, n_remaining)
        if use_antithetic:
            if n_batch % 2 == 1 and n_batch > 1:
                n_batch -= 1
            half = max(1, n_batch // 2)
            z = rng.standard_normal((2, half))
            z_pair = np.concatenate([z, -z], axis=1)
            samples_X_deg = (L2 @ z_pair).T + mu_X_deg
            if (min(batch_size, n_remaining) % 2 == 1) and (n_remaining >= 1):
                z_last = rng.standard_normal((2, 1))
                samples_last = (L2 @ z_last).T + mu_X_deg
                samples_X_deg = np.vstack([samples_X_deg, samples_last])
        else:
            z = rng.standard_normal((2, n_batch))
            samples_X_deg = (L2 @ z).T + mu_X_deg

        samples_P = np.asarray(mu_Q) + samples_X_deg

        dists = distance_metric(samples_P[:, 0], samples_P[:, 1], mu_Q[0], mu_Q[1])
        count_inside += int(np.count_nonzero(dists <= d0_meters))
        total += samples_P.shape[0]
        n_remaining -= samples_P.shape[0]

    prob = count_inside / total
    mc_stderr = float(np.sqrt(prob * (1 - prob) / total))
    cp_l, cp_u = clopper_pearson(count_inside, total, alpha=cp_alpha)

    if conservative_decision:
        fulfilled = (cp_l >= prob_threshold)
        decision_by = 'cp_lower'
    else:
        fulfilled = (prob >= prob_threshold)
        decision_by = 'point_estimate'

    return ProbabilityResult(
        fulfilled=fulfilled,
        probability=float(prob),
        method='mc_tangent',
        mc_stderr=mc_stderr,
        n_samples=int(total),
        cp_lower=float(cp_l),
        cp_upper=float(cp_u),
        sigma_cond=float(cond),
        max_std_m=float(max_std),
        delta_m=float(delta),
        decision_by=decision_by
    )


# ============================================================================
# PROFILER & UTILITIES (extended for v4)
# ============================================================================

class PerformanceProfiler:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[Dict] = []

    def _time_call(self, func, *args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, out

    def test_batch(self,
                        scenarios: List[Scenario],
                        n_repeats: int = 3,
                        default_n_mc: int = 200_000,
                        default_batch: int = 100_000,
                        rng_base_seed: int = 12345) -> List[Dict]:
        """Run fair test batch with structured Scenarios."""
        results = []
        for s_idx, sc in enumerate(scenarios):
            name = sc.name
            n_mc = sc.method_params.n_mc
            batch_size = sc.method_params.batch_size

            acc = {
                'analytic': {'probs': [], 'times': []},
                'mc_ecef': {'probs': [], 'times': [], 'stderrs': [], 'n_samples': []},
                'mc_tangent': {'probs': [], 'times': [], 'stderrs': [], 'n_samples': []}
            }

            has_analytic = sc.sigma_P_scalar is not None and sc.sigma_Q_scalar is not None

            if self.verbose:
                print(f"TEST CASE: {name} — d0={sc.d0} m, n_mc={n_mc} (repeats={n_repeats})")
                if sc.expected_behavior:
                    print(f"Expected: {sc.expected_behavior}")

            for r in range(n_repeats):
                seed = rng_base_seed + 1000 * s_idx + r

                if has_analytic:
                    t_start = time.perf_counter()
                    analytic_params = MethodParams(mode='analytic', distance_metric=sc.method_params.distance_metric)
                    res_analytic = check_geo_prob(sc.subject, sc.reference, sc.d0, analytic_params)
                    t_elapsed = time.perf_counter() - t_start
                    acc['analytic']['probs'].append(res_analytic.probability)
                    acc['analytic']['times'].append(t_elapsed)

                t_start = time.perf_counter()
                mc_params = MethodParams(
                    mode='mc_ecef',
                    n_mc=n_mc,
                    batch_size=batch_size,
                    random_state=seed,
                    distance_metric=sc.method_params.distance_metric
                )
                res_ecef = check_geo_prob(sc.subject, sc.reference, sc.d0, mc_params)
                t_elapsed = time.perf_counter() - t_start
                acc['mc_ecef']['probs'].append(res_ecef.probability)
                acc['mc_ecef']['times'].append(t_elapsed)
                acc['mc_ecef']['stderrs'].append(res_ecef.mc_stderr if res_ecef.mc_stderr is not None else 0.0)
                acc['mc_ecef']['n_samples'].append(res_ecef.n_samples if res_ecef.n_samples is not None else n_mc)
                
                if self.verbose:
                    print(f"    mc_ecef run {r}: P={res_ecef.probability:.6f}, cp_lower={res_ecef.cp_lower:.6f}, time={t_elapsed*1000:.1f} ms")

                t_start = time.perf_counter()
                mc_params = MethodParams(
                    mode='mc_tangent',
                    n_mc=n_mc,
                    batch_size=batch_size,
                    random_state=seed,
                    distance_metric=sc.method_params.distance_metric
                )
                res_tangent = check_geo_prob(sc.subject, sc.reference, sc.d0, mc_params)
                t_elapsed = time.perf_counter() - t_start
                acc['mc_tangent']['probs'].append(res_tangent.probability)
                acc['mc_tangent']['times'].append(t_elapsed)
                acc['mc_tangent']['stderrs'].append(res_tangent.mc_stderr if res_tangent.mc_stderr is not None else 0.0)
                acc['mc_tangent']['n_samples'].append(res_tangent.n_samples if res_tangent.n_samples is not None else n_mc)

            # Summarize
            summary = {'name': name, 'd0': sc.d0, 'n_mc': n_mc}
            if has_analytic:
                a_probs = np.array(acc['analytic']['probs'])
                a_times = np.array(acc['analytic']['times'])
                summary.update({
                    'analytic_prob_mean': float(a_probs.mean()),
                    'analytic_time_ms': float(a_times.mean() * 1000)
                })
            else:
                summary.update({'analytic_prob_mean': None, 'analytic_time_ms': None})

            for m in ('mc_ecef', 'mc_tangent'):
                probs = np.array(acc[m]['probs'])
                times = np.array(acc[m]['times'])
                stderrs = np.array(acc[m]['stderrs'])
                n_samples_arr = np.array(acc[m]['n_samples'])
                summary.update({
                    f'{m}_prob_mean': float(probs.mean()),
                    f'{m}_prob_std': float(probs.std()),
                    f'{m}_time_ms': float(times.mean() * 1000),
                    f'{m}_mc_stderr_mean': float(stderrs.mean()),
                    f'{m}_n_samples_mean': int(n_samples_arr.mean())
                })
            
            self.results.append(summary)
            results.append(summary)
        
        if self.verbose:
            self._print_summary_table(results, scenarios)
        
        return results
    
    def compare_metrics_on_scenario(self, 
                                    scenario: Scenario,
                                    metrics: Optional[Dict[str, DistanceMetric]] = None) -> Dict[str, ProbabilityResult]:
        """
        Run same scenario with different distance metrics.
        
        Args:
            scenario: Base scenario to test
            metrics: Dict of {name: metric} to compare. If None, uses standard set.
            
        Returns:
            Dict of {metric_name: ProbabilityResult}
        """
        if metrics is None:
            metrics = {
                'haversine': HaversineDistance(),
                'manhattan': ManhattanDistance(),
                'chebyshev': ChebyshevDistance(),
                'euclidean': EuclideanTangentDistance()
            }
        
        results = {}
        for name, metric in metrics.items():
            params = MethodParams(
                mode=scenario.method_params.mode,
                n_mc=scenario.method_params.n_mc,
                batch_size=scenario.method_params.batch_size,
                random_state=scenario.method_params.random_state,
                distance_metric=metric
            )
            result = check_geo_prob(scenario.subject, scenario.reference, scenario.d0, params)
            results[name] = result
            
            if self.verbose:
                print(f"  {name:12s}: P={result.probability:.6f}, fulfilled={result.fulfilled}")
        
        return results
    
    def _print_summary_table(self, results: List[Dict], scenarios: List[Scenario]):
        """Print a concise table summary of all results."""
        try:
            from tabulate import tabulate
        except ImportError:
            print("\nFor better table formatting, install tabulate: pip install tabulate")
            self._print_fallback_table(results)
            return
        
        print("\n" + "=" * 100)
        print("RESULTS SUMMARY TABLE")
        print("=" * 100)
        
        rows = []
        for r in results:
            scenario_name = r['name'][:18]
            d0 = r['d0']
            n_mc = r['n_mc']
            
            if r.get('analytic_prob_mean') is not None:
                rows.append([
                    scenario_name,
                    'analytic',
                    f"{r['analytic_prob_mean']:.5f}",
                    '-',
                    '-',
                    f"{r['analytic_time_ms']:.1f}",
                    f"{d0:.0f}m"
                ])
            
            for method in ['mc_ecef', 'mc_tangent']:
                if r.get(f'{method}_prob_mean') is not None:
                    prob_mean = r[f'{method}_prob_mean']
                    mc_stderr = r[f'{method}_mc_stderr_mean']
                    time_ms = r[f'{method}_time_ms']
                    n_samples = r[f'{method}_n_samples_mean']
                    
                    name_display = scenario_name if method == 'mc_ecef' else ''
                    d0_display = f"{d0:.0f}m" if method == 'mc_ecef' else ''
                    
                    rows.append([
                        name_display,
                        method,
                        f"{prob_mean:.5f}",
                        f"{mc_stderr:.5f}" if mc_stderr > 1e-6 else '<1e-6',
                        f"{n_samples:,}" if n_samples > 0 else '-',
                        f"{time_ms:.1f}",
                        d0_display
                    ])
        
        headers = ["Scenario", "Method", "Prob", "StdErr", "Samples", "Time(ms)", "d0"]
        print(tabulate(rows, headers=headers, tablefmt="simple", numalign="right"))
        print("=" * 100)
    
    def _print_fallback_table(self, results: List[Dict]):
        """Fallback table format if tabulate is not available."""
        print("\n" + "=" * 120)
        print("RESULTS SUMMARY")
        print("=" * 120)
        print(f"{'Scenario':<20} {'Method':<10} {'Mean Prob':<10} {'StdErr':<10} {'Samples':<8} {'Time(ms)':<9} {'d0(m)':<8}")
        print("-" * 120)
        
        for r in results:
            scenario_name = r['name'][:20]
            d0 = r['d0']
            
            if r.get('analytic_prob_mean') is not None:
                print(f"{scenario_name:<20} {'analytic':<10} {r['analytic_prob_mean']:<10.6f} {'-':<10} {'-':<8} {r['analytic_time_ms']:<9.1f} {d0:<8.0f}")
            
            for method in ['mc_ecef', 'mc_tangent']:
                if r.get(f'{method}_prob_mean') is not None:
                    name_field = scenario_name if method == 'mc_ecef' else ''
                    prob_mean = r[f'{method}_prob_mean']
                    mc_stderr = r[f'{method}_mc_stderr_mean']
                    time_ms = r[f'{method}_time_ms']
                    n_samples = r[f'{method}_n_samples_mean']
                    d0_field = f"{d0:.0f}" if method == 'mc_ecef' else ''
                    
                    print(f"{name_field:<20} {method:<10} {prob_mean:<10.6f} {mc_stderr:<10.6f} {n_samples:<8,} {time_ms:<9.1f} {d0_field:<8}")
        
        print("=" * 120)

    def estimate_runtime_scaling(self, target_n_mc: int = 200_000) -> Dict[str, float]:
        """Estimate runtime for target_n_mc by linear scaling."""
        estimates = {}
        for r in self.results:
            if r.get('mc_ecef_time_ms') is not None:
                n = r.get('n_mc', 1)
                t_ms = r.get('mc_ecef_time_ms', None)
                if t_ms is not None and n > 0:
                    per_sample_ms = t_ms / n
                    estimates.setdefault('mc_ecef', []).append(per_sample_ms)
            if r.get('mc_tangent_time_ms') is not None:
                n = r.get('n_mc', 1)
                t_ms = r.get('mc_tangent_time_ms', None)
                if t_ms is not None and n > 0:
                    per_sample_ms = t_ms / n
                    estimates.setdefault('mc_tangent', []).append(per_sample_ms)

        final = {}
        for method, arr in estimates.items():
            median_per_sample = float(np.median(arr))
            final[method] = float(median_per_sample * target_n_mc)
        return final

    def summary(self):
        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)
        for r in self.results:
            print(r)
        print("=" * 60)


# ============================================================================
# MAIN SCRIPT - Test scenarios and demonstrations
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import csv
    import os

    parser = argparse.ArgumentParser(description="Run geospatial probability checker v4 with distance metrics.")
    parser.add_argument("--quick", action="store_true", help="Run smaller quick tests (faster).")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per scenario (default 3).")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to write results JSON/CSV.")
    args = parser.parse_args()

    if args.quick:
        default_n_mc = 20_000
        default_batch = 10_000
        n_repeats = max(1, args.repeats)
        print("QUICK mode: using smaller Monte Carlo budgets for fast iteration.")
    else:
        default_n_mc = 200_000
        default_batch = 100_000
        n_repeats = args.repeats
    
    # Helper to create scenarios
    def create_scenario(
        name: str, 
        mu_P: Tuple[float, float], 
        sigma_P: Union[float, np.ndarray],
        mu_Q: Tuple[float, float], 
        sigma_Q: Union[float, np.ndarray],
        d0: float,
        expected_behavior: str = "",
        n_mc: int = default_n_mc,
        distance_metric: Optional[DistanceMetric] = None
    ) -> Scenario:
        """Create a Scenario using the structured format."""
        subject = Subject(
            mu=GeoPoint(*mu_P),
            Sigma=Covariance2D(sigma_P),
            id=f"subject_{name}"
        )
        reference = Reference(
            mu=GeoPoint(*mu_Q),
            Sigma=Covariance2D(sigma_Q),
            id=f"reference_{name}"
        )
        method_params = MethodParams(n_mc=n_mc, batch_size=default_batch, distance_metric=distance_metric)
        
        sigma_P_scalar = None
        sigma_Q_scalar = None
        if np.isscalar(sigma_P):
            sigma_P_scalar = float(sigma_P)
        if np.isscalar(sigma_Q):
            sigma_Q_scalar = float(sigma_Q)
        
        return Scenario(
            name=name,
            subject=subject,
            reference=reference,
            d0=d0,
            method_params=method_params,
            expected_behavior=expected_behavior,
            sigma_P_scalar=sigma_P_scalar,
            sigma_Q_scalar=sigma_Q_scalar
        )

    print("=" * 70)
    print("Geospatial Probability Checker - v4 (pluggable distance metrics)")
    print("=" * 70)
    print(f"mode: {'quick' if args.quick else 'full'}, default_n_mc={default_n_mc}, batch={default_batch}, repeats={n_repeats}")

    # Baseline test scenarios
    scenarios_baseline = [
        create_scenario(
            'isotropic_small_sigma',
            mu_P=(37.7749, -122.4194),
            sigma_P=5.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=3.0,
            d0=20.0,
            expected_behavior="High probability (~97%) - points very close with small uncertainty"
        ),
        create_scenario(
            'isotropic_large_sigma',
            mu_P=(37.7749, -122.4194),
            sigma_P=200.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=150.0,
            d0=500.0,
            expected_behavior="Medium probability (~86%) - large uncertainty but reasonable threshold"
        ),
        create_scenario(
            'deterministic_exact_same_point',
            mu_P=(37.7749, -122.4194),
            sigma_P=0.0,
            mu_Q=(37.7749, -122.4194),
            sigma_Q=0.0,
            d0=1.0,
            expected_behavior="Probability = 1.0 exactly - identical points with no uncertainty"
        ),
        create_scenario(
            'anisotropic_cross_terms',
            mu_P=(37.7749, -122.4194),
            sigma_P=np.array([[400.0, 300.0], [300.0, 250.0]]),
            mu_Q=(37.7755, -122.4185),
            sigma_Q=np.array([[100.0, -20.0], [-20.0, 80.0]]),
            d0=100.0,
            expected_behavior="Medium probability (~44%) - large anisotropic uncertainty"
        ),
    ]

    # Metric comparison scenarios
    scenarios_metric_tests = [
        create_scenario(
            'manhattan_urban_grid',
            mu_P=(40.7580, -73.9855),  # Times Square
            sigma_P=10.0,
            mu_Q=(40.7614, -73.9776),  # ~5 blocks N, 1 block E
            sigma_Q=10.0,
            d0=500.0,
            expected_behavior="Manhattan ~900m walk (blocks), haversine ~650m (crow)",
            distance_metric=HaversineDistance()  # Will compare multiple
        ),
        create_scenario(
            'chebyshev_rectangular_fence',
            mu_P=(37.7749, -122.4194),
            sigma_P=5.0,
            mu_Q=(37.7749, -122.4194),  # Same center
            sigma_Q=0.0,
            d0=50.0,
            expected_behavior="Chebyshev allows ±50m in BOTH directions (square fence)",
            distance_metric=ChebyshevDistance()
        ),
        create_scenario(
            'lp_norm_comparison',
            mu_P=(51.5074, -0.1278),  # London
            sigma_P=20.0,
            mu_Q=(51.5080, -0.1270),  # ~70m away
            sigma_Q=15.0,
            d0=100.0,
            expected_behavior="Compare L1, L2, L∞ norms on same geometry",
            distance_metric=lp_distance(2)
        ),
    ]

    # Create output directory
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    profiler = PerformanceProfiler(verbose=True)

    # Run baseline scenarios
    print("\n" + "=" * 70)
    print("BASELINE SCENARIOS")
    print("=" * 70)
    t_start_all = time.perf_counter()
    results_baseline = profiler.test_batch(scenarios_baseline, n_repeats=n_repeats)
    t_baseline = time.perf_counter() - t_start_all

    # Run metric comparison tests if requested
    print("\n" + "=" * 70)
    print("METRIC COMPARISON TESTS")
    print("=" * 70)
    
    for scenario in scenarios_metric_tests:
        print(f"\n--- {scenario.name} ---")
        print(f"Expected: {scenario.expected_behavior}")
        
        metrics_to_compare = {
            'haversine': HaversineDistance(),
            'manhattan': ManhattanDistance(),
            'euclidean': EuclideanTangentDistance(),
            'l3_norm': lp_distance(3),
            'chebyshev': ChebyshevDistance()
        }
        
        metric_results = profiler.compare_metrics_on_scenario(scenario, metrics_to_compare)
        
        # Print comparison table
        print("\nMetric Comparison:")
        print(f"{'Metric':<15} {'Probability':<12} {'Fulfilled':<10} {'Method':<12}")
        print("-" * 50)
        for metric_name, result in metric_results.items():
            print(f"{metric_name:<15} {result.probability:<12.6f} {str(result.fulfilled):<10} {result.method:<12}")
    
    # Demonstrate weighted distance
    print("\n" + "=" * 70)
    print("WEIGHTED DISTANCE EXAMPLE")
    print("=" * 70)
    
    def delivery_zone_cost(lat: float, lon: float) -> float:
        """Example cost function: downtown vs suburbs"""
        # Simplistic: close to (37.7749, -122.4194) = downtown = 1.0x
        # Far = suburbs = 1.5x cost
        dist_from_downtown = np.sqrt((lat - 37.7749)**2 + (lon + 122.4194)**2)
        if dist_from_downtown < 0.01:  # ~1km radius
            return 1.0
        else:
            return 1.5
    
    weighted_metric = weighted_haversine(delivery_zone_cost)
    scenario_weighted = create_scenario(
        'weighted_delivery_cost',
        mu_P=(37.7749, -122.4194),  # Downtown
        sigma_P=50.0,
        mu_Q=(37.7850, -122.4100),  # Suburbs (~1.5km away)
        sigma_Q=30.0,
        d0=2000.0,  # 2km "cost-distance" threshold
        expected_behavior="Weighted distance accounts for delivery zones",
        distance_metric=weighted_metric
    )
    
    result_weighted = check_geo_prob(
        scenario_weighted.subject,
        scenario_weighted.reference,
        scenario_weighted.d0,
        scenario_weighted.method_params
    )
    print(f"Weighted distance result: P={result_weighted.probability:.6f}, fulfilled={result_weighted.fulfilled}")
    
    # Compare to unweighted
    scenario_unweighted = create_scenario(
        'unweighted_baseline',
        mu_P=(37.7749, -122.4194),
        sigma_P=50.0,
        mu_Q=(37.7850, -122.4100),
        sigma_Q=30.0,
        d0=2000.0,
        distance_metric=HaversineDistance()
    )
    result_unweighted = check_geo_prob(
        scenario_unweighted.subject,
        scenario_unweighted.reference,
        scenario_unweighted.d0,
        scenario_unweighted.method_params
    )
    print(f"Unweighted (haversine): P={result_unweighted.probability:.6f}, fulfilled={result_unweighted.fulfilled}")
    print(f"Difference: {abs(result_weighted.probability - result_unweighted.probability):.6f}")

    t_total = time.perf_counter() - t_start_all

    # Save results
    all_results = results_baseline
    
    json_path = os.path.join(outdir, f"profiler_results_v4_{int(time.time())}.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved JSON summary to: {json_path}")

    keys = set()
    for r in all_results:
        keys.update(r.keys())
    keys = sorted(keys)

    csv_path = os.path.join(outdir, f"profiler_results_v4_{int(time.time())}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"Saved CSV summary to: {csv_path}")

    estimated = profiler.estimate_runtime_scaling(target_n_mc=default_n_mc)
    print("\nEstimated runtime scaling (ms) for target n_mc:")
    print(estimated)

    print(f"\nTotal profiling wall time: {t_total:.2f} s")

    print("\nDone!")
