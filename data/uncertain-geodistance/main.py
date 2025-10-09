# geospatial_prob_checker_v3.py - Clean API with structured data types
import time
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, Literal, Dict, List, Any

import numpy as np
from scipy.stats import rice, beta

# Earth radius in meters (mean radius for spherical approximation)
R_EARTH = 6371000.0


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
            'cp_alpha': self.cp_alpha
        }


@dataclass
class Scenario:
    """Structured scenario for profiling."""
    name: str
    subject: Subject
    reference: Reference
    d0: float
    method_params: MethodParams = field(default_factory=MethodParams)
    
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
    # Diagnostic fields:
    cp_lower: Optional[float] = None
    cp_upper: Optional[float] = None
    sigma_cond: Optional[float] = None
    max_std_m: Optional[float] = None
    delta_m: Optional[float] = None
    decision_by: Optional[str] = None


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
    """
    Conservative heuristic to decide if tangent-plane MC is acceptable.

    Returns:
        method: 'mc_tangent' or 'mc_ecef'
        diagnostics: dict with 'max_std_m', 'cond', 'delta_m', 'reason'
    """
    eigs = np.linalg.eigvalsh(Sigma_X_m2)
    eigs = np.clip(eigs, a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs.max()))
    # condition number - handle rank-deficient / zero carefully
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')

    is_local = (max_std < max_std_thresh) and (delta_m < delta_thresh) and (cond < cond_thresh)

    reason = f"max_std={max_std:.3f}m, delta={delta_m:.1f}m, cond={cond:.3f}"
    chosen = 'mc_tangent' if is_local else 'mc_ecef'
    return chosen, {'max_std_m': max_std, 'cond': cond, 'delta_m': delta_m, 'reason': reason}


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
        method_params: MethodParams object with computation settings
    """
    
    if method_params is None:
        method_params = MethodParams()
    
    # Convert to Subject/Reference if needed
    if not isinstance(subject, Subject):
        if isinstance(subject, GeoPoint):
            subject = Subject(subject, Covariance2D(0.0))
        elif isinstance(subject, (tuple, list)) and len(subject) == 3:
            # Handle (lat, lon, sigma) format
            lat, lon, sigma = subject
            subject = Subject(GeoPoint(lat, lon), Covariance2D(sigma))
        elif isinstance(subject, (tuple, list)) and len(subject) == 2:
            # Handle (lat, lon) format
            subject = Subject(GeoPoint(*subject), Covariance2D(0.0))
        else:
            subject = Subject(GeoPoint(*subject), Covariance2D(0.0))
    
    if not isinstance(reference, Reference):
        if isinstance(reference, GeoPoint):
            reference = Reference(reference, Covariance2D(0.0))
        elif isinstance(reference, (tuple, list)) and len(reference) == 3:
            # Handle (lat, lon, sigma) format
            lat, lon, sigma = reference
            reference = Reference(GeoPoint(lat, lon), Covariance2D(sigma))
        elif isinstance(reference, (tuple, list)) and len(reference) == 2:
            # Handle (lat, lon) format
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
        delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
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
        # compute delta for heuristic
        delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
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
        
        # Extract scalar sigmas
        sigma_P = subject.Sigma.max_std()
        sigma_Q = reference.Sigma.max_std()
        return _compute_analytic_rice(mu_P, mu_Q, sigma_P, sigma_Q, d0_meters, method_params.prob_threshold)

    if mode == 'mc_ecef':
        return _compute_mc_ecef(
            mu_P, mu_Q, Sigma_P_mat, Sigma_Q_mat, d0_meters,
            method_params.prob_threshold, method_params.n_mc, method_params.batch_size, 
            method_params.random_state, method_params.conservative_decision,
            method_params.use_antithetic, method_params.cp_alpha
        )

    if mode == 'mc_tangent':
        # compute delta for diagnostics
        delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        return _compute_mc_tangent(
            mu_P, mu_Q, Sigma_X_m2, d0_meters,
            method_params.prob_threshold, method_params.n_mc, method_params.batch_size, 
            method_params.random_state, method_params.conservative_decision, float(delta),
            method_params.use_antithetic, method_params.cp_alpha
        )

    raise ValueError(f"Unknown mode: {mode}")


def _compute_analytic_rice(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    sigma_P: float,
    sigma_Q: float,
    d0_meters: float,
    prob_threshold: float
) -> ProbabilityResult:
    sigma_X = np.sqrt(sigma_P ** 2 + sigma_Q ** 2)
    delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])

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
    cp_alpha: float = 0.05
) -> ProbabilityResult:
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

    # diagnostics for return
    Sigma_X_m2 = Sigma_P_mat + Sigma_Q_mat
    eigs_m2 = np.clip(np.linalg.eigvalsh(Sigma_X_m2), a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs_m2.max()))
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')
    delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])

    # Precompute Cholesky L for 3x3 ECEF covariance to speed sampling
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
            # make n_batch even for pairing where possible
            if n_batch % 2 == 1 and n_batch > 1:
                n_batch -= 1

            half = max(1, n_batch // 2)
            z = rng.standard_normal((3, half))
            z_pair = np.concatenate([z, -z], axis=1)  # shape (3, 2*half)
            samples_X_ecef = (L @ z_pair).T + mu_X_ecef  # shape (n_batch, 3)
            # if we reduced n_batch for pairing, and there are leftover slots overall, generate an odd sample
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

        dists = haversine_distance_m(lat_samples, lon_samples, mu_Q[0], mu_Q[1])
        count_inside += int(np.count_nonzero(dists <= d0_meters))
        total += samples_X_ecef.shape[0]
        n_remaining -= samples_X_ecef.shape[0]

    prob = count_inside / total
    mc_stderr = float(np.sqrt(prob * (1 - prob) / total)) if total > 0 else None
    cp_l, cp_u = clopper_pearson(count_inside, total, alpha=cp_alpha)

    # conservative decision option: use cp_lower vs point estimate
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
    cp_alpha: float = 0.05
) -> ProbabilityResult:
    rng = np.random.default_rng(random_state)

    lat_center = (mu_P[0] + mu_Q[0]) / 2.0
    scale_to_deg = enu_to_radians_scale(lat_center)
    Sigma_X_deg2 = _ensure_psd(scale_to_deg @ Sigma_X_m2 @ scale_to_deg.T)

    # diagnostics
    eigs = np.clip(np.linalg.eigvalsh(Sigma_X_m2), a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs.max()))
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')

    # Precompute cholesky on the 2x2 Sigma_X_deg2
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

        dists = haversine_distance_m(samples_P[:, 0], samples_P[:, 1], mu_Q[0], mu_Q[1])
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


# ----------------- Profiler & runtime estimator -----------------
class PerformanceProfiler:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[Dict] = []

    def _time_call(self, func, *args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, out

    def fair_test_batch(self,
                        scenarios: List[Scenario],
                        n_repeats: int = 3,
                        default_n_mc: int = 200_000,
                        default_batch: int = 100_000,
                        rng_base_seed: int = 12345) -> List[Dict]:
        """
        Run fair test batch with structured Scenarios.
        """
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
                print(f"\nFAIR TEST: {name} — d0={sc.d0} m, n_mc={n_mc} (repeats={n_repeats})")

            for r in range(n_repeats):
                seed = rng_base_seed + 1000 * s_idx + r

                if has_analytic:
                    t_start = time.perf_counter()
                    analytic_params = MethodParams(mode='analytic')
                    res_analytic = check_geo_prob(sc.subject, sc.reference, sc.d0, analytic_params)
                    t_elapsed = time.perf_counter() - t_start
                    acc['analytic']['probs'].append(res_analytic.probability)
                    acc['analytic']['times'].append(t_elapsed)

                t_start = time.perf_counter()
                mc_params = MethodParams(
                    mode='mc_ecef',
                    n_mc=n_mc,
                    batch_size=batch_size,
                    random_state=seed
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
                    random_state=seed
                )
                res_tangent = check_geo_prob(sc.subject, sc.reference, sc.d0, mc_params)
                t_elapsed = time.perf_counter() - t_start
                acc['mc_tangent']['probs'].append(res_tangent.probability)
                acc['mc_tangent']['times'].append(t_elapsed)
                acc['mc_tangent']['stderrs'].append(res_tangent.mc_stderr if res_tangent.mc_stderr is not None else 0.0)
                acc['mc_tangent']['n_samples'].append(res_tangent.n_samples if res_tangent.n_samples is not None else n_mc)

            # summarize
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
            
            if self.verbose:
                print("\n  Results summary:")
                if has_analytic:
                    print(f"    Analytic (mean): P={summary['analytic_prob_mean']:.6f}, time={summary['analytic_time_ms']:.2f} ms")
                for m in ('mc_ecef', 'mc_tangent'):
                    print(f"    {m}: P={summary[f'{m}_prob_mean']:.6f} ± {summary[f'{m}_prob_std']:.6f}, "
                          f"stderr_mean={summary[f'{m}_mc_stderr_mean']:.6f}, time={summary[f'{m}_time_ms']:.1f} ms")
            
            self.results.append(summary)
            results.append(summary)
        return results

    def estimate_runtime_scaling(self, target_n_mc: int = 200_000) -> Dict[str, float]:
        """
        Estimate runtime for target_n_mc by linear scaling with observed per-sample times
        stored in self.results. Returns map method -> estimated_ms.
        """
        estimates = {}
        # find rows with mc_ecef_time_ms and n_mc info
        for r in self.results:
            if r.get('mc_ecef_time_ms') is not None:
                # compute per-sample ms
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


# ----------------- Example run configuration -----------------
if __name__ == "__main__":
    import argparse
    import json
    import csv
    import os

    parser = argparse.ArgumentParser(description="Run geospatial probability checker with clean structured API.")
    parser.add_argument("--quick", action="store_true", help="Run smaller quick tests (faster).")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per scenario (default 3).")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to write results JSON/CSV.")
    args = parser.parse_args()

    # Quick mode: smaller MC budget for iterative dev
    if args.quick:
        default_n_mc = 20_000
        default_batch = 10_000
        n_repeats = max(1, args.repeats)
        print("QUICK mode: using smaller Monte Carlo budgets for fast iteration.")
    else:
        default_n_mc = 200_000
        default_batch = 100_000
        n_repeats = args.repeats

    print("=" * 70)
    print("Geospatial Probability Checker - v3 (clean structured API)")
    print("=" * 70)
    print(f"mode: {'quick' if args.quick else 'full'}, default_n_mc={default_n_mc}, batch={default_batch}, repeats={n_repeats}")

    # Helper function to create scenarios with new structured format
    def create_scenario(name: str, 
                       mu_P: Tuple[float, float], 
                       sigma_P: Union[float, np.ndarray],
                       mu_Q: Tuple[float, float], 
                       sigma_Q: Union[float, np.ndarray],
                       d0: float,
                       n_mc: int = default_n_mc) -> Scenario:
        """Create a Scenario using the new structured format."""
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
        method_params = MethodParams(n_mc=n_mc, batch_size=default_batch)
        
        # For backward compatibility with analytics, store scalar sigmas if isotropic
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
            sigma_P_scalar=sigma_P_scalar,
            sigma_Q_scalar=sigma_Q_scalar
        )

    # Helper function to create identity matrix  
    def I(sigma: float) -> np.ndarray:
        return np.eye(2) * (sigma ** 2)

    # Create structured scenarios using new API
    scenarios = [
        # --- Isotropic / analytic-available (small / large sigma) ---
        create_scenario(
            'isotropic_small_sigma',
            mu_P=(37.7749, -122.4194),
            sigma_P=5.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=3.0,
            d0=20.0
        ),
        create_scenario(
            'isotropic_large_sigma',
            mu_P=(37.7749, -122.4194),
            sigma_P=200.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=150.0,
            d0=500.0
        ),

        # --- Deterministic / degenerate ---
        create_scenario(
            'deterministic_exact_same_point',
            mu_P=(37.7749, -122.4194),
            sigma_P=0.0,
            mu_Q=(37.7749, -122.4194),
            sigma_Q=0.0,
            d0=1.0
        ),
        create_scenario(
            'deterministic_far_apart',
            mu_P=(0.0, 0.0),
            sigma_P=0.0,
            mu_Q=(10.0, 10.0),
            sigma_Q=0.0,
            d0=100.0
        ),

        # --- One uncertain, one exact ---
        create_scenario(
            'one_uncertain_small_sigma',
            mu_P=(51.5074, -0.1278),   # London
            sigma_P=2.0,
            mu_Q=(51.5074, -0.1278),
            sigma_Q=0.0,
            d0=10.0
        ),

        # --- Anisotropic / cross-terms ---
        create_scenario(
            'anisotropic_cross_terms',
            mu_P=(37.7749, -122.4194),
            sigma_P=np.array([[400.0, 300.0], [300.0, 250.0]]),
            mu_Q=(37.7755, -122.4185),
            sigma_Q=np.array([[100.0, -20.0], [-20.0, 80.0]]),
            d0=100.0
        ),
        create_scenario(
            'anisotropic_high_condition',
            mu_P=(37.7749, -122.4194),
            sigma_P=np.array([[1e6, 9.999e5], [9.999e5, 1e6]]),  # nearly singular
            mu_Q=(37.7750, -122.4195),
            sigma_Q=1.0,
            d0=1000.0
        ),

        # --- Geographic edge cases ---
        create_scenario(
            'near_north_pole',
            mu_P=(89.999, 0.0),
            sigma_P=10.0,
            mu_Q=(89.998, 10.0),
            sigma_Q=10.0,
            d0=500.0
        ),
        create_scenario(
            'longitude_wrap',
            mu_P=(0.0, 179.9),
            sigma_P=10.0,
            mu_Q=(0.0, -179.9),
            sigma_Q=10.0,
            d0=500.0
        ),
        create_scenario(
            'equator_vs_high_lat',
            mu_P=(0.0, 30.0),
            sigma_P=50.0,
            mu_Q=(60.0, 30.0),
            sigma_Q=50.0,
            d0=5000.0
        ),

        # --- extreme d0 values & large uncertainty ---
        create_scenario(
            'tiny_d0_tight_uncertainty',
            mu_P=(37.7749, -122.4194),
            sigma_P=1.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=1.0,
            d0=1.0
        ),
        create_scenario(
            'huge_d0_large_uncertainty',
            mu_P=(37.7749, -122.4194),
            sigma_P=1000.0,
            mu_Q=(38.0, -122.0),
            sigma_Q=1000.0,
            d0=10_000.0
        ),

        # --- mixed precision: Q exact, P uncertain (useful in localization) ---
        create_scenario(
            'reference_exact_many_uncertain',
            mu_P=(40.7128, -74.0060),  # NYC
            sigma_P=np.array([[25.0, 5.0], [5.0, 9.0]]),
            mu_Q=(40.7128, -74.0060),
            sigma_Q=0.0,
            d0=50.0
        ),

        # --- sensitivity sweep (small set) for sigma scaling ---
        create_scenario(
            'sensitivity_sigma_1m',
            mu_P=(34.0522, -118.2437),  # Los Angeles
            sigma_P=1.0,
            mu_Q=(34.05225, -118.24375),
            sigma_Q=1.0,
            d0=5.0
        ),
        create_scenario(
            'sensitivity_sigma_100m',
            mu_P=(34.0522, -118.2437),
            sigma_P=100.0,
            mu_Q=(34.05225, -118.24375),
            sigma_Q=50.0,
            d0=50.0
        )
    ]

    # create output directory if missing
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    profiler = PerformanceProfiler(verbose=True)

    t_start_all = time.perf_counter()
    results = profiler.fair_test_batch(scenarios, n_repeats=n_repeats)
    t_total = time.perf_counter() - t_start_all

    # save JSON
    json_path = os.path.join(outdir, f"profiler_results_v3_{int(time.time())}.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved JSON summary to: {json_path}")

    # save CSV: union of keys across result dicts
    keys = set()
    for r in results:
        keys.update(r.keys())
    keys = sorted(keys)

    csv_path = os.path.join(outdir, f"profiler_results_v3_{int(time.time())}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"Saved CSV summary to: {csv_path}")

    estimated = profiler.estimate_runtime_scaling(target_n_mc=default_n_mc)
    print("\nEstimated runtime scaling (ms) for target n_mc based on collected runs:")
    print(estimated)

    print(f"\nTotal profiling wall time: {t_total:.2f} s")
    print("Done.")