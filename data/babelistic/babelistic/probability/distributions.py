# ============================================================================
# UNCERTAINTY DISTRIBUTION IMPLEMENTATIONS
# ============================================================================

from typing import Tuple, List, Optional

import numpy as np

from ..base import UncertaintyDistribution


class GaussianDistribution(UncertaintyDistribution):
    """Multivariate Gaussian"""
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self._mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        
        # Precompute for efficiency
        from scipy.linalg import cholesky
        self._chol = cholesky(cov, lower=True)
        self._cov_inv = np.linalg.inv(cov)
        self._det = np.linalg.det(cov)
        self._dim = len(mean)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Gaussian PDF"""
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        diff = x_flat - self._mean
        exponent = -0.5 * np.sum(diff @ self._cov_inv * diff, axis=-1)
        norm = 1.0 / np.sqrt((2*np.pi)**self._dim * self._det)
        result = norm * np.exp(exponent)

        out = result.reshape(original_shape)
        # For scalar inputs return a Python float rather than a 0-d numpy array
        if isinstance(out, np.ndarray) and out.shape == ():
            return float(out.item())
        return out
    
    def sample(self, n: int) -> np.ndarray:
        """Generate samples"""
        z = np.random.randn(n, self._dim)
        return self._mean + z @ self._chol.T
    
    def mean(self) -> np.ndarray:
        return self._mean


class UniformDistribution(UncertaintyDistribution):
    """Uniform distribution over box"""
    
    def __init__(self, bounds: Tuple):
        if len(bounds) == 4:  # 2D
            self._bounds = np.array([[bounds[0], bounds[2]], 
                                     [bounds[1], bounds[3]]])
        else:
            self._bounds = np.array(bounds).reshape(2, -1)
        
        self._range = self._bounds[1] - self._bounds[0]
        self._volume = np.prod(self._range)
        self._mean = np.mean(self._bounds, axis=0)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Constant inside bounds"""
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        inside = np.all((x_flat >= self._bounds[0]) & (x_flat <= self._bounds[1]), axis=-1)
        result = inside.astype(float) / self._volume
        
        return result.reshape(original_shape)
    
    def sample(self, n: int) -> np.ndarray:
        """Uniform samples"""
        return self._bounds[0] + np.random.rand(n, len(self._range)) * self._range
    
    def mean(self) -> np.ndarray:
        return self._mean


class MixtureDistribution(UncertaintyDistribution):
    """Mixture of distributions"""
    
    def __init__(self, components: List[UncertaintyDistribution], 
                 weights: np.ndarray):
        self.components = components
        self.weights = np.asarray(weights)
        self.weights /= self.weights.sum()
        
        self._mean = sum(w * c.mean() for w, c in zip(self.weights, components))
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Mixture PDF"""
        result = np.zeros(x.shape[:-1])
        for w, c in zip(self.weights, self.components):
            result = result + w * c.pdf(x)
        return result
    
    def sample(self, n: int) -> np.ndarray:
        """Sample from mixture"""
        component_idx = np.random.choice(len(self.components), n, p=self.weights)
        samples = []
        for i, comp in enumerate(self.components):
            n_i = np.sum(component_idx == i)
            if n_i > 0:
                samples.append(comp.sample(n_i))
        return np.vstack(samples)
    
    def mean(self) -> np.ndarray:
        return self._mean


class StudentTDistribution(UncertaintyDistribution):
    """
    Multivariate Student-t distribution
    
    Properties:
        - Heavy tails compared to Gaussian
        - Controlled by degrees of freedom (df)
        - df → ∞ converges to Gaussian
        - Robust to outliers
    
    Mathematical definition:
        p(x) ∝ [1 + (x-μ)ᵀΣ⁻¹(x-μ)/df]^(-(df+d)/2)
    
    Use cases:
        - When outliers are expected
        - Financial modeling (fat tails)
        - Robust statistics
        - Conservative uncertainty estimates
    
    Examples:
        >>> # Heavy tails (df=3)
        >>> dist = StudentTDistribution(mean=[0, 0], cov=[[1, 0], [0, 1]], df=3)
        >>> 
        >>> # Nearly Gaussian (df=30)
        >>> dist = StudentTDistribution(mean=[0, 0], cov=[[1, 0], [0, 1]], df=30)
    """
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray, df: float = 5.0):
        """
        Parameters
        ----------
        mean : np.ndarray
            Mean vector
        cov : np.ndarray
            Covariance matrix
        df : float
            Degrees of freedom (>0). Lower = heavier tails
        """
        self._mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.df = df
        self._dim = len(mean)
        
        if df <= 0:
            raise ValueError(f"df must be positive, got {df}")
        
        # Precompute for efficiency
        from scipy.linalg import cholesky
        self._chol = cholesky(cov, lower=True)
        self._cov_inv = np.linalg.inv(cov)
        self._det = np.linalg.det(cov)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Student-t PDF"""
        from scipy.special import gamma
        
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        diff = x_flat - self._mean
        
        # Mahalanobis distance squared
        mahal_sq = np.sum(diff @ self._cov_inv * diff, axis=-1)
        
        # Student-t normalization constant
        nu = self.df
        d = self._dim
        
        norm_const = (gamma((nu + d) / 2.0) / 
                     (gamma(nu / 2.0) * (nu * np.pi)**(d/2.0) * np.sqrt(self._det)))
        
        # Student-t PDF
        result = norm_const * (1 + mahal_sq / nu)**(-(nu + d) / 2.0)
        
        return result.reshape(original_shape)
    
    def sample(self, n: int) -> np.ndarray:
        """Generate samples from multivariate Student-t"""
        # Student-t = Gaussian / sqrt(chi-square/df)
        # Sample from Gaussian
        z = np.random.randn(n, self._dim)
        gaussian_samples = self._mean + z @ self._chol.T
        
        # Sample chi-square scaling
        chi2_samples = np.random.chisquare(self.df, n)
        scale = np.sqrt(self.df / chi2_samples)
        
        # Apply scaling
        return self._mean + (gaussian_samples - self._mean) * scale[:, None]
    
    def mean(self) -> np.ndarray:
        return self._mean


class LogNormalDistribution(UncertaintyDistribution):
    """
    Log-normal distribution: X = exp(Y) where Y ~ Normal(μ, Σ)
    
    Properties:
        - Always positive
        - Right-skewed (long tail)
        - Multiplicative processes
        - Geometric interpretation
    
    Mathematical definition:
        If Y ~ N(μ, Σ), then X = exp(Y) ~ LogNormal
    
    Use cases:
        - Distances (always positive)
        - Multiplicative errors
        - Financial returns
        - Growth processes
        - When zeros/negatives impossible
    
    Examples:
        >>> # Log-normal with median at [1, 1]
        >>> dist = LogNormalDistribution(mean=[0, 0], cov=[[0.1, 0], [0, 0.1]])
        >>> # median = exp(mean) = [1, 1]
    """
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Parameters
        ----------
        mean : np.ndarray
            Mean of underlying Gaussian (NOT mean of log-normal!)
        cov : np.ndarray
            Covariance of underlying Gaussian
        """
        self._log_mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self._dim = len(mean)
        
        from scipy.linalg import cholesky
        self._chol = cholesky(cov, lower=True)
        self._cov_inv = np.linalg.inv(cov)
        self._det = np.linalg.det(cov)
        
        # True mean of log-normal: exp(μ + σ²/2)
        self._true_mean = np.exp(self._log_mean + np.diag(cov) / 2.0)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Log-normal PDF"""
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Check for non-positive values
        valid_mask = np.all(x_flat > 0, axis=-1)
        result = np.zeros(len(x_flat))
        
        if not valid_mask.any():
            return result.reshape(original_shape)
        
        x_valid = x_flat[valid_mask]
        log_x = np.log(x_valid)
        
        # PDF of underlying Gaussian
        diff = log_x - self._log_mean
        exponent = -0.5 * np.sum(diff @ self._cov_inv * diff, axis=-1)
        
        norm = 1.0 / np.sqrt((2*np.pi)**self._dim * self._det)
        
        # Jacobian: product of 1/x_i
        jacobian = 1.0 / np.prod(x_valid, axis=-1)
        
        result[valid_mask] = norm * np.exp(exponent) * jacobian
        
        return result.reshape(original_shape)
    
    def sample(self, n: int) -> np.ndarray:
        """Generate samples: exp(Gaussian samples)"""
        z = np.random.randn(n, self._dim)
        gaussian_samples = self._log_mean + z @ self._chol.T
        return np.exp(gaussian_samples)
    
    def mean(self) -> np.ndarray:
        """True mean of log-normal distribution"""
        return self._true_mean


class EmpiricalDistribution(UncertaintyDistribution):
    """
    Empirical distribution from samples (non-parametric)
    
    Properties:
        - No parametric assumptions
        - Uses kernel density estimation (KDE) for PDF
        - Exact for sampling (just resample)
        - Data-driven
    
    Use cases:
        - When you have measurement data but no model
        - Arbitrary/complex distributions
        - Historical data-based forecasting
        - Bootstrap-style uncertainty
    
    Examples:
        >>> # From GPS measurements
        >>> measurements = np.random.randn(1000, 2) * 10  # 1000 GPS fixes
        >>> dist = EmpiricalDistribution(measurements, bandwidth=5.0)
    """
    
    def __init__(self, samples: np.ndarray, bandwidth: Optional[float] = None):
        """
        Parameters
        ----------
        samples : np.ndarray (n_samples, n_dimensions)
            Empirical samples
        bandwidth : float, optional
            KDE bandwidth. If None, uses Scott's rule
        """
        self.samples = np.asarray(samples)
        if self.samples.ndim == 1:
            self.samples = self.samples[:, None]
        
        self._n_samples, self._dim = self.samples.shape
        self._mean = np.mean(samples, axis=0)
        
        # Set bandwidth (Scott's rule: n^(-1/(d+4)))
        if bandwidth is None:
            self.bandwidth = self._n_samples ** (-1.0 / (self._dim + 4))
            self.bandwidth *= np.std(samples)
        else:
            self.bandwidth = bandwidth
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate PDF using Gaussian KDE
        
        p(x) ≈ (1/n) Σ K((x - x_i) / h)
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Gaussian kernel KDE
        result = np.zeros(len(x_flat))
        
        h = self.bandwidth
        norm = 1.0 / ((2 * np.pi * h**2)**(self._dim / 2.0))
        
        for i, x_query in enumerate(x_flat):
            # Distance to all samples
            diff = self.samples - x_query
            dist_sq = np.sum(diff**2, axis=1)
            
            # Sum of kernels
            kernel_sum = np.sum(np.exp(-dist_sq / (2 * h**2)))
            result[i] = norm * kernel_sum / self._n_samples
        
        return result.reshape(original_shape)
    
    def sample(self, n: int) -> np.ndarray:
        """Resample from empirical distribution (with replacement)"""
        idx = np.random.choice(self._n_samples, size=n, replace=True)
        
        # Add small jitter for continuous approximation
        jitter = np.random.randn(n, self._dim) * self.bandwidth * 0.1
        
        return self.samples[idx] + jitter
    
    def mean(self) -> np.ndarray:
        return self._mean


class RegionDistribution(UncertaintyDistribution):
    """
    Treat a region as a uniform distribution over its interior.
    
    This allows region-to-region queries like:
    "What's the probability that a random point in region A
     is within distance D of region B?"
    """
    
    def __init__(self, region, bounds, n_samples=10000):
        self.region = region
        # Sample points uniformly from the region
        self.samples = self._sample_region_uniform(region, bounds, n_samples)
    
    def _sample_region_uniform(self, region, bounds, n):
        # Rejection sampling
        samples = []
        while len(samples) < n:
            # Sample uniformly from bounding box
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[2], bounds[3])
            point = np.array([x, y])
            
            # Accept if inside region
            if region.indicator(point) > 0.5:
                samples.append(point)
        
        return np.array(samples)
    
    def pdf(self, x):
        # Uniform over region interior
        indicator_vals = self.region.indicator(x)
        # Normalize by region area (approximate)
        area = len(self.samples) / 10000  # rough estimate
        return indicator_vals / area
    
    def sample(self, n):
        # Resample from cached samples
        idx = np.random.choice(len(self.samples), n, replace=True)
        return self.samples[idx]
    
    def mean(self):
        return np.mean(self.samples, axis=0)