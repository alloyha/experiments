# ============================================================================
# KERNEL IMPLEMENTATIONS
# ============================================================================

from typing import Protocol

import numpy as np

from .base import MetricSpace, Region, UncertaintyDistribution, Kernel

class GaussianKernel(Kernel):
    """Gaussian kernel: exp(-d²/(2h²))"""
    
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        return np.exp(-0.5 * (distance / bandwidth)**2)
    
    def support_radius(self, bandwidth: float) -> float:
        return 3.0 * bandwidth  # 3-sigma cutoff
    
    def is_compact(self) -> bool:
        return False  # Gaussian has infinite support (but we truncate)


class EpanechnikovKernel(Kernel):
    """Epanechnikov kernel: (1 - (d/h)²)₊"""
    
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        t = distance / bandwidth
        result = np.zeros_like(t)
        mask = t <= 1.0
        result[mask] = 1.0 - t[mask]**2
        return result
    
    def support_radius(self, bandwidth: float) -> float:
        return bandwidth
    
    def is_compact(self) -> bool:
        return True


class UniformKernel(Kernel):
    """Uniform (box) kernel"""
    
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        return (distance <= bandwidth).astype(float)
    
    def support_radius(self, bandwidth: float) -> float:
        return bandwidth
    
    def is_compact(self) -> bool:
        return True


class TriangularKernel(Kernel):
    """
    Triangular (tent) kernel: (1 - |d/h|)₊
    
    Properties:
        - Linear decay from 1 at center to 0 at bandwidth
        - Compact support [-h, h]
        - Continuous but not differentiable at center
        - Good balance between smoothness and locality
    
    Use cases:
        - Linear interpolation-style smoothing
        - When you want smoother than uniform but simpler than Gaussian
        - Image processing, signal filtering
    """
    
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        t = distance / bandwidth
        result = np.zeros_like(t)
        mask = t <= 1.0
        result[mask] = 1.0 - t[mask]
        return result
    
    def support_radius(self, bandwidth: float) -> float:
        return bandwidth
    
    def is_compact(self) -> bool:
        return True


class QuarticKernel(Kernel):
    """
    Quartic (biweight) kernel: (1 - (d/h)²)²₊
    
    Properties:
        - Smooth (differentiable) decay
        - Compact support [-h, h]
        - Optimal for certain estimation problems
        - More weight near center than Epanechnikov
    
    Mathematical form:
        K(t) = (15/16)(1 - t²)² for |t| ≤ 1, else 0
    
    Use cases:
        - Kernel density estimation
        - Statistical smoothing where smoothness matters
        - Robust regression
    """
    
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        t = distance / bandwidth
        result = np.zeros_like(t)
        mask = t <= 1.0
        result[mask] = (15.0/16.0) * (1.0 - t[mask]**2)**2
        return result
    
    def support_radius(self, bandwidth: float) -> float:
        return bandwidth
    
    def is_compact(self) -> bool:
        return True


class MaternKernel(Kernel):
    """
    Matérn kernel: General family of correlation kernels
    
    Mathematical definition:
        K(r) = (2^(1-ν)/Γ(ν)) * (√(2ν)r/ρ)^ν * K_ν(√(2ν)r/ρ)
    
    Where:
        - ν (nu): Smoothness parameter
        - ρ (rho): Length scale (= bandwidth)
        - K_ν: Modified Bessel function of second kind
    
    Special cases:
        - ν = 0.5: Exponential kernel (not smooth)
        - ν = 1.5: Once differentiable
        - ν = 2.5: Twice differentiable (recommended default)
        - ν → ∞: Squared exponential (Gaussian-like)
    
    Properties:
        - Controls smoothness precisely via ν
        - Widely used in Gaussian processes
        - Exponential decay (not compact support)
    
    Use cases:
        - Spatial statistics and geostatistics
        - Gaussian process regression
        - When you need specific smoothness properties
        - Machine learning applications
    
    Examples:
        >>> kernel = MaternKernel(nu=2.5)  # Twice differentiable (standard)
        >>> kernel = MaternKernel(nu=1.5)  # Once differentiable (rougher)
    """
    
    def __init__(self, nu: float = 2.5):
        """
        Parameters
        ----------
        nu : float
            Smoothness parameter. Common values: 0.5, 1.5, 2.5, 3.5
            Higher ν = smoother kernel
        """
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.nu = nu
    
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Evaluate Matérn kernel at given distances.
        
        Uses simplified forms for common nu values for efficiency.
        """
        # Normalized distance
        d = distance / bandwidth
        
        # Special cases for efficiency
        if np.isclose(self.nu, 0.5):
            # Exponential kernel
            return np.exp(-d)
        elif np.isclose(self.nu, 1.5):
            # Once differentiable
            sqrt3_d = np.sqrt(3) * d
            return (1 + sqrt3_d) * np.exp(-sqrt3_d)
        elif np.isclose(self.nu, 2.5):
            # Twice differentiable (most common)
            sqrt5_d = np.sqrt(5) * d
            return (1 + sqrt5_d + sqrt5_d**2 / 3.0) * np.exp(-sqrt5_d)
        else:
            # General case (requires scipy)
            from scipy.special import kv, gamma
            
            d_safe = np.maximum(d, 1e-10)  # Avoid division by zero
            sqrt_2nu_d = np.sqrt(2 * self.nu) * d_safe
            
            # Matérn formula
            prefactor = 2**(1 - self.nu) / gamma(self.nu)
            bessel_term = kv(self.nu, sqrt_2nu_d)
            result = prefactor * sqrt_2nu_d**self.nu * bessel_term
            
            # Handle d=0 case (should be 1)
            result = np.where(d < 1e-10, 1.0, result)
            return result
    
    def support_radius(self, bandwidth: float) -> float:
        """Matérn has infinite support, but use 3*bandwidth for practical cutoff"""
        return 3.0 * bandwidth
    
    def is_compact(self) -> bool:
        return False  # Exponential decay, not compact