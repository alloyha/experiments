import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CORE ABSTRACTIONS
# ============================================================================

class MetricSpace(ABC):
    """Abstract metric space with distance and integration support"""
    
    @abstractmethod
    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute distance d(x1, x2). Supports broadcasting."""
        pass
    
    @abstractmethod
    def area_element(self, x: np.ndarray) -> np.ndarray:
        """Compute local area/volume element for integration"""
        pass
    
    @abstractmethod
    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:
        """Create computational grid with points and weights"""
        pass


class Region(ABC):
    """Abstract region representation"""
    
    @abstractmethod
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """Membership function: 1 inside, 0 outside, [0,1] on boundary"""
        pass
    
    @abstractmethod
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample n points from boundary for uncertainty analysis"""
        pass
    
    @abstractmethod
    def bounds(self) -> Tuple:
        """Return bounding box for grid creation"""
        pass


class UncertaintyDistribution(ABC):
    """Abstract probability distribution for query points"""
    
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density at points x"""
        pass
    
    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Generate n samples from distribution"""
        pass
    
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Distribution mean"""
        pass


class Kernel(ABC):
    """Abstract smoothing kernel"""
    
    @abstractmethod
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:
        """Evaluate kernel at given distances"""
        pass
    
    @abstractmethod
    def support_radius(self, bandwidth: float) -> float:
        """Return radius beyond which kernel ≈ 0"""
        pass
    
    @abstractmethod
    def is_compact(self) -> bool:
        """Whether kernel has compact support"""
        pass


class ConvolutionStrategy(ABC):
    """Abstract convolution method"""
    
    @abstractmethod
    def convolve(self, 
                 indicator: np.ndarray,
                 kernel: Kernel,
                 bandwidth: float,
                 grid: Dict[str, np.ndarray],
                 metric_space: MetricSpace) -> np.ndarray:
        """Compute w(x) = (I_R * K)(x)"""
        pass


class Integrator(ABC):
    """Abstract numerical integration"""
    
    @abstractmethod
    def integrate(self,
                  integrand: np.ndarray,
                  weights: np.ndarray) -> float:
        """Compute ∫ f(x) dx using quadrature weights"""
        pass
    
    @abstractmethod
    def estimate_error(self,
                       integrand: np.ndarray,
                       weights: np.ndarray) -> float:
        """Estimate integration error (if possible)"""
        pass