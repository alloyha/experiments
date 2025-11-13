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
    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:   # pragma: no cover
        """Compute distance d(x1, x2). Supports broadcasting."""
        pass
    
    @abstractmethod
    def area_element(self, x: np.ndarray) -> np.ndarray:   # pragma: no cover
        """Compute local area/volume element for integration"""
        pass
    
    @abstractmethod
    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:   # pragma: no cover
        """Create computational grid with points and weights"""
        pass


class DistanceFunction(ABC):
    """
    Abstract probability distance/divergence measure.
    Quantifies dissimilarity between two probability distributions.
    """
    
    @abstractmethod
    def compute(self, p: np.ndarray, q: np.ndarray, 
                weights: Optional[np.ndarray] = None) -> float:  # pragma: no cover
        """
        Compute distance/divergence between distributions p and q.
        
        Parameters
        ----------
        p, q : np.ndarray
            Probability densities (need not be normalized if using weights)
        weights : np.ndarray, optional
            Integration weights for discrete approximation
            
        Returns
        -------
        distance : float
            Non-negative distance/divergence value
        """
        pass
    
    @abstractmethod
    def is_metric(self) -> bool:  # pragma: no cover
        """Whether this satisfies metric axioms (symmetry, triangle inequality)"""
        pass


class Region(ABC):
    """Abstract region representation"""
    
    @abstractmethod
    def indicator(self, x: np.ndarray) -> np.ndarray:   # pragma: no cover
        """Membership function: 1 inside, 0 outside, [0,1] on boundary"""
        pass
    
    @abstractmethod
    def sample_boundary(self, n: int) -> np.ndarray:   # pragma: no cover
        """Sample n points from boundary for uncertainty analysis"""
        pass
    
    @abstractmethod
    def bounds(self) -> Tuple:   # pragma: no cover
        """Return bounding box for grid creation"""
        pass
    
    @abstractmethod
    def sample_uniform(self, n_samples: int) -> np.ndarray:   # pragma: no cover
        """Sample points uniformly from region"""
        pass
    
    @abstractmethod
    def area(self) -> float:   # pragma: no cover
        """Compute region area"""
        pass
    
    @abstractmethod
    def centroid(self) -> np.ndarray:   # pragma: no cover
        """Compute centroid"""
        pass


class UncertaintyDistribution(ABC):
    """Abstract probability distribution for query points"""
    
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Probability density at points x"""
        pass
    
    @abstractmethod
    def sample(self, n: int) -> np.ndarray:  # pragma: no cover
        """Generate n samples from distribution"""
        pass
    
    @abstractmethod
    def mean(self) -> np.ndarray:  # pragma: no cover
        """Distribution mean"""
        pass


class Kernel(ABC):
    """Abstract smoothing kernel"""
    
    @abstractmethod
    def evaluate(self, distance: np.ndarray, bandwidth: float) -> np.ndarray:  # pragma: no cover
        """Evaluate kernel at given distances"""
        pass
    
    @abstractmethod
    def support_radius(self, bandwidth: float) -> float:  # pragma: no cover
        """Return radius beyond which kernel ≈ 0"""
        pass
    
    @abstractmethod
    def is_compact(self) -> bool:  # pragma: no cover
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
                 metric_space: MetricSpace) -> np.ndarray:  # pragma: no cover
        """Compute w(x) = (I_R * K)(x)"""
        pass


class Integrator(ABC):
    """Abstract numerical integration"""
    
    @abstractmethod
    def integrate(self,
                  integrand: np.ndarray,
                  weights: np.ndarray) -> float:  # pragma: no cover
        """Compute ∫ f(x) dx using quadrature weights"""
        pass
    
    @abstractmethod
    def estimate_error(self,
                       integrand: np.ndarray,
                       weights: np.ndarray) -> float:  # pragma: no cover
        """Estimate integration error (if possible)"""
        pass

