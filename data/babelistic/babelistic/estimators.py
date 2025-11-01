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

from .base import Region, Integrator, MetricSpace
from .geometry.metric_spaces import (
    EuclideanSpace, 
    ManhattanSpace, 
    SphericalSpace,
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
    UncertaintyDistribution,
    GaussianDistribution,
    UniformDistribution,
    StudentTDistribution,
    LogNormalDistribution,
    MixtureDistribution,
    EmpiricalDistribution,
)
from .probability.kernels import (
    Kernel,
    GaussianKernel,
    EpanechnikovKernel,
    UniformKernel,
    QuarticKernel,
    TriangularKernel,
    MaternKernel,
)
from .computation.convolution_strategies import (
    ConvolutionStrategy,
    DirectConvolution,
    SparseConvolution,
    FFTConvolution,
)
from .computation.integrators import (
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



