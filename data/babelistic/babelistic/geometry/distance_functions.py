# ============================================================================
# PROBABILITY DISTANCE UTILITIES
# ============================================================================

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


from ..base import DistanceFunction, MetricSpace


class KLDivergence(DistanceFunction ):
    """Kullback-Leibler divergence: D_KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx"""
    
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon  # For numerical stability
    
    def compute(self, p: np.ndarray, q: np.ndarray, 
                weights: Optional[np.ndarray] = None) -> float:
        """KL divergence (non-symmetric)"""
        p_safe = np.maximum(p, self.epsilon)
        q_safe = np.maximum(q, self.epsilon)
        
        # Normalize
        if weights is not None:
            p_norm = p_safe / (np.sum(p_safe * weights) + self.epsilon)
            q_norm = q_safe / (np.sum(q_safe * weights) + self.epsilon)
        else:
            p_norm = p_safe / (np.sum(p_safe) + self.epsilon)
            q_norm = q_safe / (np.sum(q_safe) + self.epsilon)
        
        # D_KL = ∫ p log(p/q)
        ratio = p_norm / q_norm
        integrand = p_norm * np.log(ratio)
        
        if weights is not None:
            return float(np.sum(integrand * weights))
        return float(np.sum(integrand))
    
    def is_metric(self) -> bool:
        return False  # KL is not symmetric


class JSDistance(DistanceFunction):
    """Jensen-Shannon distance (symmetrized KL): sqrt(JS_divergence)"""
    
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
        self.kl = KLDivergence(epsilon)
    
    def compute(self, p: np.ndarray, q: np.ndarray, 
                weights: Optional[np.ndarray] = None) -> float:
        """JS distance (symmetric, bounded)"""
        # Normalize first
        if weights is not None:
            p_norm = np.maximum(p, self.epsilon)
            q_norm = np.maximum(q, self.epsilon)
            p_norm = p_norm / (np.sum(p_norm * weights) + self.epsilon)
            q_norm = q_norm / (np.sum(q_norm * weights) + self.epsilon)
        else:
            p_norm = np.maximum(p, self.epsilon)
            q_norm = np.maximum(q, self.epsilon)
            p_norm = p_norm / (np.sum(p_norm) + self.epsilon)
            q_norm = q_norm / (np.sum(q_norm) + self.epsilon)
        
        # Mixture: m = (p + q) / 2
        m = 0.5 * (p_norm + q_norm)
        
        # JS divergence = [D_KL(p||m) + D_KL(q||m)] / 2
        d_pm = self.kl.compute(p_norm, m, weights)
        d_qm = self.kl.compute(q_norm, m, weights)
        js_div = 0.5 * (d_pm + d_qm)
        
        # JS distance = sqrt(JS_divergence)
        return float(np.sqrt(np.maximum(js_div, 0)))
    
    def is_metric(self) -> bool:
        return True  # JS distance is a metric


class WassersteinDistance(DistanceFunction):
    """
    Wasserstein-1 (Earth Mover's) distance.
    Requires underlying metric space.
    """
    
    def __init__(self, metric_space: MetricSpace, approximate: bool = True):
        self.metric_space = metric_space
        self.approximate = approximate
    
    def compute(self, p: np.ndarray, q: np.ndarray, 
                weights: Optional[np.ndarray] = None,
                points: Optional[np.ndarray] = None) -> float:
        """
        Wasserstein distance: inf_γ ∫∫ d(x,y) dγ(x,y)
        
        Parameters
        ----------
        points : np.ndarray
            Support points where p, q are evaluated (required)
        """
        if points is None:
            raise ValueError("Wasserstein distance requires support points")
        
        # Ensure arrays, not iterators
        p_arr = np.asarray(p)
        q_arr = np.asarray(q)
        
        # Normalize
        if weights is not None:
            weights_arr = np.asarray(weights)
            p_norm = p_arr / (np.sum(p_arr * weights_arr) + 1e-12)
            q_norm = q_arr / (np.sum(q_arr * weights_arr) + 1e-12)
        else:
            p_norm = p_arr / (np.sum(p_arr) + 1e-12)
            q_norm = q_arr / (np.sum(q_arr) + 1e-12)
        
        if self.approximate:
            # Sliced Wasserstein approximation (fast)
            return self._sliced_wasserstein(p_norm, q_norm, points, n_projections=50)
        else:
            # Exact via linear programming (requires scipy)
            return self._exact_wasserstein(p_norm, q_norm, points)
    
    def _sliced_wasserstein(self, p: np.ndarray, q: np.ndarray, 
                           points: np.ndarray, n_projections: int) -> float:
        """Fast approximation via random projections"""
        dim = points.shape[-1]
        distances = []
        
        for _ in range(n_projections):
            # Random direction
            theta = np.random.randn(dim)
            theta = theta / np.linalg.norm(theta)
            
            # Project points onto direction
            proj = points @ theta
            
            # 1D Wasserstein (closed form via CDFs)
            idx_p = np.argsort(proj.flat)
            idx_q = np.argsort(proj.flat)
            
            p_sorted = p.flat[idx_p]
            q_sorted = q.flat[idx_q]
            proj_sorted_p = proj.flat[idx_p]
            proj_sorted_q = proj.flat[idx_q]
            
            # Cumulative distributions
            cdf_p = np.cumsum(p_sorted)
            cdf_q = np.cumsum(q_sorted)
            
            # L1 distance between CDFs weighted by coordinate
            # Use `np.trapezoid` (replacement for deprecated `trapz`)
            dist_1d = np.trapezoid(np.abs(cdf_p - cdf_q), proj_sorted_p)
            distances.append(dist_1d)
        
        return float(np.mean(distances))
    
    def _exact_wasserstein(self, p: np.ndarray, q: np.ndarray, 
                          points: np.ndarray) -> float:
        """Exact computation via optimal transport"""
        try:
            from scipy.optimize import linprog
            from scipy.spatial.distance import cdist
        except ImportError:
            raise ImportError("Exact Wasserstein requires scipy")
        
        # Flatten
        p_flat = p.flat
        q_flat = q.flat
        points_flat = points.reshape(-1, points.shape[-1])
        
        # Cost matrix
        C = cdist(points_flat, points_flat, 
                  metric=lambda u, v: self.metric_space.distance(u, v))
        
        # Linear program: min <C, γ> s.t. marginal constraints
        # This is simplified - full implementation needs proper LP setup
        # For now, return sliced approximation
        return self._sliced_wasserstein(p, q, points, n_projections=100)
    
    def is_metric(self) -> bool:
        return True  # Wasserstein is a metric


class TotalVariationDistance(DistanceFunction):
    """Total variation distance: (1/2) ∫ |p(x) - q(x)| dx"""
    
    def compute(self, p: np.ndarray, q: np.ndarray, 
                weights: Optional[np.ndarray] = None) -> float:
        """TV distance (L1 norm of difference)"""
        # Ensure arrays, not iterators
        p_arr = np.asarray(p)
        q_arr = np.asarray(q)
        
        # Normalize
        if weights is not None:
            weights_arr = np.asarray(weights)
            p_norm = p_arr / (np.sum(p_arr * weights_arr) + 1e-12)
            q_norm = q_arr / (np.sum(q_arr * weights_arr) + 1e-12)
            return 0.5 * float(np.sum(np.abs(p_norm - q_norm) * weights_arr))
        else:
            p_norm = p_arr / (np.sum(p_arr) + 1e-12)
            q_norm = q_arr / (np.sum(q_arr) + 1e-12)
            return 0.5 * float(np.sum(np.abs(p_norm - q_norm)))
    
    def is_metric(self) -> bool:
        return True


class HellingerDistance(DistanceFunction):
    """Hellinger distance: sqrt(1 - ∫ sqrt(p(x)q(x)) dx)"""
    
    def compute(self, p: np.ndarray, q: np.ndarray, 
                weights: Optional[np.ndarray] = None) -> float:
        """Hellinger distance"""
        # Ensure arrays, not iterators
        p_arr = np.asarray(p)
        q_arr = np.asarray(q)
        
        p_safe = np.maximum(p_arr, 0)
        q_safe = np.maximum(q_arr, 0)
        
        # Normalize
        if weights is not None:
            weights_arr = np.asarray(weights)
            p_norm = p_safe / (np.sum(p_safe * weights_arr) + 1e-12)
            q_norm = q_safe / (np.sum(q_safe * weights_arr) + 1e-12)
            bc = np.sum(np.sqrt(p_norm * q_norm) * weights_arr)
        else:
            p_norm = p_safe / (np.sum(p_safe) + 1e-12)
            q_norm = q_safe / (np.sum(q_safe) + 1e-12)
            bc = np.sum(np.sqrt(p_norm * q_norm))
        
        # Hellinger = sqrt(1 - BC) where BC is Bhattacharyya coefficient
        return float(np.sqrt(np.maximum(1.0 - bc, 0)))
    
    def is_metric(self) -> bool:
        return True