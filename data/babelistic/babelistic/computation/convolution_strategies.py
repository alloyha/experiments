# ============================================================================
# CONVOLUTION STRATEGY IMPLEMENTATIONS
# ============================================================================

from typing import Dict

import numpy as np

from ..base import ConvolutionStrategy, Kernel, MetricSpace
from ..geometry.metric_spaces import EuclideanSpace


class DirectConvolution(ConvolutionStrategy):
    """Direct summation convolution (metric-agnostic)"""
    
    def convolve(self,
                 indicator: np.ndarray,
                 kernel: Kernel,
                 bandwidth: float,
                 grid: Dict[str, np.ndarray],
                 metric_space: MetricSpace) -> np.ndarray:
        """
        Compute w(x) = ∫ I_R(y) K(d(x,y)) dy / ∫ K(d(x,y)) dy
        """
        points = grid['points']
        weights = grid['weights']
        shape = grid['shape']
        
        # Flatten for processing
        points_flat = points.reshape(-1, points.shape[-1])
        indicator_flat = indicator.reshape(-1)
        weights_flat = weights.reshape(-1)
        
        w_field = np.zeros(len(points_flat))
        
        # For each query point
        for i, x_query in enumerate(points_flat):
            # Compute distances to all points
            distances = metric_space.distance(points_flat, x_query)
            
            # Evaluate kernel
            K_vals = kernel.evaluate(distances, bandwidth)
            
            # Numerator: integrate kernel over region
            numerator = np.sum(K_vals * indicator_flat * weights_flat)
            
            # Denominator: integrate kernel over domain
            denominator = np.sum(K_vals * weights_flat)
            
            if denominator > 1e-12:
                w_field[i] = numerator / denominator
        
        return w_field.reshape(shape)


class SparseConvolution(ConvolutionStrategy):
    """Sparse convolution using kernel support"""
    
    def convolve(self,
                 indicator: np.ndarray,
                 kernel: Kernel,
                 bandwidth: float,
                 grid: Dict[str, np.ndarray],
                 metric_space: MetricSpace) -> np.ndarray:
        """Optimize by only evaluating within kernel support"""
        
        points = grid['points']
        weights = grid['weights']
        shape = grid['shape']
        
        points_flat = points.reshape(-1, points.shape[-1])
        indicator_flat = indicator.reshape(-1)
        weights_flat = weights.reshape(-1)
        
        support_radius = kernel.support_radius(bandwidth)
        w_field = np.zeros(len(points_flat))
        
        # Find region support points
        region_points = points_flat[indicator_flat > 0.5]
        
        for i, x_query in enumerate(points_flat):
            # Only compute distances to nearby points
            distances = metric_space.distance(points_flat, x_query)
            
            # Mask for points within support
            support_mask = distances <= support_radius
            
            if not support_mask.any():
                continue
            
            # Evaluate kernel only on support
            K_vals = np.zeros_like(distances)
            K_vals[support_mask] = kernel.evaluate(distances[support_mask], bandwidth)
            
            numerator = np.sum(K_vals * indicator_flat * weights_flat)
            denominator = np.sum(K_vals * weights_flat)
            
            if denominator > 1e-12:
                w_field[i] = numerator / denominator
        
        return w_field.reshape(shape)


class FFTConvolution(ConvolutionStrategy):
    """FFT-based convolution (Euclidean uniform grids only)"""
    
    def convolve(self,
                 indicator: np.ndarray,
                 kernel: Kernel,
                 bandwidth: float,
                 grid: Dict[str, np.ndarray],
                 metric_space: MetricSpace) -> np.ndarray:
        """Fast convolution via FFT (requires uniform Euclidean grid)"""
        
        if not isinstance(metric_space, EuclideanSpace):
            raise ValueError("FFT convolution requires Euclidean space")
        
        from scipy.signal import fftconvolve
        
        # Build kernel on grid
        dx = grid['x'][1] - grid['x'][0]
        dy = grid['y'][1] - grid['y'][0]
        
        # Kernel size
        support_radius = kernel.support_radius(bandwidth)
        ksize = int(np.ceil(support_radius / dx))
        kx = np.arange(-ksize, ksize+1) * dx
        ky = np.arange(-ksize, ksize+1) * dy
        KX, KY = np.meshgrid(kx, ky)
        
        distances = np.sqrt(KX**2 + KY**2)
        K_grid = kernel.evaluate(distances, bandwidth)
        K_grid /= K_grid.sum()  # Normalize
        
        # Convolve
        w_field = fftconvolve(indicator, K_grid, mode='same')
        
        return np.clip(w_field, 0.0, 1.0)
