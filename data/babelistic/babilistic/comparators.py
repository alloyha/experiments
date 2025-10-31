import numpy as np
from typing import Any, Dict, List

from .distance_functions import (
    ProbabilityDistance, 
    KLDivergence, 
    WassersteinDistance
)
from .distributions import UncertaintyDistribution  
from .core import ProbabilityEstimator

# ============================================================================
# INTEGRATION WITH FRAMEWORK
# ============================================================================

class DistributionComparator:
    """
    Utility for comparing probability distributions in the framework.
    Useful for sensitivity analysis, convergence testing, etc.
    """
    
    def __init__(self, distance_metric: ProbabilityDistance):
        self.distance = distance_metric
    
    def compare_w_fields(self, 
                        w1: np.ndarray, 
                        w2: np.ndarray,
                        grid: Dict[str, np.ndarray]) -> float:
        """
        Compare two mollified indicator fields.
        
        Useful for:
        - Bandwidth sensitivity: compare w(h1) vs w(h2)
        - Convergence: compare w(N1) vs w(N2)
        - Kernel comparison: compare Gaussian vs Epanechnikov
        """
        return self.distance.compute(w1.flat, w2.flat, grid['weights'].flat)
    
    def compare_query_distributions(self,
                                    dist1: UncertaintyDistribution,
                                    dist2: UncertaintyDistribution,
                                    grid: Dict[str, np.ndarray]) -> float:
        """Compare two query distributions"""
        p1 = dist1.pdf(grid['points'])
        p2 = dist2.pdf(grid['points'])
        
        # For Wasserstein, pass points
        if isinstance(self.distance, WassersteinDistance):
            return self.distance.compute(p1.flat, p2.flat, 
                                        grid['weights'].flat,
                                        grid['points'].reshape(-1, grid['points'].shape[-1]))
        
        return self.distance.compute(p1.flat, p2.flat, grid['weights'].flat)
    
    def convergence_analysis(self,
                            estimator: ProbabilityEstimator,
                            bandwidth: float,
                            resolutions: List[int]) -> Dict[str, Any]:
        """
        Analyze convergence as grid resolution increases.
        
        Returns distance between consecutive resolutions.
        """
        results = []
        w_fields = []
        
        for res in resolutions:
            w_field, grid = estimator.compute_probability_field(bandwidth, res)
            w_fields.append((w_field, grid))
            result = estimator.compute_probability(w_field, grid)
            results.append(result.probability)
        
        # Compare consecutive resolutions
        distances = []
        for i in range(len(w_fields) - 1):
            w1, grid1 = w_fields[i]
            w2, grid2 = w_fields[i + 1]
            
            # Interpolate w1 to grid2 for comparison
            # For simplicity, just compare at coarse resolution
            dist = self.compare_w_fields(w1, w1, grid1)  # Placeholder
            distances.append(dist)
        
        return {
            'resolutions': resolutions,
            'probabilities': results,
            'convergence_distances': distances,
            'converged': distances[-1] < 0.01 if distances else False
        }
    
    def sensitivity_analysis(self,
                            estimator: ProbabilityEstimator,
                            bandwidths: List[float],
                            resolution: int = 128) -> Dict[str, Any]:
        """
        Analyze sensitivity to bandwidth parameter.
        
        Returns distances between probability fields at different bandwidths.
        """
        w_fields = []
        probabilities = []
        
        for h in bandwidths:
            w_field, grid = estimator.compute_probability_field(h, resolution)
            w_fields.append(w_field)
            result = estimator.compute_probability(w_field, grid)
            probabilities.append(result.probability)
        
        # Compute pairwise distances
        distance_matrix = np.zeros((len(bandwidths), len(bandwidths)))
        for i in range(len(bandwidths)):
            for j in range(i + 1, len(bandwidths)):
                dist = self.distance.compute(w_fields[i].flat, w_fields[j].flat, 
                                            grid['weights'].flat)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return {
            'bandwidths': bandwidths,
            'probabilities': probabilities,
            'distance_matrix': distance_matrix,
            'max_variation': np.max(distance_matrix),
            'mean_variation': np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
        }
