# ============================================================================
# INTEGRATOR IMPLEMENTATIONS
# ============================================================================

import numpy as np

from ..base import Integrator

class QuadratureIntegrator(Integrator):
    """Standard quadrature integration"""
    
    def integrate(self, integrand: np.ndarray, weights: np.ndarray) -> float:
        """∫ f(x) dx ≈ Σ f(xᵢ) wᵢ"""
        return float(np.sum(integrand * weights))
    
    def estimate_error(self, integrand: np.ndarray, weights: np.ndarray) -> float:
        """Rough error estimate via variance"""
        mean_val = self.integrate(integrand, weights)
        variance = np.sum((integrand - mean_val)**2 * weights)
        return float(np.sqrt(variance / len(integrand.flat)))


class MonteCarloIntegrator(Integrator):
    """Monte Carlo integration with samples"""
    
    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples
    
    def integrate(self, integrand: np.ndarray, weights: np.ndarray) -> float:
        """Sample-based integration"""
        # Interpret weights as sampling probabilities
        total_weight = np.sum(weights)
        probs = weights.flat / total_weight
        
        # Sample indices
        idx = np.random.choice(len(probs), self.n_samples, p=probs)
        samples = integrand.flat[idx]
        
        return float(np.mean(samples) * total_weight)
    
    def estimate_error(self, integrand: np.ndarray, weights: np.ndarray) -> float:
        """Standard error: σ/√n"""
        total_weight = np.sum(weights)
        probs = weights.flat / total_weight
        idx = np.random.choice(len(probs), self.n_samples, p=probs)
        samples = integrand.flat[idx]
        
        return float(np.std(samples) * total_weight / np.sqrt(self.n_samples))
