import numpy as np

import pytest

from babilistic import (
    MonteCarloIntegrator,
    QuadratureIntegrator,
)

# ============================================================================
# TEST INTEGRATORS
# ============================================================================

class TestIntegrators:
    """Test integrator edge cases"""
    
    def test_monte_carlo_estimate_error(self):
        integrator = MonteCarloIntegrator(n_samples=1000)
        
        integrand = np.random.rand(100)
        weights = np.ones(100) / 100
        
        error = integrator.estimate_error(integrand, weights)
        assert isinstance(error, float)
        assert error >= 0

def test_quadrature_integrator_integrate_and_error():
    quad = QuadratureIntegrator()
    integrand = np.array([1.0, 1.0, 1.0])
    weights = np.array([0.2, 0.3, 0.5])

    total = quad.integrate(integrand, weights)
    assert np.isclose(total, 1.0)

    err = quad.estimate_error(integrand, weights)
    assert isinstance(err, float)
    assert err >= 0.0


def test_montecarlo_integrator_approximates_quadrature():
    # Small deterministic test using a seeded RNG for reproducibility
    np.random.seed(0)
    quad = QuadratureIntegrator()
    mc = MonteCarloIntegrator(n_samples=1000)

    # Simple integrand over three points
    integrand = np.array([0.1, 0.4, 0.5])
    weights = np.array([1.0, 2.0, 3.0])

    exact = quad.integrate(integrand, weights)
    approx = mc.integrate(integrand, weights)

    # Monte Carlo is stochastic but seeded; allow a modest tolerance
    assert np.isfinite(approx)
    assert abs(approx - exact) / max(abs(exact), 1e-12) < 0.2