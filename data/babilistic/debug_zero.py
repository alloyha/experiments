import numpy as np
from babilistic import (
    ProbabilityEstimator, EuclideanSpace, DiskRegion,
    GaussianDistribution, GaussianKernel, FFTConvolution,
    QuadratureIntegrator
)

estimator = ProbabilityEstimator(
    EuclideanSpace(2),
    DiskRegion([0, 0], 1.0),
    GaussianDistribution([0.5, 0], np.eye(2) * 1e-10),
    GaussianKernel(),
    FFTConvolution(),
    QuadratureIntegrator()
)

w_field, grid = estimator.compute_probability_field(bandwidth=0.1, resolution=32)
print('w_field min/max:', w_field.min(), w_field.max())
print('weights sum:', grid['weights'].sum())
print('region mean:', estimator.region.center if hasattr(estimator.region,'center') else None)
points = grid['points']
p_X = estimator.query.pdf(points)
print('p_X min/max:', p_X.min(), p_X.max())
total_prob = estimator.integrator.integrate(p_X, grid['weights'])
print('total_prob:', total_prob)
if total_prob>0:
    p_norm = p_X/total_prob
    print('p_norm min/max:', p_norm.min(), p_norm.max())

result = estimator.compute_probability(w_field, grid)
print('probability result:', result.probability, 'error:', result.error_estimate)
