import numpy as np
from babelistic import (
    ProbabilityEstimator, EuclideanSpace, DiskRegion,
    GaussianDistribution, GaussianKernel, DirectConvolution,
    QuadratureIntegrator
)

estimator = ProbabilityEstimator(
    EuclideanSpace(2),
    DiskRegion([10, 10], 0.01),
    GaussianDistribution([0, 0], np.eye(2)),
    GaussianKernel(),
    DirectConvolution(),
    QuadratureIntegrator()
)

w_field, grid = estimator.compute_probability_field(bandwidth=0.1, resolution=32)
points = grid['points']
print('w_field min/max:', w_field.min(), w_field.max())
print('weights sum:', grid['weights'].sum())

p_X = estimator.query.pdf(points)
print('p_X min/max:', p_X.min(), p_X.max())
total_prob = estimator.integrator.integrate(p_X, grid['weights'])
print('total_prob:', total_prob)

prob = estimator.compute_probability(w_field, grid)
print('result prob:', prob.probability, 'error:', prob.error_estimate)
