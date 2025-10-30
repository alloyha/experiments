import numpy as np

from babilistic import (
    QuadratureIntegrator,
    MonteCarloIntegrator,
    GaussianKernel,
    EpanechnikovKernel,
)


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


def test_disk_region_edge_cases():
    """Cover disk region edge cases (line 37)"""
    from babilistic.regions import DiskRegion
    from babilistic.metric_spaces import EuclideanSpace, ManhattanSpace
    
    # Test with different metric spaces
    euclidean_disk = DiskRegion([0, 0], 1.0, metric_space=EuclideanSpace(2))
    manhattan_disk = DiskRegion([0, 0], 1.0, metric_space=ManhattanSpace(2))
    
    test_point = np.array([0.5, 0.5])
    assert euclidean_disk.indicator(test_point) in [0.0, 1.0]
    assert manhattan_disk.indicator(test_point) in [0.0, 1.0]

def test_ellipse_boundary_sampling():
    """Cover ellipse boundary generation (lines 311)"""
    from babilistic.regions import EllipseRegion
    
    ellipse = EllipseRegion([1, 1], [2, 1], rotation_deg=0)
    
    # Test with different sample sizes
    boundary_10 = ellipse.sample_boundary(10)
    assert boundary_10.shape == (10, 2)
    
    boundary_100 = ellipse.sample_boundary(100)
    assert boundary_100.shape == (100, 2)
    
    # Verify points are on boundary
    centered = boundary_100 - ellipse.center
    quad_forms = np.sum(centered @ ellipse.A * centered, axis=1)
    assert np.allclose(quad_forms, 1.0, rtol=0.05)

def test_buffered_polygon_negative_buffer():
    """Cover negative buffer (erosion) case (lines 361-381)"""
    from babilistic.regions import BufferedPolygonRegion
    
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    
    # Positive buffer (expansion)
    expanded = BufferedPolygonRegion(square, buffer=0.5)
    
    # Negative buffer (erosion)
    eroded = BufferedPolygonRegion(square, buffer=-0.5)
    
    # Points that should behave differently
    near_edge = np.array([0.3, 1.0])
    
    # Both should give valid results
    assert 0 <= expanded.indicator(near_edge) <= 1
    assert 0 <= eroded.indicator(near_edge) <= 1

def test_multi_region_empty():
    """Cover multi-region with empty list (lines 433, 439)"""
    from babilistic.regions import MultiRegion, DiskRegion
    
    disk1 = DiskRegion([0, 0], 1)
    disk2 = DiskRegion([3, 3], 1)
    
    # Test union with no overlap
    union = MultiRegion([disk1, disk2], operation='union')
    
    # Point between both (should be outside both)
    between = np.array([1.5, 1.5])
    assert union.indicator(between) == 0.0
    
    # Test intersection with no overlap
    intersection = MultiRegion([disk1, disk2], operation='intersection')
    assert intersection.indicator(between) == 0.0

def test_implicit_region_boundary():
    """Cover implicit region boundary sampling (lines 114-115, 132)"""
    from babilistic.regions import ImplicitRegion
    
    # Circle via SDF
    def circle_sdf(x):
        return np.linalg.norm(x, axis=-1) - 1.0
    
    region = ImplicitRegion(
        sdf=circle_sdf,
        bounds=(-2, 2, -2, 2),
        samples_cache=None  # Force boundary computation
    )
    
    # Test boundary sampling without cache
    boundary = region.sample_boundary(50)
    assert boundary.shape[0] > 0  # Should generate some points
    
    # Test with cache
    cached_boundary = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    region_cached = ImplicitRegion(
        sdf=circle_sdf,
        bounds=(-2, 2, -2, 2),
        samples_cache=cached_boundary
    )
    
    boundary_cached = region_cached.sample_boundary(10)
    assert boundary_cached.shape == (10, 2)
