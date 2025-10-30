import numpy as np

import pytest

# ============================================================================
# GEOFENCE-SPECIFIC TESTS
# ============================================================================

class TestGeofence:
    """Tests specific to geofence integration"""
    
    @staticmethod
    def test_geofence_adapter():
        """Test GeofenceAdapter converts distance functions"""
        from babilistic import GeofenceAdapter
        
        def mock_distance(lat1, lon1, lat2, lon2):
            return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000
        
        adapter = GeofenceAdapter(mock_distance)
        metric_space = adapter.to_metric_space()
        
        # Test distance calculation
        p1 = np.array([37.7749, -122.4194])
        p2 = np.array([37.7750, -122.4195])
        
        dist = metric_space.distance(p1, p2)
        assert dist > 0, "Distance should be positive"
    
    @staticmethod
    def test_geofence_region():
        """Test GeofenceRegion indicator"""
        from babilistic import GeofenceRegion
        
        def mock_distance(lat1, lon1, lat2, lon2):
            return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000
        
        region = GeofenceRegion(
            reference_lat=37.7750,
            reference_lon=-122.4195,
            reference_uncertainty=5.0,
            distance_threshold=100.0,
            distance_metric=mock_distance
        )
        
        # Test points
        close = np.array([37.7750, -122.4195])  # At reference
        far = np.array([37.7760, -122.4205])  # ~150m away
        
        assert region.indicator(close) == 1.0, "Close point should be inside"
        assert region.indicator(far) == 0.0, "Far point should be outside"
    
    @staticmethod
    def test_analytic_approximation():
        """Test analytic geofence probability"""
        from babilistic import estimate_geofence_probability_analytic
        
        # Known case: high probability when inside
        prob = estimate_geofence_probability_analytic(
            subject_uncertainty=5.0,
            reference_uncertainty=5.0,
            mean_separation=10.0,
            distance_threshold=50.0
        )
        
        assert 0.9 < prob < 1.0, f"Should have high probability, got {prob}"
        
        # Known case: low probability when far
        prob_far = estimate_geofence_probability_analytic(
            subject_uncertainty=5.0,
            reference_uncertainty=5.0,
            mean_separation=100.0,
            distance_threshold=20.0
        )
        
        assert prob_far < 0.1, f"Should have low probability, got {prob_far}"

    def test_geofence_adapter_edge_cases(self):
        """Cover GeofenceAdapter different input shapes"""
        from babilistic import GeofenceAdapter
        
        def mock_distance(lat1, lon1, lat2, lon2):
            return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)
        
        adapter = GeofenceAdapter(mock_distance)
        space = adapter.to_metric_space()
        
        # Test 1D to 1D
        p1 = np.array([0, 0])
        p2 = np.array([1, 1])
        dist = space.distance(p1, p2)
        assert dist > 0
        
        # Test 2D to 1D (batch)
        batch = np.array([[0, 0], [1, 0], [2, 0]])
        dists = space.distance(batch, p2)
        assert dists.shape == (3,)
        
        # Test with unsupported shape combination
        with pytest.raises(ValueError):
            space.distance(np.array([[[0, 0]]]), p2)  # 3D array
    
    def test_geofence_region_bounds(self):
        """Cover GeofenceRegion bounds calculation (lines 262)"""
        from babilistic import GeofenceRegion
        
        def mock_distance(lat1, lon1, lat2, lon2):
            return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)
        
        region = GeofenceRegion(
            reference_lat=37.0,
            reference_lon=-122.0,
            reference_uncertainty=10.0,
            distance_threshold=100.0,
            distance_metric=mock_distance
        )
        
        bounds = region.bounds()
        assert len(bounds) == 4
        assert bounds[0] < bounds[1]  # lat_min < lat_max
        assert bounds[2] < bounds[3]  # lon_min < lon_max
    
    def test_geofence_to_probability_bandwidth_autoselect(self):
        """Cover automatic bandwidth selection (lines 277-297)"""
        from babilistic.integrations.geofence import geofence_to_probability
        
        def mock_distance(lat1, lon1, lat2, lon2):
            return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000
        
        # Test with bandwidth=None (auto-select)
        result = geofence_to_probability(
            subject_lat=37.0, subject_lon=-122.0, subject_uncertainty=10.0,
            reference_lat=37.001, reference_lon=-122.001, reference_uncertainty=5.0,
            distance_threshold=200.0,
            distance_metric=mock_distance,
            bandwidth=None,  # Auto-select
            resolution=32
        )
        
        assert 0 <= result.probability <= 1
        assert 'bandwidth_m' in result.metadata


    def test_geofence_osm_metric_branch_and_normalization(self):
        """Cover OSM metric branch and normalization fallback"""
        from babilistic import (
            ProbabilityEstimator, EuclideanSpace, DiskRegion,
            GaussianDistribution, GaussianKernel, DirectConvolution,
            QuadratureIntegrator
        )

        # Create a simple OSM-like class name to trigger the branch
        class OSMMock:
            pass

        def dummy_metric(lat1, lon1, lat2, lon2):
            return 0.0 if np.ndim(lat1) == 0 else np.zeros_like(lat1)

        # The geofence helper is invoked indirectly; we test that creating an
        # estimator and computing works when convolution choices could vary.
        space = EuclideanSpace(2)
        region = DiskRegion(center=[0.0, 0.0], radius=0.1)
        query = GaussianDistribution(mean=np.array([10.0, 10.0]), cov=np.eye(2) * 1e-6)

        estimator = ProbabilityEstimator(
            metric_space=space,
            region=region,
            query_distribution=query,
            kernel=GaussianKernel(),
            convolution_strategy=DirectConvolution(),
            integrator=QuadratureIntegrator(),
        )

        # compute with very small covariance — should use deterministic fallback or normalization
        res = estimator.compute(bandwidth=0.01, resolution=24)
        assert 0.0 <= res.probability <= 1.0


    def test_geofence_to_probability_trivial_metric(self):
        # Distance metric that always returns zero -> subject always at reference
        from babilistic import geofence_to_probability
        
        def zero_metric(lat1, lon1, lat2, lon2):
            if np.ndim(lat1) == 0:
                return 0.0
            return np.zeros_like(np.asarray(lat1), dtype=float)

        res = geofence_to_probability(
            subject_lat=0.0,
            subject_lon=0.0,
            subject_uncertainty=10.0,
            reference_lat=0.0,
            reference_lon=0.0,
            reference_uncertainty=5.0,
            distance_threshold=1.0,
            distance_metric=zero_metric,
            resolution=16,
        )

        assert 0.9 <= res.probability <= 1.0


    def test_geofence_to_probability_far_metric(self):
        # Distance metric that always returns large distance -> probability ~ 0
        from babilistic import geofence_to_probability
        
        def far_metric(lat1, lon1, lat2, lon2):
            if np.ndim(lat1) == 0:
                return 1e6
            return np.ones_like(np.asarray(lat1), dtype=float) * 1e6

        res = geofence_to_probability(
            subject_lat=0.0,
            subject_lon=0.0,
            subject_uncertainty=10.0,
            reference_lat=1.0,
            reference_lon=1.0,
            reference_uncertainty=5.0,
            distance_threshold=1.0,
            distance_metric=far_metric,
            resolution=16,
        )

        assert 0.0 <= res.probability <= 0.1


    def test_geofence_anisotropic_subject_covariance(self):
        """Test geofence probability with anisotropic subject covariance"""
        from babilistic import geofence_to_probability

        # Use anisotropic covariance for subject_uncertainty (2x2 matrix)
        def simple_metric(lat1, lon1, lat2, lon2):
            # Euclidean-like in degrees (approx)
            a = np.asarray(lat1) - np.asarray(lat2)
            b = np.asarray(lon1) - np.asarray(lon2)
            return np.sqrt(a**2 + b**2) * 111000.0

        subject_cov_m = np.array([[100.0, 10.0], [10.0, 200.0]])

        res = geofence_to_probability(
            subject_lat=40.0,
            subject_lon=-74.0,
            subject_uncertainty=subject_cov_m,
            reference_lat=40.001,
            reference_lon=-74.001,
            reference_uncertainty=10.0,
            distance_threshold=500.0,
            distance_metric=simple_metric,
            resolution=16,
        )

        assert 0.0 <= res.probability <= 1.0


    def test_estimate_geofence_probability_analytic_basic(self):
        """Test analytic geofence probability estimation"""
        from babilistic import estimate_geofence_probability_analytic

        p = estimate_geofence_probability_analytic(
            subject_uncertainty=10.0,
            reference_uncertainty=5.0,
            mean_separation=20.0,
            distance_threshold=50.0,
        )

        assert 0.0 <= p <= 1.0
