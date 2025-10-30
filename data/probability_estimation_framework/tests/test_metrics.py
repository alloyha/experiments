import numpy as np

# ============================================================================
# UNIT TESTS: METRIC SPACES
# ============================================================================

class TestMetricSpaces:
    """Unit tests for all MetricSpace implementations"""
    
    @staticmethod
    def test_euclidean_distance():
        """Test Euclidean distance calculation"""
        from babilistic import EuclideanSpace
        
        space = EuclideanSpace(2)
        
        # Test known distances
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])
        
        dist = space.distance(p1, p2)
        assert np.isclose(dist, 5.0), f"Expected 5.0, got {dist}"
        
        # Test array broadcasting
        points = np.array([[0, 0], [1, 1], [2, 2]])
        dists = space.distance(points, p2)
        assert dists.shape == (3,), f"Shape mismatch: {dists.shape}"
        
        # Test symmetry
        dist_12 = space.distance(p1, p2)
        dist_21 = space.distance(p2, p1)
        assert np.isclose(dist_12, dist_21), "Distance not symmetric"
        
        # Test identity
        dist_11 = space.distance(p1, p1)
        assert np.isclose(dist_11, 0.0), "Self-distance should be zero"
    
    @staticmethod
    def test_geo_distance():
        """Test spherical (haversine) distance"""
        from babilistic import GeoSpace
        
        space = GeoSpace(radius=6371000)
        
        # Test equator points (1 degree apart)
        p1 = np.array([0, 0])  # lat, lon
        p2 = np.array([0, 1])
        
        dist = space.distance(p1, p2)
        expected = 111200  # ~111.2 km per degree at equator
        assert np.isclose(dist, expected, rtol=0.01), f"Expected ~{expected}, got {dist}"
        
        # Test antipodal points (should be ~20000 km)
        p_north = np.array([90, 0])
        p_south = np.array([-90, 0])
        dist_antipodal = space.distance(p_north, p_south)
        expected_antipodal = np.pi * 6371000  # Half circumference
        assert np.isclose(dist_antipodal, expected_antipodal, rtol=0.01)
    
    @staticmethod
    def test_manhattan_distance():
        """Test Manhattan (L1) distance"""
        from babilistic import ManhattanSpace
        
        space = ManhattanSpace(2)
        
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])
        
        dist = space.distance(p1, p2)
        expected = 7.0  # |3| + |4|
        assert np.isclose(dist, expected), f"Expected {expected}, got {dist}"
    
    @staticmethod
    def test_area_elements():
        """Test area element computation"""
        from babilistic import EuclideanSpace, GeoSpace as SphericalSpace
        
        # Euclidean: constant area element
        euclidean = EuclideanSpace(2)
        points = np.array([[0, 0], [1, 1], [2, 2]])
        areas = euclidean.area_element(points)
        assert np.allclose(areas, 1.0), "Euclidean area should be constant"
        
        # Spherical: varies with latitude
        spherical = SphericalSpace()
        points_sphere = np.array([[0, 0], [45, 0], [90, 0]])  # Equator, mid, pole
        areas_sphere = spherical.area_element(points_sphere)
        
        # Area should decrease from equator to pole
        assert areas_sphere[0] > areas_sphere[1] > areas_sphere[2], \
            "Spherical area should decrease toward poles"
    
    @staticmethod
    def test_grid_creation():
        """Test grid generation for all spaces"""
        from babilistic import EuclideanSpace, GeoSpace as SphericalSpace
        
        # Euclidean grid
        euclidean = EuclideanSpace(2)
        grid = euclidean.create_grid((0, 1, 0, 1), 10)
        
        assert 'points' in grid
        assert 'weights' in grid
        assert grid['shape'] == (10, 10)
        assert grid['points'].shape == (10, 10, 2)
        
        # Spherical grid
        spherical = SphericalSpace()
        grid_sphere = spherical.create_grid((-10, 10, -10, 10), 20)
        
        assert 'LAT' in grid_sphere
        assert 'LON' in grid_sphere
        assert grid_sphere['points'].shape[-1] == 2  # [lat, lon]

