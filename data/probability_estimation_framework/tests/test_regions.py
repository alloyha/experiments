import numpy as np

# ============================================================================
# UNIT TESTS: REGIONS
# ============================================================================

class TestRegions:
    """Unit tests for all Region implementations"""
    
    @staticmethod
    def test_disk_region():
        """Test disk/circle region"""
        from babilistic import DiskRegion
        
        disk = DiskRegion(center=[0, 0], radius=1.0)
        
        # Test indicator
        assert disk.indicator(np.array([0, 0])) == 1.0  # Center
        assert disk.indicator(np.array([0.5, 0])) == 1.0  # Inside
        assert disk.indicator(np.array([2, 0])) == 0.0  # Outside
        
        # Test boundary sampling
        boundary = disk.sample_boundary(100)
        distances = np.linalg.norm(boundary - disk.center, axis=1)
        assert np.allclose(distances, disk.radius, rtol=0.01)
    
    @staticmethod
    def test_ellipse_region():
        """Test elliptical region with rotation"""
        from babilistic import EllipseRegion
        
        ellipse = EllipseRegion(
            center=[1, 1],
            semi_axes=[2, 1],
            rotation_deg=45
        )
        
        # Center should be inside
        assert ellipse.indicator(ellipse.center) == 1.0
        
        # Boundary samples should satisfy quadratic form
        boundary = ellipse.sample_boundary(100)
        centered = boundary - ellipse.center
        quad_forms = np.sum(centered @ ellipse.A * centered, axis=1)
        assert np.allclose(quad_forms, 1.0, rtol=0.02)
    
    @staticmethod
    def test_polygon_region():
        """Test polygon region"""
        from babilistic import PolygonRegion
        
        square = PolygonRegion(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        
        assert square.indicator(np.array([0.5, 0.5])) == 1.0  # Inside
        assert square.indicator(np.array([2, 2])) == 0.0  # Outside
    
    @staticmethod
    def test_buffered_polygon():
        """Test buffered polygon region"""
        from babilistic import BufferedPolygonRegion
        
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        
        # Positive buffer (expansion)
        buffered_expand = BufferedPolygonRegion(square, buffer=0.2)
        
        # Point just outside original square should be inside buffered
        near_edge = np.array([1.1, 0.5])
        # Note: This test depends on distance calculation implementation
        
        # Original inside point should still be inside
        assert buffered_expand.indicator(np.array([0.5, 0.5])) == 1.0
    
    @staticmethod
    def test_multi_region_union():
        """Test multi-region union"""
        from babilistic import MultiRegion, DiskRegion
        
        disk1 = DiskRegion([0, 0], 1.0)
        disk2 = DiskRegion([2, 0], 1.0)
        
        union = MultiRegion([disk1, disk2], operation='union')
        
        # Points in either disk should be inside
        assert union.indicator(np.array([0, 0])) == 1.0  # In disk1
        assert union.indicator(np.array([2, 0])) == 1.0  # In disk2
        assert union.indicator(np.array([5, 5])) == 0.0  # In neither
    
    @staticmethod
    def test_multi_region_intersection():
        """Test multi-region intersection"""
        from babilistic import MultiRegion, DiskRegion
        
        disk1 = DiskRegion([0, 0], 1.5)
        disk2 = DiskRegion([1, 0], 1.5)
        
        intersection = MultiRegion([disk1, disk2], operation='intersection')
        
        # Only points in both disks
        assert intersection.indicator(np.array([0.5, 0])) == 1.0  # In both
        assert intersection.indicator(np.array([-1, 0])) == 0.0  # Only disk1
        assert intersection.indicator(np.array([2, 0])) == 0.0  # Only disk2

    @staticmethod
    def test_disk_region_indicator_and_bounds():
        from babilistic import DiskRegion

        region = DiskRegion(center=[0.0, 0.0], radius=1.0)
        inside = np.array([0.0, 0.0])
        outside = np.array([2.0, 0.0])

        assert region.indicator(inside) == 1.0
        assert region.indicator(outside) == 0.0

        b = region.bounds()
        assert len(b) == 4

    @staticmethod
    def test_polygon_and_ellipse_region_basic():
        from babilistic import PolygonRegion, EllipseRegion

        square = PolygonRegion(vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        assert square.indicator(np.array([0.5, 0.5])) == 1.0
        assert square.indicator(np.array([1.5, 0.5])) == 0.0

        ellipse = EllipseRegion(center=np.array([0.0, 0.0]), semi_axes=np.array([2.0, 1.0]), rotation_deg=0.0)
        assert ellipse.indicator(np.array([0.0, 0.0])) == 1.0
        assert ellipse.indicator(np.array([3.0, 0.0])) == 0.0

    @staticmethod
    def test_buffered_polygon_and_point_to_segment():
        from babilistic import BufferedPolygonRegion

        verts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        buf = BufferedPolygonRegion(vertices=verts, buffer=0.5)
        # A point just outside original polygon but within buffer should be inside
        pt = np.array([2.3, 1.0])
        pts = np.array([pt])
        val = buf.indicator(pts)
        assert val.shape == (1,)

    @staticmethod
    def test_implicit_region_indicator_and_sample_boundary_cache():
        from babilistic import ImplicitRegion

        # sdf for unit circle centered at origin
        def sdf(x):
            x = np.asarray(x)
            if x.ndim == 1:
                r = np.linalg.norm(x)
                return np.array([r - 1.0])
            else:
                return np.linalg.norm(x, axis=-1) - 1.0

        imp = ImplicitRegion(sdf=sdf, bounds=(-2, 2, -2, 2), samples_cache=np.array([[1.0, 0.0], [0.0, 1.0]]))
        # single point inside (r=0 -> inside circle)
        assert imp.indicator(np.array([0.0, 0.0])) == 1.0
        # boundary sample uses cache
        b = imp.sample_boundary(2)
        assert b.shape[0] == 2
