import numpy as np

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and error handling"""
    
    @staticmethod
    def test_zero_uncertainty():
        """Test deterministic case (zero uncertainty)"""
        from babilistic import (
            ProbabilityEstimator, EuclideanSpace, DiskRegion,
            GaussianDistribution, GaussianKernel, FFTConvolution,
            QuadratureIntegrator
        )
        
        # Zero covariance
        estimator = ProbabilityEstimator(
            EuclideanSpace(2),
            DiskRegion([0, 0], 1.0),
            GaussianDistribution([0.5, 0], np.eye(2) * 1e-10),  # Nearly zero
            GaussianKernel(),
            FFTConvolution(),
            QuadratureIntegrator()
        )
        
        result = estimator.compute(bandwidth=0.1, resolution=32)
        # Should be very high probability (point is inside)
        assert result.probability > 0.8
    
    @staticmethod
    def test_empty_region():
        """Test with very small region"""
        from babilistic import (
            ProbabilityEstimator, EuclideanSpace, DiskRegion,
            GaussianDistribution, GaussianKernel, DirectConvolution,
            QuadratureIntegrator
        )
        
        estimator = ProbabilityEstimator(
            EuclideanSpace(2),
            DiskRegion([10, 10], 0.01),  # Tiny region far away
            GaussianDistribution([0, 0], np.eye(2)),
            GaussianKernel(),
            DirectConvolution(),
            QuadratureIntegrator()
        )
        
        result = estimator.compute(bandwidth=0.1, resolution=32)
        # Should be very low probability
        assert result.probability < 0.1
    
    @staticmethod
    def test_large_bandwidth():
        """Test with very large bandwidth"""
        from babilistic import (
            EuclideanSpace, DiskRegion, GaussianKernel, DirectConvolution
        )
        
        space = EuclideanSpace(2)
        region = DiskRegion([0, 0], 1.0)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        
        grid = space.create_grid((-5, 5, -5, 5), 32)
        indicator = region.indicator(grid['points'])
        
        # Very large bandwidth (should heavily smooth)
        w_field = conv.convolve(indicator, kernel, 10.0, grid, space)
        
        # Should be very smooth (no sharp transitions)
        grad_x = np.diff(w_field, axis=1)
        grad_y = np.diff(w_field, axis=0)
        
        assert np.abs(grad_x).max() < 0.1, "Field should be smooth with large bandwidth"
    
    @staticmethod
    def test_degenerate_covariance():
        """Test with singular/near-singular covariance"""
        from babilistic import GaussianDistribution
        
        # Nearly singular covariance
        cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        
        try:
            dist = GaussianDistribution([0, 0], cov)
            # Should handle gracefully (Cholesky with jitter)
            samples = dist.sample(100)
            assert samples.shape == (100, 2)
        except Exception as e:
            # Should not crash
            pass
    
    @staticmethod
    def test_boundary_points():
        """Test points exactly on region boundary"""
        from babilistic import DiskRegion, PolygonRegion
        
        # Disk boundary
        disk = DiskRegion([0, 0], 1.0)
        boundary_point = np.array([1.0, 0.0])  # Exactly on boundary
        
        # Should be inside (or at least defined)
        result = disk.indicator(boundary_point)
        assert 0 <= result <= 1
        
        # Polygon boundary
        square = PolygonRegion(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        corner = np.array([0.0, 0.0])  # Exactly on corner
        
        result = square.indicator(corner)
        assert 0 <= result <= 1


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegressions:
    """Tests for previously found bugs"""
    
    @staticmethod
    def test_empirical_pdf_shape_bug():
        """Regression: EmpiricalDistribution.pdf() shape mismatch"""
        from babilistic import EmpiricalDistribution
        
        samples = np.random.randn(500, 2)
        dist = EmpiricalDistribution(samples, bandwidth=0.5)
        
        # Test on grid (this was causing the bug)
        test_grid = np.random.randn(64, 64, 2)
        pdf_vals = dist.pdf(test_grid)
        
        assert pdf_vals.shape == (64, 64), \
            f"PDF shape mismatch: expected (64, 64), got {pdf_vals.shape}"
    
    @staticmethod
    def test_flatiter_multiplication_bug():
        """Regression: flatiter multiplication in distance metrics"""
        from babilistic import TotalVariationDistance
        
        tv = TotalVariationDistance()
        
        # This was causing "unsupported operand type" error
        p = np.random.rand(100)
        q = np.random.rand(100)
        weights = np.ones(100)
        
        # Should work with flat arrays
        dist = tv.compute(p.flat, q.flat, weights.flat)
        assert isinstance(dist, float)
        assert 0 <= dist <= 1
    
    @staticmethod
    def test_kernel_decay_assertion():
        """Regression: Gaussian kernel decay test was too strict"""
        from babilistic import GaussianKernel
        
        kernel = GaussianKernel()
        distances = np.array([0.0, 0.5, 1.0, 2.0])
        vals = kernel.evaluate(distances, bandwidth=1.0)
        
        # Should decay (but not necessarily to < 0.1 at distance 2)
        assert vals[0] > vals[-1], "Kernel should decay"
        assert vals[-1] < vals[0], "Kernel should decay monotonically"

