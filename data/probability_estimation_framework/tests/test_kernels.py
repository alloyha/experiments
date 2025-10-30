import numpy as np

# ============================================================================
# UNIT TESTS: KERNELS
# ============================================================================

class TestKernels:
    """Unit tests for all Kernel implementations"""
    
    @staticmethod
    def test_kernel_normalization():
        """Verify all kernels are properly normalized"""
        from babilistic import (
            GaussianKernel, EpanechnikovKernel, TriangularKernel,
            QuarticKernel, UniformKernel, MaternKernel
        )
        
        kernels = [
            ("Gaussian", GaussianKernel()),
            ("Epanechnikov", EpanechnikovKernel()),
            ("Triangular", TriangularKernel()),
            ("Quartic", QuarticKernel()),
            ("Uniform", UniformKernel()),
            ("Matern(2.5)", MaternKernel(nu=2.5))
        ]
        
        bandwidth = 1.0
        
        for name, kernel in kernels:
            # Value at origin should be maximum
            val_at_zero = kernel.evaluate(np.array([0.0]), bandwidth)[0]
            val_away = kernel.evaluate(np.array([0.5]), bandwidth)[0]
            
            assert val_at_zero >= val_away, \
                f"{name}: kernel should be maximum at origin"
            
            # Compact support kernels should be zero beyond bandwidth
            if kernel.is_compact():
                support = kernel.support_radius(bandwidth)
                val_beyond = kernel.evaluate(np.array([support * 1.1]), bandwidth)[0]
                assert np.isclose(val_beyond, 0.0), \
                    f"{name}: compact kernel not zero beyond support"
    
    @staticmethod
    def test_kernel_monotonicity():
        """Test that kernels decay monotonically"""
        from babilistic import GaussianKernel, MaternKernel
        
        distances = np.linspace(0, 3, 100)
        
        for kernel in [GaussianKernel(), MaternKernel(nu=2.5)]:
            vals = kernel.evaluate(distances, bandwidth=1.0)
            
            # Check monotonic decrease
            diffs = np.diff(vals)
            assert np.all(diffs <= 0), "Kernel should be monotonically decreasing"
    
    @staticmethod
    def test_matern_special_cases():
        """Test Matérn kernel special cases"""
        from babilistic import MaternKernel
        
        distances = np.array([0.0, 0.5, 1.0, 2.0])
        
        # Test different nu values
        for nu in [0.5, 1.5, 2.5]:
            kernel = MaternKernel(nu=nu)
            vals = kernel.evaluate(distances, bandwidth=1.0)
            
            assert np.isclose(vals[0], 1.0), f"Matérn({nu}) should be 1 at origin"
            assert vals[-1] < vals[0], f"Matérn({nu}) should decay"
    
    @staticmethod
    def test_kernel_properties():
        """Test mathematical properties of kernels"""
        from babilistic import EpanechnikovKernel, QuarticKernel
        
        # Epanechnikov: (1 - t²) for |t| ≤ 1
        epan = EpanechnikovKernel()
        assert np.isclose(epan.evaluate(np.array([0.0]), 1.0)[0], 1.0)
        assert np.isclose(epan.evaluate(np.array([0.5]), 1.0)[0], 0.75)
        assert np.isclose(epan.evaluate(np.array([1.0]), 1.0)[0], 0.0)
        
        # Quartic: (15/16)(1 - t²)² for |t| ≤ 1
        quartic = QuarticKernel()
        assert np.isclose(quartic.evaluate(np.array([0.0]), 1.0)[0], 15.0/16.0)


def test_gaussian_kernel_evaluate_and_support():
    from babilistic import GaussianKernel

    k = GaussianKernel()
    d = np.array([0.0, 1.0, 3.0])
    vals = k.evaluate(d, bandwidth=1.0)

    assert np.isclose(vals[0], 1.0)
    assert vals[1] < vals[0]
    # 3-sigma cutoff -> support_radius should be 3.0
    assert np.isclose(k.support_radius(1.0), 3.0)


def test_epanechnikov_kernel_support_and_boundary():
    from babilistic import EpanechnikovKernel

    k = EpanechnikovKernel()
    d = np.array([0.0, 1.0, 1.5])
    vals = k.evaluate(d, bandwidth=1.0)

    # At d=0 -> max value
    assert vals[0] > 0.0
    # At boundary d=1.0 -> value == 0
    assert np.isclose(vals[1], 0.0)
    # Outside support -> zero
    assert vals[2] == 0.0
    assert np.isclose(k.support_radius(1.0), 1.0)