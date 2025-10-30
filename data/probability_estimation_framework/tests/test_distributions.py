import numpy as np
import pytest

# ============================================================================
# UNIT TESTS: DISTRIBUTIONS
# ============================================================================

class TestDistributions:
    """Unit tests for all Distribution implementations"""
    
    @staticmethod
    def test_gaussian_properties():
        """Test Gaussian distribution properties"""
        from babilistic import GaussianDistribution
        
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        
        dist = GaussianDistribution(mean, cov)
        
        # PDF at mean
        pdf_mean = dist.pdf(mean)
        expected = 1.0 / (2 * np.pi)
        assert np.isclose(pdf_mean, expected, rtol=0.01)
        
        # Samples
        samples = dist.sample(10000)
        assert samples.shape == (10000, 2)
        assert np.allclose(samples.mean(axis=0), mean, atol=0.1)
        assert np.allclose(np.cov(samples.T), cov, atol=0.1)
    
    @staticmethod
    def test_student_t_heavy_tails():
        """Test Student-t has heavier tails than Gaussian"""
        from babilistic import StudentTDistribution, GaussianDistribution
        
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        
        student = StudentTDistribution(mean, cov, df=3)
        gaussian = GaussianDistribution(mean, cov)
        
        # At origin, both should be similar
        pdf_t_center = student.pdf(mean)
        pdf_g_center = gaussian.pdf(mean)
        
        # Far from center, Student-t should have higher PDF (heavier tails)
        far_point = np.array([5.0, 5.0])
        pdf_t_tail = student.pdf(far_point)
        pdf_g_tail = gaussian.pdf(far_point)
        
        assert pdf_t_tail > pdf_g_tail, "Student-t should have heavier tails"
    
    @staticmethod
    def test_lognormal_positivity():
        """Test log-normal is always positive"""
        from babilistic import LogNormalDistribution
        
        dist = LogNormalDistribution(mean=np.array([0.0, 0.0]), cov=np.eye(2) * 0.5)
        
        # PDF should be zero for non-positive values
        negative_point = np.array([-1.0, 1.0])
        pdf_neg = dist.pdf(negative_point)
        assert pdf_neg == 0.0, "Log-normal PDF should be zero for negative values"
        
        # Samples should all be positive
        samples = dist.sample(1000)
        assert np.all(samples > 0), "All log-normal samples must be positive"
    
    @staticmethod
    def test_empirical_distribution():
        """Test empirical distribution from samples"""
        from babilistic import EmpiricalDistribution
        
        # Generate synthetic data
        true_mean = np.array([2.0, -1.0])
        samples = np.random.randn(1000, 2) + true_mean
        
        dist = EmpiricalDistribution(samples, bandwidth=0.5)
        
        # Mean should match
        assert np.allclose(dist.mean(), true_mean, atol=0.2)
        
        # PDF should be highest near data
        pdf_near = dist.pdf(true_mean)
        pdf_far = dist.pdf(np.array([10.0, 10.0]))
        assert pdf_near > pdf_far, "PDF should be higher near data"
        
        # Resampling should give similar distribution
        resamples = dist.sample(500)
        assert np.allclose(resamples.mean(axis=0), true_mean, atol=0.5)
    
    @staticmethod
    def test_mixture_distribution():
        """Test mixture distribution composition"""
        from babilistic import MixtureDistribution, GaussianDistribution
        
        comp1 = GaussianDistribution(np.array([-2, 0]), np.eye(2) * 0.5)
        comp2 = GaussianDistribution(np.array([2, 0]), np.eye(2) * 0.5)
        
        mixture = MixtureDistribution([comp1, comp2], weights=np.array([0.3, 0.7]))
        
        # Mean should be weighted average
        expected_mean = 0.3 * comp1.mean() + 0.7 * comp2.mean()
        assert np.allclose(mixture.mean(), expected_mean)
        
        # PDF should be weighted sum
        test_point = np.array([0.0, 0.0])
        pdf_mix = mixture.pdf(test_point)
        pdf_expected = 0.3 * comp1.pdf(test_point) + 0.7 * comp2.pdf(test_point)
        assert np.isclose(pdf_mix, pdf_expected)

    @staticmethod
    def test_gaussian_pdf_single_and_grid_and_mean():
        from babilistic import GaussianDistribution

        mean = np.array([0.5, -0.2])
        cov = np.array([[0.2, 0.05], [0.05, 0.1]])
        g = GaussianDistribution(mean, cov)

        # Single point (1D input) -> scalar pdf
        p0 = g.pdf(np.array([0.5, -0.2]))
        assert np.isfinite(p0)
        assert p0 > 0.0

        # Grid input -> shape (N,)
        grid = np.array([[0.5, -0.2], [0.6, -0.1], [1.0, 0.0]])
        p_grid = g.pdf(grid)
        assert p_grid.shape == (3,)
        assert np.all(np.isfinite(p_grid))

        # mean() returns the same mean
        assert np.allclose(g.mean(), mean)

    @staticmethod
    def test_gaussian_near_singular_covariance():
        from babilistic import GaussianDistribution

        mean = np.array([0.0, 0.0])
        cov = np.eye(2) * 1e-10
        g = GaussianDistribution(mean, cov)

        # PDF at mean should be finite (may be large) and mean() should match
        p0 = g.pdf(np.array([0.0, 0.0]))
        assert np.isfinite(p0)
        assert np.allclose(g.mean(), mean)

    @staticmethod
    def test_studentt_pdf_and_mean():
        from babilistic import StudentTDistribution

        mean = np.array([1.0, 2.0])
        cov = np.eye(2) * 0.5

        t1 = StudentTDistribution(mean=mean, cov=cov, df=3.0)
        t2 = StudentTDistribution(mean=mean, cov=cov, df=30.0)

        # PDF evaluated at mean should be finite and positive
        p1 = t1.pdf(np.array([1.0, 2.0]))
        p2 = t2.pdf(np.array([1.0, 2.0]))
        assert np.isfinite(p1) and p1 > 0.0
        assert np.isfinite(p2) and p2 > 0.0

        # mean() returns provided mean
        assert np.allclose(t1.mean(), mean)
        assert np.allclose(t2.mean(), mean)

    @staticmethod
    def test_empirical_kde_pdf_and_sample_shape():
        from babilistic import EmpiricalDistribution

        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        ed = EmpiricalDistribution(samples, bandwidth=0.5)

        # PDF at a sample point
        p_sample = ed.pdf(np.array([0.0, 0.0]))
        assert np.isfinite(p_sample)

        # Sampling returns correct shape and finite values
        s = ed.sample(5)
        assert s.shape == (5, 2)
        assert np.all(np.isfinite(s))

def test_gaussian_array_shapes():
        """Cover different array shape handling (line 55)"""
        from babilistic.distributions import GaussianDistribution
        
        dist = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2))
        
        # Test 1D input
        single_point = np.array([0.5, 0.5])
        pdf_single = dist.pdf(single_point)
        assert isinstance(pdf_single, (float, np.floating))
        
        # Test 2D array (batch)
        batch_points = np.random.randn(10, 2)
        pdf_batch = dist.pdf(batch_points)
        assert pdf_batch.shape == (10,)
        
        # Test 3D grid
        grid = np.random.randn(5, 5, 2)
        pdf_grid = dist.pdf(grid)
        assert pdf_grid.shape == (5, 5)
    
def test_student_t_edge_cases():
    """Cover Student-t edge cases (lines 99-105)"""
    from babilistic.distributions import StudentTDistribution
    
    # Very low df (heavy tails)
    dist_heavy = StudentTDistribution([0, 0], np.eye(2), df=1.0)
    pdf_heavy = dist_heavy.pdf(np.array([5, 5]))  # Far from mean
    assert pdf_heavy > 0
    
    # High df (near Gaussian)
    dist_light = StudentTDistribution([0, 0], np.eye(2), df=100.0)
    pdf_light = dist_light.pdf(np.array([0, 0]))
    assert pdf_light > 0
    
    # Test invalid df
    with pytest.raises(ValueError):
        StudentTDistribution([0, 0], np.eye(2), df=-1)

def test_lognormal_edge_values():
    """Cover log-normal boundary cases (lines 191-199)"""
    from babilistic.distributions import LogNormalDistribution
    
    dist = LogNormalDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.1)
    
    # Test with zeros (should be 0 or very small)
    pdf_zero = dist.pdf(np.array([0.0, 1.0]))
    assert pdf_zero >= 0
    
    # Test with negatives (should be 0)
    pdf_neg = dist.pdf(np.array([-1.0, -1.0]))
    assert pdf_neg == 0.0
    
    # Test with large values
    pdf_large = dist.pdf(np.array([100.0, 100.0]))
    assert pdf_large >= 0

def test_empirical_kde_bandwidth():
    """Cover empirical distribution bandwidth selection (lines 264-278)"""
    from babilistic.distributions import EmpiricalDistribution
    
    samples = np.random.randn(100, 2)
    
    # Test with automatic bandwidth (Scott's rule)
    dist_auto = EmpiricalDistribution(samples, bandwidth=None)
    assert dist_auto.bandwidth > 0
    
    # Test with manual bandwidth
    dist_manual = EmpiricalDistribution(samples, bandwidth=0.5)
    assert dist_manual.bandwidth == 0.5
    
    # Test PDF computation
    test_point = np.array([0, 0])
    pdf = dist_manual.pdf(test_point)
    assert pdf > 0

def test_mixture_edge_cases():
    """Cover mixture distribution edge cases"""
    from babilistic.distributions import MixtureDistribution, GaussianDistribution
    
    comp1 = GaussianDistribution([0, 0], np.eye(2))
    comp2 = GaussianDistribution([2, 2], np.eye(2))
    
    # Test with single component
    single_mix = MixtureDistribution([comp1], weights=np.array([1.0]))
    pdf = single_mix.pdf(np.array([0, 0]))
    assert pdf > 0
    
    # Test with unequal weights
    unequal_mix = MixtureDistribution([comp1, comp2], weights=np.array([0.9, 0.1]))
    pdf = unequal_mix.pdf(np.array([0, 0]))
    assert pdf > 0