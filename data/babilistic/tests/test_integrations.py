import numpy as np


from babilistic import (
    ProbabilityEstimator,
    EuclideanSpace,
    GeoSpace,
    ManhattanSpace,
    DiskRegion, 
    EllipseRegion,
    PolygonRegion,
    BufferedPolygonRegion,
    TotalVariationDistance,
    WassersteinDistance,
    GaussianDistribution,
    MixtureDistribution,
    StudentTDistribution,
    LogNormalDistribution,
    EmpiricalDistribution,
    UniformKernel,
    GaussianKernel,
    EpanechnikovKernel,
    TriangularKernel,
    QuarticKernel,
    DirectConvolution,
    SparseConvolution,
    FFTConvolution,
    QuadratureIntegrator,
    MonteCarloIntegrator,
    DistributionComparator,
)



# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_full_pipeline_euclidean(self):
        """Test complete pipeline with Euclidean space"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0.5, 0]), cov=np.eye(2) * 0.1)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.1, resolution=64)
        
        assert 0 <= result.probability <= 1
        assert result.error_estimate >= 0
        assert result.w_field.shape == result.grid['shape']
    
    def test_full_pipeline_geospace(self):
        """Test complete pipeline with GeoSpace"""
        space = GeoSpace(radius=6371.0)
        region = DiskRegion(center=np.array([37.7749, -122.4194]), radius=1.0, metric_space=space)
        query = GaussianDistribution(
            mean=np.array([37.7749, -122.4194]),
            cov=np.diag([0.01, 0.01])
        )
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.5, resolution=32)
        
        assert 0 <= result.probability <= 1
    
    def test_mixture_distribution_pipeline(self):
        """Test with mixture distribution"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=2.0)
        
        comp1 = GaussianDistribution(mean=np.array([-0.5, 0]), cov=np.eye(2) * 0.1)
        comp2 = GaussianDistribution(mean=np.array([0.5, 0]), cov=np.eye(2) * 0.1)
        query = MixtureDistribution(
            components=[comp1, comp2],
            weights=np.array([0.6, 0.4])
        )
        
        kernel = EpanechnikovKernel()
        conv = SparseConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.3, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_student_t_distribution_pipeline(self):
        """Test with Student-t distribution (heavy tails)"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = StudentTDistribution(
            mean=np.array([0, 0]),
            cov=np.eye(2) * 0.2,
            df=3.0
        )
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.2, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_lognormal_distribution_pipeline(self):
        """Test with LogNormal distribution"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([1, 1]), radius=2.0)
        query = LogNormalDistribution(
            mean=np.array([0, 0]),
            cov=np.eye(2) * 0.1
        )
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.2, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_empirical_distribution_pipeline(self):
        """Test with Empirical distribution"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.5)
        
        # Generate samples from bimodal distribution
        samples = np.vstack([
            np.random.randn(50, 2) * 0.3 - 0.5,
            np.random.randn(50, 2) * 0.3 + 0.5
        ])
        query = EmpiricalDistribution(samples, bandwidth=0.2)
        
        kernel = TriangularKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.2, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_ellipse_region_pipeline(self):
        """Test with EllipseRegion"""
        space = EuclideanSpace(2)
        region = EllipseRegion(
            center=np.array([0, 0]),
            semi_axes=np.array([2.0, 1.0]),
            rotation_deg=30.0
        )
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.3)
        kernel = QuarticKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.3, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_polygon_region_pipeline(self):
        """Test with PolygonRegion"""
        space = EuclideanSpace(2)
        vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        region = PolygonRegion(vertices)
        query = GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.2)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.2, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_buffered_polygon_positive_buffer(self):
        """Test BufferedPolygonRegion with positive buffer"""
        space = EuclideanSpace(2)
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        region = BufferedPolygonRegion(vertices, buffer=0.2)
        query = GaussianDistribution(mean=np.array([0.5, 0.5]), cov=np.eye(2) * 0.1)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.1, resolution=64)
        
        assert 0 <= result.probability <= 1
    
    def test_buffered_polygon_negative_buffer(self):
        """Test BufferedPolygonRegion with negative buffer (erosion)"""
        space = EuclideanSpace(2)
        vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        region = BufferedPolygonRegion(vertices, buffer=-0.3)
        query = GaussianDistribution(mean=np.array([1, 1]), cov=np.eye(2) * 0.1)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.1, resolution=64)
        
        assert 0 <= result.probability <= 1

    def test_full_system_all_components(self):
        """Test complete system with all component types"""
        
        # Test all combinations
        test_configs = [
            # (space, region_center, query_mean, kernel, conv, integrator)
            (EuclideanSpace(2), [0, 0], [0.5, 0], GaussianKernel(), DirectConvolution(), QuadratureIntegrator()),
            (EuclideanSpace(2), [0, 0], [0, 0], EpanechnikovKernel(), SparseConvolution(), MonteCarloIntegrator(500)),
            (ManhattanSpace(2), [0, 0], [0.5, 0.5], UniformKernel(), DirectConvolution(), QuadratureIntegrator()),
            (GeoSpace(6371), [37.7749, -122.4194], [37.7750, -122.4195], TriangularKernel(), DirectConvolution(), QuadratureIntegrator()),
        ]
        
        for space, center, query_mean, kernel, conv, integrator in test_configs:
            region = DiskRegion(center=np.array(center), radius=1.0, metric_space=space)
            query = GaussianDistribution(mean=np.array(query_mean), cov=np.eye(2) * 0.1)
            
            estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
            result = estimator.compute(bandwidth=0.3, resolution=32)
            
            assert 0 <= result.probability <= 1
            assert result.error_estimate >= 0
            assert result.w_field is not None
            assert result.grid is not None
            assert 'space' in result.metadata