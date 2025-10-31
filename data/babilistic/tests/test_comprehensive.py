import numpy as np


from babilistic import *

class TestAllComponentsWorking:
    """Verify all major components work after fixes"""
    
    def test_all_metric_spaces(self):
        """Test all metric space types work"""
        spaces = [
            EuclideanSpace(2),
            ManhattanSpace(2),
            SphericalSpace(radius=6371.0),
            GeoSpace(radius=6371.0),
        ]
        
        for space in spaces:
            # Should be able to create grid
            grid = space.create_grid((0, 1, 0, 1), 10)
            assert 'points' in grid
            assert 'weights' in grid
    
    def test_all_regions(self):
        """Test all region types work"""
        regions = [
            DiskRegion(center=np.array([0, 0]), radius=1.0),
            PolygonRegion(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])),
            EllipseRegion(center=np.array([0, 0]), semi_axes=np.array([2, 1])),
            ImplicitRegion(lambda x: np.linalg.norm(x, axis=-1) - 1.0, bounds=(-2, 2, -2, 2)),
            BufferedPolygonRegion(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), buffer=0.1),
            MultiRegion([DiskRegion(np.array([0, 0]), 1.0)], operation='union'),
        ]
        
        test_point = np.array([0.5, 0.5])
        for region in regions:
            result = region.indicator(test_point)
            assert isinstance(result, (float, np.floating, np.ndarray))
    
    def test_all_distributions(self):
        """Test all distribution types work"""
        distributions = [
            GaussianDistribution(mean=np.array([0, 0]), cov=np.eye(2)),
            UniformDistribution(bounds=(-1, 1, -1, 1)),
            MixtureDistribution(
                [GaussianDistribution(np.array([0, 0]), np.eye(2))],
                weights=np.array([1.0])
            ),
            StudentTDistribution(mean=np.array([0, 0]), cov=np.eye(2), df=5.0),
            LogNormalDistribution(mean=np.array([0, 0]), cov=np.eye(2) * 0.1),
            EmpiricalDistribution(np.random.randn(100, 2)),
        ]
        
        test_point = np.array([0.5, 0.5])
        for dist in distributions:
            pdf_val = dist.pdf(test_point)
            assert pdf_val >= 0
    
    def test_all_kernels(self):
        """Test all kernel types work"""
        kernels = [
            GaussianKernel(),
            EpanechnikovKernel(),
            UniformKernel(),
            TriangularKernel(),
            QuarticKernel(),
            MaternKernel(nu=2.5),
        ]
        
        distances = np.array([0, 0.5, 1.0, 2.0])
        for kernel in kernels:
            vals = kernel.evaluate(distances, bandwidth=1.0)
            assert len(vals) == len(distances)
            assert np.all(vals >= 0)
    
    def test_complete_pipeline(self):
        """Test complete probability estimation pipeline"""
        space = EuclideanSpace(2)
        region = DiskRegion(center=np.array([0, 0]), radius=1.0)
        query = GaussianDistribution(mean=np.array([0.5, 0]), cov=np.eye(2) * 0.2)
        kernel = GaussianKernel()
        conv = DirectConvolution()
        integrator = QuadratureIntegrator()
        
        estimator = ProbabilityEstimator(space, region, query, kernel, conv, integrator)
        result = estimator.compute(bandwidth=0.3, resolution=64)
        
        assert 0 <= result.probability <= 1
        assert result.error_estimate >= 0
        assert result.w_field is not None
        assert result.grid is not None