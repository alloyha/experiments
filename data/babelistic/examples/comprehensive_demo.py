"""
Comprehensive examples for the `babelistic` package.

These examples were extracted from `babelistic/main.py` and placed here
so they are separate from the library code. Run this file directly to
execute the demonstrations.
"""

import numpy as np

from babelistic import (
    EuclideanSpace,
    PolygonRegion,
    DiskRegion,
    EllipseRegion,
    BufferedPolygonRegion,
    MultiRegion,
    GaussianDistribution,
    EmpiricalDistribution,
    StudentTDistribution,
    EpanechnikovKernel,
    GaussianKernel,
    TriangularKernel,
    QuarticKernel,
    MaternKernel,
    FFTConvolution,
    DirectConvolution,
    QuadratureIntegrator,
    ProbabilityEstimator,
    DistributionComparator,
    JSDistance,
    KLDivergence,
    WassersteinDistance,
    geofence_to_probability,
)


def example_custom_components():
    """Example: Compose custom configuration"""
    print("\n" + "="*70)
    print("EXAMPLE: Custom Component Composition")
    print("="*70 + "\n")

    # Scenario: Ship navigation with exclusion zone

    # 1. Define space (Euclidean ocean surface in km)
    space = EuclideanSpace(2)

    # 2. Define region (irregular polygon - rocks/shallow water)
    hazard_zone = PolygonRegion(vertices=np.array([
        [5, 2], [8, 1], [10, 4], [9, 7], [6, 8], [3, 6]
    ]))

    # 3. Define query (ship position with GPS uncertainty)
    ship_position = GaussianDistribution(
        mean=np.array([7.0, 5.0]),  # km
        cov=np.array([[0.5, 0.1], [0.1, 0.3]])  # km²
    )

    # 4. Choose kernel (Epanechnikov for compact support)
    kernel = EpanechnikovKernel()

    # 5. Choose convolution (FFT for speed)
    conv = FFTConvolution()

    # 6. Choose integrator (quadrature for accuracy)
    integrator = QuadratureIntegrator()

    # Assemble
    estimator = ProbabilityEstimator(
        metric_space=space,
        region=hazard_zone,
        query_distribution=ship_position,
        kernel=kernel,
        convolution_strategy=conv,
        integrator=integrator
    )

    # Compute risk
    result = estimator.compute(bandwidth=0.8, resolution=128)

    print(f"Ship Navigation Risk Assessment:")
    print(f"  Ship position: {ship_position.mean()} km")
    print(f"  P(ship in hazard zone) = {result.probability:.4f}")
    print(f"  Risk level: {'HIGH' if result.probability > 0.3 else 'MODERATE' if result.probability > 0.1 else 'LOW'}")
    print(f"  Components used:")
    for key, value in result.metadata.items():
        print(f"    - {key}: {value}")

    print("\n✓ Custom configuration executed successfully\n")


def example_probability_distance_analysis():
    """Example: Use probability distances for sensitivity analysis"""
    print("\n" + "="*70)
    print("EXAMPLE: Probability Distance for Sensitivity Analysis")
    print("="*70 + "\n")

    # Setup problem
    space = EuclideanSpace(2)
    region = DiskRegion(center=[1.0, 0.5], radius=0.8)
    query = GaussianDistribution(mean=np.array([0.8, 0.4]), 
                                  cov=np.array([[0.3, 0.05], [0.05, 0.2]]))

    kernel = GaussianKernel()
    conv = FFTConvolution()
    integrator = QuadratureIntegrator()

    estimator = ProbabilityEstimator(
        space, region, query, kernel, conv, integrator
    )

    # Analyze sensitivity to bandwidth using different distance metrics
    bandwidths = [0.05, 0.1, 0.2, 0.4, 0.8]

    print("Bandwidth Sensitivity Analysis:")
    print("-" * 70)

    for distance_name, distance_metric in [
        ("JS Distance", JSDistance()),
        ("Total Variation", None),
        ("Hellinger", None),
    ]:
        # For demo we only run JS distance (the others are similar)
        if distance_metric is None:
            continue
        print(f"\nUsing {distance_name}:")
        comparator = DistributionComparator(distance_metric)

        analysis = comparator.sensitivity_analysis(estimator, bandwidths, resolution=96)

        print(f"  Probabilities: {[f'{p:.4f}' for p in analysis['probabilities']]}")
        print(f"  Max variation: {analysis['max_variation']:.4f}")
        print(f"  Mean variation: {analysis['mean_variation']:.4f}")

        # Compare adjacent bandwidths
        print(f"  Distance between consecutive bandwidths:")
        for i in range(len(bandwidths) - 1):
            d = analysis['distance_matrix'][i, i+1]
            print(f"    h={bandwidths[i]:.2f} → h={bandwidths[i+1]:.2f}: {d:.4f}")

    print("\n" + "="*70)
    print("Interpretation:")
    print("  - Small distances → insensitive to bandwidth choice")
    print("  - Large distances → results depend strongly on bandwidth")
    print("  - Different metrics capture different aspects of variation")
    print("="*70 + "\n")

    # Compare two very different query distributions
    print("Distribution Comparison Example:")
    print("-" * 70)

    query2 = GaussianDistribution(mean=np.array([2.0, 2.0]),  # Far away
                                   cov=np.array([[0.3, 0], [0, 0.3]]))

    grid = space.create_grid(region.bounds(), 64)

    for distance_name, distance_metric in [
        ("KL Divergence (asymmetric)", KLDivergence()),
        ("JS Distance (symmetric)", JSDistance()),
        ("Wasserstein (geometric)", WassersteinDistance(space, approximate=True))
    ]:
        comparator = DistributionComparator(distance_metric)
        distance = comparator.compare_query_distributions(query, query2, grid)

        print(f"  {distance_name:30s}: {distance:.4f}")

    print("\n✓ Probability distance analysis completed\n")


def example_geofence_scenarios():
    """Example: Practical geofence scenarios"""
    print("\n" + "="*70)
    print("EXAMPLE: Geofence Scenarios")
    print("="*70 + "\n")

    # Mock distance function
    def mock_haversine(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1 = np.deg2rad(lat1)
        lam1 = np.deg2rad(lon1)
        phi2 = np.deg2rad(lat2)
        lam2 = np.deg2rad(lon2)
        dphi = phi2 - phi1
        dlam = lam2 - lam1
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return R * c

    # Scenario 1: Delivery radius compliance
    print("📦 Scenario 1: Delivery Radius Compliance")
    print("-" * 70)
    print("A delivery drone must stay within 500m of warehouse")
    print("GPS uncertainty: drone=15m, warehouse=5m")

    result1 = geofence_to_probability(
        subject_lat=37.7749,
        subject_lon=-122.4194,
        subject_uncertainty=15.0,
        reference_lat=37.7755,
        reference_lon=-122.4200,
        reference_uncertainty=5.0,
        distance_threshold=500.0,
        distance_metric=mock_haversine,
        resolution=96
    )

    print(f"Result: P(within 500m) = {result1.probability:.4f}")
    print(f"Status: {'✅ COMPLIANT' if result1.probability > 0.95 else '⚠️  UNCERTAIN' if result1.probability > 0.80 else '❌ NON-COMPLIANT'}")

    # Scenario 2: Restricted airspace
    print("\n✈️  Scenario 2: Restricted Airspace Avoidance")
    print("-" * 70)
    print("Aircraft must stay >2km from airport center")
    print("GPS uncertainty: aircraft=50m, airport=0m (known)")

    result2 = geofence_to_probability(
        subject_lat=37.6200,
        subject_lon=-122.3700,
        subject_uncertainty=50.0,
        reference_lat=37.6213,
        reference_lon=-122.3790,  # ~1.8km away
        reference_uncertainty=0.0,
        distance_threshold=2000.0,
        distance_metric=mock_haversine,
        resolution=96
    )

    # For exclusion zone, we want probability of being INSIDE to be LOW
    exclusion_prob = result2.probability
    compliance_prob = 1.0 - exclusion_prob

    print(f"Result: P(inside restricted) = {exclusion_prob:.4f}")
    print(f"        P(safely outside) = {compliance_prob:.4f}")
    print(f"Status: {'✅ SAFE' if exclusion_prob < 0.05 else '⚠️  UNCERTAIN' if exclusion_prob < 0.20 else '❌ VIOLATION RISK'}")

    # Scenario 3: Emergency response time
    print("\n🚑 Scenario 3: Emergency Response Coverage")
    print("-" * 70)
    print("Ambulance must reach patient within 5km")
    print("Location uncertainty: ambulance=30m, patient=100m")

    result3 = geofence_to_probability(
        subject_lat=40.7580,
        subject_lon=-73.9855,
        subject_uncertainty=30.0,
        reference_lat=40.7650,
        reference_lon=-73.9950,  # ~4.5km diagonal
        reference_uncertainty=100.0,
        distance_threshold=5000.0,
        distance_metric=mock_haversine,
        resolution=96
    )

    print(f"Result: P(within 5km) = {result3.probability:.4f}")
    print(f"Status: {'✅ IN RANGE' if result3.probability > 0.90 else '⚠️  UNCERTAIN' if result3.probability > 0.60 else '❌ OUT OF RANGE'}")

    # Scenario 4: Comparison of distance metrics
    print("\n🗺️  Scenario 4: Distance Metric Comparison")
    print("-" * 70)
    print("Urban delivery: comparing geometric vs Manhattan distance")

    # Manhattan distance mock
    def mock_manhattan(lat1, lon1, lat2, lon2):
        lat_diff_m = np.abs(lat1 - lat2) * 111132.954
        lat_mid = (lat1 + lat2) / 2.0 if np.isscalar(lat1) else lat1
        lon_scale = 111132.954 * np.cos(np.deg2rad(lat_mid))
        lon_diff_m = np.abs(lon1 - lon2) * lon_scale
        return lat_diff_m + lon_diff_m

    result4a = geofence_to_probability(
        subject_lat=40.7580,
        subject_lon=-73.9855,
        subject_uncertainty=20.0,
        reference_lat=40.7590,
        reference_lon=-73.9865,
        reference_uncertainty=15.0,
        distance_threshold=250.0,
        distance_metric=mock_haversine,
        resolution=64
    )

    result4b = geofence_to_probability(
        subject_lat=40.7580,
        subject_lon=-73.9855,
        subject_uncertainty=20.0,
        reference_lat=40.7590,
        reference_lon=-73.9865,
        reference_uncertainty=15.0,
        distance_threshold=250.0,
        distance_metric=mock_manhattan,
        resolution=64
    )

    print(f"Haversine (straight line): P(within 250m) = {result4a.probability:.4f}")
    print(f"Manhattan (grid streets):  P(within 250m) = {result4b.probability:.4f}")
    print(f"Difference: {abs(result4a.probability - result4b.probability):.4f}")
    print("Insight: Manhattan distance accounts for grid navigation constraints")

    print("\n" + "="*70)


def example_new_components():
    """Example: Showcase new kernels, distributions, and regions"""
    print("\n" + "="*70)
    print("EXAMPLE: New Components Showcase")
    print("="*70 + "\n")

    # Example 1: Matérn kernel for spatial correlation
    print("🌍 Example 1: Spatial Correlation with Matérn Kernel")
    print("-" * 70)
    print("Modeling spatially correlated GPS errors")

    space = EuclideanSpace(2)
    region = DiskRegion(center=[10.0, 5.0], radius=50.0)  # 50m radius zone

    # GPS with correlated errors (Student-t for robustness)
    query = StudentTDistribution(
        mean=np.array([12.0, 6.0]),
        cov=np.array([[25.0, 5.0], [5.0, 20.0]]),  # 5m × 4.5m ellipse
        df=5.0  # Heavy tails for outliers
    )

    # Matérn(2.5) for smooth spatial correlation
    kernel = MaternKernel(nu=2.5)

    estimator = ProbabilityEstimator(
        space, region, query, kernel, 
        DirectConvolution(), QuadratureIntegrator()
    )

    result1 = estimator.compute(bandwidth=8.0, resolution=64)

    print(f"Result: P(within 50m zone) = {result1.probability:.4f}")
    print(f"Components: Matérn(ν=2.5) kernel + Student-t(df=5) distribution")
    print(f"Interpretation: {result1.probability*100:.1f}% chance of compliance")

    # Example 2: Elliptical safety zone (anisotropic)
    print("\n🚁 Example 2: Elliptical Safety Zone")
    print("-" * 70)
    print("Helicopter landing zone with directional wind uncertainty")

    # Ellipse oriented along wind direction (45°)
    landing_zone = EllipseRegion(
        center=np.array([100.0, 100.0]),
        semi_axes=np.array([30.0, 15.0]),  # 30m × 15m
        rotation_deg=45.0  # Wind direction
    )

    # Aircraft position with log-normal uncertainty (always positive offsets)
    # Use empirical for simplicity
    aircraft_samples = np.random.randn(1000, 2) * np.array([10, 8]) + np.array([102, 98])
    aircraft_dist = EmpiricalDistribution(aircraft_samples, bandwidth=5.0)

    estimator2 = ProbabilityEstimator(
        space, landing_zone, aircraft_dist,
        TriangularKernel(), DirectConvolution(), QuadratureIntegrator()
    )

    result2 = estimator2.compute(bandwidth=6.0, resolution=64)

    print(f"Result: P(in landing zone) = {result2.probability:.4f}")
    print(f"Zone: 30m × 15m ellipse at 45° (wind-aligned)")
    print(f"Status: {'✅ SAFE TO LAND' if result2.probability > 0.95 else '⚠️  CAUTION' if result2.probability > 0.75 else '❌ ABORT'}")

    # Example 3: Multi-region composite geofence
    print("\n🏭 Example 3: Industrial Site Multi-Region")
    print("-" * 70)
    print("Worker must stay in EITHER building A OR building B")

    building_a = PolygonRegion(np.array([
        [0, 0], [50, 0], [50, 40], [0, 40]
    ]))

    building_b = PolygonRegion(np.array([
        [70, 0], [120, 0], [120, 30], [70, 30]
    ]))

    # Union: worker can be in either building
    safe_zone = MultiRegion([building_a, building_b], operation='union')

    # Worker position with Gaussian uncertainty
    worker = GaussianDistribution(
        mean=np.array([55.0, 20.0]),  # Between buildings
        cov=np.eye(2) * 100.0  # 10m standard deviation
    )

    estimator3 = ProbabilityEstimator(
        space, safe_zone, worker,
        QuarticKernel(), DirectConvolution(), QuadratureIntegrator()
    )

    result3 = estimator3.compute(bandwidth=8.0, resolution=96)

    print(f"Result: P(in safe zone) = {result3.probability:.4f}")
    print(f"Configuration: Union of 2 rectangular buildings")
    print(f"Worker position: between buildings with 10m uncertainty")

    # Example 4: Buffered hazard zone
    print("\n☢️  Example 4: Buffered Hazard Exclusion Zone")
    print("-" * 70)
    print("Contaminated area with 20m safety buffer")

    # Contaminated polygon
    contaminated_area = np.array([
        [50, 50], [80, 50], [80, 70], [50, 70]
    ])

    # Add buffer for safety margin
    hazard_buffered = BufferedPolygonRegion(contaminated_area, buffer=20.0)

    # Person with moderate uncertainty
    person = GaussianDistribution(
        mean=np.array([85.0, 60.0]),  # Near edge
        cov=np.array([[64.0, 10.0], [10.0, 49.0]])  # 8m × 7m ellipse
    )

    estimator4 = ProbabilityEstimator(
        space, hazard_buffered, person,
        GaussianKernel(), DirectConvolution(), QuadratureIntegrator()
    )

    result4 = estimator4.compute(bandwidth=6.0, resolution=96)

    print(f"Result: P(in hazard buffer) = {result4.probability:.4f}")
    print(f"Safety assessment: P(safe) = {1 - result4.probability:.4f}")
    print(f"Status: {'✅ SAFE' if result4.probability < 0.05 else '⚠️  CAUTION' if result4.probability < 0.25 else '❌ DANGER'}")

    # Comparison table
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    print(f"{ 'Scenario':<40} {'Probability':<12} {'Status':<15}")
    print("-" * 70)
    print(f"{'1. Spatial correlation (Matérn)':<40} {result1.probability:<12.4f} {'Compliant' if result1.probability > 0.8 else 'Review'}")
    print(f"{'2. Elliptical landing zone':<40} {result2.probability:<12.4f} {'Safe' if result2.probability > 0.95 else 'Caution'}")
    print(f"{'3. Multi-region union':<40} {result3.probability:<12.4f} {'In zone' if result3.probability > 0.5 else 'Outside'}")
    print(f"{'4. Buffered hazard (exclusion)':<40} {result4.probability:<12.4f} {'Danger' if result4.probability > 0.25 else 'Safe'}")
    print("="*70)


if __name__ == '__main__':
    # Optionally run library tests first
    try:
        from babelistic import run_all_tests
        passed, failed = run_all_tests()
    except Exception:
        passed = failed = 0

    # Run demo examples only if tests passed (or if running standalone)
    if failed == 0:
        example_custom_components()
        example_probability_distance_analysis()
        example_geofence_scenarios()
        example_new_components()
    else:
        print("Skipping examples because test-suite reported failures.")
