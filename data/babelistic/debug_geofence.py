import numpy as np
from babelistic import geofence_to_probability, GeofenceAdapter, GeofenceRegion, GaussianDistribution, QuadratureIntegrator

def mock_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000

result = geofence_to_probability(
    subject_lat=37.7749, subject_lon=-122.4194, subject_uncertainty=10.0,
    reference_lat=37.7750, reference_lon=-122.4195, reference_uncertainty=5.0,
    distance_threshold=50.0,
    distance_metric=mock_distance,
    resolution=32
)

print('probability:', result.probability)
print('metadata:', result.metadata)

# More diagnostics: recreate internal steps
from babelistic import GeofenceAdapter, GeofenceRegion, GeofenceMetricSpace, GaussianKernel, QuadratureIntegrator
adapter = GeofenceAdapter(mock_distance)
space = adapter.to_metric_space()
region = GeofenceRegion(reference_lat=37.7750, reference_lon=-122.4195,
                        reference_uncertainty=5.0, distance_threshold=50.0,
                        distance_metric=mock_distance)

# Build query as geofence_to_probability does: mean/cov in radians
lat_scale = 111320.0
lon_scale = lat_scale * np.cos(np.radians(37.7749))
var_lat_deg = (10.0 / lat_scale) ** 2
var_lon_deg = (10.0 / lon_scale) ** 2
deg2rad = (np.pi / 180.0) ** 2
subject_cov_rad = np.diag([var_lat_deg * deg2rad, var_lon_deg * deg2rad])
query = GaussianDistribution(mean=np.radians(np.array([37.7749, -122.4194])), cov=subject_cov_rad)

grid = space.create_grid(region.bounds(), resolution=32)
w_field = result.w_field
points = grid['points']  # points are in radians internally

# Evaluate query pdf (per rad^2) on radian grid
p_X = query.pdf(points)
total_mass = QuadratureIntegrator().integrate(p_X, grid['weights'])
numerator = QuadratureIntegrator().integrate(p_X * w_field, grid['weights'])
print('w_field min/max:', w_field.min(), w_field.max())
print('p_X min/max:', p_X.min(), p_X.max())
print('weights sum:', grid['weights'].sum())
print('integral p_X over grid (should be ~1):', total_mass)
print('integral p_X * w_field (probability):', numerator)
