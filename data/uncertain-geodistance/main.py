import time
import inspect
from dataclasses import dataclass, field
from typing import (
    Union, 
    Tuple, 
    Optional, 
    Literal, 
    Dict, 
    List, 
    Any, 
    Callable, 
    Protocol,
)

import numpy as np
from scipy.stats import rice, beta

# Optional dependencies:
# - geopy: for VincentyDistance (pip install geopy)
# - requests: for OSMRoutingDistance (pip install requests)

# Earth radius in meters (mean radius for spherical approximation)
R_EARTH = 6371000.0


# ============================================================================
# DISTANCE METRICS
# ============================================================================

class DistanceMetric(Protocol):
    """
    Protocol for distance functions on lat/lon points.
    
    All distance metrics must support:
    - Scalar inputs: distance(lat1, lon1, lat2, lon2) -> float
    - Array inputs: distance(lat1_array, lon1_array, lat2, lon2) -> array
      (for Monte Carlo sampling efficiency)
    """
    def __call__(self, 
                 lat1: Union[float, np.ndarray], 
                 lon1: Union[float, np.ndarray],
                 lat2: float, 
                 lon2: float) -> Union[float, np.ndarray]:
        """Compute distance in meters"""
        ...


class HaversineDistance:
    """
    Great circle distance on perfect sphere (default).
    
    Mathematical definition:
        Uses haversine formula for great circle arc length.
        Treats Earth as sphere with radius R_EARTH.
    
    Properties:
        Accuracy: ±0.5% vs true WGS84 ellipsoid
        Speed: ~10 flops (very fast)
        Symmetric: Yes
        Triangle inequality: Yes
    
    Use cases:
        - General purpose geospatial calculations
        - Works globally (equator to poles)
        - Default for most applications
    
    Limitations:
        - Ignores Earth's oblateness (0.3% error)
        - Not suitable for survey-grade precision
    
    Examples:
        >>> metric = HaversineDistance()
        >>> d = metric(37.7749, -122.4194, 37.7750, -122.4195)
        >>> print(f"Distance: {d:.2f} meters")
    """
    def __call__(self, lat1, lon1, lat2, lon2):
        return haversine_distance_m(lat1, lon1, lat2, lon2)


class GreatCircleDistance:
    """
    Alias for HaversineDistance (more intuitive name).
    
    Great circle = shortest path on sphere surface.
    """
    def __init__(self):
        self._haversine = HaversineDistance()
    
    def __call__(self, lat1, lon1, lat2, lon2):
        return self._haversine(lat1, lon1, lat2, lon2)


class VincentyDistance:
    """
    Geodesic distance on WGS84 ellipsoid (high precision).
    
    Mathematical definition:
        Uses Vincenty's formulae for geodesics on oblate ellipsoid.
        Accounts for Earth's actual shape (flattening ~1/298.257).
    
    Properties:
        Accuracy: ±0.5mm (industry standard)
        Speed: ~100 flops (iterative, 10x slower than haversine)
        Symmetric: Yes
        Triangle inequality: Yes
    
    Use cases:
        - Survey-grade GPS applications
        - Long distances where 0.3% matters
        - Polar regions (haversine breaks down)
        - Legal/cadastral boundaries
    
    Limitations:
        - Requires geopy library (optional dependency)
        - Slower than haversine
        - Can fail for antipodal points (opposite sides of Earth)
    
    Dependencies:
        pip install geopy
    
    Examples:
        >>> metric = VincentyDistance()
        >>> d = metric(0.0, 0.0, 0.0, 180.0)  # Half Earth circumference
    """
    def __init__(self, cache_size: int = 10000):
        try:
            from geopy.distance import geodesic
            self._geodesic = geodesic
        except ImportError:
            raise ImportError(
                "VincentyDistance requires geopy. Install: pip install geopy"
            )
        
        # Cache for performance
        from functools import lru_cache
        self._compute_cached = lru_cache(maxsize=cache_size)(self._compute_single)
    
    def _compute_single(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute single distance (cached)"""
        return self._geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def __call__(self, lat1, lon1, lat2, lon2):
        if np.isscalar(lat1):
            # Round to 6 decimals (~10cm) for cache hits
            return self._compute_cached(
                round(float(lat1), 6), round(float(lon1), 6),
                round(float(lat2), 6), round(float(lon2), 6)
            )
        else:
            # Vectorized for arrays
            return np.array([
                self._compute_cached(
                    round(float(la1), 6), round(float(lo1), 6),
                    round(float(lat2), 6), round(float(lon2), 6)
                )
                for la1, lo1 in zip(lat1, lon1)
            ])


class LpDistance:
    """
    Generalized Lp (Minkowski) distance in local tangent plane.
    
    Mathematical definition:
        d = (|Δlat_m|^p + |Δlon_m|^p)^(1/p)
        where Δlat_m, Δlon_m are differences in meters using local scale.
    
    Special cases:
        p=1: Manhattan distance (L1, city-block)
        p=2: Euclidean distance (L2, straight line in tangent plane)
        p=∞: Chebyshev distance (L∞, max of components)
    
    Properties:
        Accuracy: Good for small distances (<10km)
        Speed: ~20 flops (fast)
        Symmetric: Yes
        Triangle inequality: Yes (for p ≥ 1)
    
    Use cases:
        - Urban navigation (Manhattan for grid streets)
        - Rectangular geofences (Chebyshev)
        - Custom norms (p=3, p=1.5, etc.)
    
    Limitations:
        - Tangent plane approximation (breaks down for large distances)
        - Not suitable for polar regions
        - Ignores Earth's curvature
    
    Args:
        p: Norm parameter (1 ≤ p ≤ ∞)
    
    Examples:
        >>> metric = LpDistance(p=1)  # Manhattan
        >>> metric = LpDistance(p=2)  # Euclidean
        >>> metric = LpDistance(p=float('inf'))  # Chebyshev
    """
    def __init__(self, p: float = 2.0):
        if p < 1:
            raise ValueError(f"p must be ≥ 1, got {p}")
        self.p = p
    
    def __call__(self, lat1, lon1, lat2, lon2):
        # Convert to meters using local scale
        lat_diff_m = np.abs(lat1 - lat2) * 111132.954
        
        # Longitude scale depends on latitude (use midpoint or average)
        if np.isscalar(lat1):
            lat_mid = (lat1 + lat2) / 2.0
        else:
            lat_mid = lat1  # For arrays, use sample latitude
        
        lon_scale = 111132.954 * np.cos(np.deg2rad(lat_mid))
        lon_diff_m = np.abs(lon1 - lon2) * lon_scale
        
        if np.isinf(self.p):
            # L∞ norm (Chebyshev)
            return np.maximum(lat_diff_m, lon_diff_m)
        else:
            # Lp norm
            return (lat_diff_m**self.p + lon_diff_m**self.p)**(1.0/self.p)


class ManhattanDistance(LpDistance):
    """
    L1 / city-block / taxicab distance.
    
    Mathematical definition:
        d = |Δlat_m| + |Δlon_m|
    
    Properties:
        Always ≥ great circle distance (upper bound)
    
    Use cases:
        - Urban navigation on grid streets (Manhattan, Barcelona)
        - Situations where diagonal movement impossible
        - Conservative distance estimates
    
    Examples:
        >>> metric = ManhattanDistance()
        >>> # Distance is sum of north-south + east-west
    """
    def __init__(self):
        super().__init__(p=1.0)


class EuclideanTangentDistance(LpDistance):
    """
    L2 / Euclidean distance in tangent plane.
    
    Mathematical definition:
        d = sqrt(Δlat_m² + Δlon_m²)
    
    Properties:
        Nearly identical to haversine for distances <10km
        Simpler calculation (no trig)
    
    Use cases:
        - Very local scenarios (indoor, warehouse)
        - When flat Earth assumption acceptable
        - Performance-critical local calculations
    
    Examples:
        >>> metric = EuclideanTangentDistance()
        >>> # Treats Earth as flat locally
    """
    def __init__(self):
        super().__init__(p=2.0)


class ChebyshevDistance(LpDistance):
    """
    L∞ / max / rectangular distance.
    
    Mathematical definition:
        d = max(|Δlat_m|, |Δlon_m|)
    
    Properties:
        Defines square/rectangular regions
        Distance = largest component
    
    Use cases:
        - Rectangular geofences
        - Bounding box checks
        - When constraint is "both lat AND lon within tolerance"
    
    Examples:
        >>> metric = ChebyshevDistance()
        >>> # d0=100 means ±100m in BOTH directions (square fence)
    """
    def __init__(self):
        super().__init__(p=float('inf'))


class WeightedDistance:
    """
    Distance scaled by a cost/pricing/difficulty function.
    
    Mathematical definition:
        d_weighted = d_geometric × cost_factor(location)
    
    The cost function can represent:
        - Terrain difficulty (elevation, obstacles)
        - Traffic patterns (congestion zones)
        - Delivery pricing zones
        - Risk assessment (danger areas)
        - Travel time (instead of distance)
    
    Properties:
        Preserves properties of base metric if cost > 0
        Can break triangle inequality if cost varies spatially
    
    Use cases:
        - Delivery cost optimization
        - Hiking difficulty estimation
        - Drone battery consumption
        - Accessibility planning
    
    Args:
        base_metric: Underlying geometric distance
        cost_function: Callable(lat, lon) -> float multiplier (>0)
    
    Examples:
        >>> def delivery_cost(lat, lon):
        ...     zone = get_zone(lat, lon)
        ...     return {'downtown': 1.0, 'suburbs': 1.5}[zone]
        >>> 
        >>> metric = WeightedDistance(HaversineDistance(), delivery_cost)
        >>> # Now d0=5000 means "5000 cost-units", not meters
    """
    def __init__(self, 
                 base_metric: DistanceMetric,
                 cost_function: Callable[[float, float], float]):
        self.base_metric = base_metric
        self.cost_function = cost_function
    
    def __call__(self, lat1, lon1, lat2, lon2):
        base_dist = self.base_metric(lat1, lon1, lat2, lon2)
        
        # Apply cost at midpoint (or could integrate along path for more accuracy)
        if np.isscalar(lat1):
            lat_mid = (lat1 + lat2) / 2.0
            lon_mid = (lon1 + lon2) / 2.0
            cost = self.cost_function(lat_mid, lon_mid)
        else:
            # Vectorized for arrays
            lat_mid = (lat1 + lat2) / 2.0
            lon_mid = (lon1 + lon2) / 2.0
            cost = np.array([
                self.cost_function(float(la), float(lo)) 
                for la, lo in zip(lat_mid, lon_mid)
            ])
        
        return base_dist * cost


class CustomDistance:
    """
    User-defined distance function wrapper.
    
    Allows arbitrary distance calculations to be plugged into the framework.
    
    Properties:
        Depends entirely on user implementation
    
    Use cases:
        - Road network distances (via routing API)
        - Graph distances
        - Custom business logic
    
    Args:
        distance_func: Callable with signature:
                      (lat1, lon1, lat2, lon2) -> distance_in_meters
                      Must support both scalar and array inputs for lat1/lon1
    
    Examples:
        >>> def road_distance(lat1, lon1, lat2, lon2):
        ...     # Call routing API
        ...     route = get_route((lat1, lon1), (lat2, lon2))
        ...     return route.distance_meters
        >>> 
        >>> metric = CustomDistance(road_distance)
        >>> params = MethodParams(distance_metric=metric)
    """
    def __init__(self, distance_func: Callable):
        self.distance_func = distance_func
        
        # Validate signature
        sig = inspect.signature(distance_func)
        if len(sig.parameters) != 4:
            raise ValueError(
                f"distance_func must take 4 parameters (lat1, lon1, lat2, lon2), "
                f"got {len(sig.parameters)}"
            )
    
    def __call__(self, lat1, lon1, lat2, lon2):
        return self.distance_func(lat1, lon1, lat2, lon2)


# ============================================================================
# OSM-BASED ROUTING DISTANCES AND TRAVEL TIMES
# ============================================================================

class OSMRoutingDistance:
    """
    Real road network distance via OpenStreetMap routing services.
    
    Uses routing engines like OSRM, GraphHopper, or Valhalla to compute
    actual driving/walking distances along road networks.
    
    Mathematical definition:
        d = shortest_path_distance_via_road_network(start, end)
    
    Properties:
        Accuracy: Real-world routing distances
        Speed: ~100-1000ms per request (network dependent)
        Symmetric: Generally yes (unless one-way streets differ)
        Triangle inequality: May not hold (road network constraints)
    
    Use cases:
        - Delivery route optimization
        - Travel time estimation
        - Urban logistics planning
        - Accessibility analysis
    
    Limitations:
        - Requires internet connection
        - Rate limited by API providers
        - Slower than geometric calculations
        - May fail for remote/unmapped areas
    
    Args:
        routing_engine: 'osrm', 'graphhopper', 'valhalla', or 'mapbox'
        profile: 'driving', 'walking', 'cycling', etc.
        api_key: Required for some services (GraphHopper, Mapbox)
        cache_size: LRU cache size for repeated requests
        timeout: Request timeout in seconds
        return_type: 'distance' or 'time' - what to return as the "distance"
    
    Examples:
        >>> # Free OSRM (driving distance)
        >>> metric = OSMRoutingDistance('osrm', 'driving')
        >>> 
        >>> # GraphHopper with API key (travel time as "distance")
        >>> metric = OSMRoutingDistance('graphhopper', 'driving', api_key='your_key', return_type='time')
        >>> 
        >>> # Walking time for accessibility analysis
        >>> metric = OSMRoutingDistance('osrm', 'walking', return_type='time')
    """
    
    def __init__(self, 
                 routing_engine: str = 'osrm',
                 profile: str = 'driving',
                 api_key: Optional[str] = None,
                 cache_size: int = 1000,
                 timeout: int = 10,
                 return_type: str = 'distance'):
        
        self.routing_engine = routing_engine.lower()
        self.profile = profile.lower()
        self.api_key = api_key
        self.timeout = timeout
        self.return_type = return_type.lower()
        
        if self.return_type not in ['distance', 'time']:
            raise ValueError(f"return_type must be 'distance' or 'time', got '{self.return_type}'")
        
        # Set up caching
        from functools import lru_cache
        self._route_cached = lru_cache(maxsize=cache_size)(self._compute_route)
        
        # Validate and set up routing service
        self._setup_routing_service()
    
    def _setup_routing_service(self):
        """Configure the routing service URLs and parameters."""
        routing_configs = {
            'osrm': {
                'url': "http://router.project-osrm.org",
                'supported_profiles': ['driving', 'walking', 'cycling'],
                'requires_key': False
            },
            'graphhopper': {
                'url': "https://graphhopper.com/api/1",
                'supported_profiles': None,  # Supports many profiles
                'requires_key': True,
                'signup_url': "https://www.graphhopper.com/"
            },
            'mapbox': {
                'url': "https://api.mapbox.com",
                'supported_profiles': None,
                'requires_key': True,
                'signup_url': "https://www.mapbox.com/"
            },
            'valhalla': {
                'url': "https://api.maptiler.com/routing",
                'supported_profiles': None,
                'requires_key': False
            }
        }
        
        config = routing_configs.get(self.routing_engine)
        if not config:
            raise ValueError(f"Unsupported routing engine: {self.routing_engine}")
        
        self.base_url = config['url']
        
        # Check API key requirements
        if config['requires_key'] and not self.api_key:
            signup_url = config.get('signup_url', 'the provider website')
            raise ValueError(f"{self.routing_engine.title()} requires an API key. Get one at: {signup_url}")
        
        # Check profile support for OSRM
        if config['supported_profiles'] and self.profile not in config['supported_profiles']:
            supported = ', '.join(config['supported_profiles'])
            raise ValueError(f"OSRM profile '{self.profile}' not supported. Use: {supported}")
    
    def _get_fallback_result(self, lat1: float, lon1: float, lat2: float, lon2: float, error: Exception) -> float:
        """Get fallback result when routing API fails."""
        fallback_distance = haversine_distance_m(lat1, lon1, lat2, lon2)
        if self.return_type == 'time':
            speed_kmh = {'driving': 50, 'walking': 5, 'cycling': 15}.get(self.profile, 50)
            fallback_time = (fallback_distance / 1000) / speed_kmh * 3600  # seconds
            print(f"Warning: Routing API failed ({error}), falling back to estimated time")
            return fallback_time
        else:
            print(f"Warning: Routing API failed ({error}), falling back to haversine")
            return fallback_distance

    def _route_by_engine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Route using the configured engine."""
        routing_methods = {
            'osrm': self._route_osrm,
            'graphhopper': self._route_graphhopper,
            'mapbox': self._route_mapbox,
            'valhalla': self._route_valhalla
        }
        
        method = routing_methods.get(self.routing_engine)
        if method is None:
            raise ValueError(f"Unsupported routing engine: {self.routing_engine}")
        
        return method(lat1, lon1, lat2, lon2)

    def _compute_route(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute route distance or time (cached)."""
        try:
            import requests
        except ImportError:
            raise ImportError("OSM routing requires requests library: pip install requests")
        
        # Round coordinates to reduce cache misses
        lat1, lon1 = round(lat1, 6), round(lon1, 6)
        lat2, lon2 = round(lat2, 6), round(lon2, 6)
        
        # Handle identical points
        if lat1 == lat2 and lon1 == lon2:
            return 0.0
        
        try:
            return self._route_by_engine(lat1, lon1, lat2, lon2)
        except Exception as e:
            return self._get_fallback_result(lat1, lon1, lat2, lon2, e)
    
    def _route_osrm(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Route via OSRM API."""
        import requests
        
        # OSRM expects lon,lat order
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        url = f"{self.base_url}/route/v1/{self.profile}/{coords}"
        params = {'overview': 'false', 'steps': 'false'}
        
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        if data['code'] != 'Ok':
            raise RuntimeError(f"OSRM error: {data.get('message', 'Unknown error')}")
        
        route = data['routes'][0]
        if self.return_type == 'time':
            return float(route['duration'])  # seconds
        else:
            return float(route['distance'])  # meters
    
    def _route_graphhopper(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Route via GraphHopper API."""
        import requests
        
        url = f"{self.base_url}/route"
        params = {
            'point': [f"{lat1},{lon1}", f"{lat2},{lon2}"],
            'vehicle': self.profile,
            'key': self.api_key,
            'calc_points': 'false'
        }
        
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        if 'paths' not in data or not data['paths']:
            raise RuntimeError("GraphHopper: No route found")
        
        path = data['paths'][0]
        if self.return_type == 'time':
            return float(path['time']) / 1000  # Convert ms to seconds
        else:
            return float(path['distance'])  # meters
    
    def _route_mapbox(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Route via Mapbox Directions API."""
        import requests
        
        # Mapbox expects lon,lat order
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        url = f"{self.base_url}/directions/v5/mapbox/{self.profile}/{coords}"
        params = {
            'access_token': self.api_key,
            'overview': 'false',
            'steps': 'false'
        }
        
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        if data['code'] != 'Ok' or not data['routes']:
            raise RuntimeError(f"Mapbox error: {data.get('message', 'No route found')}")
        
        route = data['routes'][0]
        if self.return_type == 'time':
            return float(route['duration'])  # seconds
        else:
            return float(route['distance'])  # meters
    
    def _route_valhalla(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Route via Valhalla API."""
        import requests
        
        url = f"{self.base_url}/v1/{self.profile}"
        json_data = {
            'locations': [
                {'lat': lat1, 'lon': lon1},
                {'lat': lat2, 'lon': lon2}
            ],
            'costing': self.profile,
            'directions_options': {'units': 'kilometers'}
        }
        
        response = requests.post(url, json=json_data, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        if 'trip' not in data or not data['trip']['legs']:
            raise RuntimeError("Valhalla: No route found")
        
        leg = data['trip']['legs'][0]
        if self.return_type == 'time':
            return float(leg['time'])  # seconds
        else:
            return float(leg['length']) * 1000  # Convert km to meters
    
    def __call__(self, lat1, lon1, lat2, lon2):
        """Main distance/time calculation interface."""
        if np.isscalar(lat1):
            return self._route_cached(float(lat1), float(lon1), float(lat2), float(lon2))
        else:
            # Vectorized for Monte Carlo sampling
            results = []
            for la1, lo1 in zip(lat1, lon1):
                results.append(self._route_cached(float(la1), float(lo1), float(lat2), float(lon2)))
            return np.array(results)


class OSMTravelTime(OSMRoutingDistance):
    """
    Travel time via OSM routing (time as "distance" for probability calculations).
    
    This class treats travel time as the "distance" metric for probability calculations.
    Useful for multi-modal transportation where time is the key constraint.
    
    Mathematical definition:
        t = travel_time_via_road_network(start, end)  [in seconds]
    
    Use cases:
        - Multi-modal transportation planning
        - Time-based accessibility analysis  
        - Public transit integration
        - Delivery time optimization
    
    Examples:
        >>> # 15-minute walking accessibility
        >>> walking_time = OSMTravelTime('osrm', 'walking')
        >>> result = check_geo_prob(home, transit, d0=900,  # 15 minutes = 900 seconds
        ...                        MethodParams(distance_metric=walking_time))
        >>> 
        >>> # 30-minute driving commute
        >>> driving_time = OSMTravelTime('osrm', 'driving')
        >>> result = check_geo_prob(home, work, d0=1800,  # 30 minutes = 1800 seconds
        ...                        MethodParams(distance_metric=driving_time))
    """
    
    def __init__(self, routing_engine: str = 'osrm', profile: str = 'driving', **kwargs):
        # Force return_type to 'time'
        kwargs['return_type'] = 'time'
        super().__init__(routing_engine, profile, **kwargs)


class MultiModalTravelTime:
    """
    Multi-modal travel time combining different transportation modes.
    
    This class models realistic multi-modal journeys where people use
    different transportation modes for different segments of their trip.
    
    Mathematical definition:
        t_total = sum(t_i * mode_factor_i) for each segment i
    
    Properties:
        Accounts for mode switches, waiting times, and realistic routing
        Can model complex transportation scenarios
    
    Use cases:
        - Urban transportation planning
        - Public transit + walking accessibility
        - Bike-share + public transit
        - Park-and-ride scenarios
        - Last-mile delivery optimization
    
    Args:
        mode_segments: List of (mode, distance_threshold, routing_engine) tuples
        transfer_time: Time penalty for switching modes (seconds)
        
    Examples:
        >>> # Walk to transit, then bus/train for long distance
        >>> multimodal = MultiModalTravelTime([
        ...     ('walking', 500, 'osrm'),    # Walk up to 500m
        ...     ('transit', float('inf'), 'graphhopper')  # Transit for rest
        ... ], transfer_time=120)  # 2-minute transfer penalty
        >>> 
        >>> # Bike-share + walking for urban trips
        >>> bike_walk = MultiModalTravelTime([
        ...     ('cycling', 2000, 'osrm'),   # Bike up to 2km  
        ...     ('walking', float('inf'), 'osrm')   # Walk the rest
        ... ], transfer_time=60)  # 1-minute to find/return bike
    """
    
    def __init__(self, 
                 mode_segments: List[Tuple[str, float, str]] = None,
                 transfer_time: float = 120,  # seconds
                 api_key: Optional[str] = None):
        
        if mode_segments is None:
            # Default: walk short distances, drive longer ones
            mode_segments = [
                ('walking', 800, 'osrm'),      # Walk up to 800m
                ('driving', float('inf'), 'osrm')  # Drive the rest
            ]
        
        self.mode_segments = mode_segments
        self.transfer_time = transfer_time
        self.api_key = api_key
        
        # Create routing instances for each mode
        self.routers = {}
        for mode, _, engine in mode_segments:
            if mode not in self.routers:
                self.routers[mode] = OSMTravelTime(engine, mode, api_key=api_key)
    
    def __call__(self, lat1, lon1, lat2, lon2):
        """Calculate multi-modal travel time."""
        if np.isscalar(lat1):
            return self._compute_multimodal_time(float(lat1), float(lon1), float(lat2), float(lon2))
        else:
            # Vectorized for Monte Carlo
            times = []
            for la1, lo1 in zip(lat1, lon1):
                times.append(self._compute_multimodal_time(float(la1), float(lo1), float(lat2), float(lon2)))
            return np.array(times)
    
    def _compute_multimodal_time(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute multi-modal travel time for a single trip."""
        total_distance = haversine_distance_m(lat1, lon1, lat2, lon2)
        
        # Select appropriate mode based on distance thresholds
        selected_mode = None
        transfers = 0
        
        for mode, threshold, engine in self.mode_segments:
            if total_distance <= threshold:
                selected_mode = mode
                break
                
        if selected_mode is None:
            # Use last mode if distance exceeds all thresholds
            selected_mode = self.mode_segments[-1][0]
            transfers = len(self.mode_segments) - 1  # Assume we used all previous modes
        
        # Calculate base travel time
        base_time = self.routers[selected_mode](lat1, lon1, lat2, lon2)
        
        # Add transfer penalties
        total_time = base_time + (transfers * self.transfer_time)
        
        return float(total_time)


# Specialized multi-modal classes
class TransitAccessibilityTime(MultiModalTravelTime):
    """Walking + public transit accessibility."""
    
    def __init__(self, walking_threshold: float = 800, **kwargs):
        mode_segments = [
            ('walking', walking_threshold, 'osrm'),
            ('transit', float('inf'), 'osrm')  # Simplified as driving for now
        ]
        super().__init__(mode_segments, transfer_time=180, **kwargs)  # 3-min wait


class BikeShareTime(MultiModalTravelTime):
    """Bike-share + walking for urban mobility."""
    
    def __init__(self, bike_threshold: float = 3000, **kwargs):
        mode_segments = [
            ('cycling', bike_threshold, 'osrm'),
            ('walking', float('inf'), 'osrm')
        ]
        super().__init__(mode_segments, transfer_time=90, **kwargs)  # 1.5-min bike pickup/return


class ParkAndRideTime(MultiModalTravelTime):
    """Drive to parking + walk/transit to destination."""
    
    def __init__(self, parking_threshold: float = 5000, **kwargs):
        mode_segments = [
            ('driving', parking_threshold, 'osrm'),
            ('walking', float('inf'), 'osrm')
        ]
        super().__init__(mode_segments, transfer_time=300, **kwargs)  # 5-min parking


class OSMWalkingDistance(OSMRoutingDistance):
    """Walking distance via OSM routing (pedestrian paths)."""
    def __init__(self, routing_engine: str = 'osrm', **kwargs):
        super().__init__(routing_engine, profile='walking', **kwargs)


class OSMDrivingDistance(OSMRoutingDistance):
    """Driving distance via OSM routing (car routing).""" 
    def __init__(self, routing_engine: str = 'osrm', **kwargs):
        super().__init__(routing_engine, profile='driving', **kwargs)


class OSMCyclingDistance(OSMRoutingDistance):
    """Cycling distance via OSM routing (bike paths)."""
    def __init__(self, routing_engine: str = 'osrm', **kwargs):
        super().__init__(routing_engine, profile='cycling', **kwargs)


# Convenience factory functions
def osm_driving_distance(routing_engine: str = 'osrm', api_key: Optional[str] = None) -> OSMRoutingDistance:
    """
    Create driving distance metric using OSM routing.
    
    Args:
        routing_engine: 'osrm' (free), 'graphhopper', 'mapbox', 'valhalla'
        api_key: Required for GraphHopper and Mapbox
    
    Examples:
        >>> # Free OSRM
        >>> metric = osm_driving_distance('osrm')
        >>> 
        >>> # GraphHopper with API key  
        >>> metric = osm_driving_distance('graphhopper', api_key='your_key')
    """
    return OSMDrivingDistance(routing_engine, api_key=api_key)


def osm_walking_distance(routing_engine: str = 'osrm', api_key: Optional[str] = None) -> OSMRoutingDistance:
    """Create walking distance metric using OSM routing."""
    return OSMWalkingDistance(routing_engine, api_key=api_key)


# Time-based routing factory functions
def osm_travel_time(profile: str = 'driving', routing_engine: str = 'osrm', api_key: Optional[str] = None) -> OSMTravelTime:
    """
    Create travel time metric using OSM routing.
    
    Args:
        profile: 'driving', 'walking', 'cycling'
        routing_engine: 'osrm', 'graphhopper', 'mapbox', 'valhalla'
        api_key: Required for GraphHopper and Mapbox
        
    Returns time in seconds as the "distance" for probability calculations.
    
    Examples:
        >>> # 15-minute walking accessibility (900 seconds)
        >>> walking_time = osm_travel_time('walking', 'osrm')
        >>> result = check_geo_prob(home, transit, d0=900, 
        ...                        MethodParams(distance_metric=walking_time))
        >>> 
        >>> # 30-minute driving commute (1800 seconds)  
        >>> driving_time = osm_travel_time('driving', 'osrm')
        >>> result = check_geo_prob(home, work, d0=1800,
        ...                        MethodParams(distance_metric=driving_time))
    """
    return OSMTravelTime(routing_engine, profile, api_key=api_key)


def multimodal_travel_time(modes: List[Tuple[str, float]] = None, 
                          transfer_time: float = 120,
                          routing_engine: str = 'osrm',
                          api_key: Optional[str] = None) -> MultiModalTravelTime:
    """
    Create multi-modal travel time metric.
    
    Args:
        modes: List of (mode, distance_threshold) tuples
        transfer_time: Time penalty for mode switches (seconds)
        routing_engine: Base routing engine to use
        api_key: API key if needed
        
    Examples:
        >>> # Walk short distances, drive longer ones
        >>> multimodal = multimodal_travel_time([
        ...     ('walking', 800),
        ...     ('driving', float('inf'))
        ... ], transfer_time=60)
        >>> 
        >>> # Bike + walk combination  
        >>> bike_walk = multimodal_travel_time([
        ...     ('cycling', 2000),
        ...     ('walking', float('inf'))
        ... ], transfer_time=90)
    """
    if modes is None:
        modes = [('walking', 800), ('driving', float('inf'))]
    
    mode_segments = [(mode, thresh, routing_engine) for mode, thresh in modes]
    return MultiModalTravelTime(mode_segments, transfer_time, api_key)
def lp_distance(p: float) -> LpDistance:
    """
    Factory function for creating Lp distances.
    
    Examples:
        >>> metric = lp_distance(1)      # Manhattan
        >>> metric = lp_distance(2)      # Euclidean
        >>> metric = lp_distance(3)      # L3 norm
        >>> metric = lp_distance(np.inf) # Chebyshev
    """
    return LpDistance(p)


def weighted_haversine(cost_func: Callable[[float, float], float]) -> WeightedDistance:
    """
    Create a cost-weighted haversine distance.
    
    Args:
        cost_func: Function(lat, lon) -> multiplier
        
    Example:
        >>> def urban_congestion(lat, lon):
        ...     if in_downtown(lat, lon):
        ...         return 2.0  # 2x "farther" due to traffic
        ...     return 1.0
        >>> 
        >>> metric = weighted_haversine(urban_congestion)
    """
    return WeightedDistance(HaversineDistance(), cost_func)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class GeoPoint:
    """Represents a geodetic point (lat, lon)."""
    lat: float
    lon: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lon)
    
    def __repr__(self) -> str:
        return f"GeoPoint(lat={self.lat:.6f}, lon={self.lon:.6f})"


@dataclass
class Covariance2D:
    """Wrapper for 2×2 ENU covariance matrix."""
    _matrix: np.ndarray = field(init=False)
    
    def __init__(self, data: Union[float, np.ndarray, 'Covariance2D']):
        if isinstance(data, Covariance2D):
            self._matrix = data._matrix.copy()
        else:
            self._matrix = self._process_input(data)
    
    @staticmethod
    def _process_input(inp: Union[float, np.ndarray]) -> np.ndarray:
        """Convert input to 2x2 covariance matrix."""
        if np.isscalar(inp):
            sigma = float(inp)
            if sigma < 0:
                raise ValueError(f"Sigma must be non-negative, got {sigma}")
            return np.eye(2) * (sigma ** 2)
        
        arr = np.asarray(inp, dtype=float)
        if arr.shape == (2,):
            # Diagonal variances
            return np.diag(arr)
        elif arr.shape == (2, 2):
            return arr.copy()
        else:
            raise ValueError(f"Invalid covariance shape: {arr.shape}")
    
    @classmethod
    def from_input(cls, inp: Any) -> 'Covariance2D':
        """Create Covariance2D from various input types."""
        return cls(inp)
    
    def as_matrix(self) -> np.ndarray:
        """Return the 2x2 covariance matrix."""
        return self._matrix.copy()
    
    def is_isotropic(self, rtol: float = 1e-6) -> bool:
        """Check if covariance is isotropic (circular)."""
        eigs = np.linalg.eigvalsh(self._matrix)
        return np.allclose(eigs[0], eigs[1], rtol=rtol)
    
    def max_std(self) -> float:
        """Return maximum standard deviation."""
        eigs = np.clip(np.linalg.eigvalsh(self._matrix), a_min=0.0, a_max=None)
        return float(np.sqrt(eigs.max()))
    
    def cond(self) -> float:
        """Return condition number."""
        try:
            return float(np.linalg.cond(self._matrix))
        except Exception:
            return float('inf')
    
    def __repr__(self) -> str:
        if self.is_isotropic():
            std = self.max_std()
            return f"Covariance2D(isotropic, σ={std:.3f}m)"
        else:
            return f"Covariance2D(anisotropic, max_σ={self.max_std():.3f}m)"


@dataclass
class Subject:
    """Subject point with uncertainty."""
    mu: GeoPoint
    Sigma: Covariance2D
    id: Optional[str] = None
    
    def __repr__(self) -> str:
        id_str = f", id='{self.id}'" if self.id else ""
        return f"Subject({self.mu}, {self.Sigma}{id_str})"


@dataclass
class Reference:
    """Reference point with uncertainty."""
    mu: GeoPoint
    Sigma: Covariance2D
    id: Optional[str] = None
    
    def __repr__(self) -> str:
        id_str = f", id='{self.id}'" if self.id else ""
        return f"Reference({self.mu}, {self.Sigma}{id_str})"


@dataclass
class MethodParams:
    """Parameters for computation methods."""
    mode: Literal['auto', 'analytic', 'mc_ecef', 'mc_tangent'] = 'auto'
    prob_threshold: float = 0.95
    n_mc: int = 200_000
    batch_size: int = 100_000
    conservative_decision: bool = True
    random_state: Optional[int] = None
    use_antithetic: bool = True
    cp_alpha: float = 0.05
    distance_metric: Optional[DistanceMetric] = None  # NEW in v4
    
    def get_distance_metric(self) -> DistanceMetric:
        """Get the distance metric, defaulting to haversine."""
        return self.distance_metric if self.distance_metric is not None else HaversineDistance()
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'prob_threshold': self.prob_threshold,
            'n_mc': self.n_mc,
            'batch_size': self.batch_size,
            'conservative_decision': self.conservative_decision,
            'random_state': self.random_state,
            'use_antithetic': self.use_antithetic,
            'cp_alpha': self.cp_alpha,
            'distance_metric': type(self.distance_metric).__name__ if self.distance_metric else 'HaversineDistance'
        }


@dataclass
class Scenario:
    """Structured scenario for profiling."""
    name: str
    subject: Subject
    reference: Reference
    d0: float
    method_params: MethodParams = field(default_factory=MethodParams)
    expected_behavior: str = ""
    
    # For analytics compatibility
    sigma_P_scalar: Optional[float] = None
    sigma_Q_scalar: Optional[float] = None


@dataclass
class ProbabilityResult:
    """Result from probability computation."""
    fulfilled: bool
    probability: float
    method: str
    mc_stderr: Optional[float] = None
    n_samples: Optional[int] = None
    cp_lower: Optional[float] = None
    cp_upper: Optional[float] = None
    sigma_cond: Optional[float] = None
    max_std_m: Optional[float] = None
    delta_m: Optional[float] = None
    decision_by: Optional[str] = None


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def latlon_to_ecef(lat_deg: float, lon_deg: float, R: float = R_EARTH) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)

    x = R * np.cos(phi) * np.cos(lam)
    y = R * np.cos(phi) * np.sin(lam)
    z = R * np.sin(phi)

    return np.array([x, y, z], dtype=float)


def geodetic_to_ecef_jacobian(lat_deg: float, lon_deg: float, R: float = R_EARTH) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)

    dxdphi = -R * np.sin(phi) * np.cos(lam)
    dydphi = -R * np.sin(phi) * np.sin(lam)
    dzdphi = R * np.cos(phi)

    dxdlam = -R * np.cos(phi) * np.sin(lam)
    dydlam = R * np.cos(phi) * np.cos(lam)
    dzdlam = 0.0

    return np.array([[dxdphi, dxdlam],
                     [dydphi, dydlam],
                     [dzdphi, dzdlam]], dtype=float)


def haversine_distance_m(lat1: Union[float, np.ndarray],
                         lon1: Union[float, np.ndarray],
                         lat2: float,
                         lon2: float,
                         R: float = R_EARTH) -> Union[float, np.ndarray]:
    phi1 = np.deg2rad(lat1)
    lam1 = np.deg2rad(lon1)
    phi2 = np.deg2rad(lat2)
    lam2 = np.deg2rad(lon2)

    dphi = phi2 - phi1
    dlam = lam2 - lam1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def enu_to_radians_scale(lat_deg: float) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)
    meters_per_deg_lat = 111132.954
    meters_per_deg_lon = 111132.954 * np.cos(lat_rad)

    return np.diag([1.0 / meters_per_deg_lat, 1.0 / meters_per_deg_lon])


def _ensure_psd(mat: np.ndarray) -> np.ndarray:
    """Force symmetric PSD by clipping small negative eigenvalues to zero."""
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    return (eigvecs * eigvals_clipped) @ eigvecs.T


def _add_jitter(mat: np.ndarray, rel: float = 1e-12) -> np.ndarray:
    trace = np.trace(mat)
    jitter = rel * max(trace, 1.0)
    return mat + np.eye(mat.shape[0]) * jitter


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Return the (lower, upper) Clopper-Pearson (1-alpha) interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lower), float(upper)


def _choose_method_heuristic(Sigma_X_m2: np.ndarray, delta_m: float,
                             max_std_thresh: float = 200.0, delta_thresh: float = 5000.0,
                             cond_thresh: float = 20.0) -> Tuple[str, Dict]:
    """Conservative heuristic to decide if tangent-plane MC is acceptable."""
    eigs = np.linalg.eigvalsh(Sigma_X_m2)
    eigs = np.clip(eigs, a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs.max()))
    
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')

    is_local = (max_std < max_std_thresh) and (delta_m < delta_thresh) and (cond < cond_thresh)

    reason = f"max_std={max_std:.3f}m, delta={delta_m:.1f}m, cond={cond:.3f}"
    chosen = 'mc_tangent' if is_local else 'mc_ecef'
    return chosen, {'max_std_m': max_std, 'cond': cond, 'delta_m': delta_m, 'reason': reason}


# ============================================================================
# MAIN API FUNCTION
# ============================================================================

def _convert_to_subject(subject_input: Union[Subject, Tuple[float, float], GeoPoint]) -> Subject:
    """Convert various input types to Subject object."""
    if isinstance(subject_input, Subject):
        return subject_input
    if isinstance(subject_input, GeoPoint):
        return Subject(subject_input, Covariance2D(0.0))
    if isinstance(subject_input, (tuple, list)):
        if len(subject_input) == 3:
            lat, lon, sigma = subject_input
            return Subject(GeoPoint(lat, lon), Covariance2D(sigma))
        elif len(subject_input) == 2:
            return Subject(GeoPoint(*subject_input), Covariance2D(0.0))
    return Subject(GeoPoint(*subject_input), Covariance2D(0.0))


def _convert_to_reference(reference_input: Union[Reference, Tuple[float, float], GeoPoint]) -> Reference:
    """Convert various input types to Reference object."""
    if isinstance(reference_input, Reference):
        return reference_input
    if isinstance(reference_input, GeoPoint):
        return Reference(reference_input, Covariance2D(0.0))
    if isinstance(reference_input, (tuple, list)):
        if len(reference_input) == 3:
            lat, lon, sigma = reference_input
            return Reference(GeoPoint(lat, lon), Covariance2D(sigma))
        elif len(reference_input) == 2:
            return Reference(GeoPoint(*reference_input), Covariance2D(0.0))
    return Reference(GeoPoint(*reference_input), Covariance2D(0.0))


def _validate_inputs(d0_meters: float, method_params: MethodParams) -> None:
    """Validate input parameters."""
    if d0_meters < 0:
        raise ValueError("d0_meters must be non-negative")
    if not 0 <= method_params.prob_threshold <= 1:
        raise ValueError("prob_threshold must be in [0,1]")
    if method_params.n_mc < 100:
        raise ValueError("n_mc must be >= 100")


def _handle_deterministic_case(
    mu_P: np.ndarray, 
    mu_Q: np.ndarray, 
    d0_meters: float, 
    prob_threshold: float, 
    distance_metric: DistanceMetric
) -> ProbabilityResult:
    """Handle case where both points have no uncertainty."""
    delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
    prob = 1.0 if delta <= d0_meters else 0.0
    return ProbabilityResult(
        fulfilled=(prob >= prob_threshold),
        probability=float(prob),
        method='deterministic',
        mc_stderr=0.0,
        n_samples=0,
        cp_lower=float(prob),
        cp_upper=float(prob),
        sigma_cond=None,
        max_std_m=0.0,
        delta_m=float(delta),
        decision_by='point_estimate'
    )


def _choose_computation_method(
    mode: str,
    is_isotropic: bool,
    Sigma_X_m2: np.ndarray,
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    distance_metric: DistanceMetric
) -> str:
    """Choose the appropriate computation method."""
    if mode == 'auto':
        delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        chosen, _ = _choose_method_heuristic(Sigma_X_m2, float(delta))
        return 'analytic' if is_isotropic else chosen
    return mode


def _execute_computation(
    method: str,
    subject: Subject,
    reference: Reference,
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    Sigma_P_mat: np.ndarray,
    Sigma_Q_mat: np.ndarray,
    Sigma_X_m2: np.ndarray,
    d0_meters: float,
    method_params: MethodParams,
    distance_metric: DistanceMetric,
    is_isotropic: bool
) -> ProbabilityResult:
    """Execute the chosen computation method."""
    if method == 'analytic':
        if not is_isotropic:
            raise ValueError("analytic mode requires both uncertainties to be isotropic")
        sigma_P = subject.Sigma.max_std()
        sigma_Q = reference.Sigma.max_std()
        return _compute_analytic_rice(mu_P, mu_Q, sigma_P, sigma_Q, d0_meters, 
                                      method_params.prob_threshold, distance_metric)
    
    if method == 'mc_ecef':
        return _compute_mc_ecef(
            mu_P, mu_Q, Sigma_P_mat, Sigma_Q_mat, d0_meters,
            method_params.prob_threshold, method_params.n_mc, method_params.batch_size, 
            method_params.random_state, method_params.conservative_decision,
            method_params.use_antithetic, method_params.cp_alpha, distance_metric
        )
    
    if method == 'mc_tangent':
        delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        return _compute_mc_tangent(
            mu_P, mu_Q, Sigma_X_m2, d0_meters,
            method_params.prob_threshold, method_params.n_mc, method_params.batch_size, 
            method_params.random_state, method_params.conservative_decision, float(delta),
            method_params.use_antithetic, method_params.cp_alpha, distance_metric
        )
    
    raise ValueError(f"Unknown mode: {method}")


def check_geo_prob(
    subject: Union[Subject, Tuple[float, float], GeoPoint],
    reference: Union[Reference, Tuple[float, float], GeoPoint],
    d0_meters: float,
    method_params: Optional[MethodParams] = None
) -> ProbabilityResult:
    """
    Compute probability that distance between two uncertain points is ≤ d0_meters.
    
    Args:
        subject: Subject object or (lat, lon) tuple for first point
        reference: Reference object or (lat, lon) tuple for second point  
        d0_meters: Distance threshold in meters
        method_params: MethodParams object with computation settings including distance_metric
        
    Distance Metrics (set via method_params.distance_metric):
        - HaversineDistance() [default]: Great circle on sphere
        - VincentyDistance(): High-precision WGS84 ellipsoid (requires geopy)
        - ManhattanDistance(): L1 / city-block distance
        - ChebyshevDistance(): L∞ / rectangular distance
        - EuclideanTangentDistance(): L2 in tangent plane
        - lp_distance(p): Generalized Lp norm
        - WeightedDistance(base, cost_func): Cost-weighted distance
        - CustomDistance(func): User-defined distance function
        
    Examples:
        # Default (haversine)
        >>> result = check_geo_prob(subject, reference, d0=100)
        
        # Urban navigation (Manhattan)
        >>> params = MethodParams(distance_metric=ManhattanDistance())
        >>> result = check_geo_prob(subject, reference, d0=100, params)
        
        # Rectangular geofence (Chebyshev)
        >>> params = MethodParams(distance_metric=ChebyshevDistance())
        >>> result = check_geo_prob(subject, reference, d0=100, params)
        
        # Custom weighted distance
        >>> def delivery_cost(lat, lon):
        ...     return 1.5 if in_suburbs(lat, lon) else 1.0
        >>> metric = weighted_haversine(delivery_cost)
        >>> params = MethodParams(distance_metric=metric)
        >>> result = check_geo_prob(subject, reference, d0=5000, params)
    """
    
    # Initialize defaults and validate inputs
    if method_params is None:
        method_params = MethodParams()
    
    _validate_inputs(d0_meters, method_params)
    
    # Convert inputs to standard types
    subject = _convert_to_subject(subject)
    reference = _convert_to_reference(reference)
    distance_metric = method_params.get_distance_metric()
    
    # Extract coordinates and uncertainties
    mu_P = np.array(subject.mu.to_tuple())
    mu_Q = np.array(reference.mu.to_tuple())
    Sigma_P_mat = _ensure_psd(subject.Sigma.as_matrix())
    Sigma_Q_mat = _ensure_psd(reference.Sigma.as_matrix())
    Sigma_X_m2 = Sigma_P_mat + Sigma_Q_mat

    # Handle deterministic case (no uncertainty)
    if np.allclose(Sigma_X_m2, 0.0, atol=1e-14):
        return _handle_deterministic_case(mu_P, mu_Q, d0_meters, method_params.prob_threshold, distance_metric)

    # Choose computation method
    is_isotropic = subject.Sigma.is_isotropic() and reference.Sigma.is_isotropic()
    method = _choose_computation_method(method_params.mode, is_isotropic, Sigma_X_m2, mu_P, mu_Q, distance_metric)
    
    # Execute computation
    return _execute_computation(
        method, subject, reference, mu_P, mu_Q, Sigma_P_mat, Sigma_Q_mat, Sigma_X_m2,
        d0_meters, method_params, distance_metric, is_isotropic
    )


def _compute_analytic_rice(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    sigma_P: float,
    sigma_Q: float,
    d0_meters: float,
    prob_threshold: float,
    distance_metric: DistanceMetric = None
) -> ProbabilityResult:
    if distance_metric is None:
        distance_metric = HaversineDistance()
    
    sigma_X = np.sqrt(sigma_P ** 2 + sigma_Q ** 2)
    delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])

    if sigma_X == 0.0:
        prob = 1.0 if delta <= d0_meters else 0.0
    else:
        b = delta / sigma_X
        x = d0_meters / sigma_X
        prob = float(rice.cdf(x, b))

    fulfilled = prob >= prob_threshold
    return ProbabilityResult(
        fulfilled=fulfilled,
        probability=prob,
        method='analytic_rice',
        mc_stderr=None,
        n_samples=None,
        cp_lower=None,
        cp_upper=None,
        sigma_cond=None,
        max_std_m=float(sigma_X),
        delta_m=float(delta),
        decision_by='point_estimate'
    )


def _setup_ecef_transformations(mu_P: np.ndarray, mu_Q: np.ndarray, 
                                Sigma_P_mat: np.ndarray, Sigma_Q_mat: np.ndarray) -> tuple:
    """Set up ECEF coordinate transformations and covariance matrices."""
    mu_P_ecef = latlon_to_ecef(mu_P[0], mu_P[1])
    mu_Q_ecef = latlon_to_ecef(mu_Q[0], mu_Q[1])

    Jp = geodetic_to_ecef_jacobian(mu_P[0], mu_P[1])
    Jq = geodetic_to_ecef_jacobian(mu_Q[0], mu_Q[1])

    phi_p = np.deg2rad(mu_P[0])
    phi_q = np.deg2rad(mu_Q[0])
    A_p = np.array([[0.0, 1.0 / R_EARTH],
                    [1.0 / (R_EARTH * np.cos(phi_p)), 0.0]], dtype=float)
    A_q = np.array([[0.0, 1.0 / R_EARTH],
                    [1.0 / (R_EARTH * np.cos(phi_q)), 0.0]], dtype=float)

    Sigma_P_phi_lambda = _ensure_psd(A_p @ Sigma_P_mat @ A_p.T)
    Sigma_Q_phi_lambda = _ensure_psd(A_q @ Sigma_Q_mat @ A_q.T)

    Sigma_P_ecef = _ensure_psd(Jp @ Sigma_P_phi_lambda @ Jp.T)
    Sigma_Q_ecef = _ensure_psd(Jq @ Sigma_Q_phi_lambda @ Jq.T)

    mu_X_ecef = mu_P_ecef - mu_Q_ecef
    Sigma_X_ecef = _ensure_psd(Sigma_P_ecef + Sigma_Q_ecef)
    
    return mu_X_ecef, Sigma_X_ecef, mu_Q_ecef


def _compute_ecef_diagnostics(mu_P: np.ndarray, mu_Q: np.ndarray, 
                             Sigma_P_mat: np.ndarray, Sigma_Q_mat: np.ndarray,
                             distance_metric: DistanceMetric) -> tuple:
    """Compute diagnostic metrics for ECEF computation."""
    Sigma_X_m2 = Sigma_P_mat + Sigma_Q_mat
    eigs_m2 = np.clip(np.linalg.eigvalsh(Sigma_X_m2), a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs_m2.max()))
    
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')
    
    delta = distance_metric(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
    return max_std, cond, delta


def _generate_ecef_samples(rng, L: np.ndarray, mu_X_ecef: np.ndarray, 
                          n_batch: int, use_antithetic: bool) -> np.ndarray:
    """Generate ECEF samples using normal distribution."""
    if use_antithetic and n_batch > 1:
        # Adjust batch size for antithetic pairs
        if n_batch % 2 == 1:
            n_batch -= 1

        half = max(1, n_batch // 2)
        z = rng.standard_normal((3, half))
        z_pair = np.concatenate([z, -z], axis=1)
        samples_X_ecef = (L @ z_pair).T + mu_X_ecef
        
        # Handle odd remainder
        if n_batch < len(z_pair[0]) * 2:
            z_last = rng.standard_normal((3, 1))
            samples_last = (L @ z_last).T + mu_X_ecef
            samples_X_ecef = np.vstack([samples_X_ecef, samples_last])
    else:
        z = rng.standard_normal((3, n_batch))
        samples_X_ecef = (L @ z).T + mu_X_ecef
    
    return samples_X_ecef


def _ecef_to_latlon_batch(P_ecef: np.ndarray) -> tuple:
    """Convert batch of ECEF coordinates to lat/lon."""
    x, y, zc = P_ecef[:, 0], P_ecef[:, 1], P_ecef[:, 2]
    r = np.sqrt(x * x + y * y + zc * zc)
    lat_samples = np.rad2deg(np.arcsin(np.clip(zc / r, -1.0, 1.0)))
    lon_samples = np.rad2deg(np.arctan2(y, x))
    return lat_samples, lon_samples


def _compute_mc_ecef(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    Sigma_P_mat: np.ndarray,
    Sigma_Q_mat: np.ndarray,
    d0_meters: float,
    prob_threshold: float,
    n_mc: int,
    batch_size: int,
    random_state: Optional[int],
    conservative_decision: bool = True,
    use_antithetic: bool = True,
    cp_alpha: float = 0.05,
    distance_metric: DistanceMetric = None
) -> ProbabilityResult:
    """Monte Carlo sampling in ECEF coordinates."""
    if distance_metric is None:
        distance_metric = HaversineDistance()
    
    rng = np.random.default_rng(random_state)
    
    # Set up coordinate transformations
    mu_X_ecef, Sigma_X_ecef, mu_Q_ecef = _setup_ecef_transformations(mu_P, mu_Q, Sigma_P_mat, Sigma_Q_mat)
    
    # Compute diagnostics
    max_std, cond, delta = _compute_ecef_diagnostics(mu_P, mu_Q, Sigma_P_mat, Sigma_Q_mat, distance_metric)

    # Cholesky for sampling
    Sigma_for_chol = Sigma_X_ecef.copy()
    try:
        L = np.linalg.cholesky(Sigma_for_chol)
    except np.linalg.LinAlgError:
        Sigma_for_chol = _add_jitter(Sigma_for_chol, rel=1e-10)
        L = np.linalg.cholesky(Sigma_for_chol)

    count_inside = 0
    total = 0
    n_remaining = n_mc

    # Monte Carlo sampling loop
    while n_remaining > 0:
        n_batch = min(batch_size, n_remaining)
        
        # Generate samples
        samples_X_ecef = _generate_ecef_samples(rng, L, mu_X_ecef, n_batch, use_antithetic)
        P_ecef = mu_Q_ecef + samples_X_ecef

        # Convert to lat/lon
        lat_samples, lon_samples = _ecef_to_latlon_batch(P_ecef)

        # Compute distances and count
        dists = distance_metric(lat_samples, lon_samples, mu_Q[0], mu_Q[1])
        count_inside += int(np.count_nonzero(dists <= d0_meters))
        total += samples_X_ecef.shape[0]
        n_remaining -= samples_X_ecef.shape[0]

    # Calculate final results
    prob = count_inside / total
    mc_stderr = float(np.sqrt(prob * (1 - prob) / total)) if total > 0 else None
    cp_l, cp_u = clopper_pearson(count_inside, total, alpha=cp_alpha)

    if conservative_decision:
        fulfilled = (cp_l >= prob_threshold)
        decision_by = 'cp_lower'
    else:
        fulfilled = (prob >= prob_threshold)
        decision_by = 'point_estimate'

    return ProbabilityResult(
        fulfilled=fulfilled,
        probability=float(prob),
        method='mc_ecef',
        mc_stderr=mc_stderr,
        n_samples=int(total),
        cp_lower=float(cp_l),
        cp_upper=float(cp_u),
        sigma_cond=float(cond),
        max_std_m=float(max_std),
        delta_m=float(delta),
        decision_by=decision_by
    )


def _setup_tangent_plane_sampling(mu_P: np.ndarray, mu_Q: np.ndarray, 
                                  Sigma_X_m2: np.ndarray) -> tuple:
    """Set up tangent plane coordinate system for sampling."""
    lat_center = (mu_P[0] + mu_Q[0]) / 2.0
    scale_to_deg = enu_to_radians_scale(lat_center)
    Sigma_X_deg2 = _ensure_psd(scale_to_deg @ Sigma_X_m2 @ scale_to_deg.T)
    mu_X_deg = np.array(mu_P) - np.array(mu_Q)
    return Sigma_X_deg2, mu_X_deg


def _compute_tangent_diagnostics(Sigma_X_m2: np.ndarray) -> tuple:
    """Compute diagnostic metrics for tangent plane computation."""
    eigs = np.clip(np.linalg.eigvalsh(Sigma_X_m2), a_min=0.0, a_max=None)
    max_std = float(np.sqrt(eigs.max()))
    
    try:
        cond = float(np.linalg.cond(Sigma_X_m2))
    except Exception:
        cond = float('inf')
    
    return max_std, cond


def _generate_tangent_samples(rng, L2: np.ndarray, mu_X_deg: np.ndarray,
                             n_batch: int, use_antithetic: bool) -> np.ndarray:
    """Generate samples in tangent plane coordinates."""
    if use_antithetic and n_batch > 1:
        if n_batch % 2 == 1:
            n_batch -= 1
        half = max(1, n_batch // 2)
        z = rng.standard_normal((2, half))
        z_pair = np.concatenate([z, -z], axis=1)
        samples_X_deg = (L2 @ z_pair).T + mu_X_deg
        
        # Handle odd remainder
        if n_batch < len(z_pair[0]) * 2:
            z_last = rng.standard_normal((2, 1))
            samples_last = (L2 @ z_last).T + mu_X_deg
            samples_X_deg = np.vstack([samples_X_deg, samples_last])
    else:
        z = rng.standard_normal((2, n_batch))
        samples_X_deg = (L2 @ z).T + mu_X_deg
    
    return samples_X_deg


def _compute_mc_tangent(
    mu_P: np.ndarray,
    mu_Q: np.ndarray,
    Sigma_X_m2: np.ndarray,
    d0_meters: float,
    prob_threshold: float,
    n_mc: int,
    batch_size: int,
    random_state: Optional[int],
    conservative_decision: bool,
    delta: float,
    use_antithetic: bool = True,
    cp_alpha: float = 0.05,
    distance_metric: DistanceMetric = None
) -> ProbabilityResult:
    """Monte Carlo sampling in tangent plane coordinates."""
    if distance_metric is None:
        distance_metric = HaversineDistance()
    
    rng = np.random.default_rng(random_state)
    
    # Set up tangent plane sampling
    Sigma_X_deg2, mu_X_deg = _setup_tangent_plane_sampling(mu_P, mu_Q, Sigma_X_m2)
    
    # Compute diagnostics
    max_std, cond = _compute_tangent_diagnostics(Sigma_X_m2)

    # Cholesky decomposition for sampling
    Sigma_for_chol = Sigma_X_deg2.copy()
    try:
        L2 = np.linalg.cholesky(Sigma_for_chol)
    except np.linalg.LinAlgError:
        Sigma_for_chol = _add_jitter(Sigma_for_chol, rel=1e-10)
        L2 = np.linalg.cholesky(Sigma_for_chol)

    count_inside = 0
    total = 0
    n_remaining = n_mc

    # Monte Carlo sampling loop
    while n_remaining > 0:
        n_batch = min(batch_size, n_remaining)
        
        # Generate samples in tangent plane
        samples_X_deg = _generate_tangent_samples(rng, L2, mu_X_deg, n_batch, use_antithetic)
        samples_P = np.asarray(mu_Q) + samples_X_deg

        # Compute distances and count
        dists = distance_metric(samples_P[:, 0], samples_P[:, 1], mu_Q[0], mu_Q[1])
        count_inside += int(np.count_nonzero(dists <= d0_meters))
        total += samples_P.shape[0]
        n_remaining -= samples_P.shape[0]

    # Calculate final results
    prob = count_inside / total
    mc_stderr = float(np.sqrt(prob * (1 - prob) / total))
    cp_l, cp_u = clopper_pearson(count_inside, total, alpha=cp_alpha)

    if conservative_decision:
        fulfilled = (cp_l >= prob_threshold)
        decision_by = 'cp_lower'
    else:
        fulfilled = (prob >= prob_threshold)
        decision_by = 'point_estimate'

    return ProbabilityResult(
        fulfilled=fulfilled,
        probability=float(prob),
        method='mc_tangent',
        mc_stderr=mc_stderr,
        n_samples=int(total),
        cp_lower=float(cp_l),
        cp_upper=float(cp_u),
        sigma_cond=float(cond),
        max_std_m=float(max_std),
        delta_m=float(delta),
        decision_by=decision_by
    )


# ============================================================================
# PROFILER & UTILITIES (extended for v4)
# ============================================================================

class PerformanceProfiler:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[Dict] = []

    def _time_call(self, func, *args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, out

    def _run_analytic_test(self, scenario: Scenario, seed: int) -> tuple:
        """Run analytic test for a scenario."""
        t_start = time.perf_counter()
        analytic_params = MethodParams(mode='analytic', distance_metric=scenario.method_params.distance_metric)
        res_analytic = check_geo_prob(scenario.subject, scenario.reference, scenario.d0, analytic_params)
        t_elapsed = time.perf_counter() - t_start
        return res_analytic.probability, t_elapsed

    def _run_monte_carlo_test(self, scenario: Scenario, mode: str, seed: int) -> tuple:
        """Run Monte Carlo test (ecef or tangent) for a scenario."""
        n_mc = scenario.method_params.n_mc
        batch_size = scenario.method_params.batch_size
        
        t_start = time.perf_counter()
        mc_params = MethodParams(
            mode=mode,
            n_mc=n_mc,
            batch_size=batch_size,
            random_state=seed,
            distance_metric=scenario.method_params.distance_metric
        )
        result = check_geo_prob(scenario.subject, scenario.reference, scenario.d0, mc_params)
        t_elapsed = time.perf_counter() - t_start
        
        stderr = result.mc_stderr if result.mc_stderr is not None else 0.0
        n_samples = result.n_samples if result.n_samples is not None else n_mc
        
        return result.probability, t_elapsed, stderr, n_samples

    def _initialize_accumulator(self):
        """Initialize result accumulator for test batch."""
        return {
            'analytic': {'probs': [], 'times': []},
            'mc_ecef': {'probs': [], 'times': [], 'stderrs': [], 'n_samples': []},
            'mc_tangent': {'probs': [], 'times': [], 'stderrs': [], 'n_samples': []}
        }

    def _accumulate_results(self, acc: dict, mode: str, prob: float, time_elapsed: float, 
                           stderr: float = None, n_samples: int = None):
        """Accumulate results for a specific mode."""
        acc[mode]['probs'].append(prob)
        acc[mode]['times'].append(time_elapsed)
        if stderr is not None:
            acc[mode]['stderrs'].append(stderr)
        if n_samples is not None:
            acc[mode]['n_samples'].append(n_samples)

    def _create_scenario_summary(self, scenario: Scenario, acc: dict, has_analytic: bool) -> dict:
        """Create summary statistics for a scenario."""
        summary = {
            'name': scenario.name, 
            'd0': scenario.d0, 
            'n_mc': scenario.method_params.n_mc
        }
        
        # Analytic results
        if has_analytic:
            a_probs = np.array(acc['analytic']['probs'])
            a_times = np.array(acc['analytic']['times'])
            summary.update({
                'analytic_prob_mean': float(a_probs.mean()),
                'analytic_time_ms': float(a_times.mean() * 1000)
            })
        else:
            summary.update({'analytic_prob_mean': None, 'analytic_time_ms': None})

        # Monte Carlo results
        for method in ('mc_ecef', 'mc_tangent'):
            probs = np.array(acc[method]['probs'])
            times = np.array(acc[method]['times'])
            stderrs = np.array(acc[method]['stderrs'])
            n_samples_arr = np.array(acc[method]['n_samples'])
            summary.update({
                f'{method}_prob_mean': float(probs.mean()),
                f'{method}_prob_std': float(probs.std()),
                f'{method}_time_ms': float(times.mean() * 1000),
                f'{method}_mc_stderr_mean': float(stderrs.mean()),
                f'{method}_n_samples_mean': int(n_samples_arr.mean())
            })
        
        return summary

    def test_batch(self,
                        scenarios: List[Scenario],
                        n_repeats: int = 3,
                        default_n_mc: int = 200_000,
                        default_batch: int = 100_000,
                        rng_base_seed: int = 12345) -> List[Dict]:
        """Run fair test batch with structured Scenarios."""
        results = []
        
        for s_idx, scenario in enumerate(scenarios):
            acc = self._initialize_accumulator()
            has_analytic = scenario.sigma_P_scalar is not None and scenario.sigma_Q_scalar is not None

            if self.verbose:
                print(f"TEST CASE: {scenario.name} — d0={scenario.d0} m, n_mc={scenario.method_params.n_mc} (repeats={n_repeats})")
                if scenario.expected_behavior:
                    print(f"Expected: {scenario.expected_behavior}")

            for r in range(n_repeats):
                seed = rng_base_seed + 1000 * s_idx + r

                # Run analytic test if possible
                if has_analytic:
                    prob, time_elapsed = self._run_analytic_test(scenario, seed)
                    self._accumulate_results(acc, 'analytic', prob, time_elapsed)

                # Run Monte Carlo ECEF test
                prob, time_elapsed, stderr, n_samples = self._run_monte_carlo_test(scenario, 'mc_ecef', seed)
                self._accumulate_results(acc, 'mc_ecef', prob, time_elapsed, stderr, n_samples)
                
                if self.verbose:
                    print(f"    mc_ecef run {r}: P={prob:.6f}, time={time_elapsed*1000:.1f} ms")

                # Run Monte Carlo tangent test
                prob, time_elapsed, stderr, n_samples = self._run_monte_carlo_test(scenario, 'mc_tangent', seed)
                self._accumulate_results(acc, 'mc_tangent', prob, time_elapsed, stderr, n_samples)

            # Create summary
            summary = self._create_scenario_summary(scenario, acc, has_analytic)
            self.results.append(summary)
            results.append(summary)
        
        if self.verbose:
            self._print_summary_table(results, scenarios)
        
        return results
    
    def compare_metrics_on_scenario(self, 
                                    scenario: Scenario,
                                    metrics: Optional[Dict[str, DistanceMetric]] = None) -> Dict[str, ProbabilityResult]:
        """
        Run same scenario with different distance metrics.
        
        Args:
            scenario: Base scenario to test
            metrics: Dict of {name: metric} to compare. If None, uses standard set.
            
        Returns:
            Dict of {metric_name: ProbabilityResult}
        """
        if metrics is None:
            metrics = {
                'haversine': HaversineDistance(),
                'manhattan': ManhattanDistance(),
                'chebyshev': ChebyshevDistance(),
                'euclidean': EuclideanTangentDistance()
            }
        
        results = {}
        for name, metric in metrics.items():
            params = MethodParams(
                mode=scenario.method_params.mode,
                n_mc=scenario.method_params.n_mc,
                batch_size=scenario.method_params.batch_size,
                random_state=scenario.method_params.random_state,
                distance_metric=metric
            )
            result = check_geo_prob(scenario.subject, scenario.reference, scenario.d0, params)
            results[name] = result
            
            if self.verbose:
                print(f"  {name:12s}: P={result.probability:.6f}, fulfilled={result.fulfilled}")
        
        return results
    
    def _print_summary_table(self, results: List[Dict], scenarios: List[Scenario]):
        """Print a concise table summary of all results."""
        try:
            from tabulate import tabulate
        except ImportError:
            print("\nFor better table formatting, install tabulate: pip install tabulate")
            self._print_fallback_table(results)
            return
        
        print("\n" + "=" * 100)
        print("RESULTS SUMMARY TABLE")
        print("=" * 100)
        
        rows = []
        for r in results:
            scenario_name = r['name'][:18]
            d0 = r['d0']
            n_mc = r['n_mc']
            
            if r.get('analytic_prob_mean') is not None:
                rows.append([
                    scenario_name,
                    'analytic',
                    f"{r['analytic_prob_mean']:.5f}",
                    '-',
                    '-',
                    f"{r['analytic_time_ms']:.1f}",
                    f"{d0:.0f}m"
                ])
            
            for method in ['mc_ecef', 'mc_tangent']:
                if r.get(f'{method}_prob_mean') is not None:
                    prob_mean = r[f'{method}_prob_mean']
                    mc_stderr = r[f'{method}_mc_stderr_mean']
                    time_ms = r[f'{method}_time_ms']
                    n_samples = r[f'{method}_n_samples_mean']
                    
                    name_display = scenario_name if method == 'mc_ecef' else ''
                    d0_display = f"{d0:.0f}m" if method == 'mc_ecef' else ''
                    
                    rows.append([
                        name_display,
                        method,
                        f"{prob_mean:.5f}",
                        f"{mc_stderr:.5f}" if mc_stderr > 1e-6 else '<1e-6',
                        f"{n_samples:,}" if n_samples > 0 else '-',
                        f"{time_ms:.1f}",
                        d0_display
                    ])
        
        headers = ["Scenario", "Method", "Prob", "StdErr", "Samples", "Time(ms)", "d0"]
        print(tabulate(rows, headers=headers, tablefmt="simple", numalign="right"))
        print("=" * 100)
    
    def _print_fallback_table(self, results: List[Dict]):
        """Fallback table format if tabulate is not available."""
        print("\n" + "=" * 120)
        print("RESULTS SUMMARY")
        print("=" * 120)
        print(f"{'Scenario':<20} {'Method':<10} {'Mean Prob':<10} {'StdErr':<10} {'Samples':<8} {'Time(ms)':<9} {'d0(m)':<8}")
        print("-" * 120)
        
        for r in results:
            scenario_name = r['name'][:20]
            d0 = r['d0']
            
            if r.get('analytic_prob_mean') is not None:
                print(f"{scenario_name:<20} {'analytic':<10} {r['analytic_prob_mean']:<10.6f} {'-':<10} {'-':<8} {r['analytic_time_ms']:<9.1f} {d0:<8.0f}")
            
            for method in ['mc_ecef', 'mc_tangent']:
                if r.get(f'{method}_prob_mean') is not None:
                    name_field = scenario_name if method == 'mc_ecef' else ''
                    prob_mean = r[f'{method}_prob_mean']
                    mc_stderr = r[f'{method}_mc_stderr_mean']
                    time_ms = r[f'{method}_time_ms']
                    n_samples = r[f'{method}_n_samples_mean']
                    d0_field = f"{d0:.0f}" if method == 'mc_ecef' else ''
                    
                    print(f"{name_field:<20} {method:<10} {prob_mean:<10.6f} {mc_stderr:<10.6f} {n_samples:<8,} {time_ms:<9.1f} {d0_field:<8}")
        
        print("=" * 120)

    def estimate_runtime_scaling(self, target_n_mc: int = 200_000) -> Dict[str, float]:
        """Estimate runtime for target_n_mc by linear scaling."""
        estimates = {}
        for r in self.results:
            if r.get('mc_ecef_time_ms') is not None:
                n = r.get('n_mc', 1)
                t_ms = r.get('mc_ecef_time_ms', None)
                if t_ms is not None and n > 0:
                    per_sample_ms = t_ms / n
                    estimates.setdefault('mc_ecef', []).append(per_sample_ms)
            if r.get('mc_tangent_time_ms') is not None:
                n = r.get('n_mc', 1)
                t_ms = r.get('mc_tangent_time_ms', None)
                if t_ms is not None and n > 0:
                    per_sample_ms = t_ms / n
                    estimates.setdefault('mc_tangent', []).append(per_sample_ms)

        final = {}
        for method, arr in estimates.items():
            median_per_sample = float(np.median(arr))
            final[method] = float(median_per_sample * target_n_mc)
        return final

    def summary(self):
        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)
        for r in self.results:
            print(r)
        print("=" * 60)


# ============================================================================
# MAIN SCRIPT - Test scenarios and demonstrations
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import csv
    import os

    parser = argparse.ArgumentParser(description="Run geospatial probability checker v4 with distance metrics.")
    parser.add_argument("--quick", action="store_true", help="Run smaller quick tests (faster).")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per scenario (default 3).")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to write results JSON/CSV.")
    args = parser.parse_args()

    if args.quick:
        default_n_mc = 20_000
        default_batch = 10_000
        n_repeats = max(1, args.repeats)
        print("QUICK mode: using smaller Monte Carlo budgets for fast iteration.")
    else:
        default_n_mc = 200_000
        default_batch = 100_000
        n_repeats = args.repeats
    
    # Helper to create scenarios
    def create_scenario(
        name: str, 
        mu_P: Tuple[float, float], 
        sigma_P: Union[float, np.ndarray],
        mu_Q: Tuple[float, float], 
        sigma_Q: Union[float, np.ndarray],
        d0: float,
        expected_behavior: str = "",
        n_mc: int = default_n_mc,
        distance_metric: Optional[DistanceMetric] = None
    ) -> Scenario:
        """Create a Scenario using the structured format."""
        subject = Subject(
            mu=GeoPoint(*mu_P),
            Sigma=Covariance2D(sigma_P),
            id=f"subject_{name}"
        )
        reference = Reference(
            mu=GeoPoint(*mu_Q),
            Sigma=Covariance2D(sigma_Q),
            id=f"reference_{name}"
        )
        method_params = MethodParams(n_mc=n_mc, batch_size=default_batch, distance_metric=distance_metric)
        
        sigma_P_scalar = None
        sigma_Q_scalar = None
        if np.isscalar(sigma_P):
            sigma_P_scalar = float(sigma_P)
        if np.isscalar(sigma_Q):
            sigma_Q_scalar = float(sigma_Q)
        
        return Scenario(
            name=name,
            subject=subject,
            reference=reference,
            d0=d0,
            method_params=method_params,
            expected_behavior=expected_behavior,
            sigma_P_scalar=sigma_P_scalar,
            sigma_Q_scalar=sigma_Q_scalar
        )

    print("=" * 70)
    print("Geospatial Probability Checker - v4 (pluggable distance metrics)")
    print("=" * 70)
    print(f"mode: {'quick' if args.quick else 'full'}, default_n_mc={default_n_mc}, batch={default_batch}, repeats={n_repeats}")

    # Baseline test scenarios
    scenarios_baseline = [
        create_scenario(
            'isotropic_small_sigma',
            mu_P=(37.7749, -122.4194),
            sigma_P=5.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=3.0,
            d0=20.0,
            expected_behavior="High probability (~97%) - points very close with small uncertainty"
        ),
        create_scenario(
            'isotropic_large_sigma',
            mu_P=(37.7749, -122.4194),
            sigma_P=200.0,
            mu_Q=(37.77495, -122.41945),
            sigma_Q=150.0,
            d0=500.0,
            expected_behavior="Medium probability (~86%) - large uncertainty but reasonable threshold"
        ),
        create_scenario(
            'deterministic_exact_same_point',
            mu_P=(37.7749, -122.4194),
            sigma_P=0.0,
            mu_Q=(37.7749, -122.4194),
            sigma_Q=0.0,
            d0=1.0,
            expected_behavior="Probability = 1.0 exactly - identical points with no uncertainty"
        ),
        create_scenario(
            'anisotropic_cross_terms',
            mu_P=(37.7749, -122.4194),
            sigma_P=np.array([[400.0, 300.0], [300.0, 250.0]]),
            mu_Q=(37.7755, -122.4185),
            sigma_Q=np.array([[100.0, -20.0], [-20.0, 80.0]]),
            d0=100.0,
            expected_behavior="Medium probability (~44%) - large anisotropic uncertainty"
        ),
    ]

    # Metric comparison scenarios (improved with strategic threshold positioning)
    scenarios_metric_tests = [
        create_scenario(
            'manhattan_vs_haversine_optimal',
            mu_P=(40.7580, -73.9855),  # Times Square
            sigma_P=15.0,  # Increased uncertainty for interesting probabilities
            mu_Q=(40.7614, -73.9776),  # ~5 blocks N, 1 block E
            sigma_Q=15.0,
            d0=750.0,  # Strategic: between haversine (~640m) and manhattan (~900m)
            expected_behavior="Threshold between metrics: Haversine P~0.95, Manhattan P~0.30",
            distance_metric=HaversineDistance()
        ),
        create_scenario(
            'chebyshev_square_fence',
            mu_P=(37.7749, -122.4194),
            sigma_P=30.0,
            mu_Q=(37.7760, -122.4205),  # ~120m N, ~80m E (anisotropic)
            sigma_Q=20.0,
            d0=100.0,  # Less than max component (120m) but strategic
            expected_behavior="Rectangular fence: Chebyshev P~0.55, Euclidean P~0.80",
            distance_metric=ChebyshevDistance()
        ),
        create_scenario(
            'lp_norm_hierarchy',
            mu_P=(51.5074, -0.1278),  # London
            sigma_P=25.0,
            mu_Q=(51.5082, -0.1268),  # ~90m N, ~70m E ≈ 115m Euclidean
            sigma_Q=20.0,
            d0=120.0,  # Just above Euclidean distance to show clear ordering
            expected_behavior="Hierarchy: L1 P~0.25 < L2 P~0.65 < L3 P~0.80 < L∞ P~0.92",
            distance_metric=lp_distance(2)
        ),
        create_scenario(
            'edge_case_manhattan_vs_chebyshev_optimal',
            mu_P=(0.0, 0.0),
            sigma_P=8.0,  # Reduced uncertainty for better separation
            mu_Q=(0.0008, 0.0005),  # ~89m N, ~56m E (anisotropic for clear difference)
            sigma_Q=8.0, 
            d0=75.0,  # Strategic threshold between Manhattan (145m) and Chebyshev (89m)
            expected_behavior="Manhattan P~0.15 (sum=145m), Chebyshev P~0.85 (max=89m)",
            distance_metric=ManhattanDistance()
        ),
        create_scenario(
            'practical_rectangular_vs_circular_geofence',
            mu_P=(34.0522, -118.2437),  # LA center
            sigma_P=12.0,  # Reduced uncertainty 
            mu_Q=(34.0530, -118.2427),  # ~89m N, ~89m E (equal components)
            sigma_Q=8.0,  
            d0=100.0,  # Strategic: between Chebyshev (89m) and Euclidean (126m)
            expected_behavior="Chebyshev P~0.75 (square), Euclidean P~0.25 (circle)",
            distance_metric=ChebyshevDistance()
        ),
        create_scenario(
            'edge_case_pure_diagonal_movement',
            mu_P=(51.5074, -0.1278),  # London
            sigma_P=5.0,  # Small uncertainty for clear deterministic behavior
            mu_Q=(51.5083, -0.1269),  # ~100m N, ~63m E 
            sigma_Q=5.0,
            d0=140.0,  # Strategic: between Euclidean (118m) and Manhattan (163m)
            expected_behavior="Euclidean P~0.95, Manhattan P~0.15 (clear diagonal advantage)",
            distance_metric=EuclideanTangentDistance()
        ),
    ]

    # Create output directory
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    profiler = PerformanceProfiler(verbose=True)

    # Run baseline scenarios
    print("\n" + "=" * 70)
    print("BASELINE SCENARIOS")
    print("=" * 70)
    t_start_all = time.perf_counter()
    results_baseline = profiler.test_batch(scenarios_baseline, n_repeats=n_repeats)
    t_baseline = time.perf_counter() - t_start_all

    # Run metric comparison tests if requested
    print("\n" + "=" * 70)
    print("METRIC COMPARISON TESTS")
    print("=" * 70)
    print("Key principle: Thresholds positioned strategically between metric distances")
    print("to maximize separation and demonstrate clear differences.")
    print("=" * 70)
    
    for scenario in scenarios_metric_tests:
        print(f"\n--- {scenario.name} ---")
        print(f"Expected: {scenario.expected_behavior}")
        
        metrics_to_compare = {
            'haversine': HaversineDistance(),
            'manhattan': ManhattanDistance(),
            'euclidean': EuclideanTangentDistance(),
            'l3_norm': lp_distance(3),
            'chebyshev': ChebyshevDistance()
        }
        
        metric_results = profiler.compare_metrics_on_scenario(scenario, metrics_to_compare)
        
        # Validate metric ordering for Lp norms (key improvement from improvements_v4.py)
        if 'lp_norm' in scenario.name:
            print("\n📊 Lp Norm Ordering Validation:")
            expected_order = ['manhattan', 'euclidean', 'l3_norm', 'chebyshev']
            valid_ordering = True
            for i in range(len(expected_order) - 1):
                m1, m2 = expected_order[i], expected_order[i+1]
                if m1 in metric_results and m2 in metric_results:
                    p1, p2 = metric_results[m1].probability, metric_results[m2].probability
                    symbol = "✅" if p1 <= p2 else "❌"
                    print(f"   {symbol} P({m1}) = {p1:.4f} {'≤' if p1 <= p2 else '>'} P({m2}) = {p2:.4f}")
                    if p1 > p2:
                        valid_ordering = False
            
            if valid_ordering:
                print("   ✅ All ordering constraints satisfied!")
            else:
                print("   ⚠️  Some ordering violated (may be MC noise)")
        
        # Print comparison table with probability span analysis
        print("\nMetric Comparison:")
        print(f"{'Metric':<15} {'Probability':<12} {'Fulfilled':<10} {'Method':<12}")
        print("-" * 50)
        sorted_results = sorted(metric_results.items(), key=lambda x: x[1].probability)
        for metric_name, result in sorted_results:
            print(f"{metric_name:<15} {result.probability:<12.6f} {str(result.fulfilled):<10} {result.method:<12}")
        
        # Analyze probability span (key insight from improvements)
        probs = [r.probability for r in metric_results.values()]
        span = max(probs) - min(probs)
        print(f"\nProbability span: {span:.6f} (range: [{min(probs):.6f}, {max(probs):.6f}])")
        if span > 0.15:
            print("✅ Excellent separation - metrics clearly distinguishable")
        elif span > 0.08:
            print("✅ Good separation - metrics show meaningful differences") 
        elif span > 0.03:
            print("⚠️  Moderate separation - differences visible but subtle")
        else:
            print("❌ Poor separation - scenario may need adjustment")
    
    # Demonstrate weighted distance
    print("\n" + "=" * 70)
    print("WEIGHTED DISTANCE EXAMPLE")
    print("=" * 70)
    
    def delivery_zone_cost(lat: float, lon: float) -> float:
        """Example cost function: downtown vs suburbs"""
        # Simplistic: close to (37.7749, -122.4194) = downtown = 1.0x
        # Far = suburbs = 1.5x cost
        dist_from_downtown = np.sqrt((lat - 37.7749)**2 + (lon + 122.4194)**2)
        if dist_from_downtown < 0.01:  # ~1km radius
            return 1.0
        else:
            return 1.5
    
    weighted_metric = weighted_haversine(delivery_zone_cost)
    scenario_weighted = create_scenario(
        'weighted_delivery_cost',
        mu_P=(37.7749, -122.4194),  # Downtown
        sigma_P=50.0,
        mu_Q=(37.7850, -122.4100),  # Suburbs (~1.5km away)
        sigma_Q=30.0,
        d0=2000.0,  # 2km "cost-distance" threshold
        expected_behavior="Weighted distance accounts for delivery zones",
        distance_metric=weighted_metric
    )
    
    result_weighted = check_geo_prob(
        scenario_weighted.subject,
        scenario_weighted.reference,
        scenario_weighted.d0,
        scenario_weighted.method_params
    )
    print(f"Weighted distance result: P={result_weighted.probability:.6f}, fulfilled={result_weighted.fulfilled}")
    
    # Compare to unweighted
    scenario_unweighted = create_scenario(
        'unweighted_baseline',
        mu_P=(37.7749, -122.4194),
        sigma_P=50.0,
        mu_Q=(37.7850, -122.4100),
        sigma_Q=30.0,
        d0=2000.0,
        distance_metric=HaversineDistance()
    )
    result_unweighted = check_geo_prob(
        scenario_unweighted.subject,
        scenario_unweighted.reference,
        scenario_unweighted.d0,
        scenario_unweighted.method_params
    )
    print(f"Unweighted (haversine): P={result_unweighted.probability:.6f}, fulfilled={result_unweighted.fulfilled}")
    print(f"Difference: {abs(result_weighted.probability - result_unweighted.probability):.6f}")

    # Demonstrate OSM routing distance (if requests is available)
    print("\n" + "=" * 70)
    print("OSM ROUTING DISTANCE EXAMPLE")
    print("=" * 70)
    
    try:
        import requests
        
        print("Testing OSM routing distance (this requires internet connection)...")
        
        # Create OSM driving distance metric
        osm_metric = osm_driving_distance('osrm')  # Free OSRM service
        
        scenario_osm = create_scenario(
            'osm_driving_route',
            mu_P=(40.7580, -73.9855),  # Times Square, NYC
            sigma_P=30.0,
            mu_Q=(40.7614, -73.9776),  # 5 blocks away
            sigma_Q=20.0,
            d0=1200.0,  # 1.2km threshold
            expected_behavior="Real driving distance via street network",
            distance_metric=osm_metric
        )
        
        # Test single route calculation first
        try:
            driving_dist = osm_metric(40.7580, -73.9855, 40.7614, -73.9776)
            haversine_dist = haversine_distance_m(40.7580, -73.9855, 40.7614, -73.9776)
            
            print(f"Route comparison (Times Square to +5 blocks):")
            print(f"  OSM driving distance: {driving_dist:.0f}m")
            print(f"  Haversine distance:   {haversine_dist:.0f}m") 
            print(f"  Routing factor:       {driving_dist/haversine_dist:.2f}x")
            
            # Only run full probability calculation if single route works
            if driving_dist > 0:
                print("\nRunning probability calculation with OSM routing...")
                result_osm = check_geo_prob(
                    scenario_osm.subject,
                    scenario_osm.reference, 
                    scenario_osm.d0,
                    MethodParams(n_mc=5000, distance_metric=osm_metric)  # Smaller n_mc for API limits
                )
                print(f"OSM driving route result: P={result_osm.probability:.6f}, fulfilled={result_osm.fulfilled}")
                
                # Compare to geometric
                result_geometric = check_geo_prob(
                    scenario_osm.subject,
                    scenario_osm.reference,
                    scenario_osm.d0,
                    MethodParams(n_mc=5000, distance_metric=HaversineDistance())
                )
                print(f"Geometric (haversine):    P={result_geometric.probability:.6f}, fulfilled={result_geometric.fulfilled}")
                print(f"Difference: {abs(result_osm.probability - result_geometric.probability):.6f}")
                
                # Demonstrate time-based routing
                print("\n⏱️  TIME-BASED ROUTING EXAMPLE:")
                
                travel_time_metric = osm_travel_time('driving', 'osrm')
                
                # 20-minute commute scenario (1200 seconds)
                scenario_time = create_scenario(
                    'commute_time_accessibility',
                    mu_P=(40.7580, -73.9855),  # Times Square
                    sigma_P=50.0,  # Larger uncertainty for time analysis
                    mu_Q=(40.7829, -73.9654),  # Central Park
                    sigma_Q=30.0,
                    d0=1200.0,  # 20 minutes = 1200 seconds
                    expected_behavior="20-minute driving commute accessibility",
                    distance_metric=travel_time_metric
                )
                
                result_time = check_geo_prob(
                    scenario_time.subject,
                    scenario_time.reference,
                    scenario_time.d0,
                    MethodParams(n_mc=3000, distance_metric=travel_time_metric)  # Smaller for API limits
                )
                
                print(f"20-minute accessibility: P={result_time.probability:.6f}, fulfilled={result_time.fulfilled}")
                
                # Compare with multi-modal
                print("\n🚶🚗 MULTI-MODAL EXAMPLE:")
                multimodal_metric = multimodal_travel_time([
                    ('walking', 500),     # Walk up to 500m
                    ('driving', float('inf'))  # Drive longer distances
                ], transfer_time=180)  # 3-minute transfer (finding parking, etc.)
                
                result_multimodal = check_geo_prob(
                    scenario_time.subject,
                    scenario_time.reference,
                    scenario_time.d0,
                    MethodParams(n_mc=2000, distance_metric=multimodal_metric)
                )
                
                print(f"Multi-modal (walk+drive): P={result_multimodal.probability:.6f}, fulfilled={result_multimodal.fulfilled}")
                print(f"Time penalty from transfers: {abs(result_time.probability - result_multimodal.probability):.3f}")
                
        except Exception as route_error:
            print(f"Route calculation failed: {route_error}")
            print("This may be due to API rate limits or network issues.")
            
    except ImportError:
        print("OSM routing requires 'requests' library: pip install requests")
        print("Skipping OSM routing example.")
    except Exception as e:
        print(f"OSM routing example skipped: {e}")

    t_total = time.perf_counter() - t_start_all

    # Save results
    all_results = results_baseline
    
    json_path = os.path.join(outdir, f"profiler_results_v4_{int(time.time())}.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved JSON summary to: {json_path}")

    keys = set()
    for r in all_results:
        keys.update(r.keys())
    keys = sorted(keys)

    csv_path = os.path.join(outdir, f"profiler_results_v4_{int(time.time())}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"Saved CSV summary to: {csv_path}")

    estimated = profiler.estimate_runtime_scaling(target_n_mc=default_n_mc)
    print("\nEstimated runtime scaling (ms) for target n_mc:")
    print(estimated)

    print(f"Total profiling wall time: {t_total:.2f} s")

    # Application Guidelines (from improvements)
    print("\n" + "=" * 70)
    print("📚 DISTANCE METRIC APPLICATION GUIDELINES")
    print("=" * 70)
    print("Based on test results and geometric properties:")
    print("   • Urban navigation (grid streets) → ManhattanDistance")
    print("   • Circular geofences → HaversineDistance or EuclideanTangentDistance") 
    print("   • Rectangular safe zones → ChebyshevDistance")
    print("   • Cost-based routing → WeightedDistance with custom cost function")
    print("   • Real road routing → OSMRoutingDistance (driving/walking/cycling)")
    print("   • Travel time analysis → OSMTravelTime (time as distance)")
    print("   • Multi-modal transport → MultiModalTravelTime (realistic transfers)")
    print("   • Transit accessibility → TransitAccessibilityTime (walk + transit)")
    print("   • Bike-share systems → BikeShareTime (cycling + walking)")
    print("   • Delivery optimization → OSMDrivingDistance with traffic zones")
    print("   • Pedestrian navigation → OSMWalkingDistance")
    print("   • Default/general purpose → HaversineDistance (backward compatible)")
    print("   • Survey-grade precision → VincentyDistance (requires geopy)")
    
    # Distance calculation reference (from improvements)
    print("\n📐 DISTANCE REFERENCE - NYC Grid Example:")
    lat1, lon1 = 40.7580, -73.9855  # Times Square
    lat2, lon2 = 40.7614, -73.9776  # +5 blocks N, +1 block E
    
    lat_diff_m = abs(lat2 - lat1) * 111132.954
    lon_diff_m = abs(lon2 - lon1) * 111132.954 * np.cos(np.deg2rad((lat1+lat2)/2))
    manhattan_dist = lat_diff_m + lon_diff_m
    euclidean_dist = np.sqrt(lat_diff_m**2 + lon_diff_m**2)
    chebyshev_dist = max(lat_diff_m, lon_diff_m)
    
    print(f"   Haversine:  ~640m (great circle)")
    print(f"   Manhattan:  {manhattan_dist:.0f}m (sum of components)")  
    print(f"   Euclidean:  {euclidean_dist:.0f}m (Pythagorean)")
    print(f"   Chebyshev:  {chebyshev_dist:.0f}m (max component)")
    print(f"   Ratio Manhattan/Haversine: {manhattan_dist/640:.2f}x")
    print("=" * 70)

    print("\nDone!")
