#!/usr/bin/env python3
"""
Enhanced OSM Routing Demo - Real-world distance and time-based geospatial probability analysis.

This demo shows how to integrate OpenStreetMap routing services for realistic
distance and travel time calculations in geospatial probability analysis.
"""

import sys
import time
from pathlib import Path

# Add the current directory to path to import main.py
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import (
        # Core probability functions
        check_geo_prob,
        Subject, Reference, GeoPoint, Covariance2D, MethodParams,
        
        # Geometric distance metrics
        haversine_distance_m, HaversineDistance, ManhattanDistance, EuclideanTangentDistance,
        
        # OSM routing classes
        OSMRoutingDistance, OSMTravelTime, 
        
        # Convenience functions
        osm_driving_distance, osm_travel_time,
        
        # Multi-modal classes (if available)
        MultiModalTravelTime, CustomDistance
    )
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure main.py is in the same directory.")
    sys.exit(1)


def demo_basic_osm_routing():
    """Demonstrate basic OSM routing vs geometric distances."""
    
    print("=" * 70)
    print("OSM ROUTING DISTANCE DEMONSTRATION") 
    print("=" * 70)
    
    # Times Square to Central Park (shorter urban route)
    lat1, lon1 = 40.7580, -73.9855  # Times Square
    lat2, lon2 = 40.7829, -73.9654  # Central Park
    
    print("Route: Times Square → Central Park")
    print(f"From: ({lat1}, {lon1})")
    print(f"To:   ({lat2}, {lon2})")
    print()
    
    # Show geometric baseline
    print("📐 GEOMETRIC DISTANCES:")
    haversine_dist = haversine_distance_m(lat1, lon1, lat2, lon2)
    manhattan_metric = ManhattanDistance()
    manhattan_dist = manhattan_metric(lat1, lon1, lat2, lon2)
    euclidean_metric = EuclideanTangentDistance()
    euclidean_dist = euclidean_metric(lat1, lon1, lat2, lon2)
    
    print(f"  Haversine (crow flies): {haversine_dist:.0f}m")
    print(f"  Manhattan (grid):       {manhattan_dist:.0f}m") 
    print(f"  Euclidean (tangent):    {euclidean_dist:.0f}m")
    print()
    
    # Test OSM routing distances
    print("🗺️  OSM ROUTING DISTANCES:")
    distance_tests = [
        ("OSRM Driving", 'osrm', 'driving'),
        ("OSRM Walking", 'osrm', 'walking'),
        ("OSRM Cycling", 'osrm', 'cycling'),
    ]
    
    osrm_distances = {}
    for name, engine, profile in distance_tests:
        try:
            start_time = time.time()
            metric = OSMRoutingDistance(engine, profile, timeout=5)
            distance = metric(lat1, lon1, lat2, lon2)
            elapsed = (time.time() - start_time) * 1000
            
            crow_ratio = distance / haversine_dist
            print(f"  {name:<15}: {distance:.0f}m ({crow_ratio:.2f}x crow flies, {elapsed:.0f}ms)")
            osrm_distances[profile] = distance
            
        except Exception as e:
            print(f"  {name:<15}: Failed ({e})")
    print()
    
    return osrm_distances, haversine_dist, lat1, lon1, lat2, lon2


def demo_travel_times(osrm_distances, haversine_dist, lat1, lon1, lat2, lon2):
    """Demonstrate travel time calculations with realistic comparisons."""
    
    print("⏱️  OSM TRAVEL TIMES:")
    print("⚠️  NOTE: Free OSRM demo server returns identical times for all profiles.")
    print("    Production routing services (GraphHopper, Mapbox) provide realistic differentiation.")
    print()
    
    time_tests = [
        ("Driving Time", 'osrm', 'driving'),
        ("Walking Time", 'osrm', 'walking'),
        ("Cycling Time", 'osrm', 'cycling'),
    ]
    
    actual_travel_times = {}
    for name, engine, profile in time_tests:
        try:
            start_time = time.time()
            metric = OSMTravelTime(engine, profile, timeout=5)
            travel_time = metric(lat1, lon1, lat2, lon2)
            elapsed = (time.time() - start_time) * 1000
            
            minutes = travel_time / 60
            print(f"  {name:<15}: {travel_time:.0f}sec ({minutes:.1f}min, {elapsed:.0f}ms)")
            
            # Show speed calculation
            distance = osrm_distances.get(profile, haversine_dist * 1.4)
            speed_ms = distance / travel_time if travel_time > 0 else 0
            speed_kmh = speed_ms * 3.6
            print(f"  {'':15}   Distance: {distance:.0f}m, Speed: {speed_kmh:.1f}km/h")
            actual_travel_times[profile] = travel_time
            
        except Exception as e:
            print(f"  {name:<15}: Failed ({e})")
    print()
    
    # Show realistic travel times for comparison
    print("🚀 REALISTIC TRAVEL TIMES (for comparison):")
    print("    Here's what production routing services would typically return:")
    
    base_distance = osrm_distances.get('driving', haversine_dist * 1.4)
    realistic_times = {
        'driving': base_distance / (35 * 1000/3600),    # 35 km/h in NYC traffic
        'cycling': base_distance / (15 * 1000/3600),    # 15 km/h cycling
        'walking': base_distance / (5 * 1000/3600),     # 5 km/h walking
    }
    
    for mode, time_sec in realistic_times.items():
        minutes = time_sec / 60
        speed_kmh = (base_distance / time_sec) * 3.6
        print(f"  {mode.title()} (realistic): {time_sec:.0f}sec ({minutes:.1f}min, {speed_kmh:.1f}km/h)")
    print()
    
    return actual_travel_times, realistic_times


def demo_probability_calculations(lat1, lon1, lat2, lon2, realistic_times):
    """Demonstrate probability calculations with time-based metrics."""
    
    print("🎯 PROBABILITY CALCULATION EXAMPLES:")
    print()
    
    # Example 1: Distance-based delivery scenario
    print("📦 Example 1: Delivery Route Planning")
    threshold_distance = 5000  # 5km delivery radius
    print(f"Scenario: Delivery uncertainty with {threshold_distance/1000}km threshold")
    
    start_time = time.time()
    osm_metric = OSMRoutingDistance('osrm', 'driving')
    subject = Subject(GeoPoint(lat1, lon1), Covariance2D(30.0))
    reference = Reference(GeoPoint(lat2, lon2), Covariance2D(20.0))
    
    try:
        result_osm = check_geo_prob(subject, reference, d0_meters=threshold_distance,
                                   method_params=MethodParams(distance_metric=osm_metric))
        osm_time = time.time() - start_time
        print(f"  OSM driving route: P={result_osm.probability:.4f} (time: {osm_time:.2f}s)")
    except Exception as e:
        print(f"  OSM driving route: Failed ({e})")
        result_osm = None
        osm_time = 0
    
    # Compare with geometric
    start_time = time.time()
    result_geom = check_geo_prob(subject, reference, d0_meters=threshold_distance,
                                method_params=MethodParams(distance_metric=haversine_distance_m))
    geom_time = time.time() - start_time
    print(f"  Geometric (haversine): P={result_geom.probability:.4f} (time: {geom_time:.4f}s)")
    
    if result_osm:
        prob_diff = abs(result_osm.probability - result_geom.probability)
        print(f"  Probability difference: {prob_diff:.4f}")
        if geom_time > 0:
            print(f"  OSM is {osm_time/geom_time:.1f}x slower than geometric")
    print()
    
    # Example 2: Time-based accessibility analysis
    print("🕐 Example 2: Time-Based Accessibility Analysis")
    threshold_minutes = 8
    threshold_seconds = threshold_minutes * 60
    print(f"Scenario: Commute accessibility with {threshold_minutes}-minute threshold")
    
    # Use realistic driving time
    def realistic_driving_time(lat1, lon1, lat2, lon2):
        distance = haversine_distance_m(lat1, lon1, lat2, lon2) * 1.3  # Route factor
        return distance / (35 * 1000/3600)  # 35 km/h in city traffic
    
    custom_driving_time = CustomDistance(realistic_driving_time)
    
    try:
        result_time = check_geo_prob(subject, reference, d0_meters=threshold_seconds,
                                    method_params=MethodParams(distance_metric=custom_driving_time))
        print(f"  Realistic driving time: P={result_time.probability:.4f}")
        
        # Compare different time thresholds
        time_thresholds = [5*60, 10*60, 15*60, 20*60]  # 5, 10, 15, 20 minutes
        print(f"  Time threshold sensitivity:")
        for thresh_min in [5, 10, 15, 20]:
            thresh_sec = thresh_min * 60
            result = check_geo_prob(subject, reference, d0_meters=thresh_sec,
                                   method_params=MethodParams(distance_metric=custom_driving_time))
            print(f"    {thresh_min:2d} minutes: P={result.probability:.4f}")
            
    except Exception as e:
        print(f"  Time-based calculation failed: {e}")
    print()


def demo_multimodal_transportation(lat1, lon1, lat2, lon2):
    """Demonstrate multi-modal transportation scenarios."""
    
    print("🚶🚗 MULTI-MODAL TRANSPORTATION EXAMPLES:")
    print()
    
    # Example 1: Walk + Drive combination
    print("Example 1: Walk short distances, drive longer ones")
    
    walking_threshold = 500  # Walk up to 500m
    transfer_time = 120      # 2 minutes to get to/from car
    
    def multimodal_walk_drive_time(lat1, lon1, lat2, lon2):
        """Realistic multi-modal time: walk short, drive long distances."""
        distance = haversine_distance_m(lat1, lon1, lat2, lon2) * 1.3  # Route factor
        
        if distance <= walking_threshold:
            # Walk the whole way
            return distance / (5 * 1000/3600)  # 5 km/h walking
        else:
            # Walk to car + drive + walk from car (simplified)
            walking_time = (walking_threshold / (5 * 1000/3600)) * 2  # To and from
            driving_distance = distance - walking_threshold
            driving_time = driving_distance / (35 * 1000/3600)  # 35 km/h
            return walking_time + driving_time + transfer_time
    
    custom_multimodal = CustomDistance(multimodal_walk_drive_time)
    
    # Example 2: Bike + Walk combination  
    def bike_walk_time(lat1, lon1, lat2, lon2):
        """Bike moderate distances, walk short ones."""
        distance = haversine_distance_m(lat1, lon1, lat2, lon2) * 1.2  # Bike route factor
        
        if distance <= 300:
            # Walk very short distances
            return distance / (5 * 1000/3600)
        elif distance <= 3000:
            # Bike moderate distances
            return distance / (15 * 1000/3600)  # 15 km/h cycling
        else:
            # Walk to bike + bike + walk from bike
            bike_distance = distance - 600  # 300m walk on each end
            walk_time = 600 / (5 * 1000/3600)
            bike_time = bike_distance / (15 * 1000/3600)
            return walk_time + bike_time + 60  # 1 minute to get/return bike
    
    custom_bike_walk = CustomDistance(bike_walk_time)
    
    subject = Subject(GeoPoint(lat1, lon1), Covariance2D(25.0))
    reference = Reference(GeoPoint(lat2, lon2), Covariance2D(15.0))
    
    threshold_minutes = 15
    threshold_seconds = threshold_minutes * 60
    
    print(f"Comparing {threshold_minutes}-minute accessibility:")
    
    try:
        # Pure walking
        def pure_walking_time(lat1, lon1, lat2, lon2):
            distance = haversine_distance_m(lat1, lon1, lat2, lon2) * 1.2
            return distance / (5 * 1000/3600)
        walking_metric = CustomDistance(pure_walking_time)
        
        result_walking = check_geo_prob(subject, reference, d0_meters=threshold_seconds,
                                       method_params=MethodParams(distance_metric=walking_metric))
        print(f"  Pure walking:     P={result_walking.probability:.4f}")
        
        # Multi-modal walk+drive
        result_multimodal = check_geo_prob(subject, reference, d0_meters=threshold_seconds,
                                          method_params=MethodParams(distance_metric=custom_multimodal))
        print(f"  Walk + drive:     P={result_multimodal.probability:.4f}")
        
        # Bike + walk
        result_bike = check_geo_prob(subject, reference, d0_meters=threshold_seconds,
                                    method_params=MethodParams(distance_metric=custom_bike_walk))
        print(f"  Bike + walk:      P={result_bike.probability:.4f}")
        
        # Pure driving  
        def pure_driving_time(lat1, lon1, lat2, lon2):
            distance = haversine_distance_m(lat1, lon1, lat2, lon2) * 1.3
            return distance / (35 * 1000/3600)
        driving_metric = CustomDistance(pure_driving_time)
        
        result_driving = check_geo_prob(subject, reference, d0_meters=threshold_seconds,
                                       method_params=MethodParams(distance_metric=driving_metric))
        print(f"  Pure driving:     P={result_driving.probability:.4f}")
        
        print()
        print("  Analysis:")
        best_prob = max(result_walking.probability, result_multimodal.probability, 
                       result_bike.probability, result_driving.probability)
        
        if result_multimodal.probability == best_prob:
            print("  → Multi-modal walk+drive provides best accessibility")
        elif result_bike.probability == best_prob:
            print("  → Bike+walk provides best accessibility")
        elif result_driving.probability == best_prob:
            print("  → Pure driving provides best accessibility")
        else:
            print("  → Pure walking provides best accessibility")
            
    except Exception as e:
        print(f"  Multi-modal comparison failed: {e}")
    print()


def show_integration_examples():
    """Show code examples for integrating OSM routing."""
    
    print("=" * 70)
    print("🚀 INTEGRATION EXAMPLES")
    print("=" * 70)
    
    print("""
# 1. Basic distance-based usage
from main import osm_driving_distance, check_geo_prob, Subject, Reference, GeoPoint, Covariance2D

metric = osm_driving_distance('osrm')  # Free OSRM
subject = Subject(GeoPoint(40.7580, -73.9855), Covariance2D(20.0))
reference = Reference(GeoPoint(40.7829, -73.9654), Covariance2D(15.0))
result = check_geo_prob(subject, reference, d0_meters=3000,
                       method_params=MethodParams(distance_metric=metric))

# 2. Time-based accessibility analysis
from main import osm_travel_time

# 15-minute walking accessibility (900 seconds)
walking_time = osm_travel_time('walking', 'osrm')
        result = check_geo_prob(home, transit_stop, d0_meters=900,
                       method_params=MethodParams(distance_metric=walking_time))# 30-minute driving commute (1800 seconds)
driving_time = osm_travel_time('driving', 'osrm')
result = check_geo_prob(home, work, d0_meters=1800,
                       method_params=MethodParams(distance_metric=driving_time))

# 3. Custom multi-modal transportation
from main import CustomDistance

def transit_access_time(lat1, lon1, lat2, lon2):
    \"\"\"Walk to transit + transit time + walk from transit.\"\"\"
    # Simplified: walk to nearest transit (avg 400m) + travel + walk from transit
    distance = haversine_distance_m(lat1, lon1, lat2, lon2)
    
    walk_to_transit = 400 / (5 * 1000/3600)  # 400m at 5 km/h
    transit_time = distance / (25 * 1000/3600)  # 25 km/h average transit speed
    walk_from_transit = 400 / (5 * 1000/3600)
    waiting_time = 300  # 5 minutes average wait
    
    return walk_to_transit + transit_time + walk_from_transit + waiting_time

transit_metric = CustomDistance(transit_access_time)

# 4. Different routing profiles
walking = OSMRoutingDistance('osrm', 'walking')
cycling = OSMRoutingDistance('osrm', 'cycling')
driving = OSMRoutingDistance('osrm', 'driving')

# Return travel time instead of distance
walking_time = OSMTravelTime('osrm', 'walking')
cycling_time = OSMTravelTime('osrm', 'cycling')
driving_time = OSMTravelTime('osrm', 'driving')

# 5. Commercial routing services (require API keys)
# graphhopper = OSMRoutingDistance('graphhopper', 'driving', api_key='your_key')
# mapbox = OSMRoutingDistance('mapbox', 'driving', api_key='your_key')

# 6. Time-based geofencing (isochrones)
# Check if destination is within 20-minute drive time
time_fence = osm_travel_time('driving', 'osrm')
within_20min = check_geo_prob(origin, destination, d0_meters=1200,  # 20 min = 1200 sec
                             method_params=MethodParams(distance_metric=time_fence))

# 7. Complex multi-modal with realistic factors
def realistic_urban_commute(lat1, lon1, lat2, lon2):
    \"\"\"Realistic urban commute: walk + transit + walk with weather/traffic factors.\"\"\"
    distance = haversine_distance_m(lat1, lon1, lat2, lon2)
    
    if distance < 500:
        # Walk short distances
        return distance / (5 * 1000/3600)
    elif distance < 15000:
        # Transit for medium distances
        walk_access = 600 / (5 * 1000/3600)  # Walk to/from stops
        transit_time = distance / (20 * 1000/3600)  # 20 km/h effective transit speed
        wait_time = 420  # 7 minutes average wait (includes transfer)
        return walk_access + transit_time + wait_time
    else:
        # Drive long distances
        return distance * 1.3 / (45 * 1000/3600)  # 45 km/h highway + city

urban_commute = CustomDistance(realistic_urban_commute)


📚 KEY CONCEPTS:
  • Distance metrics: Return meters for spatial analysis
  • Time metrics: Return seconds for temporal analysis
  • Multi-modal: Combines modes with transfer penalties and realistic routing
  • Thresholds: d0 in meters for distance, seconds for time
  • Use cases: Delivery (distance), commuting (time), accessibility (time)

🕐 TIME THRESHOLD EXAMPLES:
  • 5 minutes = 300 seconds (quick errands)
  • 15 minutes = 900 seconds (walking to transit)
  • 30 minutes = 1800 seconds (typical commute)
  • 1 hour = 3600 seconds (regional accessibility)
  • 2 hours = 7200 seconds (inter-city travel)

🌐 PRODUCTION ROUTING SERVICES:
  • OSRM: Free demo (limited), self-hosted (unlimited)
  • GraphHopper: Commercial API with free tier
  • Mapbox: Commercial API with generous free tier
  • Valhalla: Open source, MapTiler provides hosted service

⚡ PERFORMANCE TIPS:
  • Use caching for repeated calculations (built-in LRU cache)
  • Batch requests when possible
  • Consider fallback to geometric distances for API failures
  • Use appropriate timeouts to balance accuracy vs speed
""")


def main():
    """Run the complete OSM routing demonstration."""
    
    try:
        # Basic OSM routing demo
        osrm_distances, haversine_dist, lat1, lon1, lat2, lon2 = demo_basic_osm_routing()
        
        # Travel time demonstrations
        actual_times, realistic_times = demo_travel_times(osrm_distances, haversine_dist, lat1, lon1, lat2, lon2)
        
        # Probability calculation examples
        demo_probability_calculations(lat1, lon1, lat2, lon2, realistic_times)
        
        # Multi-modal transportation examples
        demo_multimodal_transportation(lat1, lon1, lat2, lon2)
        
        # Integration examples
        show_integration_examples()
        
    except ImportError:
        print("❌ OSM routing requires 'requests' library:")
        print("   pip install requests")
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()