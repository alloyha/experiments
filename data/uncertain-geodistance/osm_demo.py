#!/usr/bin/env python3
"""
Demonstration of OSM routing integration with the geospatial probability checker.

This script shows how to use real road network distances instead of geometric distances.

Requirements:
    pip install requests

Usage:
    python osm_demo.py
"""

from main import *
import time

def demo_osm_routing():
    """Demonstrate OSM routing capabilities."""
    
    print("=" * 70)
    print("OSM ROUTING DISTANCE DEMONSTRATION")
    print("=" * 70)
    
    # Test coordinates: Times Square to Central Park
    lat1, lon1 = 40.7580, -73.9855  # Times Square
    lat2, lon2 = 40.7829, -73.9654  # Central Park (Bethesda Fountain)
    
    print(f"Route: Times Square → Central Park")
    print(f"From: ({lat1:.4f}, {lon1:.4f})")
    print(f"To:   ({lat2:.4f}, {lon2:.4f})")
    print()
    
    # Calculate geometric distances
    haversine_dist = haversine_distance_m(lat1, lon1, lat2, lon2)
    manhattan_dist = ManhattanDistance()(lat1, lon1, lat2, lon2)
    euclidean_dist = EuclideanTangentDistance()(lat1, lon1, lat2, lon2)
    
    print("📐 GEOMETRIC DISTANCES:")
    print(f"  Haversine (crow flies): {haversine_dist:.0f}m")
    print(f"  Manhattan (grid):       {manhattan_dist:.0f}m")
    print(f"  Euclidean (tangent):    {euclidean_dist:.0f}m")
    print()
    
    # Test OSM routing (if available)
    try:
        import requests
        
        print("🗺️  OSM ROUTING DISTANCES:")
        
        # Test different routing engines
        routing_tests = [
            ('OSRM (free)', 'osrm', 'driving'),
            ('OSRM Walking', 'osrm', 'walking'),
            ('OSRM Cycling', 'osrm', 'cycling'),
        ]
        
        for name, engine, profile in routing_tests:
            try:
                start_time = time.time()
                metric = OSMRoutingDistance(engine, profile, timeout=5)
                distance = metric(lat1, lon1, lat2, lon2)
                elapsed = (time.time() - start_time) * 1000
                
                ratio = distance / haversine_dist
                print(f"  {name:<15}: {distance:.0f}m ({ratio:.2f}x crow flies, {elapsed:.0f}ms)")
                
            except Exception as e:
                print(f"  {name:<15}: Failed ({e})")
        
        print()
        
        # Probability calculation example
        print("🎯 PROBABILITY CALCULATION EXAMPLE:")
        print("Scenario: Delivery uncertainty with 500m threshold")
        
        # Create scenario with driving distance
        driving_metric = osm_driving_distance('osrm')
        
        scenario = Scenario(
            name='osm_delivery',
            subject=Subject(GeoPoint(lat1, lon1), Covariance2D(25.0)),  # 25m uncertainty
            reference=Reference(GeoPoint(lat2, lon2), Covariance2D(15.0)),  # 15m uncertainty  
            d0=3000.0,  # 3km threshold
            method_params=MethodParams(n_mc=5000, distance_metric=driving_metric)
        )
        
        try:
            start_time = time.time()
            result_osm = check_geo_prob(scenario.subject, scenario.reference, scenario.d0, scenario.method_params)
            osm_time = time.time() - start_time
            
            # Compare with haversine
            scenario.method_params.distance_metric = HaversineDistance()
            start_time = time.time()
            result_geo = check_geo_prob(scenario.subject, scenario.reference, scenario.d0, scenario.method_params)
            geo_time = time.time() - start_time
            
            print(f"  OSM driving route: P={result_osm.probability:.4f} (time: {osm_time:.2f}s)")
            print(f"  Geometric (haversine): P={result_geo.probability:.4f} (time: {geo_time:.2f}s)")
            print(f"  Probability difference: {abs(result_osm.probability - result_geo.probability):.4f}")
            print(f"  Speed ratio: {osm_time/geo_time:.1f}x slower")
            
        except Exception as e:
            print(f"  Probability calculation failed: {e}")
        
    except ImportError:
        print("❌ OSM routing requires 'requests' library:")
        print("   pip install requests")
    except Exception as e:
        print(f"❌ OSM routing failed: {e}")
    
    print()
    print("=" * 70)
    print("🚀 INTEGRATION EXAMPLES")
    print("=" * 70)
    
    print("""
# 1. Basic usage
from main import osm_driving_distance, check_geo_prob, Subject, Reference, GeoPoint, Covariance2D

metric = osm_driving_distance('osrm')  # Free OSRM
subject = Subject(GeoPoint(40.7580, -73.9855), Covariance2D(20.0))
reference = Reference(GeoPoint(40.7829, -73.9654), Covariance2D(15.0))
result = check_geo_prob(subject, reference, d0=3000, 
                       MethodParams(distance_metric=metric))

# 2. Different routing profiles
walking = OSMWalkingDistance('osrm')
cycling = OSMCyclingDistance('osrm') 
driving = OSMDrivingDistance('osrm')

# 3. Commercial routing services (require API keys)
graphhopper = osm_driving_distance('graphhopper', api_key='your_key')
mapbox = OSMRoutingDistance('mapbox', 'driving', api_key='your_key')

# 4. Combine with weighted distance for complex scenarios
def traffic_multiplier(lat, lon):
    # Apply traffic congestion factors
    if in_downtown(lat, lon):
        return 1.5  # 50% slower
    return 1.0

traffic_aware = WeightedDistance(driving, traffic_multiplier)
""")


if __name__ == "__main__":
    demo_osm_routing()