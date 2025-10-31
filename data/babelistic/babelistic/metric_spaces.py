# ============================================================================
# METRIC SPACE IMPLEMENTATIONS
# ============================================================================

from typing import Tuple, Dict

import numpy as np


from .base import MetricSpace

class EuclideanSpace(MetricSpace):
    """Standard Euclidean R^n space"""
    
    def __init__(self, dim: int = 2):
        self.dim = dim
    
    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Euclidean distance: ||x1 - x2||_2"""
        diff = x1 - x2
        if diff.ndim == 1:
            return np.linalg.norm(diff)
        return np.linalg.norm(diff, axis=-1)
    
    def area_element(self, x: np.ndarray) -> np.ndarray:
        """Constant area element in Euclidean space"""
        return np.ones(x.shape[:-1] if x.ndim > 1 else 1)
    
    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:
        """Create uniform Cartesian grid"""
        if len(bounds) == 4:  # 2D: (xmin, xmax, ymin, ymax)
            x = np.linspace(bounds[0], bounds[1], resolution)
            y = np.linspace(bounds[2], bounds[3], resolution)
            X, Y = np.meshgrid(x, y)
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            
            points = np.stack([X, Y], axis=-1)
            weights = np.ones_like(X) * dx * dy
            
            return {
                'points': points,
                'weights': weights,
                'X': X, 'Y': Y,
                'x': x, 'y': y,
                'shape': X.shape
            }
        else:
            raise NotImplementedError("Only 2D implemented")


class SphericalSpace(MetricSpace):
    """Sphere with geodesic distance (e.g., Earth surface)"""
    
    def __init__(self, radius: float = 6371.0):
        self.radius = radius  # km for Earth
    
    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Haversine distance on sphere"""
        lon1, lat1 = np.radians(x1[..., 0]), np.radians(x1[..., 1])
        lon2, lat2 = np.radians(x2[..., 0]), np.radians(x2[..., 1])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        
        return self.radius * c
    
    def area_element(self, x: np.ndarray) -> np.ndarray:
        """Area element on a sphere (pure spherical coordinates).

        This implementation assumes points are provided as (lon, lat)
        in degrees (consistent with `distance`). It returns the factor
        R^2 * cos(lat) (latitude in degrees). The caller should multiply
        by dlon*dlat in radians when integrating over a grid.
        """
        x_arr = np.asarray(x)

        # Expect coordinates in (lon, lat) order. Extract latitude.
        lat = x_arr[..., 1]
        return self.radius ** 2 * np.cos(np.radians(lat))
    
    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:
        """Create lon/lat grid with proper area weighting"""
        lon_min, lon_max, lat_min, lat_max = bounds
        
        # Adaptive latitude resolution
        lat_mid = (lat_min + lat_max) / 2
        lat_res = int(resolution * (lat_max - lat_min) / 
                     ((lon_max - lon_min) * np.cos(np.radians(lat_mid))))
        lat_res = max(lat_res, 60)
        
        lon = np.linspace(lon_min, lon_max, resolution)
        lat = np.linspace(lat_min, lat_max, lat_res)
        LON, LAT = np.meshgrid(lon, lat)
        
        dlon = np.radians(lon[1] - lon[0])
        dlat = np.radians(lat[1] - lat[0])
        
        points = np.stack([LON, LAT], axis=-1)
        weights = self.area_element(points) * dlon * dlat
        
        return {
            'points': points,
            'weights': weights,
            'LON': LON, 'LAT': LAT,
            'lon': lon, 'lat': lat,
            'shape': LON.shape
        }


class ManhattanSpace(MetricSpace):
    """L1 metric space (taxicab geometry)"""
    
    def __init__(self, dim: int = 2):
        self.dim = dim
    
    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Manhattan distance: sum(|x1 - x2|)"""
        diff = np.abs(x1 - x2)
        if diff.ndim == 1:
            return np.sum(diff)
        return np.sum(diff, axis=-1)
    
    def area_element(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape[:-1] if x.ndim > 1 else 1)
    
    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:
        """Reuse Euclidean grid (only metric differs)"""
        return EuclideanSpace(self.dim).create_grid(bounds, resolution)


class GeoSpace(MetricSpace):
    """Geographical space with (lat, lon) in degrees using haversine distance.

    This class is intended for geographic computations where coordinates are
    provided as (latitude, longitude) in degrees. The area element and grid
    creation use spherical approximations (R * cos(lat)).
    """

    def __init__(self, radius: float = 6371.0):
        # radius in kilometers (default Earth)
        self.radius = radius

    def distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Haversine distance for inputs in degrees.

        Accepts both single point (1D) and array inputs with broadcasting.
        Points must be in (lat, lon) order.
        """
        # Convert to arrays
        a1 = np.asarray(x1)
        a2 = np.asarray(x2)

        # Helper to get lat/lon radians with flexible shapes
        def _to_radians(arr):
            return np.radians(arr[..., 0]), np.radians(arr[..., 1])

        lat1, lon1 = _to_radians(a1)
        lat2, lon2 = _to_radians(a2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        return self.radius * c

    def area_element(self, x: np.ndarray) -> np.ndarray:
        """Local area element: R^2 cos(lat) for lat in degrees.

        Expects points as (lat, lon) in degrees; returns area per radian
        square (i.e., R^2 * cos(lat)). Caller must multiply by dlat*dlon in
        radians when integrating over a grid.
        """
        lat = np.radians(np.asarray(x)[..., 0])
        return self.radius ** 2 * np.cos(lat)

    def create_grid(self, bounds: Tuple, resolution: int) -> Dict[str, np.ndarray]:
        """Create lat/lon grid.

        Bounds: (lat_min, lat_max, lon_min, lon_max) in degrees.
        resolution: number of latitude points; longitude resolution adapts to
        maintain approximate spacing in degrees adjusted by cos(lat).
        """
        lat_min, lat_max, lon_min, lon_max = bounds

        # Choose longitude resolution scaled by latitude extent
        lat_mid = (lat_min + lat_max) / 2.0
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Avoid division by zero; ensure at least 4 longitudes
        lon_res = max(4, int(resolution * (lon_range / max(lat_range * np.cos(np.radians(lat_mid)), 1e-6))))

        lat = np.linspace(lat_min, lat_max, resolution)
        lon = np.linspace(lon_min, lon_max, lon_res)
        LAT, LON = np.meshgrid(lat, lon, indexing='ij')

        dlat = np.radians(lat[1] - lat[0]) if len(lat) > 1 else 0.0
        dlon = np.radians(lon[1] - lon[0]) if len(lon) > 1 else 0.0

        points = np.stack([LAT, LON], axis=-1)
        weights = self.area_element(points) * dlat * dlon

        return {
            'points': points,
            'weights': weights,
            'LAT': LAT, 'LON': LON,
            'lat': lat, 'lon': lon,
            'shape': LAT.shape
        }
