# ============================================================================
# REGION IMPLEMENTATIONS
# ============================================================================

from typing import Tuple, Callable, Optional, List

import matplotlib
import numpy as np

from .base import Region, MetricSpace
from .metric_spaces import EuclideanSpace


class DiskRegion(Region):
    """Circular/spherical region"""
    
    def __init__(self, center: np.ndarray, radius: float, 
                 metric_space: Optional[MetricSpace] = None):
        self.center = np.asarray(center)
        self.radius = radius
        self.metric = metric_space or EuclideanSpace(len(center))
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """1 if inside, 0 if outside"""
        dist = self.metric.distance(x, self.center)
        # Use strict inside (<) to avoid boundary ambiguities when combining regions
        return (dist < self.radius).astype(float)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample points on circle/sphere"""
        if len(self.center) == 2:
            theta = np.linspace(0, 2*np.pi, n, endpoint=False)
            points = self.center + self.radius * np.column_stack([
                np.cos(theta), np.sin(theta)
            ])
            return points
        raise NotImplementedError("Only 2D implemented")
    
    def bounds(self) -> Tuple:
        """Bounding box"""
        margin = self.radius * 1.5
        return (self.center[0] - margin, self.center[0] + margin,
                self.center[1] - margin, self.center[1] + margin)


class PolygonRegion(Region):
    """Convex polygon region"""
    
    def __init__(self, vertices: np.ndarray):
        self.vertices = np.asarray(vertices)
        # Compute convex hull for robust indicator
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        self.hull_vertices = vertices[hull.vertices]
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """Point-in-polygon test"""
        from matplotlib.path import Path
        
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        path = Path(self.hull_vertices)
        inside = path.contains_points(x_flat)
        
        return inside.reshape(original_shape).astype(float)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample uniformly along edges"""
        n_edges = len(self.hull_vertices)
        samples_per_edge = n // n_edges
        
        points = []
        for i in range(n_edges):
            v1 = self.hull_vertices[i]
            v2 = self.hull_vertices[(i + 1) % n_edges]
            t = np.linspace(0, 1, samples_per_edge, endpoint=False)
            edge_points = v1 + t[:, None] * (v2 - v1)
            points.append(edge_points)
        
        return np.vstack(points)
    
    def bounds(self) -> Tuple:
        """Bounding box"""
        margin = 0.2 * np.ptp(self.vertices, axis=0).mean()
        xmin, ymin = self.vertices.min(axis=0)
        xmax, ymax = self.vertices.max(axis=0)
        return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)


class ImplicitRegion(Region):
    """Region defined by implicit function f(x) <= 0"""
    
    def __init__(self, 
                 sdf: Callable[[np.ndarray], np.ndarray],
                 bounds: Tuple,
                 samples_cache: Optional[np.ndarray] = None):
        self.sdf = sdf  # Signed distance function
        self._bounds = bounds
        self._samples_cache = samples_cache
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """1 where sdf <= 0"""
        x_arr = np.asarray(x)

        # Handle single-point input (1D) by reshaping to (1, D)
        if x_arr.ndim == 1:
            vals = self.sdf(x_arr.reshape(1, -1))
            # sdf may return array-like; take first element
            val0 = np.asarray(vals).ravel()[0]
            return float(val0 <= 0)

        # For batched inputs, assume sdf accepts (N, D) arrays
        vals = self.sdf(x_arr)
        return (np.asarray(vals) <= 0).astype(float)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample near zero level set (requires cache or marching)"""
        if self._samples_cache is not None:
            idx = np.random.choice(len(self._samples_cache), n, replace=True)
            return self._samples_cache[idx]
        
        # Fallback: grid sampling near boundary
        from scipy.spatial.distance import cdist
        grid = EuclideanSpace().create_grid(self._bounds, 100)
        pts = grid['points'].reshape(-1, 2)
        vals = self.sdf(pts)
        boundary_mask = np.abs(vals) < 0.1 * np.ptp(vals)
        boundary_pts = pts[boundary_mask]
        
        if len(boundary_pts) < n:
            return boundary_pts
        idx = np.random.choice(len(boundary_pts), n, replace=False)
        return boundary_pts[idx]
    
    def bounds(self) -> Tuple:
        return self._bounds



class EllipseRegion(Region):
    """
    Elliptical region with arbitrary orientation
    
    Mathematical definition:
        (x - c)ᵀ A (x - c) ≤ 1
        where A defines the ellipse shape and orientation
    
    Properties:
        - Generalizes disk (isotropic case)
        - Can be rotated
        - Specified by center, axes, and rotation
    
    Use cases:
        - Anisotropic uncertainty regions
        - Covariance-based regions
        - Oriented geofences
    
    Examples:
        >>> # Ellipse with semi-axes 2 and 1, rotated 45°
        >>> region = EllipseRegion(
        ...     center=[0, 0],
        ...     semi_axes=[2.0, 1.0],
        ...     rotation_deg=45.0
        ... )
    """
    
    def __init__(self, 
                 center: np.ndarray,
                 semi_axes: np.ndarray,
                 rotation_deg: float = 0.0):
        """
        Parameters
        ----------
        center : np.ndarray
            Ellipse center [x, y]
        semi_axes : np.ndarray
            Semi-axes lengths [a, b] where a ≥ b
        rotation_deg : float
            Rotation angle in degrees (counter-clockwise)
        """
        self.center = np.asarray(center)
        self.semi_axes = np.asarray(semi_axes)
        self.rotation_deg = rotation_deg
        
        # Build transformation matrix
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotation matrix (rotates by theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])

        # For the quadratic form A such that (x-c)^T A (x-c) <= 1,
        # when the ellipse axes are rotated by R and semi-axes are a,b,
        # the correct A is R @ diag(1/a^2, 1/b^2) @ R.T
        inv_sq = np.diag(1.0 / (self.semi_axes ** 2))
        self.A = R @ inv_sq @ R.T
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """1 if inside ellipse, 0 otherwise"""
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Centered points
        diff = x_flat - self.center
        
        # Quadratic form: (x-c)ᵀ A (x-c) ≤ 1
        quad_form = np.sum(diff @ self.A * diff, axis=-1)
        
        return (quad_form <= 1.0).astype(float).reshape(original_shape)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample points on ellipse boundary"""
        # Parametric: x(t) = center + R @ [a*cos(t), b*sin(t)]
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Points on axis-aligned ellipse
        points_local = np.column_stack([
            self.semi_axes[0] * np.cos(t),
            self.semi_axes[1] * np.sin(t)
        ])
        
        # Rotate
        theta = np.radians(self.rotation_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        points_rotated = points_local @ R.T
        
        return self.center + points_rotated
    
    def bounds(self) -> Tuple:
        """Bounding box"""
        # Maximum extent in each direction
        theta = np.radians(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        a, b = self.semi_axes
        
        # Axis-aligned bounding box of rotated ellipse
        dx = np.sqrt((a * cos_t)**2 + (b * sin_t)**2)
        dy = np.sqrt((a * sin_t)**2 + (b * cos_t)**2)
        
        margin = 0.1 * max(dx, dy)
        
        return (
            self.center[0] - dx - margin,
            self.center[0] + dx + margin,
            self.center[1] - dy - margin,
            self.center[1] + dy + margin
        )


class BufferedPolygonRegion(Region):
    """
    Polygon with buffer zone (dilation)
    
    Creates a region that is the original polygon expanded by a buffer distance.
    
    Properties:
        - Smooth boundaries (if buffer > 0)
        - Useful for safety margins
        - Can shrink (negative buffer) or grow (positive)
    
    Use cases:
        - Safety zones around hazards
        - Right-of-way buffers
        - Tolerance regions
        - Environmental setbacks
    
    Examples:
        >>> # Square with 50m buffer
        >>> vertices = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> region = BufferedPolygonRegion(vertices, buffer=50.0)
    """
    
    def __init__(self, vertices: np.ndarray, buffer: float = 0.0):
        """
        Parameters
        ----------
        vertices : np.ndarray (n_vertices, 2)
            Polygon vertices
        buffer : float
            Buffer distance (can be negative for erosion)
        """
        self.vertices = np.asarray(vertices)
        self.buffer = buffer
        
        # Compute convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        self.hull_vertices = vertices[hull.vertices]
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """
        Membership via distance to polygon boundary
        
        Inside if: signed_distance < buffer
        """
        from matplotlib.path import Path
        
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Check if inside base polygon
        path = Path(self.hull_vertices)
        inside_base = path.contains_points(x_flat)
        
        if self.buffer == 0:
            return inside_base.astype(float).reshape(original_shape)
        
        # For buffered region, compute distance to boundary
        # Points inside: distance is positive up to buffer
        # Points outside: distance is negative up to -buffer
        
        result = np.zeros(len(x_flat))
        
        for i, point in enumerate(x_flat):
            # Distance to closest edge
            min_dist = float('inf')
            
            for j in range(len(self.hull_vertices)):
                v1 = self.hull_vertices[j]
                v2 = self.hull_vertices[(j + 1) % len(self.hull_vertices)]
                
                # Distance to edge segment
                dist = self._point_to_segment_distance(point, v1, v2)
                min_dist = min(min_dist, dist)
            
            # Inside if distance to boundary < buffer
            if inside_base[i]:
                # Inside base polygon: always inside if buffer >= 0
                result[i] = 1.0 if self.buffer >= 0 else float(min_dist < abs(self.buffer))
            else:
                # Outside base polygon: inside if within buffer distance
                result[i] = float(min_dist < self.buffer) if self.buffer > 0 else 0.0
        
        return result.reshape(original_shape)
    
    def _point_to_segment_distance(self, point, v1, v2):
        """Distance from point to line segment"""
        # Vector from v1 to v2
        seg = v2 - v1
        seg_len_sq = np.dot(seg, seg)
        
        if seg_len_sq < 1e-10:
            return np.linalg.norm(point - v1)
        
        # Projection parameter
        t = np.clip(np.dot(point - v1, seg) / seg_len_sq, 0, 1)
        
        # Closest point on segment
        closest = v1 + t * seg
        
        return np.linalg.norm(point - closest)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample on buffered boundary"""
        # For simplicity, sample on hull + offset
        base_boundary = []
        
        for i in range(len(self.hull_vertices)):
            v1 = self.hull_vertices[i]
            v2 = self.hull_vertices[(i + 1) % len(self.hull_vertices)]
            
            # Edge direction and normal
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            
            # Sample along edge
            n_edge = max(1, n // len(self.hull_vertices))
            t = np.linspace(0, 1, n_edge, endpoint=False)
            
            for ti in t:
                point_on_edge = v1 + ti * edge
                buffered_point = point_on_edge + self.buffer * normal
                base_boundary.append(buffered_point)
        
        return np.array(base_boundary)
    
    def bounds(self) -> Tuple:
        """Bounding box including buffer"""
        margin = abs(self.buffer) + 0.1 * np.ptp(self.vertices, axis=0).mean()
        xmin, ymin = self.vertices.min(axis=0)
        xmax, ymax = self.vertices.max(axis=0)
        
        return (
            xmin - margin,
            xmax + margin,
            ymin - margin,
            ymax + margin
        )


class MultiRegion(Region):
    """
    Union or intersection of multiple regions
    
    Properties:
        - Combines arbitrary regions
        - Boolean operations (AND, OR)
        - Hierarchical composition
    
    Use cases:
        - Complex geofences (union of circles)
        - Exclusion zones (intersection of constraints)
        - Composite safety regions
    
    Examples:
        >>> # Union of two circles
        >>> region1 = DiskRegion([0, 0], 1.0)
        >>> region2 = DiskRegion([1.5, 0], 1.0)
        >>> union = MultiRegion([region1, region2], operation='union')
        >>> 
        >>> # Intersection (AND)
        >>> intersection = MultiRegion([region1, region2], operation='intersection')
    """
    
    def __init__(self, 
                 regions: List[Region],
                 operation: str = 'union'):
        """
        Parameters
        ----------
        regions : List[Region]
            List of regions to combine
        operation : str
            'union' (OR) or 'intersection' (AND)
        """
        if not regions:
            raise ValueError("Must provide at least one region")
        
        self.regions = regions
        self.operation = operation.lower()
        
        if self.operation not in ['union', 'intersection']:
            raise ValueError(f"operation must be 'union' or 'intersection', got '{operation}'")
    
    def indicator(self, x: np.ndarray) -> np.ndarray:
        """Combined indicator based on operation"""
        # Evaluate all regions
        indicators = [region.indicator(x) for region in self.regions]
        indicators = np.array(indicators)
        
        if self.operation == 'union':
            # Union: 1 if ANY region contains point
            # Probabilistic: max or sum (we use max for boolean logic)
            return np.max(indicators, axis=0)
        else:  # intersection
            # Intersection: 1 if ALL regions contain point
            # Probabilistic: min or product (we use min for boolean logic)
            return np.min(indicators, axis=0)
    
    def sample_boundary(self, n: int) -> np.ndarray:
        """Sample from boundaries of constituent regions"""
        # Distribute samples across regions
        samples_per_region = max(1, n // len(self.regions))
        
        all_samples = []
        for region in self.regions:
            samples = region.sample_boundary(samples_per_region)
            all_samples.append(samples)
        
        return np.vstack(all_samples)
    
    def bounds(self) -> Tuple:
        """Combined bounding box"""
        all_bounds = [region.bounds() for region in self.regions]
        
        xmin = min(b[0] for b in all_bounds)
        xmax = max(b[1] for b in all_bounds)
        ymin = min(b[2] for b in all_bounds)
        ymax = max(b[3] for b in all_bounds)
        
        return (xmin, xmax, ymin, ymax)

