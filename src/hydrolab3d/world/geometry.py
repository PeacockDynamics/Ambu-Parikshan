"""Lightweight world geometry used by navigation, sensing, and rendering.

The module intentionally provides geometry queries only.  It does not alter
the validated vehicle equations or inject unvalidated collision forces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _point(value, name="point") -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return point


@dataclass(frozen=True)
class WaterVolume:
    """Axis-aligned navigable volume with an upward-positive world z-axis."""

    minimum: np.ndarray
    maximum: np.ndarray
    surface_z: float = 0.0

    def __post_init__(self):
        minimum, maximum = _point(self.minimum, "minimum"), _point(self.maximum, "maximum")
        if np.any(minimum >= maximum):
            raise ValueError("minimum must be strictly smaller than maximum on every axis")
        if not np.isfinite(float(self.surface_z)):
            raise ValueError("surface_z must be finite")
        object.__setattr__(self, "minimum", minimum.copy())
        object.__setattr__(self, "maximum", maximum.copy())

    def contains(self, point) -> bool:
        point = _point(point)
        return bool(np.all(point >= self.minimum) and np.all(point <= self.maximum) and point[2] <= self.surface_z)


@dataclass(frozen=True)
class FlatSeabed:
    """Horizontal seabed represented by its world-frame z coordinate."""

    z: float

    def height(self, x: float, y: float) -> float:
        if not np.isfinite(float(x)) or not np.isfinite(float(y)) or not np.isfinite(float(self.z)):
            raise ValueError("seabed coordinates and z must be finite")
        return float(self.z)


@dataclass(frozen=True)
class HeightField:
    """Regular-grid bathymetry with bilinear height interpolation."""

    x: np.ndarray
    y: np.ndarray
    heights: np.ndarray

    def __post_init__(self):
        x, y, heights = np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float), np.asarray(self.heights, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            raise ValueError("height-field x and y must be strictly increasing vectors with at least two values")
        if heights.shape != (len(y), len(x)) or not np.all(np.isfinite(heights)):
            raise ValueError("height-field heights must be finite with shape (len(y), len(x))")
        object.__setattr__(self, "x", x.copy()); object.__setattr__(self, "y", y.copy()); object.__setattr__(self, "heights", heights.copy())

    def height(self, x: float, y: float) -> float:
        x, y = float(x), float(y)
        if not np.isfinite(x) or not np.isfinite(y) or x < self.x[0] or x > self.x[-1] or y < self.y[0] or y > self.y[-1]:
            raise ValueError("height query lies outside the height-field domain")
        return float(np.interp(y, self.y, [np.interp(x, self.x, row) for row in self.heights]))


@dataclass(frozen=True)
class SphereObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "center", _point(self.center, "center").copy())
        if not np.isfinite(float(self.radius)) or self.radius <= 0:
            raise ValueError("radius must be finite and strictly positive")

    def contains(self, point, radius: float = 0.0) -> bool:
        radius = float(radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and non-negative")
        return bool(np.linalg.norm(_point(point) - self.center) <= self.radius + radius)


@dataclass(frozen=True)
class BoxObstacle:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self):
        minimum, maximum = _point(self.minimum, "minimum"), _point(self.maximum, "maximum")
        if np.any(minimum >= maximum):
            raise ValueError("box minimum must be strictly smaller than maximum")
        object.__setattr__(self, "minimum", minimum.copy()); object.__setattr__(self, "maximum", maximum.copy())

    def contains(self, point, radius: float = 0.0) -> bool:
        """Conservative axis-aligned margin query, not contact dynamics."""
        radius = float(radius)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and non-negative")
        point = _point(point)
        return bool(np.all(point >= self.minimum - radius) and np.all(point <= self.maximum + radius))


@dataclass
class WorldGeometry:
    """Composition root for read-only world geometry queries."""

    volume: WaterVolume
    seabed: FlatSeabed | HeightField
    obstacles: list[SphereObstacle | BoxObstacle] = field(default_factory=list)

    def seabed_height(self, x: float, y: float) -> float:
        return self.seabed.height(x, y)

    def altitude(self, position) -> float:
        position = _point(position, "position")
        return float(position[2] - self.seabed_height(position[0], position[1]))

    def collisions(self, position, radius: float = 0.0) -> list[int]:
        """Return obstacle/seabed query hits; water-volume walls are not hits."""
        position = _point(position, "position")
        hits = [index for index, obstacle in enumerate(self.obstacles) if obstacle.contains(position, radius)]
        if position[2] - float(radius) <= self.seabed_height(position[0], position[1]):
            hits.append(-1)  # -1 denotes seabed contact.
        return hits
