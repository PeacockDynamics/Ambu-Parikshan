"""Selective-fidelity geometry-based sonar observations.

This is intentionally a range/bearing detector, not an acoustic propagation
model.  It is kept out of the historical public sensor export until an
application chooses to configure it explicitly.
"""

from __future__ import annotations

import numpy as np

from hydrolab3d.utils.transforms import rotation_body_to_world


def seabed_altitude(pose, world) -> float:
    """Return vertical clearance above the configured seabed in metres."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite vector with shape (6,)")
    return world.altitude(pose[:3])


def range_bearing_detections(pose, world, max_range: float) -> list[dict[str, float]]:
    """Return obstacle-center range, azimuth and elevation in the body frame."""
    pose = np.asarray(pose, dtype=float)
    max_range = float(max_range)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite vector with shape (6,)")
    if not np.isfinite(max_range) or max_range <= 0:
        raise ValueError("max_range must be finite and strictly positive")
    rotation = rotation_body_to_world(pose[3], pose[4], pose[5])
    detections = []
    for index, obstacle in enumerate(world.obstacles):
        center = obstacle.center if hasattr(obstacle, "center") else (obstacle.minimum + obstacle.maximum) / 2.0
        relative_body = rotation.T @ (center - pose[:3])
        distance = float(np.linalg.norm(relative_body))
        if distance <= max_range:
            detections.append({
                "obstacle_index": index,
                "range": distance,
                "bearing": float(np.arctan2(relative_body[1], relative_body[0])),
                "elevation": float(np.arctan2(relative_body[2], np.hypot(relative_body[0], relative_body[1]))),
            })
    return detections
