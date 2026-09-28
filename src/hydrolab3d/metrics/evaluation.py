"""Small, pure metrics for trajectories and multi-vehicle runs."""

import numpy as np


def trajectory_length(positions) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite with shape (n, 3)")
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def final_waypoint_error(position, waypoint) -> float:
    position, waypoint = np.asarray(position, dtype=float), np.asarray(waypoint, dtype=float)
    if position.shape != (3,) or waypoint.shape != (3,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(waypoint)):
        raise ValueError("position and waypoint must be finite vectors with shape (3,)")
    return float(np.linalg.norm(waypoint - position))


def minimum_inter_vehicle_separation(positions) -> float:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3) for n >= 2")
    return float(min(np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i)))
