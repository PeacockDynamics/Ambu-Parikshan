"""Optional ROS 2 conversion boundary; importing HydroLab never needs ROS."""

from __future__ import annotations

import numpy as np


def observation_dict_to_ros_fields(observation):
    """Return serializable standard-message field values without importing ROS."""
    required = {"depth", "heading", "dvl", "gyroscope", "accelerometer"}
    missing = required - set(observation)
    if missing: raise ValueError(f"observation is missing fields: {sorted(missing)}")
    return {"depth": float(observation["depth"]), "heading": float(observation["heading"]), "dvl": np.asarray(observation["dvl"], dtype=float).copy(), "angular_velocity": np.asarray(observation["gyroscope"], dtype=float).copy(), "linear_acceleration": np.asarray(observation["accelerometer"], dtype=float).copy()}


def ros2_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError: return False
    return True


def require_ros2():
    if not ros2_available():
        raise RuntimeError("ROS 2 support requires rclpy and ROS 2 message packages; install them in a ROS 2 environment")
