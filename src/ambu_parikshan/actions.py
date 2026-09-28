"""Application action sources that feed the single HydroLab action boundary."""

from __future__ import annotations

import numpy as np
from hydrolab3d.control import PIDController, waypoint_guidance
from hydrolab3d.sensors import wrap_angle_pi


class ActionSource:
    """Small protocol-like base for application action generators."""

    def reset(self) -> None:
        """Reset source state before a new deterministic run."""

    def action(self, state: dict[str, object]) -> np.ndarray:
        """Return one finite generalized body-frame wrench."""
        raise NotImplementedError


class ZeroAction(ActionSource):
    def action(self, state: dict[str, object]) -> np.ndarray:
        return np.zeros(6, dtype=float)


class ConstantWrench(ActionSource):
    def __init__(self, wrench: object):
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("wrench must be a finite vector with shape (6,)")
        self._wrench = wrench.copy()

    def action(self, state: dict[str, object]) -> np.ndarray:
        return self._wrench.copy()


class WaypointAction(ActionSource):
    """Use the validated guidance/PID primitives to produce a body wrench.

    This initial application controller uses the supplied state estimate; the
    headless scenario runtime currently supplies ground truth because no
    localization system exists, matching the notebook's explicit boundary.
    """
    def __init__(self, config):
        self._waypoint = np.asarray(config["waypoint"], dtype=float)
        self._surge = float(config["surge_force"])
        self._depth = PIDController(float(config["heave_kp"]), 0, 0, output_limits=(-float(config["max_heave_force"]), float(config["max_heave_force"])))
        self._yaw = PIDController(float(config["yaw_kp"]), 0, 0, output_limits=(-float(config["max_yaw_moment"]), float(config["max_yaw_moment"])))

    def reset(self): self._depth.reset(); self._yaw.reset()

    def action(self, state):
        pose = np.asarray(state["pose"], dtype=float); guidance = waypoint_guidance(pose[:3], self._waypoint)
        # HydroLab uses +z upward while depth is positive downward.  Notebook
        # 09 therefore maps a positive depth-controller command to -Fz.
        heave, _ = self._depth.update(guidance["desired_depth"] - (-pose[2]), 0.02)
        yaw, _ = self._yaw.update(wrap_angle_pi(guidance["desired_heading"] - pose[5]), 0.02)
        surge = self._surge if guidance["horizontal_distance"] > .1 else 0.0
        return np.array([surge, 0, -heave, 0, 0, yaw], dtype=float)


def action_source(config: dict[str, object]) -> ActionSource:
    """Build an action source from validated scenario action configuration."""
    if config["mode"] == "zero":
        return ZeroAction()
    if config["mode"] == "constant_wrench":
        return ConstantWrench(config["wrench"])
    if config["mode"] == "waypoint":
        return WaypointAction(config)
    raise ValueError(f"unsupported action mode: {config['mode']!r}")
