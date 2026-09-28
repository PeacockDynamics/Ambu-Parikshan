"""Doppler velocity log sensor models for HydroLab-3D."""

import numpy as np

from hydrolab3d.sensors.sensor import (
    measure_with_bias_noise,
)
from hydrolab3d.world.currents import (
    relative_velocity_6dof,
)


def ideal_dvl_measurement(
    pose,
    body_velocity,
    current_velocity_world,
):
    """
    Return ideal body-frame water-relative translational velocity.

    Parameters
    ----------
    pose : array-like
        Pose [x, y, z, roll, pitch, yaw].

    body_velocity : array-like
        Generalized body velocity [u, v, w, p, q, r].

    current_velocity_world : array-like
        World-frame ocean-current velocity [vc_x, vc_y, vc_z].

    Returns
    -------
    np.ndarray
        Body-frame translational velocity relative to the water,
        [u_r, v_r, w_r], in m/s.
    """
    pose = np.asarray(
        pose,
        dtype=float,
    )

    body_velocity = np.asarray(
        body_velocity,
        dtype=float,
    )

    current_velocity_world = np.asarray(
        current_velocity_world,
        dtype=float,
    )

    if pose.shape != (6,):
        raise ValueError(
            "pose must have shape (6,)"
        )

    if body_velocity.shape != (6,):
        raise ValueError(
            "body_velocity must have shape (6,)"
        )

    if current_velocity_world.shape != (3,):
        raise ValueError(
            "current_velocity_world must have shape (3,)"
        )

    if not np.all(np.isfinite(pose)):
        raise ValueError(
            "pose must contain only finite values"
        )

    if not np.all(
        np.isfinite(body_velocity)
    ):
        raise ValueError(
            "body_velocity must contain only finite values"
        )

    if not np.all(
        np.isfinite(current_velocity_world)
    ):
        raise ValueError(
            "current_velocity_world must contain only finite values"
        )

    relative_velocity = relative_velocity_6dof(
        pose,
        body_velocity,
        current_velocity_world,
    )

    relative_velocity = np.asarray(
        relative_velocity,
        dtype=float,
    )

    if relative_velocity.shape != (6,):
        raise RuntimeError(
            "relative_velocity_6dof returned unexpected shape"
        )

    return relative_velocity[:3].copy()


def dvl_measurement(
    pose,
    body_velocity,
    current_velocity_world,
    *,
    bias=0.0,
    noise_std=0.0,
    rng=None,
):
    """
    Return imperfect DVL water-relative velocity measurement.
    """
    ideal = ideal_dvl_measurement(
        pose,
        body_velocity,
        current_velocity_world,
    )

    measured = measure_with_bias_noise(
        ideal,
        bias=bias,
        noise_std=noise_std,
        rng=rng,
    )

    if measured.shape != (3,):
        raise ValueError(
            "DVL bias and noise_std must be "
            "scalar or broadcast-compatible with shape (3,)"
        )

    return measured
