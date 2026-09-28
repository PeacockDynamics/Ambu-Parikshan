"""Inertial sensor models for HydroLab-3D."""

import numpy as np

from hydrolab3d.sensors.sensor import (
    measure_with_bias_noise,
)
from hydrolab3d.utils.transforms import (
    rotation_body_to_world,
)


def ideal_gyroscope_measurement(
    body_velocity,
):
    """
    Return ideal body-frame angular velocity.

    Parameters
    ----------
    body_velocity : array-like
        Generalized body velocity [u, v, w, p, q, r].

    Returns
    -------
    np.ndarray
        Body angular velocity [p, q, r] in rad/s.
    """
    body_velocity = np.asarray(
        body_velocity,
        dtype=float,
    )

    if body_velocity.shape != (6,):
        raise ValueError(
            "body_velocity must have shape (6,)"
        )

    if not np.all(
        np.isfinite(body_velocity)
    ):
        raise ValueError(
            "body_velocity must contain only finite values"
        )

    return body_velocity[3:6].copy()


def gyroscope_measurement(
    body_velocity,
    *,
    bias=0.0,
    noise_std=0.0,
    rng=None,
):
    """
    Return imperfect body-frame gyroscope measurement.
    """
    ideal = ideal_gyroscope_measurement(
        body_velocity
    )

    measured = measure_with_bias_noise(
        ideal,
        bias=bias,
        noise_std=noise_std,
        rng=rng,
    )

    if measured.shape != (3,):
        raise ValueError(
            "gyroscope bias and noise_std must be "
            "scalar or broadcast-compatible with shape (3,)"
        )

    return measured


def ideal_accelerometer_measurement(
    pose,
    body_velocity,
    body_acceleration,
    *,
    gravity=9.81,
):
    """
    Return ideal body-frame accelerometer specific force.

    Parameters
    ----------
    pose : array-like
        Pose [x, y, z, roll, pitch, yaw].

    body_velocity : array-like
        Generalized body velocity [u, v, w, p, q, r].

    body_acceleration : array-like
        Generalized body velocity derivative
        [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot].

    gravity : float, optional
        Positive gravitational acceleration magnitude in m/s^2.

    Returns
    -------
    np.ndarray
        Body-frame specific force in m/s^2.
    """
    pose = np.asarray(
        pose,
        dtype=float,
    )

    body_velocity = np.asarray(
        body_velocity,
        dtype=float,
    )

    body_acceleration = np.asarray(
        body_acceleration,
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

    if body_acceleration.shape != (6,):
        raise ValueError(
            "body_acceleration must have shape (6,)"
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
        np.isfinite(body_acceleration)
    ):
        raise ValueError(
            "body_acceleration must contain only finite values"
        )

    gravity = float(gravity)

    if (
        not np.isfinite(gravity)
        or gravity <= 0.0
    ):
        raise ValueError(
            "gravity must be finite and strictly positive"
        )

    roll, pitch, yaw = pose[3:]

    R_bw = rotation_body_to_world(
        roll,
        pitch,
        yaw,
    )

    gravity_world = np.array([
        0.0,
        0.0,
        -gravity,
    ])

    velocity_body = body_velocity[:3]
    omega_body = body_velocity[3:]
    velocity_dot_body = body_acceleration[:3]

    inertial_acceleration_body = (
        velocity_dot_body
        + np.cross(
            omega_body,
            velocity_body,
        )
    )

    gravity_body = (
        R_bw.T
        @ gravity_world
    )

    return (
        inertial_acceleration_body
        - gravity_body
    )


def accelerometer_measurement(
    pose,
    body_velocity,
    body_acceleration,
    *,
    gravity=9.81,
    bias=0.0,
    noise_std=0.0,
    rng=None,
):
    """
    Return imperfect body-frame accelerometer measurement.
    """
    ideal = ideal_accelerometer_measurement(
        pose,
        body_velocity,
        body_acceleration,
        gravity=gravity,
    )

    measured = measure_with_bias_noise(
        ideal,
        bias=bias,
        noise_std=noise_std,
        rng=rng,
    )

    if measured.shape != (3,):
        raise ValueError(
            "accelerometer bias and noise_std must be "
            "scalar or broadcast-compatible with shape (3,)"
        )

    return measured
