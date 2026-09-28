"""Heading sensor models for HydroLab-3D."""

import numpy as np

from hydrolab3d.sensors.sensor import (
    measure_with_bias_noise,
)


def wrap_angle_pi(
    angle,
):
    """
    Wrap an angle in radians to the interval (-pi, pi].
    """
    angle = float(angle)

    if not np.isfinite(angle):
        raise ValueError(
            "angle must be finite"
        )

    wrapped = (
        (angle + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    if np.isclose(
        wrapped,
        -np.pi,
        rtol=0.0,
        atol=1e-15,
    ):
        wrapped = np.pi

    return float(wrapped)


def ideal_heading_measurement(
    pose,
):
    """
    Return ideal wrapped yaw heading.

    Parameters
    ----------
    pose : array-like
        Pose [x, y, z, roll, pitch, yaw].

    Returns
    -------
    float
        Heading in radians, wrapped to (-pi, pi].
    """
    pose = np.asarray(
        pose,
        dtype=float,
    )

    if pose.shape != (6,):
        raise ValueError(
            "pose must have shape (6,)"
        )

    if not np.all(np.isfinite(pose)):
        raise ValueError(
            "pose must contain only finite values"
        )

    return wrap_angle_pi(
        pose[5]
    )


def heading_measurement(
    pose,
    *,
    bias=0.0,
    noise_std=0.0,
    rng=None,
):
    """
    Return imperfect wrapped heading measurement.

    Parameters
    ----------
    pose : array-like
        Pose [x, y, z, roll, pitch, yaw].

    bias : scalar, optional
        Additive heading bias in radians.

    noise_std : scalar, optional
        Gaussian heading-noise standard deviation in radians.

    rng : numpy.random.Generator or None, optional
        Explicit random generator when noise is nonzero.

    Returns
    -------
    float
        Heading measurement in radians, wrapped to (-pi, pi].
    """
    ideal = ideal_heading_measurement(
        pose
    )

    measured = measure_with_bias_noise(
        ideal,
        bias=bias,
        noise_std=noise_std,
        rng=rng,
    )

    measured = np.asarray(
        measured,
        dtype=float,
    )

    if measured.shape != ():
        raise ValueError(
            "heading bias and noise_std must be scalar"
        )

    return wrap_angle_pi(
        float(measured)
    )
