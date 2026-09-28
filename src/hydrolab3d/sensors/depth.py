"""Depth sensor models for HydroLab-3D."""

import numpy as np

from hydrolab3d.sensors.sensor import (
    measure_with_bias_noise,
)


def ideal_depth_measurement(
    pose,
    *,
    surface_z=0.0,
):
    """
    Return ideal depth below the modeled water surface.

    Parameters
    ----------
    pose : array-like
        Pose [x, y, z, roll, pitch, yaw].

    surface_z : float, optional
        World-frame z coordinate of the water surface.

    Returns
    -------
    float
        Positive depth below the water surface in meters.

    Raises
    ------
    ValueError
        If pose is malformed, inputs are non-finite, or the
        vehicle lies above the modeled water surface.
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

    surface_z = float(surface_z)

    if not np.isfinite(surface_z):
        raise ValueError(
            "surface_z must be finite"
        )

    depth = (
        surface_z
        - pose[2]
    )

    tolerance = 1e-12

    if depth < -tolerance:
        raise ValueError(
            "vehicle is above the modeled water surface"
        )

    if abs(depth) <= tolerance:
        depth = 0.0

    return float(depth)


def depth_measurement(
    pose,
    *,
    surface_z=0.0,
    bias=0.0,
    noise_std=0.0,
    rng=None,
):
    """
    Return imperfect depth measurement.

    Parameters
    ----------
    pose : array-like
        Pose [x, y, z, roll, pitch, yaw].

    surface_z : float, optional
        World-frame z coordinate of the water surface.

    bias : scalar, optional
        Additive depth bias in meters.

    noise_std : scalar, optional
        Gaussian depth-noise standard deviation in meters.

    rng : numpy.random.Generator or None, optional
        Explicit random generator when noise is nonzero.

    Returns
    -------
    float
        Depth measurement in meters.
    """
    ideal = ideal_depth_measurement(
        pose,
        surface_z=surface_z,
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
            "depth bias and noise_std must be scalar"
        )

    return float(measured)
