"""Ocean-current utilities for HydroLab-3D."""

import numpy as np


def uniform_current_velocity(current_velocity):
    """
    Return a validated uniform ocean-current velocity.

    The current velocity is expressed in the HydroLab-3D world frame.

    Parameters
    ----------
    current_velocity : array-like
        Uniform world-frame current velocity [vc_x, vc_y, vc_z] in m/s.

    Returns
    -------
    np.ndarray
        Independent finite 3-vector containing the current velocity.

    Raises
    ------
    ValueError
        If current_velocity does not have shape (3,) or contains
        non-finite values.
    """
    current_velocity = np.asarray(
        current_velocity,
        dtype=float,
    )

    if current_velocity.shape != (3,):
        raise ValueError(
            "current_velocity must have shape (3,)"
        )

    if not np.all(np.isfinite(current_velocity)):
        raise ValueError(
            "current_velocity must contain only finite values"
        )

    return current_velocity.copy()


def relative_water_velocity(
    vehicle_velocity,
    current_velocity,
):
    """
    Compute vehicle velocity relative to the surrounding water.

    Both velocity vectors are expressed in the HydroLab-3D world frame.

    Parameters
    ----------
    vehicle_velocity : array-like
        Vehicle world-frame velocity [vx, vy, vz] in m/s.

    current_velocity : array-like
        Ocean-current world-frame velocity [vc_x, vc_y, vc_z] in m/s.

    Returns
    -------
    np.ndarray
        Water-relative velocity:

            v_rel = v_vehicle - v_current

    Raises
    ------
    ValueError
        If either input does not have shape (3,) or contains
        non-finite values.
    """
    vehicle_velocity = np.asarray(
        vehicle_velocity,
        dtype=float,
    )

    current_velocity = np.asarray(
        current_velocity,
        dtype=float,
    )

    if vehicle_velocity.shape != (3,):
        raise ValueError(
            "vehicle_velocity must have shape (3,)"
        )

    if current_velocity.shape != (3,):
        raise ValueError(
            "current_velocity must have shape (3,)"
        )

    if not np.all(np.isfinite(vehicle_velocity)):
        raise ValueError(
            "vehicle_velocity must contain only finite values"
        )

    if not np.all(np.isfinite(current_velocity)):
        raise ValueError(
            "current_velocity must contain only finite values"
        )

    return vehicle_velocity - current_velocity
# =====================================================================
# 6-DOF body-frame current utilities
# =====================================================================


def current_velocity_body(
    pose,
    current_velocity_world,
):
    """Transform world-frame translational current into body coordinates."""
    import numpy as np

    from hydrolab3d.utils.transforms import (
        rotation_body_to_world,
    )

    pose = np.asarray(
        pose,
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

    if current_velocity_world.shape != (3,):
        raise ValueError(
            "current_velocity_world must have shape (3,)"
        )

    if not np.all(np.isfinite(pose)):
        raise ValueError(
            "pose must contain only finite values"
        )

    if not np.all(
        np.isfinite(current_velocity_world)
    ):
        raise ValueError(
            "current_velocity_world must contain only finite values"
        )

    roll, pitch, yaw = pose[3:]

    R_bw = rotation_body_to_world(
        roll,
        pitch,
        yaw,
    )

    return (
        R_bw.T
        @ current_velocity_world
    )


def generalized_current_velocity(
    pose,
    current_velocity_world,
):
    """Return [v_c,B, 0, 0, 0]."""
    import numpy as np

    nu_c = np.zeros(
        6,
        dtype=float,
    )

    nu_c[:3] = current_velocity_body(
        pose,
        current_velocity_world,
    )

    return nu_c


def relative_velocity_6dof(
    pose,
    body_velocity,
    current_velocity_world,
):
    """Return generalized water-relative body velocity."""
    import numpy as np

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

    return (
        body_velocity
        -
        generalized_current_velocity(
            pose,
            current_velocity_world,
        )
    )
