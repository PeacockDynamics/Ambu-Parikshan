"""Reduced translational thruster models for HydroLab-3D."""

import numpy as np

from hydrolab3d.utils.transforms import rotation_body_to_world


def thruster_force_body(
    command,
    max_thrust,
    direction_body,
):
    """
    Convert a normalized thruster command into body-frame force.

    Parameters
    ----------
    command : float
        Requested normalized command. Values outside [-1, 1] are
        saturated.
    max_thrust : float
        Positive maximum thrust magnitude in newtons.
    direction_body : array-like, shape (3,)
        Body-frame unit vector defining positive thrust direction.

    Returns
    -------
    np.ndarray
        Body-frame thrust force [Fx, Fy, Fz] in newtons.
    """

    if (
        isinstance(command, (bool, np.bool_))
        or not np.isscalar(command)
    ):
        raise ValueError(
            "command must be a finite scalar."
        )

    command = float(command)

    if not np.isfinite(command):
        raise ValueError(
            "command must be finite."
        )

    if (
        isinstance(max_thrust, (bool, np.bool_))
        or not np.isscalar(max_thrust)
    ):
        raise ValueError(
            "max_thrust must be a finite positive scalar."
        )

    max_thrust = float(max_thrust)

    if (
        not np.isfinite(max_thrust)
        or max_thrust <= 0.0
    ):
        raise ValueError(
            "max_thrust must be finite and strictly positive."
        )

    direction_body = np.asarray(
        direction_body,
        dtype=float,
    )

    if direction_body.shape != (3,):
        raise ValueError(
            "direction_body must have shape (3,)."
        )

    if not np.all(
        np.isfinite(direction_body)
    ):
        raise ValueError(
            "direction_body must contain only finite values."
        )

    direction_norm = np.linalg.norm(
        direction_body
    )

    if direction_norm == 0.0:
        raise ValueError(
            "direction_body must not be the zero vector."
        )

    if not np.isclose(
        direction_norm,
        1.0,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "direction_body must be a unit vector; "
            f"received norm {direction_norm:.12g}."
        )

    saturated_command = np.clip(
        command,
        -1.0,
        1.0,
    )

    thrust_magnitude = (
        saturated_command
        * max_thrust
    )

    return (
        thrust_magnitude
        * direction_body
    )


def thruster_force_world(
    command,
    max_thrust,
    direction_body,
    phi,
    theta,
    psi,
):
    """
    Convert a normalized thruster command into world-frame force.

    The thruster direction is defined in the vehicle body frame and
    transformed into the HydroLab-3D world frame using the validated
    body-to-world rotation convention.

    Parameters
    ----------
    command : float
        Requested normalized command.
    max_thrust : float
        Positive maximum thrust magnitude in newtons.
    direction_body : array-like, shape (3,)
        Body-frame unit vector defining positive thrust direction.
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
    psi : float
        Yaw angle in radians.

    Returns
    -------
    np.ndarray
        World-frame thrust force [Fx, Fy, Fz] in newtons.
    """

    force_body = thruster_force_body(
        command=command,
        max_thrust=max_thrust,
        direction_body=direction_body,
    )

    rotation = rotation_body_to_world(
        phi=phi,
        theta=theta,
        psi=psi,
    )

    return rotation @ force_body
# =====================================================================
# 6-DOF generalized actuator wrench
# =====================================================================


def thruster_wrench_body(
    command,
    max_thrust,
    direction_body,
    application_point_body,
):
    """Convert body thrust into [force, r x F] generalized wrench."""
    import numpy as np

    application_point_body = np.asarray(
        application_point_body,
        dtype=float,
    )

    if application_point_body.shape != (3,):
        raise ValueError(
            "application_point_body must have shape (3,)"
        )

    if not np.all(
        np.isfinite(application_point_body)
    ):
        raise ValueError(
            "application_point_body must contain only finite values"
        )

    force_body = thruster_force_body(
        command,
        max_thrust,
        direction_body,
    )

    moment_body = np.cross(
        application_point_body,
        force_body,
    )

    return np.concatenate(
        [
            force_body,
            moment_body,
        ]
    )
