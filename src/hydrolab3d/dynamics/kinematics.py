"""Basic translational kinematics for HydroLab-3D."""

import numpy as np


def integrate_position(position, velocity, dt):
    """
    Advance a 3D position by one constant-velocity timestep.

    Parameters
    ----------
    position : np.ndarray
        Current world-frame position [x, y, z] in meters.
    velocity : np.ndarray
        Current world-frame velocity [vx, vy, vz] in meters/second.
    dt : float
        Simulation timestep in seconds.

    Returns
    -------
    np.ndarray
        Updated world-frame position after one timestep.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if position.shape != (3,):
        raise ValueError("position must have shape (3,)")

    if velocity.shape != (3,):
        raise ValueError("velocity must have shape (3,)")

    if not np.all(np.isfinite(position)):
        raise ValueError("position must contain only finite values")

    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must contain only finite values")

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar")

    return position + velocity * dt
# =====================================================================
# 6-DOF UUV kinematics
# =====================================================================


def euler_rate_matrix(
    roll,
    pitch,
    singularity_tolerance=1e-9,
):
    """Map body angular velocity to Z-Y-X Euler-angle rates."""
    import numpy as np

    roll = float(roll)
    pitch = float(pitch)

    if not np.isfinite(roll) or not np.isfinite(pitch):
        raise ValueError(
            "roll and pitch must be finite"
        )

    singularity_tolerance = float(
        singularity_tolerance
    )

    if (
        not np.isfinite(singularity_tolerance)
        or singularity_tolerance <= 0.0
    ):
        raise ValueError(
            "singularity_tolerance must be finite and positive"
        )

    c_phi = np.cos(roll)
    s_phi = np.sin(roll)
    c_theta = np.cos(pitch)

    if abs(c_theta) <= singularity_tolerance:
        raise ValueError(
            "Euler-rate transformation is singular "
            "or too close to singular at this pitch angle"
        )

    t_theta = np.tan(pitch)

    return np.array(
        [
            [
                1.0,
                s_phi * t_theta,
                c_phi * t_theta,
            ],
            [
                0.0,
                c_phi,
                -s_phi,
            ],
            [
                0.0,
                s_phi / c_theta,
                c_phi / c_theta,
            ],
        ],
        dtype=float,
    )


def kinematic_matrix_6dof(pose):
    """Construct J(eta) for eta_dot = J(eta) @ nu."""
    import numpy as np

    from hydrolab3d.utils.transforms import (
        rotation_body_to_world,
    )

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

    roll, pitch, yaw = pose[3:]

    R_bw = rotation_body_to_world(
        roll,
        pitch,
        yaw,
    )

    T_euler = euler_rate_matrix(
        roll,
        pitch,
    )

    J = np.zeros(
        (6, 6),
        dtype=float,
    )

    J[:3, :3] = R_bw
    J[3:, 3:] = T_euler

    return J


def pose_rate_6dof(
    pose,
    body_velocity,
):
    """Compute the six-component pose derivative."""
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
        kinematic_matrix_6dof(pose)
        @ body_velocity
    )
