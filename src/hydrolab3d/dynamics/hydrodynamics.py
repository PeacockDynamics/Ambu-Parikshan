"""Six-degree-of-freedom marine hydrodynamics utilities."""

import numpy as np

from hydrolab3d.dynamics.buoyancy import (
    buoyancy_force,
    gravity_force,
)
from hydrolab3d.utils.transforms import (
    rotation_body_to_world,
)


def skew(vector):
    """Return S(a) such that S(a) @ b == cross(a, b)."""
    vector = np.asarray(
        vector,
        dtype=float,
    )

    if vector.shape != (3,):
        raise ValueError(
            "vector must have shape (3,)"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "vector must contain only finite values"
        )

    x, y, z = vector

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rigid_body_mass_matrix(
    mass,
    inertia,
):
    """Construct CG-centered diagonal rigid-body inertia."""
    mass = float(mass)

    inertia = np.asarray(
        inertia,
        dtype=float,
    )

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            "mass must be finite and strictly positive"
        )

    if inertia.shape != (3,):
        raise ValueError(
            "inertia must have shape (3,)"
        )

    if not np.all(np.isfinite(inertia)):
        raise ValueError(
            "inertia must contain only finite values"
        )

    if np.any(inertia <= 0.0):
        raise ValueError(
            "principal inertias must be strictly positive"
        )

    M_rb = np.zeros(
        (6, 6),
        dtype=float,
    )

    M_rb[:3, :3] = (
        mass * np.eye(3)
    )

    M_rb[3:, 3:] = np.diag(
        inertia
    )

    return M_rb


def added_mass_matrix(
    added_inertia,
):
    """Construct diagonal positive-effective added inertia."""
    added_inertia = np.asarray(
        added_inertia,
        dtype=float,
    )

    if added_inertia.shape != (6,):
        raise ValueError(
            "added_inertia must have shape (6,)"
        )

    if not np.all(
        np.isfinite(added_inertia)
    ):
        raise ValueError(
            "added_inertia must contain only finite values"
        )

    if np.any(added_inertia < 0.0):
        raise ValueError(
            "added-inertia magnitudes must be nonnegative"
        )

    return np.diag(
        added_inertia
    )


def total_mass_matrix(
    mass,
    inertia,
    added_inertia,
):
    """Return M_RB + M_A."""
    return (
        rigid_body_mass_matrix(
            mass,
            inertia,
        )
        +
        added_mass_matrix(
            added_inertia,
        )
    )


def coriolis_from_block_diagonal_mass(
    mass_matrix,
    generalized_velocity,
):
    """Construct the selected skew-symmetric 6-DOF Coriolis matrix."""
    mass_matrix = np.asarray(
        mass_matrix,
        dtype=float,
    )

    generalized_velocity = np.asarray(
        generalized_velocity,
        dtype=float,
    )

    if mass_matrix.shape != (6, 6):
        raise ValueError(
            "mass_matrix must have shape (6, 6)"
        )

    if generalized_velocity.shape != (6,):
        raise ValueError(
            "generalized_velocity must have shape (6,)"
        )

    if not np.all(np.isfinite(mass_matrix)):
        raise ValueError(
            "mass_matrix must contain only finite values"
        )

    if not np.all(
        np.isfinite(generalized_velocity)
    ):
        raise ValueError(
            "generalized_velocity must contain only finite values"
        )

    if not np.allclose(
        mass_matrix[:3, 3:],
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):
        raise ValueError(
            "mass_matrix must use the Notebook 07 "
            "block-diagonal representation"
        )

    if not np.allclose(
        mass_matrix[3:, :3],
        0.0,
        atol=1e-15,
        rtol=0.0,
    ):
        raise ValueError(
            "mass_matrix must use the Notebook 07 "
            "block-diagonal representation"
        )

    momentum = (
        mass_matrix
        @ generalized_velocity
    )

    linear_momentum = momentum[:3]
    angular_momentum = momentum[3:]

    C = np.zeros(
        (6, 6),
        dtype=float,
    )

    C[:3, 3:] = -skew(
        linear_momentum
    )

    C[3:, :3] = -skew(
        linear_momentum
    )

    C[3:, 3:] = -skew(
        angular_momentum
    )

    return C


def rigid_body_coriolis_matrix(
    mass,
    inertia,
    body_velocity,
):
    """Construct C_RB(nu)."""
    return coriolis_from_block_diagonal_mass(
        rigid_body_mass_matrix(
            mass,
            inertia,
        ),
        body_velocity,
    )


def added_mass_coriolis_matrix(
    added_inertia,
    relative_velocity,
):
    """Construct C_A(nu_r)."""
    return coriolis_from_block_diagonal_mass(
        added_mass_matrix(
            added_inertia,
        ),
        relative_velocity,
    )


def hydrodynamic_damping_wrench(
    relative_velocity,
    linear_damping,
    quadratic_damping,
):
    """Return diagonal linear + quadratic hydrodynamic damping."""
    relative_velocity = np.asarray(
        relative_velocity,
        dtype=float,
    )

    linear_damping = np.asarray(
        linear_damping,
        dtype=float,
    )

    quadratic_damping = np.asarray(
        quadratic_damping,
        dtype=float,
    )

    if relative_velocity.shape != (6,):
        raise ValueError(
            "relative_velocity must have shape (6,)"
        )

    if linear_damping.shape != (6,):
        raise ValueError(
            "linear_damping must have shape (6,)"
        )

    if quadratic_damping.shape != (6,):
        raise ValueError(
            "quadratic_damping must have shape (6,)"
        )

    if not np.all(
        np.isfinite(relative_velocity)
    ):
        raise ValueError(
            "relative_velocity must contain only finite values"
        )

    if not np.all(
        np.isfinite(linear_damping)
    ):
        raise ValueError(
            "linear_damping must contain only finite values"
        )

    if not np.all(
        np.isfinite(quadratic_damping)
    ):
        raise ValueError(
            "quadratic_damping must contain only finite values"
        )

    if np.any(linear_damping < 0.0):
        raise ValueError(
            "linear damping coefficients must be nonnegative"
        )

    if np.any(quadratic_damping < 0.0):
        raise ValueError(
            "quadratic damping coefficients must be nonnegative"
        )

    return (
        -linear_damping
        * relative_velocity
        -
        quadratic_damping
        * np.abs(relative_velocity)
        * relative_velocity
    )


def hydrostatic_wrench(
    pose,
    mass,
    fluid_density,
    displaced_volume,
    center_of_gravity,
    center_of_buoyancy,
    gravity=9.81,
):
    """Return physical gravity + buoyancy wrench in body coordinates."""
    pose = np.asarray(
        pose,
        dtype=float,
    )

    center_of_gravity = np.asarray(
        center_of_gravity,
        dtype=float,
    )

    center_of_buoyancy = np.asarray(
        center_of_buoyancy,
        dtype=float,
    )

    if pose.shape != (6,):
        raise ValueError(
            "pose must have shape (6,)"
        )

    if center_of_gravity.shape != (3,):
        raise ValueError(
            "center_of_gravity must have shape (3,)"
        )

    if center_of_buoyancy.shape != (3,):
        raise ValueError(
            "center_of_buoyancy must have shape (3,)"
        )

    if not np.all(np.isfinite(pose)):
        raise ValueError(
            "pose must contain only finite values"
        )

    if not np.all(
        np.isfinite(center_of_gravity)
    ):
        raise ValueError(
            "center_of_gravity must contain only finite values"
        )

    if not np.all(
        np.isfinite(center_of_buoyancy)
    ):
        raise ValueError(
            "center_of_buoyancy must contain only finite values"
        )

    mass = float(mass)
    fluid_density = float(
        fluid_density
    )
    displaced_volume = float(
        displaced_volume
    )
    gravity = float(gravity)

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            "mass must be finite and strictly positive"
        )

    if (
        not np.isfinite(fluid_density)
        or fluid_density <= 0.0
    ):
        raise ValueError(
            "fluid_density must be finite and strictly positive"
        )

    if (
        not np.isfinite(displaced_volume)
        or displaced_volume <= 0.0
    ):
        raise ValueError(
            "displaced_volume must be finite and strictly positive"
        )

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

    F_g_world = gravity_force(
        mass,
        gravity,
    )

    F_b_world = buoyancy_force(
        fluid_density,
        displaced_volume,
        gravity,
    )

    F_g_body = (
        R_bw.T @ F_g_world
    )

    F_b_body = (
        R_bw.T @ F_b_world
    )

    force_body = (
        F_g_body
        + F_b_body
    )

    moment_body = (
        np.cross(
            center_of_gravity,
            F_g_body,
        )
        +
        np.cross(
            center_of_buoyancy,
            F_b_body,
        )
    )

    return np.concatenate(
        [
            force_body,
            moment_body,
        ]
    )
