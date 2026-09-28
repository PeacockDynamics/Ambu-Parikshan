"""Complete lightweight six-degree-of-freedom UUV dynamics."""

import numpy as np

from hydrolab3d.dynamics.hydrodynamics import (
    added_mass_coriolis_matrix,
    added_mass_matrix,
    hydrodynamic_damping_wrench,
    hydrostatic_wrench,
    rigid_body_coriolis_matrix,
    rigid_body_mass_matrix,
)
from hydrolab3d.dynamics.kinematics import (
    euler_rate_matrix,
    pose_rate_6dof,
)
from hydrolab3d.world.currents import (
    current_velocity_body,
    relative_velocity_6dof,
)


def current_generalized_acceleration(
    pose,
    body_velocity,
    current_velocity_world,
):
    """Body derivative of a uniform world-fixed translational current."""
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

    v_c_body = current_velocity_body(
        pose,
        current_velocity_world,
    )

    omega_body = body_velocity[3:]

    nu_c_dot = np.zeros(
        6,
        dtype=float,
    )

    nu_c_dot[:3] = -np.cross(
        omega_body,
        v_c_body,
    )

    return nu_c_dot


def six_dof_state_derivative(
    pose,
    body_velocity,
    *,
    mass,
    inertia,
    added_inertia,
    linear_damping,
    quadratic_damping,
    fluid_density,
    displaced_volume,
    center_of_gravity,
    center_of_buoyancy,
    current_velocity_world,
    actuator_wrench,
    environmental_wrench=None,
    gravity=9.81,
):
    """Compute eta_dot and nu_dot for the complete UUV model."""
    pose = np.asarray(
        pose,
        dtype=float,
    )

    body_velocity = np.asarray(
        body_velocity,
        dtype=float,
    )

    actuator_wrench = np.asarray(
        actuator_wrench,
        dtype=float,
    )

    current_velocity_world = np.asarray(
        current_velocity_world,
        dtype=float,
    )

    if environmental_wrench is None:
        environmental_wrench = np.zeros(
            6,
            dtype=float,
        )
    else:
        environmental_wrench = np.asarray(
            environmental_wrench,
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

    if actuator_wrench.shape != (6,):
        raise ValueError(
            "actuator_wrench must have shape (6,)"
        )

    if environmental_wrench.shape != (6,):
        raise ValueError(
            "environmental_wrench must have shape (6,)"
        )

    if current_velocity_world.shape != (3,):
        raise ValueError(
            "current_velocity_world must have shape (3,)"
        )

    for name, value in [
        ("pose", pose),
        ("body_velocity", body_velocity),
        ("actuator_wrench", actuator_wrench),
        (
            "environmental_wrench",
            environmental_wrench,
        ),
        (
            "current_velocity_world",
            current_velocity_world,
        ),
    ]:
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} must contain only finite values"
            )

    eta_dot = pose_rate_6dof(
        pose,
        body_velocity,
    )

    M_rb = rigid_body_mass_matrix(
        mass,
        inertia,
    )

    M_a = added_mass_matrix(
        added_inertia,
    )

    M = M_rb + M_a

    nu_r = relative_velocity_6dof(
        pose,
        body_velocity,
        current_velocity_world,
    )

    nu_c_dot = current_generalized_acceleration(
        pose,
        body_velocity,
        current_velocity_world,
    )

    C_rb = rigid_body_coriolis_matrix(
        mass,
        inertia,
        body_velocity,
    )

    C_a = added_mass_coriolis_matrix(
        added_inertia,
        nu_r,
    )

    coriolis_rb_wrench = (
        C_rb @ body_velocity
    )

    coriolis_a_wrench = (
        C_a @ nu_r
    )

    damping_wrench = (
        hydrodynamic_damping_wrench(
            nu_r,
            linear_damping,
            quadratic_damping,
        )
    )

    hydrostatic = hydrostatic_wrench(
        pose,
        mass,
        fluid_density,
        displaced_volume,
        center_of_gravity,
        center_of_buoyancy,
        gravity,
    )

    added_mass_current_correction = (
        M_a @ nu_c_dot
    )

    net_wrench = (
        actuator_wrench
        + environmental_wrench
        + hydrostatic
        + damping_wrench
        - coriolis_rb_wrench
        - coriolis_a_wrench
        + added_mass_current_correction
    )

    nu_dot = np.linalg.solve(
        M,
        net_wrench,
    )

    if not np.all(np.isfinite(eta_dot)):
        raise FloatingPointError(
            "eta_dot contains non-finite values"
        )

    if not np.all(np.isfinite(nu_dot)):
        raise FloatingPointError(
            "nu_dot contains non-finite values"
        )

    diagnostics = {
        "M_RB": M_rb,
        "M_A": M_a,
        "M": M,
        "nu_r": nu_r,
        "nu_c_dot": nu_c_dot,
        "C_RB": C_rb,
        "C_A": C_a,
        "coriolis_rb_wrench": (
            coriolis_rb_wrench
        ),
        "coriolis_a_wrench": (
            coriolis_a_wrench
        ),
        "damping_wrench": (
            damping_wrench
        ),
        "hydrostatic_wrench": (
            hydrostatic
        ),
        "actuator_wrench": (
            actuator_wrench.copy()
        ),
        "environmental_wrench": (
            environmental_wrench.copy()
        ),
        "added_mass_current_correction": (
            added_mass_current_correction
        ),
        "net_wrench": net_wrench,
    }

    return (
        eta_dot,
        nu_dot,
        diagnostics,
    )


def integrate_six_dof_euler(
    pose,
    body_velocity,
    dt,
    **dynamics_kwargs,
):
    """Advance the UUV state by one explicit-Euler timestep."""
    pose = np.asarray(
        pose,
        dtype=float,
    )

    body_velocity = np.asarray(
        body_velocity,
        dtype=float,
    )

    dt = float(dt)

    if pose.shape != (6,):
        raise ValueError(
            "pose must have shape (6,)"
        )

    if body_velocity.shape != (6,):
        raise ValueError(
            "body_velocity must have shape (6,)"
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

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and strictly positive"
        )

    eta_dot, nu_dot, diagnostics = (
        six_dof_state_derivative(
            pose,
            body_velocity,
            **dynamics_kwargs,
        )
    )

    next_pose = (
        pose
        + eta_dot * dt
    )

    next_body_velocity = (
        body_velocity
        + nu_dot * dt
    )

    if not np.all(np.isfinite(next_pose)):
        raise FloatingPointError(
            "integrated pose contains non-finite values"
        )

    if not np.all(
        np.isfinite(next_body_velocity)
    ):
        raise FloatingPointError(
            "integrated body velocity contains non-finite values"
        )

    euler_rate_matrix(
        next_pose[3],
        next_pose[4],
    )

    diagnostics = dict(
        diagnostics
    )

    diagnostics["eta_dot"] = eta_dot
    diagnostics["nu_dot"] = nu_dot
    diagnostics["dt"] = dt

    return (
        next_pose,
        next_body_velocity,
        diagnostics,
    )


def simulate_six_dof(
    initial_pose,
    initial_body_velocity,
    duration,
    dt,
    dynamics_kwargs,
):
    """Run a deterministic fixed-step 6-DOF UUV simulation."""
    initial_pose = np.asarray(
        initial_pose,
        dtype=float,
    )

    initial_body_velocity = np.asarray(
        initial_body_velocity,
        dtype=float,
    )

    duration = float(duration)
    dt = float(dt)

    if initial_pose.shape != (6,):
        raise ValueError(
            "initial_pose must have shape (6,)"
        )

    if initial_body_velocity.shape != (6,):
        raise ValueError(
            "initial_body_velocity must have shape (6,)"
        )

    if not np.all(
        np.isfinite(initial_pose)
    ):
        raise ValueError(
            "initial_pose must contain only finite values"
        )

    if not np.all(
        np.isfinite(initial_body_velocity)
    ):
        raise ValueError(
            "initial_body_velocity must contain only finite values"
        )

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(
            "duration must be finite and strictly positive"
        )

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and strictly positive"
        )

    n_steps = int(
        round(
            duration / dt
        )
    )

    if not np.isclose(
        n_steps * dt,
        duration,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "duration must be an integer multiple of dt"
        )

    times = (
        np.arange(
            n_steps + 1,
            dtype=float,
        )
        * dt
    )

    poses = np.empty(
        (n_steps + 1, 6),
        dtype=float,
    )

    velocities = np.empty(
        (n_steps + 1, 6),
        dtype=float,
    )

    poses[0] = initial_pose
    velocities[0] = initial_body_velocity

    pose = initial_pose.copy()
    velocity = initial_body_velocity.copy()

    for k in range(n_steps):
        pose, velocity, _ = (
            integrate_six_dof_euler(
                pose,
                velocity,
                dt,
                **dynamics_kwargs,
            )
        )

        poses[k + 1] = pose
        velocities[k + 1] = velocity

    if not np.all(np.isfinite(poses)):
        raise FloatingPointError(
            "simulation produced non-finite pose values"
        )

    if not np.all(
        np.isfinite(velocities)
    ):
        raise FloatingPointError(
            "simulation produced non-finite velocity values"
        )

    return (
        times,
        poses,
        velocities,
    )
