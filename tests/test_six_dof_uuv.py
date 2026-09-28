import numpy as np
import pytest

from hydrolab3d.dynamics.drag import (
    quadratic_drag_force,
)
from hydrolab3d.dynamics.hydrodynamics import (
    added_mass_coriolis_matrix,
    hydrodynamic_damping_wrench,
    hydrostatic_wrench,
    rigid_body_coriolis_matrix,
    total_mass_matrix,
)
from hydrolab3d.dynamics.kinematics import (
    euler_rate_matrix,
    kinematic_matrix_6dof,
)
from hydrolab3d.dynamics.thrusters import (
    thruster_wrench_body,
)
from hydrolab3d.vehicles.uuv import (
    integrate_six_dof_euler,
    simulate_six_dof,
    six_dof_state_derivative,
)
from hydrolab3d.world.currents import (
    current_velocity_body,
    relative_velocity_6dof,
    relative_water_velocity,
)


MASS = 20.0

INERTIA = np.array(
    [
        1.2,
        1.6,
        2.0,
    ]
)

ADDED = np.array(
    [
        5.0,
        8.0,
        10.0,
        0.20,
        0.30,
        0.40,
    ]
)

LINEAR = np.array(
    [
        2.0,
        3.0,
        4.0,
        0.10,
        0.15,
        0.20,
    ]
)

QUADRATIC = np.array(
    [
        5.0,
        6.0,
        7.0,
        0.30,
        0.40,
        0.50,
    ]
)

RHO = 1000.0

NEUTRAL_VOLUME = (
    MASS / RHO
)


def canonical_args():
    return {
        "mass": MASS,
        "inertia": INERTIA,
        "added_inertia": ADDED,
        "linear_damping": LINEAR,
        "quadratic_damping": QUADRATIC,
        "fluid_density": RHO,
        "displaced_volume": NEUTRAL_VOLUME,
        "center_of_gravity": np.zeros(3),
        "center_of_buoyancy": np.zeros(3),
        "current_velocity_world": np.zeros(3),
        "actuator_wrench": np.zeros(6),
    }


def test_identity_kinematic_matrix_is_identity():
    np.testing.assert_allclose(
        kinematic_matrix_6dof(
            np.zeros(6)
        ),
        np.eye(6),
        atol=1e-15,
        rtol=0.0,
    )


def test_euler_rate_matrix_rejects_pitch_singularity():
    with pytest.raises(ValueError):
        euler_rate_matrix(
            0.0,
            np.pi / 2.0,
        )


def test_canonical_total_mass_matrix():
    expected = np.diag(
        [
            25.0,
            28.0,
            30.0,
            1.4,
            1.9,
            2.4,
        ]
    )

    actual = total_mass_matrix(
        MASS,
        INERTIA,
        ADDED,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-15,
        rtol=0.0,
    )


def test_total_mass_matrix_is_positive_definite():
    M = total_mass_matrix(
        MASS,
        INERTIA,
        ADDED,
    )

    assert np.all(
        np.linalg.eigvalsh(M) > 0.0
    )


def test_rigid_body_coriolis_is_skew_symmetric_and_power_free():
    nu = np.array(
        [
            1.2,
            -0.4,
            0.3,
            0.2,
            -0.1,
            0.15,
        ]
    )

    C = rigid_body_coriolis_matrix(
        MASS,
        INERTIA,
        nu,
    )

    np.testing.assert_allclose(
        C.T,
        -C,
        atol=1e-14,
        rtol=0.0,
    )

    assert abs(
        nu @ C @ nu
    ) < 1e-12


def test_added_mass_coriolis_is_skew_symmetric_and_power_free():
    nu_r = np.array(
        [
            0.8,
            -0.5,
            0.25,
            0.12,
            -0.08,
            0.18,
        ]
    )

    C = added_mass_coriolis_matrix(
        ADDED,
        nu_r,
    )

    np.testing.assert_allclose(
        C.T,
        -C,
        atol=1e-14,
        rtol=0.0,
    )

    assert abs(
        nu_r @ C @ nu_r
    ) < 1e-12


def test_hydrodynamic_damping_is_dissipative():
    nu_r = np.array(
        [
            1.2,
            -0.5,
            0.3,
            0.2,
            -0.1,
            0.15,
        ]
    )

    tau_d = hydrodynamic_damping_wrench(
        nu_r,
        LINEAR,
        QUADRATIC,
    )

    assert (
        nu_r @ tau_d
    ) < 0.0


def test_quadratic_damping_reduces_to_notebook03_drag():
    speed = 1.7
    coefficient = 5.0

    legacy = quadratic_drag_force(
        np.array(
            [
                speed,
                0.0,
                0.0,
            ]
        ),
        coefficient,
    )

    generalized = hydrodynamic_damping_wrench(
        np.array(
            [
                speed,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        np.zeros(6),
        np.array(
            [
                coefficient,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        generalized[:3],
        legacy,
        atol=1e-15,
        rtol=0.0,
    )


def test_neutral_hydrostatics_are_zero_for_coincident_centers():
    wrench = hydrostatic_wrench(
        np.array(
            [
                1.0,
                -2.0,
                3.0,
                0.2,
                -0.15,
                0.4,
            ]
        ),
        MASS,
        RHO,
        NEUTRAL_VOLUME,
        np.zeros(3),
        np.zeros(3),
    )

    np.testing.assert_allclose(
        wrench,
        np.zeros(6),
        atol=1e-12,
        rtol=0.0,
    )


def test_cb_above_cg_generates_restoring_roll_moment():
    pose = np.array(
        [
            0.0,
            0.0,
            0.0,
            np.deg2rad(10.0),
            0.0,
            0.0,
        ]
    )

    wrench = hydrostatic_wrench(
        pose,
        MASS,
        RHO,
        NEUTRAL_VOLUME,
        np.zeros(3),
        np.array(
            [
                0.0,
                0.0,
                0.05,
            ]
        ),
    )

    assert wrench[3] < 0.0


def test_world_current_transforms_to_expected_body_direction():
    pose = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            np.pi / 2.0,
        ]
    )

    current_body = current_velocity_body(
        pose,
        np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        current_body,
        np.array(
            [
                0.0,
                -1.0,
                0.0,
            ]
        ),
        atol=1e-14,
        rtol=0.0,
    )


def test_relative_velocity_reduces_to_notebook06_at_identity():
    vehicle_velocity = np.array(
        [
            1.1,
            -0.3,
            0.4,
        ]
    )

    current = np.array(
        [
            0.5,
            -0.25,
            0.1,
        ]
    )

    legacy = relative_water_velocity(
        vehicle_velocity,
        current,
    )

    generalized = relative_velocity_6dof(
        np.zeros(6),
        np.array(
            [
                1.1,
                -0.3,
                0.4,
                0.2,
                -0.1,
                0.05,
            ]
        ),
        current,
    )

    np.testing.assert_allclose(
        generalized[:3],
        legacy,
        atol=1e-15,
        rtol=0.0,
    )


def test_offset_thruster_produces_expected_pitch_moment():
    wrench = thruster_wrench_body(
        0.5,
        40.0,
        np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
        np.array(
            [
                0.0,
                0.0,
                0.25,
            ]
        ),
    )

    np.testing.assert_allclose(
        wrench,
        np.array(
            [
                20.0,
                0.0,
                0.0,
                0.0,
                5.0,
                0.0,
            ]
        ),
        atol=1e-15,
        rtol=0.0,
    )


def test_complete_model_reduces_to_force_over_mass():
    _, nu_dot, _ = six_dof_state_derivative(
        np.zeros(6),
        np.zeros(6),
        mass=MASS,
        inertia=INERTIA,
        added_inertia=np.zeros(6),
        linear_damping=np.zeros(6),
        quadratic_damping=np.zeros(6),
        fluid_density=RHO,
        displaced_volume=NEUTRAL_VOLUME,
        center_of_gravity=np.zeros(3),
        center_of_buoyancy=np.zeros(3),
        current_velocity_world=np.zeros(3),
        actuator_wrench=np.array(
            [
                12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        nu_dot,
        np.array(
            [
                0.6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        atol=1e-14,
        rtol=0.0,
    )


def test_complete_neutral_equilibrium_is_zero():
    eta_dot, nu_dot, _ = (
        six_dof_state_derivative(
            np.zeros(6),
            np.zeros(6),
            **canonical_args(),
        )
    )

    np.testing.assert_allclose(
        eta_dot,
        np.zeros(6),
        atol=1e-14,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        nu_dot,
        np.zeros(6),
        atol=1e-14,
        rtol=0.0,
    )


def test_explicit_euler_uses_beginning_of_step_velocity():
    args = canonical_args()

    args["added_inertia"] = np.zeros(6)
    args["linear_damping"] = np.zeros(6)
    args["quadratic_damping"] = np.zeros(6)

    args["actuator_wrench"] = np.array(
        [
            12.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    pose_next, nu_next, _ = (
        integrate_six_dof_euler(
            np.zeros(6),
            np.zeros(6),
            0.1,
            **args,
        )
    )

    np.testing.assert_allclose(
        pose_next,
        np.zeros(6),
        atol=1e-15,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        nu_next,
        np.array(
            [
                0.06,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        atol=1e-15,
        rtol=0.0,
    )


def test_forward_thrust_simulation_approaches_analytical_terminal_speed():
    args = canonical_args()

    args["actuator_wrench"] = (
        thruster_wrench_body(
            0.5,
            40.0,
            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
            np.zeros(3),
        )
    )

    times, poses, velocities = (
        simulate_six_dof(
            np.zeros(6),
            np.zeros(6),
            20.0,
            0.01,
            args,
        )
    )

    expected_terminal = (
        -2.0 + np.sqrt(404.0)
    ) / 10.0

    assert len(times) == 2001

    assert abs(
        velocities[-1, 0]
        - expected_terminal
    ) < 0.03

    assert np.all(
        np.isfinite(poses)
    )

    assert np.all(
        np.isfinite(velocities)
    )


def test_simulation_is_deterministic():
    args = canonical_args()

    args["actuator_wrench"] = (
        thruster_wrench_body(
            0.25,
            40.0,
            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
            np.zeros(3),
        )
    )

    result_a = simulate_six_dof(
        np.zeros(6),
        np.zeros(6),
        1.0,
        0.01,
        args,
    )

    result_b = simulate_six_dof(
        np.zeros(6),
        np.zeros(6),
        1.0,
        0.01,
        args,
    )

    for array_a, array_b in zip(
        result_a,
        result_b,
    ):
        np.testing.assert_array_equal(
            array_a,
            array_b,
        )
