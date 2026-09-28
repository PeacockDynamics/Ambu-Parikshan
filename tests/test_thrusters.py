import numpy as np
import pytest

from hydrolab3d.dynamics.thrusters import (
    thruster_force_body,
    thruster_force_world,
)


FORWARD = np.array([
    1.0,
    0.0,
    0.0,
])


def test_zero_command_gives_zero_thrust():
    force = thruster_force_body(
        command=0.0,
        max_thrust=100.0,
        direction_body=FORWARD,
    )

    np.testing.assert_allclose(
        force,
        np.zeros(3),
        rtol=0.0,
        atol=1e-12,
    )


def test_unsaturated_command_scales_linearly():
    force = thruster_force_body(
        command=0.25,
        max_thrust=100.0,
        direction_body=FORWARD,
    )

    np.testing.assert_allclose(
        force,
        np.array([25.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "command, expected_x",
    [
        (2.0, 100.0),
        (-2.0, -100.0),
        (-0.5, -50.0),
    ],
)
def test_saturation_and_reverse_thrust(
    command,
    expected_x,
):
    force = thruster_force_body(
        command=command,
        max_thrust=100.0,
        direction_body=FORWARD,
    )

    np.testing.assert_allclose(
        force,
        np.array([expected_x, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_body_direction_is_respected():
    force = thruster_force_body(
        command=0.4,
        max_thrust=50.0,
        direction_body=np.array([
            0.0,
            0.0,
            1.0,
        ]),
    )

    np.testing.assert_allclose(
        force,
        np.array([0.0, 0.0, 20.0]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "command": [0.5],
            "max_thrust": 100.0,
            "direction_body": FORWARD,
        },
        {
            "command": np.nan,
            "max_thrust": 100.0,
            "direction_body": FORWARD,
        },
        {
            "command": 0.5,
            "max_thrust": 0.0,
            "direction_body": FORWARD,
        },
        {
            "command": 0.5,
            "max_thrust": -1.0,
            "direction_body": FORWARD,
        },
        {
            "command": 0.5,
            "max_thrust": np.inf,
            "direction_body": FORWARD,
        },
        {
            "command": 0.5,
            "max_thrust": 100.0,
            "direction_body": [1.0, 0.0],
        },
        {
            "command": 0.5,
            "max_thrust": 100.0,
            "direction_body": [0.0, 0.0, 0.0],
        },
        {
            "command": 0.5,
            "max_thrust": 100.0,
            "direction_body": [1.0, np.nan, 0.0],
        },
        {
            "command": 0.5,
            "max_thrust": 100.0,
            "direction_body": [2.0, 0.0, 0.0],
        },
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        thruster_force_body(**kwargs)


def test_identity_orientation_preserves_thrust():
    force = thruster_force_world(
        command=0.5,
        max_thrust=100.0,
        direction_body=FORWARD,
        phi=0.0,
        theta=0.0,
        psi=0.0,
    )

    np.testing.assert_allclose(
        force,
        np.array([50.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_positive_90_degree_yaw_rotates_forward_to_world_y():
    force = thruster_force_world(
        command=0.5,
        max_thrust=100.0,
        direction_body=FORWARD,
        phi=0.0,
        theta=0.0,
        psi=np.pi / 2.0,
    )

    np.testing.assert_allclose(
        force,
        np.array([0.0, 50.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_arbitrary_rotation_preserves_thrust_magnitude():
    force = thruster_force_world(
        command=0.5,
        max_thrust=100.0,
        direction_body=FORWARD,
        phi=np.deg2rad(20.0),
        theta=np.deg2rad(-35.0),
        psi=np.deg2rad(70.0),
    )

    np.testing.assert_allclose(
        np.linalg.norm(force),
        50.0,
        rtol=1e-12,
        atol=1e-12,
    )
