
import numpy as np
import pytest

import hydrolab3d.control as control

from hydrolab3d.control import (
    PIDController,
    waypoint_guidance,
)


# -----------------------------------------------------------------
# Public API
# -----------------------------------------------------------------

def test_control_public_api():

    assert set(
        control.__all__
    ) == {
        "PIDController",
        "waypoint_guidance",
    }


# -----------------------------------------------------------------
# PID - proportional response
# -----------------------------------------------------------------

def test_pid_proportional_response():

    pid = PIDController(
        kp=2.0,
        ki=0.0,
        kd=0.0,
        output_limits=(-10.0, 10.0),
    )

    output, diagnostics = pid.update(
        error=1.5,
        dt=0.1,
    )

    assert np.isclose(
        output,
        3.0,
    )

    assert np.isclose(
        diagnostics["proportional"],
        3.0,
    )


# -----------------------------------------------------------------
# PID - integral accumulation
# -----------------------------------------------------------------

def test_pid_integral_accumulation():

    pid = PIDController(
        kp=0.0,
        ki=2.0,
        kd=0.0,
        output_limits=(-100.0, 100.0),
    )

    for _ in range(4):

        output, _ = pid.update(
            error=1.0,
            dt=0.5,
        )

    assert np.isclose(
        pid.integral,
        2.0,
    )

    assert np.isclose(
        output,
        4.0,
    )


# -----------------------------------------------------------------
# PID - derivative response
# -----------------------------------------------------------------

def test_pid_derivative_response():

    pid = PIDController(
        kp=0.0,
        ki=0.0,
        kd=3.0,
        output_limits=(-100.0, 100.0),
    )

    first_output, first_diagnostics = pid.update(
        error=1.0,
        dt=0.5,
    )

    second_output, second_diagnostics = pid.update(
        error=2.0,
        dt=0.5,
    )

    assert np.isclose(
        first_output,
        0.0,
    )

    assert np.isclose(
        first_diagnostics["error_derivative"],
        0.0,
    )

    assert np.isclose(
        second_output,
        6.0,
    )

    assert np.isclose(
        second_diagnostics["error_derivative"],
        2.0,
    )


# -----------------------------------------------------------------
# PID - saturation
# -----------------------------------------------------------------

def test_pid_output_saturation():

    pid = PIDController(
        kp=10.0,
        ki=0.0,
        kd=0.0,
        output_limits=(-1.0, 1.0),
    )

    output, diagnostics = pid.update(
        error=5.0,
        dt=0.1,
    )

    assert np.isclose(
        diagnostics["unsaturated_output"],
        50.0,
    )

    assert np.isclose(
        output,
        1.0,
    )

    assert diagnostics["saturated"]


# -----------------------------------------------------------------
# PID - integral clamping
# -----------------------------------------------------------------

def test_pid_integral_clamping():

    pid = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        output_limits=(-10.0, 10.0),
        integral_limits=(-0.5, 0.5),
    )

    for _ in range(20):

        output, _ = pid.update(
            error=1.0,
            dt=0.1,
        )

    assert np.isclose(
        pid.integral,
        0.5,
    )

    assert np.isclose(
        output,
        0.5,
    )


# -----------------------------------------------------------------
# PID - reset
# -----------------------------------------------------------------

def test_pid_reset():

    pid = PIDController(
        kp=1.0,
        ki=1.0,
        kd=1.0,
    )

    pid.update(
        error=2.0,
        dt=0.1,
    )

    pid.reset()

    assert np.isclose(
        pid.integral,
        0.0,
    )

    assert (
        pid.previous_error
        is None
    )


# -----------------------------------------------------------------
# PID - invalid timestep
# -----------------------------------------------------------------

@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.1,
        np.nan,
        np.inf,
    ],
)
def test_pid_rejects_invalid_timestep(dt):

    pid = PIDController(
        kp=1.0,
        ki=0.0,
        kd=0.0,
    )

    with pytest.raises(
        ValueError
    ):

        pid.update(
            error=1.0,
            dt=dt,
        )


# -----------------------------------------------------------------
# PID - invalid error
# -----------------------------------------------------------------

@pytest.mark.parametrize(
    "error",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_pid_rejects_nonfinite_error(error):

    pid = PIDController(
        kp=1.0,
        ki=0.0,
        kd=0.0,
    )

    with pytest.raises(
        ValueError
    ):

        pid.update(
            error=error,
            dt=0.1,
        )


# -----------------------------------------------------------------
# PID - invalid limits
# -----------------------------------------------------------------

def test_pid_rejects_reversed_output_limits():

    with pytest.raises(
        ValueError
    ):

        PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            output_limits=(1.0, -1.0),
        )


def test_pid_rejects_reversed_integral_limits():

    with pytest.raises(
        ValueError
    ):

        PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            integral_limits=(1.0, -1.0),
        )


# -----------------------------------------------------------------
# Waypoint guidance - east
# -----------------------------------------------------------------

def test_waypoint_guidance_east():

    guidance = waypoint_guidance(
        position_world=[
            0.0,
            0.0,
            -5.0,
        ],
        waypoint_world=[
            10.0,
            0.0,
            -5.0,
        ],
    )

    assert np.isclose(
        guidance["desired_heading"],
        0.0,
    )

    assert np.isclose(
        guidance["desired_depth"],
        5.0,
    )

    assert np.isclose(
        guidance["horizontal_distance"],
        10.0,
    )

    assert np.isclose(
        guidance["distance"],
        10.0,
    )


# -----------------------------------------------------------------
# Waypoint guidance - north
# -----------------------------------------------------------------

def test_waypoint_guidance_north():

    guidance = waypoint_guidance(
        position_world=[
            0.0,
            0.0,
            -5.0,
        ],
        waypoint_world=[
            0.0,
            10.0,
            -5.0,
        ],
    )

    assert np.isclose(
        guidance["desired_heading"],
        np.pi / 2.0,
    )


# -----------------------------------------------------------------
# Waypoint guidance - southwest
# -----------------------------------------------------------------

def test_waypoint_guidance_southwest():

    guidance = waypoint_guidance(
        position_world=[
            0.0,
            0.0,
            -5.0,
        ],
        waypoint_world=[
            -10.0,
            -10.0,
            -5.0,
        ],
    )

    assert np.isclose(
        guidance["desired_heading"],
        -3.0 * np.pi / 4.0,
    )


# -----------------------------------------------------------------
# Waypoint guidance - general 3D geometry
# -----------------------------------------------------------------

def test_waypoint_guidance_general_geometry():

    guidance = waypoint_guidance(
        position_world=[
            1.0,
            2.0,
            -3.0,
        ],
        waypoint_world=[
            4.0,
            6.0,
            -15.0,
        ],
    )

    assert np.allclose(
        guidance["displacement_world"],
        [
            3.0,
            4.0,
            -12.0,
        ],
    )

    assert np.isclose(
        guidance["horizontal_distance"],
        5.0,
    )

    assert np.isclose(
        guidance["distance"],
        13.0,
    )

    assert np.isclose(
        guidance["desired_depth"],
        15.0,
    )

    assert np.isclose(
        guidance["desired_heading"],
        np.arctan2(
            4.0,
            3.0,
        ),
    )


# -----------------------------------------------------------------
# Waypoint guidance - vertical-only target
# -----------------------------------------------------------------

def test_waypoint_guidance_vertical_only():

    guidance = waypoint_guidance(
        position_world=[
            4.0,
            -2.0,
            -5.0,
        ],
        waypoint_world=[
            4.0,
            -2.0,
            -8.0,
        ],
    )

    assert np.isclose(
        guidance["horizontal_distance"],
        0.0,
    )

    assert np.isclose(
        guidance["desired_heading"],
        0.0,
    )

    assert np.isclose(
        guidance["desired_depth"],
        8.0,
    )

    assert np.isclose(
        guidance["distance"],
        3.0,
    )


# -----------------------------------------------------------------
# Waypoint guidance - invalid shapes
# -----------------------------------------------------------------

def test_waypoint_guidance_rejects_bad_position_shape():

    with pytest.raises(
        ValueError
    ):

        waypoint_guidance(
            position_world=[
                0.0,
                0.0,
            ],
            waypoint_world=[
                1.0,
                2.0,
                3.0,
            ],
        )


def test_waypoint_guidance_rejects_bad_waypoint_shape():

    with pytest.raises(
        ValueError
    ):

        waypoint_guidance(
            position_world=[
                0.0,
                0.0,
                -5.0,
            ],
            waypoint_world=[
                1.0,
                2.0,
            ],
        )


# -----------------------------------------------------------------
# Waypoint guidance - non-finite values
# -----------------------------------------------------------------

def test_waypoint_guidance_rejects_nonfinite_position():

    with pytest.raises(
        ValueError
    ):

        waypoint_guidance(
            position_world=[
                0.0,
                np.nan,
                -5.0,
            ],
            waypoint_world=[
                1.0,
                2.0,
                -5.0,
            ],
        )


def test_waypoint_guidance_rejects_nonfinite_waypoint():

    with pytest.raises(
        ValueError
    ):

        waypoint_guidance(
            position_world=[
                0.0,
                0.0,
                -5.0,
            ],
            waypoint_world=[
                1.0,
                np.inf,
                -5.0,
            ],
        )
