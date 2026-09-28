"""Regression tests for HydroLab-3D sensor models."""

import numpy as np
import pytest

import hydrolab3d.sensors as sensors
from hydrolab3d.world.currents import (
    current_velocity_body,
    relative_velocity_6dof,
)


def test_measure_with_bias_noise_seeded_reproducibility():
    truth = np.array([
        1.0,
        -2.0,
        3.0,
    ])

    bias = np.array([
        0.1,
        -0.2,
        0.3,
    ])

    noise_std = np.array([
        0.01,
        0.02,
        0.03,
    ])

    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)

    a = sensors.measure_with_bias_noise(
        truth,
        bias=bias,
        noise_std=noise_std,
        rng=rng_a,
    )

    b = sensors.measure_with_bias_noise(
        truth,
        bias=bias,
        noise_std=noise_std,
        rng=rng_b,
    )

    np.testing.assert_array_equal(
        a,
        b,
    )


def test_ideal_gyroscope_returns_body_angular_velocity():
    body_velocity = np.array([
        1.2,
        -0.4,
        0.3,
        0.08,
        -0.05,
        0.11,
    ])

    measured = sensors.ideal_gyroscope_measurement(
        body_velocity
    )

    np.testing.assert_array_equal(
        measured,
        body_velocity[3:6],
    )


def test_stationary_level_accelerometer_reads_positive_g():
    measured = sensors.ideal_accelerometer_measurement(
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
    )

    np.testing.assert_allclose(
        measured,
        np.array([
            0.0,
            0.0,
            9.81,
        ]),
        rtol=0.0,
        atol=1e-12,
    )


def test_depth_uses_positive_depth_below_surface():
    pose = np.array([
        4.0,
        -2.0,
        -12.5,
        0.2,
        -0.1,
        0.7,
    ])

    assert sensors.ideal_depth_measurement(
        pose
    ) == pytest.approx(
        12.5
    )


def test_depth_rejects_above_surface_pose():
    pose = np.array([
        0.0,
        0.0,
        0.25,
        0.0,
        0.0,
        0.0,
    ])

    with pytest.raises(ValueError):
        sensors.ideal_depth_measurement(
            pose
        )


def test_dvl_matches_water_relative_body_velocity():
    pose = np.array([
        2.0,
        -3.0,
        -10.0,
        0.22,
        -0.17,
        0.61,
    ])

    body_velocity = np.array([
        1.3,
        -0.4,
        0.25,
        0.08,
        -0.06,
        0.10,
    ])

    current_world = np.array([
        0.25,
        -0.12,
        0.05,
    ])

    expected = relative_velocity_6dof(
        pose,
        body_velocity,
        current_world,
    )[:3]

    measured = sensors.ideal_dvl_measurement(
        pose,
        body_velocity,
        current_world,
    )

    np.testing.assert_allclose(
        measured,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_dvl_zero_when_vehicle_moves_with_water():
    pose = np.array([
        1.0,
        2.0,
        -8.0,
        0.15,
        -0.12,
        0.45,
    ])

    current_world = np.array([
        0.3,
        -0.1,
        0.06,
    ])

    current_body = current_velocity_body(
        pose,
        current_world,
    )

    body_velocity = np.array([
        current_body[0],
        current_body[1],
        current_body[2],
        0.2,
        -0.1,
        0.05,
    ])

    measured = sensors.ideal_dvl_measurement(
        pose,
        body_velocity,
        current_world,
    )

    np.testing.assert_allclose(
        measured,
        np.zeros(3),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "angle",
    [
        np.pi,
        -np.pi,
        3.0 * np.pi,
        -3.0 * np.pi,
    ],
)
def test_heading_boundary_maps_to_positive_pi(angle):
    assert sensors.wrap_angle_pi(
        angle
    ) == pytest.approx(
        np.pi,
        abs=1e-15,
    )


def test_heading_bias_rewraps_across_pi():
    pose = np.array([
        0.0,
        0.0,
        -2.0,
        0.0,
        0.0,
        np.pi - 0.05,
    ])

    measured = sensors.heading_measurement(
        pose,
        bias=0.20,
        noise_std=0.0,
    )

    expected = sensors.wrap_angle_pi(
        np.pi - 0.05 + 0.20
    )

    assert measured == pytest.approx(
        expected,
        abs=1e-15,
    )

    assert -np.pi < measured <= np.pi


def test_sensor_noise_requires_explicit_rng():
    with pytest.raises(ValueError):
        sensors.gyroscope_measurement(
            np.zeros(6),
            noise_std=0.01,
        )


def test_sensor_public_api_excludes_sonar():
    expected = [
        "accelerometer_measurement",
        "depth_measurement",
        "dvl_measurement",
        "gyroscope_measurement",
        "heading_measurement",
        "ideal_accelerometer_measurement",
        "ideal_depth_measurement",
        "ideal_dvl_measurement",
        "ideal_gyroscope_measurement",
        "ideal_heading_measurement",
        "measure_with_bias_noise",
        "wrap_angle_pi",
    ]

    assert sensors.__all__ == expected

    assert not any(
        "sonar" in name.lower()
        for name in sensors.__all__
    )
