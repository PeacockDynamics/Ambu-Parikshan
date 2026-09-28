import numpy as np
import pytest

from hydrolab3d.dynamics.rigid_body import (
    acceleration_from_force,
    integrate_velocity,
)


def test_zero_force_produces_zero_acceleration():
    acceleration = acceleration_from_force(
        np.zeros(3),
        20.0,
    )

    assert np.array_equal(
        acceleration,
        np.zeros(3),
    )


def test_force_to_acceleration_known_case():
    acceleration = acceleration_from_force(
        np.array([30.0, -15.0, 7.5]),
        15.0,
    )

    expected = np.array([2.0, -1.0, 0.5])

    assert np.allclose(
        acceleration,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_acceleration_scales_inversely_with_mass():
    force = np.array([12.0, -8.0, 4.0])

    a1 = acceleration_from_force(force, 10.0)
    a2 = acceleration_from_force(force, 20.0)

    assert np.allclose(
        a2,
        0.5 * a1,
        atol=1e-12,
        rtol=0.0,
    )


def test_zero_acceleration_preserves_velocity():
    velocity = np.array([1.5, -0.75, 0.25])

    updated = integrate_velocity(
        velocity,
        np.zeros(3),
        0.1,
    )

    assert np.array_equal(
        updated,
        velocity,
    )


def test_velocity_integration_known_case():
    updated = integrate_velocity(
        np.array([1.0, -2.0, 0.5]),
        np.array([2.0, 1.0, -0.5]),
        0.2,
    )

    expected = np.array([1.4, -1.8, 0.4])

    assert np.allclose(
        updated,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "mass",
    [
        0.0,
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_invalid_mass_rejected(mass):
    with pytest.raises(ValueError):
        acceleration_from_force(
            np.ones(3),
            mass,
        )


def test_invalid_force_shape_rejected():
    with pytest.raises(ValueError):
        acceleration_from_force(
            np.ones(2),
            10.0,
        )


def test_invalid_velocity_shape_rejected():
    with pytest.raises(ValueError):
        integrate_velocity(
            np.ones(2),
            np.ones(3),
            0.1,
        )


def test_invalid_acceleration_shape_rejected():
    with pytest.raises(ValueError):
        integrate_velocity(
            np.ones(3),
            np.ones(2),
            0.1,
        )


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.1,
        np.inf,
        np.nan,
    ],
)
def test_invalid_timestep_rejected(dt):
    with pytest.raises(ValueError):
        integrate_velocity(
            np.ones(3),
            np.ones(3),
            dt,
        )
