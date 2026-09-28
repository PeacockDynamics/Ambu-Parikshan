import numpy as np
import pytest

from hydrolab3d.dynamics import quadratic_drag_force
from hydrolab3d.world import (
    relative_water_velocity,
    uniform_current_velocity,
)


def test_zero_uniform_current():
    current = uniform_current_velocity(
        np.zeros(3)
    )

    np.testing.assert_array_equal(
        current,
        np.zeros(3),
    )


def test_known_uniform_current():
    expected = np.array(
        [0.5, 0.25, -0.1],
        dtype=float,
    )

    current = uniform_current_velocity(
        expected
    )

    np.testing.assert_array_equal(
        current,
        expected,
    )


def test_uniform_current_returns_independent_array():
    source = np.array(
        [0.5, 0.25, -0.1],
        dtype=float,
    )

    current = uniform_current_velocity(
        source
    )

    assert not np.shares_memory(
        current,
        source,
    )


def test_known_relative_water_velocity():
    vehicle = np.array(
        [2.0, -1.0, 0.5],
        dtype=float,
    )

    current = np.array(
        [0.5, 0.25, -0.1],
        dtype=float,
    )

    expected = np.array(
        [1.5, -1.25, 0.6],
        dtype=float,
    )

    actual = relative_water_velocity(
        vehicle,
        current,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_stationary_vehicle_relative_velocity():
    current = np.array(
        [0.5, -0.25, 0.1],
        dtype=float,
    )

    relative = relative_water_velocity(
        np.zeros(3),
        current,
    )

    np.testing.assert_allclose(
        relative,
        -current,
        rtol=0.0,
        atol=1e-12,
    )


def test_comoving_vehicle_has_zero_relative_velocity():
    current = np.array(
        [0.5, -0.25, 0.1],
        dtype=float,
    )

    relative = relative_water_velocity(
        current,
        current,
    )

    np.testing.assert_array_equal(
        relative,
        np.zeros(3),
    )


def test_arbitrary_3d_relative_velocity():
    vehicle = np.array(
        [-1.2, 3.4, -0.8],
        dtype=float,
    )

    current = np.array(
        [0.3, -0.6, 0.2],
        dtype=float,
    )

    relative = relative_water_velocity(
        vehicle,
        current,
    )

    np.testing.assert_allclose(
        relative,
        vehicle - current,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "invalid_current",
    [
        np.ones(2),
        np.ones(4),
        np.ones((3, 1)),
    ],
)
def test_uniform_current_rejects_invalid_shape(
    invalid_current,
):
    with pytest.raises(
        ValueError,
        match=r"shape \(3,\)",
    ):
        uniform_current_velocity(
            invalid_current
        )


@pytest.mark.parametrize(
    "invalid_current",
    [
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        np.array([0.0, -np.inf, 0.0]),
    ],
)
def test_uniform_current_rejects_nonfinite_values(
    invalid_current,
):
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        uniform_current_velocity(
            invalid_current
        )


@pytest.mark.parametrize(
    "invalid_vehicle",
    [
        np.ones(2),
        np.ones(4),
        np.ones((3, 1)),
    ],
)
def test_relative_velocity_rejects_invalid_vehicle_shape(
    invalid_vehicle,
):
    with pytest.raises(
        ValueError,
        match=r"vehicle_velocity must have shape \(3,\)",
    ):
        relative_water_velocity(
            invalid_vehicle,
            np.zeros(3),
        )


@pytest.mark.parametrize(
    "invalid_current",
    [
        np.ones(2),
        np.ones(4),
        np.ones((3, 1)),
    ],
)
def test_relative_velocity_rejects_invalid_current_shape(
    invalid_current,
):
    with pytest.raises(
        ValueError,
        match=r"current_velocity must have shape \(3,\)",
    ):
        relative_water_velocity(
            np.zeros(3),
            invalid_current,
        )


@pytest.mark.parametrize(
    "invalid_vehicle",
    [
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        np.array([0.0, -np.inf, 0.0]),
    ],
)
def test_relative_velocity_rejects_nonfinite_vehicle(
    invalid_vehicle,
):
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        relative_water_velocity(
            invalid_vehicle,
            np.zeros(3),
        )


@pytest.mark.parametrize(
    "invalid_current",
    [
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        np.array([0.0, -np.inf, 0.0]),
    ],
)
def test_relative_velocity_rejects_nonfinite_current(
    invalid_current,
):
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        relative_water_velocity(
            np.zeros(3),
            invalid_current,
        )


def test_current_relative_velocity_composes_with_drag():
    vehicle = np.zeros(3)

    current = np.array(
        [0.5, 0.25, -0.1],
        dtype=float,
    )

    drag_coefficient = 3.0

    relative = relative_water_velocity(
        vehicle,
        current,
    )

    drag = quadratic_drag_force(
        relative,
        drag_coefficient,
    )

    expected_drag = (
        -drag_coefficient
        * np.linalg.norm(relative)
        * relative
    )

    np.testing.assert_allclose(
        drag,
        expected_drag,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.dot(
        drag,
        current,
    ) > 0.0
