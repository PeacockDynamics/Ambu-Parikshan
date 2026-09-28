import numpy as np
import pytest

from hydrolab3d.dynamics.drag import quadratic_drag_force


def test_zero_velocity_produces_zero_drag():
    drag = quadratic_drag_force(
        np.zeros(3),
        2.5,
    )

    assert np.array_equal(
        drag,
        np.zeros(3),
    )


def test_zero_coefficient_produces_zero_drag():
    drag = quadratic_drag_force(
        np.array([1.0, -2.0, 0.5]),
        0.0,
    )

    assert np.array_equal(
        drag,
        np.zeros(3),
    )


def test_known_one_axis_drag():
    drag = quadratic_drag_force(
        np.array([3.0, 0.0, 0.0]),
        2.0,
    )

    expected = np.array([
        -18.0,
        0.0,
        0.0,
    ])

    assert np.allclose(
        drag,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_drag_opposes_velocity():
    velocity = np.array([
        2.0,
        -1.0,
        0.5,
    ])

    drag = quadratic_drag_force(
        velocity,
        1.7,
    )

    assert np.dot(
        drag,
        velocity,
    ) < 0.0


def test_drag_has_odd_symmetry():
    velocity = np.array([
        1.25,
        -2.5,
        0.75,
    ])

    drag_forward = quadratic_drag_force(
        velocity,
        0.8,
    )

    drag_reverse = quadratic_drag_force(
        -velocity,
        0.8,
    )

    assert np.allclose(
        drag_reverse,
        -drag_forward,
        atol=1e-12,
        rtol=0.0,
    )


def test_drag_scales_quadratically_with_speed():
    velocity = np.array([
        2.0,
        -1.0,
        0.5,
    ])

    drag_base = quadratic_drag_force(
        velocity,
        1.7,
    )

    drag_double = quadratic_drag_force(
        2.0 * velocity,
        1.7,
    )

    ratio = (
        np.linalg.norm(drag_double)
        / np.linalg.norm(drag_base)
    )

    assert np.isclose(
        ratio,
        4.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_drag_is_antiparallel_to_3d_velocity():
    velocity = np.array([
        1.25,
        -2.5,
        0.75,
    ])

    drag = quadratic_drag_force(
        velocity,
        0.8,
    )

    velocity_direction = (
        velocity
        / np.linalg.norm(velocity)
    )

    drag_direction = (
        drag
        / np.linalg.norm(drag)
    )

    assert np.allclose(
        drag_direction,
        -velocity_direction,
        atol=1e-12,
        rtol=0.0,
    )


def test_drag_output_is_finite():
    drag = quadratic_drag_force(
        np.array([1.0, -2.0, 3.0]),
        1.5,
    )

    assert np.all(
        np.isfinite(drag)
    )


def test_invalid_velocity_shape_rejected():
    with pytest.raises(ValueError):
        quadratic_drag_force(
            np.ones(2),
            1.0,
        )


@pytest.mark.parametrize(
    "velocity",
    [
        np.array([1.0, np.nan, 0.0]),
        np.array([1.0, np.inf, 0.0]),
    ],
)
def test_nonfinite_velocity_rejected(
    velocity,
):
    with pytest.raises(ValueError):
        quadratic_drag_force(
            velocity,
            1.0,
        )


def test_nonscalar_coefficient_rejected():
    with pytest.raises(ValueError):
        quadratic_drag_force(
            np.ones(3),
            [1.0],
        )


@pytest.mark.parametrize(
    "drag_coefficient",
    [
        np.nan,
        np.inf,
        -1.0,
    ],
)
def test_invalid_coefficient_rejected(
    drag_coefficient,
):
    with pytest.raises(ValueError):
        quadratic_drag_force(
            np.ones(3),
            drag_coefficient,
        )
