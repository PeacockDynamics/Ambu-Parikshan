import numpy as np
import pytest

from hydrolab3d.dynamics.buoyancy import (
    buoyancy_force,
    gravity_force,
)


def test_known_gravity_force():
    force = gravity_force(
        20.0,
        9.81,
    )

    expected = np.array([
        0.0,
        0.0,
        -196.2,
    ])

    assert np.allclose(
        force,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_known_buoyancy_force():
    force = buoyancy_force(
        1000.0,
        0.02,
        9.81,
    )

    expected = np.array([
        0.0,
        0.0,
        196.2,
    ])

    assert np.allclose(
        force,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


def test_neutral_buoyancy_cancels_gravity():
    mass = 20.0
    density = 1000.0
    volume = mass / density

    net_force = (
        gravity_force(mass)
        + buoyancy_force(
            density,
            volume,
        )
    )

    assert np.allclose(
        net_force,
        np.zeros(3),
        atol=1e-12,
        rtol=0.0,
    )


def test_positive_and_negative_buoyancy_direction():
    mass = 20.0
    density = 1000.0

    neutral_volume = mass / density

    positive_force = (
        gravity_force(mass)
        + buoyancy_force(
            density,
            1.10 * neutral_volume,
        )
    )

    negative_force = (
        gravity_force(mass)
        + buoyancy_force(
            density,
            0.90 * neutral_volume,
        )
    )

    assert positive_force[2] > 0.0
    assert negative_force[2] < 0.0

    assert np.allclose(
        positive_force[:2],
        0.0,
        atol=1e-12,
        rtol=0.0,
    )

    assert np.allclose(
        negative_force[:2],
        0.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_zero_displaced_volume_produces_zero_buoyancy():
    force = buoyancy_force(
        1000.0,
        0.0,
    )

    assert np.array_equal(
        force,
        np.zeros(3),
    )


@pytest.mark.parametrize(
    "mass",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_mass_rejected(mass):
    with pytest.raises(ValueError):
        gravity_force(mass)


@pytest.mark.parametrize(
    "density",
    [
        0.0,
        -1000.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_density_rejected(density):
    with pytest.raises(ValueError):
        buoyancy_force(
            density,
            0.02,
        )


@pytest.mark.parametrize(
    "volume",
    [
        -0.01,
        np.nan,
        np.inf,
    ],
)
def test_invalid_volume_rejected(volume):
    with pytest.raises(ValueError):
        buoyancy_force(
            1000.0,
            volume,
        )


@pytest.mark.parametrize(
    "gravity",
    [
        0.0,
        -9.81,
        np.nan,
        np.inf,
    ],
)
def test_invalid_gravity_rejected(gravity):
    with pytest.raises(ValueError):
        gravity_force(
            20.0,
            gravity,
        )
