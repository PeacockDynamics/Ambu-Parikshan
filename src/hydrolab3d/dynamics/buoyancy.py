"""Gravity and buoyancy force models for HydroLab-3D."""

import numpy as np


def gravity_force(mass, gravity=9.81):
    """
    Compute world-frame gravitational force.

    HydroLab-3D convention:
        +z_W is upward

    Parameters
    ----------
    mass : float
        Vehicle mass in kilograms.
    gravity : float, optional
        Positive gravitational acceleration magnitude in m/s^2.

    Returns
    -------
    np.ndarray
        Gravity force [Fx, Fy, Fz] in newtons.
    """

    if not np.isscalar(mass):
        raise ValueError("mass must be a scalar")

    if not np.isscalar(gravity):
        raise ValueError("gravity must be a scalar")

    mass = float(mass)
    gravity = float(gravity)

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            "mass must be a positive finite scalar"
        )

    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError(
            "gravity must be a positive finite scalar"
        )

    return np.array([
        0.0,
        0.0,
        -mass * gravity,
    ])


def buoyancy_force(
    fluid_density,
    displaced_volume,
    gravity=9.81,
):
    """
    Compute world-frame hydrostatic buoyancy force.

    HydroLab-3D convention:
        +z_W is upward

    Parameters
    ----------
    fluid_density : float
        Fluid density in kg/m^3.
    displaced_volume : float
        Displaced fluid volume in m^3.
    gravity : float, optional
        Positive gravitational acceleration magnitude in m/s^2.

    Returns
    -------
    np.ndarray
        Buoyancy force [Fx, Fy, Fz] in newtons.
    """

    if not np.isscalar(fluid_density):
        raise ValueError(
            "fluid_density must be a scalar"
        )

    if not np.isscalar(displaced_volume):
        raise ValueError(
            "displaced_volume must be a scalar"
        )

    if not np.isscalar(gravity):
        raise ValueError(
            "gravity must be a scalar"
        )

    fluid_density = float(fluid_density)
    displaced_volume = float(displaced_volume)
    gravity = float(gravity)

    if (
        not np.isfinite(fluid_density)
        or fluid_density <= 0.0
    ):
        raise ValueError(
            "fluid_density must be a positive finite scalar"
        )

    if (
        not np.isfinite(displaced_volume)
        or displaced_volume < 0.0
    ):
        raise ValueError(
            "displaced_volume must be a nonnegative finite scalar"
        )

    if (
        not np.isfinite(gravity)
        or gravity <= 0.0
    ):
        raise ValueError(
            "gravity must be a positive finite scalar"
        )

    return np.array([
        0.0,
        0.0,
        fluid_density * gravity * displaced_volume,
    ])
