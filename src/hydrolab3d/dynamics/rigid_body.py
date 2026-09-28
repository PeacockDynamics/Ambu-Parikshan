"""Basic translational dynamics for HydroLab-3D."""

import numpy as np


def acceleration_from_force(force, mass):
    """
    Compute 3D translational acceleration from net force and mass.

    Parameters
    ----------
    force : array-like
        Net translational force [Fx, Fy, Fz] in newtons.
    mass : float
        Vehicle mass in kilograms.

    Returns
    -------
    np.ndarray
        Translational acceleration [ax, ay, az] in m/s^2.
    """
    force = np.asarray(force, dtype=float)

    if force.shape != (3,):
        raise ValueError("force must have shape (3,)")

    if not np.all(np.isfinite(force)):
        raise ValueError("force must contain only finite values")

    if not np.isscalar(mass):
        raise ValueError("mass must be a scalar")

    mass = float(mass)

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be a positive finite scalar")

    return force / mass


def integrate_velocity(velocity, acceleration, dt):
    """
    Advance a 3D velocity by one constant-acceleration timestep.

    Parameters
    ----------
    velocity : array-like
        Current velocity [vx, vy, vz] in m/s.
    acceleration : array-like
        Current acceleration [ax, ay, az] in m/s^2.
    dt : float
        Simulation timestep in seconds.

    Returns
    -------
    np.ndarray
        Updated velocity after one timestep.
    """
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)

    if velocity.shape != (3,):
        raise ValueError("velocity must have shape (3,)")

    if acceleration.shape != (3,):
        raise ValueError("acceleration must have shape (3,)")

    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must contain only finite values")

    if not np.all(np.isfinite(acceleration)):
        raise ValueError("acceleration must contain only finite values")

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar")

    return velocity + acceleration * dt
