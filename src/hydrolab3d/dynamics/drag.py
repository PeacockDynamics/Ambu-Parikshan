"""Hydrodynamic drag models for HydroLab-3D."""

import numpy as np


def quadratic_drag_force(velocity, drag_coefficient):
    """
    Compute reduced-order quadratic translational drag.

    The model is:

        F_D = -C_D * ||v|| * v

    where ``drag_coefficient`` is an effective reduced-order
    coefficient rather than a dimensionless textbook drag coefficient.

    Parameters
    ----------
    velocity : array-like
        3D translational velocity [vx, vy, vz] in m/s.
    drag_coefficient : float
        Nonnegative effective quadratic drag coefficient.

    Returns
    -------
    np.ndarray
        Drag-force vector [Fx, Fy, Fz] in newtons.
    """
    velocity = np.asarray(
        velocity,
        dtype=float,
    )

    if velocity.shape != (3,):
        raise ValueError(
            "velocity must have shape (3,)"
        )

    if not np.all(
        np.isfinite(velocity)
    ):
        raise ValueError(
            "velocity must contain only finite values"
        )

    if not np.isscalar(
        drag_coefficient
    ):
        raise ValueError(
            "drag_coefficient must be a scalar"
        )

    drag_coefficient = float(
        drag_coefficient
    )

    if not np.isfinite(
        drag_coefficient
    ):
        raise ValueError(
            "drag_coefficient must be finite"
        )

    if drag_coefficient < 0.0:
        raise ValueError(
            "drag_coefficient must be nonnegative"
        )

    speed = np.linalg.norm(
        velocity
    )

    return (
        -drag_coefficient
        * speed
        * velocity
    )
