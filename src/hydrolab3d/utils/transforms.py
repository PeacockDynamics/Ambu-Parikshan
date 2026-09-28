"""Reference-frame transformations for HydroLab-3D."""

import numpy as np


def rotation_x(phi: float) -> np.ndarray:
    """Active right-handed rotation about the +x axis."""
    c = np.cos(phi)
    s = np.sin(phi)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ])


def rotation_y(theta: float) -> np.ndarray:
    """Active right-handed rotation about the +y axis."""
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_z(psi: float) -> np.ndarray:
    """Active right-handed rotation about the +z axis."""
    c = np.cos(psi)
    s = np.sin(psi)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_body_to_world(
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Return the active body-to-world rotation matrix.

    Convention
    ----------
    R_B^W = R_z(psi) @ R_y(theta) @ R_x(phi)

    Angles are in radians.
    """
    return (
        rotation_z(psi)
        @ rotation_y(theta)
        @ rotation_x(phi)
    )
