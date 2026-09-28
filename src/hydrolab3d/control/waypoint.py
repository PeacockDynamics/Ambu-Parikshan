"""Minimal waypoint-reference generation for HydroLab-3D."""

import numpy as np


def waypoint_guidance(
    position_world,
    waypoint_world,
):
    """Convert a world-frame waypoint into navigation references.

    Parameters
    ----------
    position_world : array-like, shape (3,)
        Current world-frame position [x, y, z].

    waypoint_world : array-like, shape (3,)
        Desired world-frame waypoint [x_g, y_g, z_g].

    Returns
    -------
    guidance : dict
        Contains displacement, desired heading, desired depth,
        horizontal distance, and full 3D distance.

    Notes
    -----
    HydroLab-3D uses +z world upward and depth = -z.

    This function performs geometry only. It does not provide
    localization and does not modify vehicle state.
    """

    position_world = np.asarray(
        position_world,
        dtype=float,
    )

    waypoint_world = np.asarray(
        waypoint_world,
        dtype=float,
    )

    if position_world.shape != (3,):
        raise ValueError(
            "position_world must have shape (3,)"
        )

    if waypoint_world.shape != (3,):
        raise ValueError(
            "waypoint_world must have shape (3,)"
        )

    if not np.all(
        np.isfinite(
            position_world
        )
    ):
        raise ValueError(
            "position_world must contain only finite values"
        )

    if not np.all(
        np.isfinite(
            waypoint_world
        )
    ):
        raise ValueError(
            "waypoint_world must contain only finite values"
        )

    displacement_world = (
        waypoint_world
        - position_world
    )

    dx = float(
        displacement_world[0]
    )

    dy = float(
        displacement_world[1]
    )

    desired_heading = float(
        np.arctan2(
            dy,
            dx,
        )
    )

    desired_depth = float(
        -waypoint_world[2]
    )

    horizontal_distance = float(
        np.hypot(
            dx,
            dy,
        )
    )

    distance = float(
        np.linalg.norm(
            displacement_world
        )
    )

    return {
        "displacement_world":
            displacement_world,

        "desired_heading":
            desired_heading,

        "desired_depth":
            desired_depth,

        "horizontal_distance":
            horizontal_distance,

        "distance":
            distance,
    }
