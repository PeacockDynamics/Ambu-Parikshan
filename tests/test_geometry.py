import numpy as np
import pytest

from hydrolab3d.world import BoxObstacle, FlatSeabed, HeightField, SphereObstacle, WaterVolume, WorldGeometry


def test_flat_world_queries_bounds_altitude_and_collision():
    world = WorldGeometry(WaterVolume([-10, -10, -20], [10, 10, 1]), FlatSeabed(-15), [SphereObstacle([1, 0, -5], 1), BoxObstacle([-3, -1, -6], [-2, 1, -4])])
    assert world.volume.contains([0, 0, -5])
    assert not world.volume.contains([0, 0, 2])
    assert world.altitude([0, 0, -5]) == 10
    assert world.collisions([1, 0, -5]) == [0]
    assert world.collisions([0, 0, -15]) == [-1]


def test_heightfield_is_bilinear_and_rejects_outside_queries():
    field = HeightField([0, 1], [0, 1], [[-10, -8], [-6, -4]])
    assert field.height(0.5, 0.5) == pytest.approx(-7)
    with pytest.raises(ValueError, match="outside"):
        field.height(2, 0)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        WaterVolume([0, 0, 0], [0, 1, 1])
    with pytest.raises(ValueError):
        SphereObstacle([0, 0, 0], 0)
    with pytest.raises(ValueError):
        HeightField([0], [0, 1], [[0], [0]])
