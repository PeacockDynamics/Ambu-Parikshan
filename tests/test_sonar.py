import numpy as np
import pytest

from hydrolab3d.sensors.sonar import range_bearing_detections, seabed_altitude
from hydrolab3d.world import FlatSeabed, SphereObstacle, WaterVolume, WorldGeometry


def world():
    return WorldGeometry(WaterVolume([-20, -20, -20], [20, 20, 1]), FlatSeabed(-10), [SphereObstacle([3, 4, -5], 1)])


def test_lightweight_sonar_reports_body_frame_range_and_bearing():
    detections = range_bearing_detections([0, 0, -5, 0, 0, 0], world(), 10)
    assert detections[0]["range"] == pytest.approx(5)
    assert detections[0]["bearing"] == pytest.approx(np.arctan2(4, 3))
    assert seabed_altitude([0, 0, -5, 0, 0, 0], world()) == 5


def test_sonar_obeys_range_and_input_contract():
    assert range_bearing_detections([0, 0, -5, 0, 0, 0], world(), 4) == []
    with pytest.raises(ValueError):
        range_bearing_detections([0, 0], world(), 2)
