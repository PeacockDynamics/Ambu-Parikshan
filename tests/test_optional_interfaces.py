import numpy as np
import pytest

from ambu_parikshan.manual import ManualActionMapper
from ambu_parikshan.interactive import (
    COLLISION_GUARD_RADIUS,
    guard_manual_collision_translation,
    guard_manual_surface_heave,
)
from hydrolab3d.interface.ros2_interface import observation_dict_to_ros_fields, require_ros2, ros2_available
from hydrolab3d.world import FlatSeabed, SphereObstacle, WaterVolume, WorldGeometry


def test_keyboard_mapping_uses_only_standard_action_boundary():
    mapper = ManualActionMapper(); mapper.press("w"); mapper.press("q")
    assert mapper.action().tolist() == [10, 0, 0, 0, 0, 5]
    mapper.press("space"); assert np.array_equal(mapper.action(), np.zeros(6))


def test_ros_conversion_is_pure_and_ros_is_optional():
    fields = observation_dict_to_ros_fields({"depth": 2, "heading": .1, "dvl": [1, 2, 3], "gyroscope": [0, 0, 0], "accelerometer": [0, 0, 9.81]})
    assert fields["depth"] == 2
    if not ros2_available():
        with pytest.raises(RuntimeError, match="ROS 2 support"):
            require_ros2()


def test_interactive_surface_guard_only_suppresses_manual_upward_heave_near_surface():
    manual = np.array([1, 0, 10, 0, 0, 0], dtype=float)
    pose = np.array([0, 0, -1.5, 0, 0, 0], dtype=float)
    guarded = guard_manual_surface_heave(manual, pose, surface_z=0, clearance=2)
    assert guarded.tolist() == [1, 0, 0, 0, 0, 0]
    assert manual[2] == 10
    assert guard_manual_surface_heave(manual, [0, 0, -5, 0, 0, 0]).tolist() == manual.tolist()
    assert guard_manual_surface_heave([0, 0, -10, 0, 0, 0], pose).tolist() == [0, 0, -10, 0, 0, 0]


def test_interactive_collision_guard_blocks_only_translation_toward_geometry():
    world = WorldGeometry(
        WaterVolume([-20, -20, -20], [20, 20, 1]),
        FlatSeabed(-15),
        [SphereObstacle([2, 0, -5], .5)],
    )
    action = np.array([10, 4, 0, 0, 0, 5], dtype=float)
    pose = np.array([0, 0, -5, 0, 0, 0], dtype=float)
    guarded = guard_manual_collision_translation(action, pose, world, lookahead=3, vehicle_radius=.25)
    assert guarded.tolist() == [0, 4, 0, 0, 0, 5]
    assert action.tolist() == [10, 4, 0, 0, 0, 5]


def test_interactive_collision_guard_can_protect_configured_or_manual_actions_equally():
    world = WorldGeometry(
        WaterVolume([-20, -20, -20], [20, 20, 1]),
        FlatSeabed(-15),
        [SphereObstacle([2, 0, -5], .5)],
    )
    configured = np.array([10, 0, 0, 0, 0, 0], dtype=float)
    manual = np.array([0, 4, 0, 0, 0, 0], dtype=float)
    guarded = guard_manual_collision_translation(configured + manual, [0, 0, -5, 0, 0, 0], world, lookahead=3, vehicle_radius=.25)
    assert guarded.tolist() == [0, 4, 0, 0, 0, 0]


def test_interactive_collision_guard_uses_a_conservative_body_envelope_not_a_point():
    world = WorldGeometry(
        WaterVolume([-20, -20, -20], [20, 20, 1]),
        FlatSeabed(-15),
        [SphereObstacle([4, 0, -5], 1)],
    )
    # The obstacle is outside a point-like 3 m look-ahead, but is within the
    # UUV's configured body envelope and must still stop forward input.
    guarded = guard_manual_collision_translation([10, 0, 0, 0, 0, 0], [0, 0, -5, 0, 0, 0], world)
    assert COLLISION_GUARD_RADIUS >= 2.5
    assert guarded[0] == 0


def test_interactive_collision_guard_preserves_an_alternate_safe_direction_and_validates_inputs():
    world = WorldGeometry(WaterVolume([-20, -20, -20], [20, 20, 1]), FlatSeabed(-15))
    action = [10, 0, 0, 0, 0, 0]
    assert guard_manual_collision_translation(action, [0, 0, -5, 0, 0, 0], world).tolist() == action
    assert guard_manual_collision_translation(action, [0, 0, -5, 0, 0, 0], None).tolist() == action
    with pytest.raises(ValueError, match="lookahead"):
        guard_manual_collision_translation(action, [0, 0, -5, 0, 0, 0], world, lookahead=0)
