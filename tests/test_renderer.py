import numpy as np
from ambu_parikshan.renderer import (
    TrajectoryHistory,
    OrbitCamera,
    CONTROL_LEGEND,
    format_telemetry,
    seabed_grid_specification,
    telemetry_panel_text,
    telemetry_snapshot,
    texture_asset_path,
    texture_repeat_count,
    uuv_visual_specification,
    visual_states,
    visualization_available,
    water_surface_specification,
    world_axis_segments,
)
from ambu_parikshan.cli import main
from ambu_parikshan.interactive import CAMERA_KEY_BINDINGS, SYSTEM_KEY_BINDINGS
from hydrolab3d.world import FlatSeabed, WaterVolume, WorldGeometry
from unittest.mock import patch
import pytest

def test_renderer_snapshot_and_bounded_history_are_physics_independent():
    record = {"truth": {"time": 1., "pose": np.arange(6.), "body_velocity": np.zeros(6)}}
    states = visual_states(record); history = TrajectoryHistory(2)
    history.append(states); record["truth"]["pose"][0] = 99
    assert len(history.points("uuv")) == 1
    assert history.points("uuv")[0][0] != 99

def test_visualization_dependency_check_is_safe():
    assert isinstance(visualization_available(), bool)


def test_world_axes_are_origin_anchored_and_use_robotics_colors():
    axes = world_axis_segments(4)
    assert [label for _, _, _, label in axes] == ["X", "Y", "Z"]
    assert all(np.array_equal(start, np.zeros(3)) for start, _, _, _ in axes)
    assert np.array_equal(axes[0][1], [4, 0, 0])
    assert np.array_equal(axes[1][1], [0, 4, 0])
    assert np.array_equal(axes[2][1], [0, 0, 4])
    assert axes[0][2][0] > axes[0][2][1]
    assert axes[1][2][1] > axes[1][2][0]
    assert axes[2][2][2] > axes[2][2][0]
    with pytest.raises(ValueError):
        world_axis_segments(0)


def test_seabed_grid_uses_existing_world_geometry_and_has_safe_default():
    world = WorldGeometry(
        WaterVolume([-20, -10, -30], [20, 10, 1]),
        FlatSeabed(-25),
    )
    assert seabed_grid_specification(world) == {
        "z": -25.0,
        "extent": 16.0,
        "divisions": 10,
    }
    assert seabed_grid_specification() == {
        "z": -10.0,
        "extent": 20.0,
        "divisions": 10,
    }


def test_water_surface_uses_world_volume_surface_or_renderer_default():
    world = WorldGeometry(
        WaterVolume([-20, -10, -30], [20, 10, 4], surface_z=2),
        FlatSeabed(-25),
    )
    assert water_surface_specification(world) == {
        "z": 2.0,
        "extent": 17.6,
    }
    assert water_surface_specification() == {"z": 0.0, "extent": 22.0}


def test_supplied_surface_textures_resolve_and_tile_with_finite_repeats():
    assert texture_asset_path("seabed_base.png").is_file()
    assert texture_asset_path("water_surface.png").is_file()
    assert texture_repeat_count(80, 10) == 8
    assert texture_repeat_count(4, 10) == 1
    with pytest.raises(ValueError):
        texture_asset_path("../water_surface.png")
    with pytest.raises(ValueError):
        texture_repeat_count(0, 10)


def test_procedural_uuv_specification_is_x_forward_and_finite():
    uuv = uuv_visual_specification(4, .6)
    assert np.all(np.isfinite(uuv["hull_scale"]))
    assert uuv["nose"]["position"][0] > 0
    assert uuv["tail"]["position"][0] < 0
    assert all(fin[0] < 0 for fin in uuv["horizontal_fins"])
    assert uuv["vertical_fin"][0] < 0
    with pytest.raises(ValueError):
        uuv_visual_specification(0, .6)
    with pytest.raises(ValueError):
        uuv_visual_specification(4, float("nan"))


def test_orbit_camera_changes_angle_distance_and_preserves_target_ownership():
    camera = OrbitCamera()
    default_position = camera.position()
    assert np.all(np.isfinite(default_position))
    target = np.array([1.0, 2.0, -5.0])
    camera.track_target(target)
    target[0] = 99.0
    assert camera.target[0] == 1.0
    default_azimuth, default_elevation, default_distance = (
        camera.azimuth,
        camera.elevation,
        camera.distance,
    )
    camera.orbit(azimuth_delta=.2, elevation_delta=.1)
    assert camera.azimuth == pytest.approx(default_azimuth + .2)
    assert camera.elevation == pytest.approx(default_elevation + .1)
    camera.zoom(-3)
    assert camera.distance == pytest.approx(default_distance - 3)
    assert np.all(np.isfinite(camera.position()))
    camera.reset()
    assert camera.azimuth == default_azimuth
    assert camera.elevation == default_elevation
    assert camera.distance == default_distance


def test_orbit_camera_clamps_elevation_and_zoom_bounds():
    camera = OrbitCamera()
    camera.orbit(elevation_delta=100)
    assert camera.elevation == camera.maximum_elevation
    camera.orbit(elevation_delta=-200)
    assert camera.elevation == camera.minimum_elevation
    camera.zoom(-1_000)
    assert camera.distance == camera.minimum_distance
    camera.zoom(1_000)
    assert camera.distance == camera.maximum_distance
    with pytest.raises(ValueError):
        camera.track_target([0, 0])


def test_telemetry_extracts_read_only_state_and_core_diagnostic_wrenches():
    state = {
        "time": 2.5,
        "pose": np.array([1.0, -2.0, -5.0, 0.0, np.pi / 2, -np.pi / 4]),
        "body_velocity": np.array([3.0, 4.0, 0.0, .1, .2, .3]),
    }
    diagnostics = {
        "hydrostatic_wrench": np.array([0, 0, 10, 1, 2, 2.0]),
        "damping_wrench": np.array([3, 4, 0, 0, 0, 0.0]),
        "actuator_wrench": np.array([0, 0, 12, 0, 0, 0.0]),
        "net_wrench": np.array([6, 8, 0, 0, 0, 0.0]),
    }
    snapshot = telemetry_snapshot(state, diagnostics=diagnostics, current_world=[.2, 0, 0], mode="constant_wrench")
    state["pose"][0] = 99
    assert snapshot["position"][0] == 1
    assert snapshot["depth"] == 5
    assert snapshot["attitude_degrees"][1] == pytest.approx(90)
    assert snapshot["speed"] == 5
    assert snapshot["forces"] == {
        "hydrostatic": 10.0, "damping": 5.0, "control": 12.0,
        "net": 10.0, "restoring_moment": 3.0,
    }
    text = telemetry_panel_text(snapshot)
    assert "Pitch   90.00°" in text
    assert "Mode    constant_wrench" in text


def test_telemetry_handles_missing_diagnostics_and_formats_safely():
    state = {"time": 0.0, "pose": np.zeros(6), "body_velocity": np.zeros(6)}
    snapshot = telemetry_snapshot(state)
    assert all(value is None for value in snapshot["forces"].values())
    assert "Hydrostatic N/A" in telemetry_panel_text(snapshot)
    assert format_telemetry(None, "N") == "N/A"
    assert format_telemetry(float("nan"), "m") == "N/A"


def test_panel_legend_and_interactive_system_bindings_are_distinct():
    for line in (
        "W / S   Forward / Reverse", "I / K   Orbit Up / Down",
        "C       Reset Camera", "Home    Reset UUV",
    ):
        assert line in CONTROL_LEGEND
    assert CAMERA_KEY_BINDINGS == {
        "j": "left", "l": "right", "i": "up", "k": "down",
        "u": "zoom_in", "h": "zoom_out",
    }
    assert SYSTEM_KEY_BINDINGS["home"] == "reset_simulation"
    assert SYSTEM_KEY_BINDINGS["c"] == "reset_camera"



def test_cli_render_errors_are_user_facing(tmp_path):
    scenario = tmp_path / "bad.yaml"; scenario.write_text("[]")
    with pytest.raises(SystemExit) as error:
        main(["run", str(scenario), "--render"])
    assert error.value.code == 2


def test_cli_rejects_conflicting_modes_and_render_output(tmp_path):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("""name: x
simulation: {dt: 0.02, max_steps: 1}
vehicle:
  initial_pose: [0, 0, -5, 0, 0, 0]
  initial_body_velocity: [0, 0, 0, 0, 0, 0]
  dynamics: {mass: 1, inertia: [1, 1, 1], added_inertia: [0, 0, 0, 0, 0, 0], linear_damping: [0, 0, 0, 0, 0, 0], quadratic_damping: [0, 0, 0, 0, 0, 0], fluid_density: 1, displaced_volume: 1, center_of_gravity: [0, 0, 0], center_of_buoyancy: [0, 0, 0], current_velocity_world: [0, 0, 0]}
""")
    with pytest.raises(SystemExit) as error:
        main(["run", str(scenario), "--headless", "--render"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["run", str(scenario), "--render", "--output", str(tmp_path / "out")])
    assert error.value.code == 2
