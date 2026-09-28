"""Optional Panda3D engineering renderer; it only consumes snapshots."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from hydrolab3d.utils.transforms import rotation_body_to_world


CONTROL_LEGEND = """UUV CONTROL
W / S   Forward / Reverse
A / D   Sway Left / Right
R / F   Up / Down
Q / E   Yaw Left / Right
Space   Neutral

CAMERA
I / K   Orbit Up / Down
J / L   Orbit Left / Right
U / H   Zoom In / Out
C       Reset Camera

SYSTEM
Home    Reset UUV
Esc     Quit"""

_TEXTURE_DIRECTORY = Path(__file__).resolve().parents[2] / "assets" / "textures"


def texture_asset_path(filename):
    """Resolve a checked-in presentation texture without involving simulation data."""
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("texture filename must be a simple file name")
    path = _TEXTURE_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(f"required visualization texture is missing: {path}")
    return path


def texture_repeat_count(extent, metres_per_tile):
    """Return a finite repeat count for a square horizontal visual card."""
    extent, metres_per_tile = float(extent), float(metres_per_tile)
    if not np.isfinite(extent) or not np.isfinite(metres_per_tile) or extent <= 0 or metres_per_tile <= 0:
        raise ValueError("texture extent and metres_per_tile must be finite and positive")
    return max(1.0, extent / metres_per_tile)


def format_telemetry(value, unit="", precision=2):
    """Format one finite engineering value without leaking NaN/Inf into HUD."""
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    suffix = (unit if unit == "°" else f" {unit}") if unit else ""
    return f"{float(value):.{precision}f}{suffix}"


def _diagnostic_magnitude(diagnostics, key, slice_):
    if not isinstance(diagnostics, dict) or key not in diagnostics:
        return None
    value = np.asarray(diagnostics[key], dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        return None
    return float(np.linalg.norm(value[slice_]))


def telemetry_snapshot(state, *, diagnostics=None, current_world=None, mode="manual", vehicle_id="uuv"):
    """Extract copied display telemetry from truth and existing core diagnostics.

    Force values are norms of the exact wrench terms emitted by HydroLab's
    validated six-DOF derivative; no forces are recomputed in this module.
    """
    if not isinstance(state, dict):
        raise ValueError("telemetry state must be a mapping")
    pose = np.asarray(state.get("pose"), dtype=float)
    velocity = np.asarray(state.get("body_velocity"), dtype=float)
    time = float(state.get("time"))
    if pose.shape != (6,) or velocity.shape != (6,) or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)) or not np.isfinite(time):
        raise ValueError("telemetry state must contain finite pose, body_velocity, and time")
    if current_world is None:
        current = None
    else:
        current = np.asarray(current_world, dtype=float)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise ValueError("current_world must be a finite vector with shape (3,)")
        current = current.copy()
    return {
        "time": time,
        "vehicle_id": str(vehicle_id),
        "position": pose[:3].copy(),
        "depth": float(-pose[2]),
        "attitude_degrees": np.degrees(pose[3:]).copy(),
        "velocity": velocity.copy(),
        "speed": float(np.linalg.norm(velocity[:3])),
        "current_world": current,
        "mode": str(mode),
        "forces": {
            "hydrostatic": _diagnostic_magnitude(diagnostics, "hydrostatic_wrench", slice(0, 3)),
            "damping": _diagnostic_magnitude(diagnostics, "damping_wrench", slice(0, 3)),
            "control": _diagnostic_magnitude(diagnostics, "actuator_wrench", slice(0, 3)),
            "net": _diagnostic_magnitude(diagnostics, "net_wrench", slice(0, 3)),
            "restoring_moment": _diagnostic_magnitude(diagnostics, "hydrostatic_wrench", slice(3, 6)),
        },
    }


def telemetry_panel_text(snapshot):
    """Render compact, fixed-layout left-panel telemetry text."""
    position = snapshot["position"]
    roll, pitch, yaw = snapshot["attitude_degrees"]
    u, v, w, p, q, r = snapshot["velocity"]
    forces = snapshot["forces"]
    current = snapshot["current_world"]
    current_lines = ("X       N/A", "Y       N/A", "Z       N/A") if current is None else tuple(
        f"{axis}       {format_telemetry(value, 'm/s')}" for axis, value in zip("XYZ", current)
    )
    return "\n".join((
        "AMBU-PARIKSHAN",
        "SIMULATION",
        f"Time    {format_telemetry(snapshot['time'], 's')}",
        f"Vehicle {snapshot['vehicle_id']}",
        "",
        "STATE",
        f"X       {format_telemetry(position[0], 'm')}",
        f"Y       {format_telemetry(position[1], 'm')}",
        f"Z       {format_telemetry(position[2], 'm')}",
        f"Depth   {format_telemetry(snapshot['depth'], 'm')}",
        "",
        "ATTITUDE",
        f"Roll    {format_telemetry(roll, '°')}",
        f"Pitch   {format_telemetry(pitch, '°')}",
        f"Yaw     {format_telemetry(yaw, '°')}",
        "",
        "MOTION",
        f"Speed   {format_telemetry(snapshot['speed'], 'm/s')}",
        f"u       {format_telemetry(u, 'm/s')}",
        f"v       {format_telemetry(v, 'm/s')}",
        f"w       {format_telemetry(w, 'm/s')}",
        f"p/q/r   {format_telemetry(p, 'rad/s')} / {format_telemetry(q, 'rad/s')} / {format_telemetry(r, 'rad/s')}",
        "",
        "FORCES / MOMENTS",
        f"Hydrostatic {format_telemetry(forces['hydrostatic'], 'N')}",
        f"Hydro Damping {format_telemetry(forces['damping'], 'N')}",
        f"Control      {format_telemetry(forces['control'], 'N')}",
        f"Net Force    {format_telemetry(forces['net'], 'N')}",
        f"Restore Mom. {format_telemetry(forces['restoring_moment'], 'N m')}",
        "",
        "ENVIRONMENT",
        "Current (world)",
        *current_lines,
        "",
        "CONTROL",
        f"Mode    {snapshot['mode']}",
    ))


class OrbitCamera:
    """Presentation-only orbit camera with a tracked visual target.

    Angles are stored in radians.  The defaults reproduce the Step 2 fixed
    engineering composition relative to its original look-at point.
    """

    minimum_elevation = np.deg2rad(-80.0)
    maximum_elevation = np.deg2rad(80.0)
    minimum_distance = 6.0
    maximum_distance = 90.0

    def __init__(self):
        default_target = np.array([0.0, 0.0, -16.0])
        default_offset = np.array([24.0, -30.0, 34.0])
        self._default_target = default_target
        self._default_azimuth = float(np.arctan2(default_offset[1], default_offset[0]))
        self._default_elevation = float(np.arcsin(default_offset[2] / np.linalg.norm(default_offset)))
        self._default_distance = float(np.linalg.norm(default_offset))
        self.target = default_target.copy()
        self.reset()

    def reset(self):
        """Restore viewing angle and distance without changing tracked target."""
        self.azimuth = self._default_azimuth
        self.elevation = self._default_elevation
        self.distance = self._default_distance

    def track_target(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("camera target must be a finite vector with shape (3,)")
        self.target = target.copy()

    def orbit(self, azimuth_delta=0.0, elevation_delta=0.0):
        azimuth_delta, elevation_delta = float(azimuth_delta), float(elevation_delta)
        if not np.isfinite(azimuth_delta) or not np.isfinite(elevation_delta):
            raise ValueError("camera orbit deltas must be finite")
        self.azimuth += azimuth_delta
        self.elevation = float(np.clip(
            self.elevation + elevation_delta,
            self.minimum_elevation,
            self.maximum_elevation,
        ))

    def zoom(self, distance_delta=0.0):
        distance_delta = float(distance_delta)
        if not np.isfinite(distance_delta):
            raise ValueError("camera zoom delta must be finite")
        self.distance = float(np.clip(
            self.distance + distance_delta,
            self.minimum_distance,
            self.maximum_distance,
        ))

    def position(self):
        horizontal = self.distance * np.cos(self.elevation)
        offset = np.array([
            horizontal * np.cos(self.azimuth),
            horizontal * np.sin(self.azimuth),
            self.distance * np.sin(self.elevation),
        ])
        return self.target + offset


def world_axis_segments(length=6.0):
    """Return finite, origin-anchored world-frame axis line definitions.

    The colors follow the conventional robotics palette: X red, Y green and
    Z blue.  This pure helper keeps scene layout testable without Panda3D.
    """
    length = float(length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("axis length must be finite and strictly positive")
    origin = np.zeros(3, dtype=float)
    return (
        (origin, np.array([length, 0.0, 0.0]), (0.95, 0.18, 0.18, 1.0), "X"),
        (origin, np.array([0.0, length, 0.0]), (0.18, 0.85, 0.25, 1.0), "Y"),
        (origin, np.array([0.0, 0.0, length]), (0.20, 0.45, 1.0, 1.0), "Z"),
    )


def seabed_grid_specification(world=None, default_z=-10.0):
    """Choose a horizontal renderer-only seabed grid from existing geometry.

    Height-field worlds are intentionally represented by a single horizontal
    reference level here; terrain meshing is a later visualization step.
    """
    default_z = float(default_z)
    if not np.isfinite(default_z):
        raise ValueError("default_z must be finite")
    if world is None:
        return {"z": default_z, "extent": 20.0, "divisions": 10}

    minimum = np.asarray(world.volume.minimum, dtype=float)
    maximum = np.asarray(world.volume.maximum, dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("world volume bounds must be finite vectors with shape (3,)")
    center = (minimum[:2] + maximum[:2]) / 2.0
    z = float(world.seabed_height(*center))
    if not np.isfinite(z):
        raise ValueError("world seabed reference height must be finite")
    extent = min(float(np.max(maximum[:2] - minimum[:2])) * 0.4, 80.0)
    return {"z": z, "extent": max(extent, 4.0), "divisions": 10}


def water_surface_specification(world=None, default_z=0.0):
    """Choose a renderer-only water surface from the configured volume."""
    default_z = float(default_z)
    if not np.isfinite(default_z):
        raise ValueError("default_z must be finite")
    grid = seabed_grid_specification(world)
    if world is None:
        z = default_z
    else:
        z = float(world.volume.surface_z)
        if not np.isfinite(z):
            raise ValueError("world surface_z must be finite")
    return {"z": z, "extent": grid["extent"] * 1.1}


def uuv_visual_specification(hull_length=4.0, hull_radius=0.6):
    """Describe a low-poly UUV in HydroLab's local +X-forward body frame."""
    hull_length, hull_radius = float(hull_length), float(hull_radius)
    if not np.isfinite(hull_length) or not np.isfinite(hull_radius) or hull_length <= 0 or hull_radius <= 0:
        raise ValueError("UUV dimensions must be finite and strictly positive")
    half_length = hull_length / 2.0
    return {
        "hull_scale": np.array([half_length, hull_radius, hull_radius]),
        # The nose is explicitly at +X; the tail and control surfaces remain
        # at -X.  These local coordinates are never modified in update().
        "nose": {"position": np.array([half_length * 0.82, 0.0, 0.0]), "scale": np.array([hull_radius, hull_radius * 0.95, hull_radius * 0.95])},
        "tail": {"position": np.array([-half_length * 0.92, 0.0, 0.0]), "scale": np.array([hull_radius * 0.75, hull_radius * 0.65, hull_radius * 0.65])},
        "horizontal_fins": (
            np.array([-half_length * 0.62, hull_radius * 0.92, 0.0]),
            np.array([-half_length * 0.62, -hull_radius * 0.92, 0.0]),
        ),
        "vertical_fin": np.array([-half_length * 0.72, 0.0, hull_radius * 0.85]),
        "sail": np.array([hull_length * 0.1, 0.0, hull_radius * 0.95]),
        "fin_scale": np.array([hull_radius * 0.85, hull_radius * 0.65, hull_radius * 0.12]),
        "vertical_fin_scale": np.array([hull_radius * 0.85, hull_radius * 0.12, hull_radius * 0.72]),
        "sail_scale": np.array([hull_radius * 0.55, hull_radius * 0.35, hull_radius * 0.45]),
    }


def visualization_available():
    """Return whether the optional Panda3D visualization stack is available."""
    try:
        import panda3d.core  # noqa: F401
        import direct.showbase.ShowBase  # noqa: F401
    except ImportError:
        return False

    return True


def require_visualization():
    """Raise a clear error when the optional visualization dependency is absent."""
    if not visualization_available():
        raise RuntimeError(
            "Visualization requires the optional Panda3D dependency. "
            "Install with: pip install -e '.[visualization]'"
        )


def visual_states(record):
    """Convert a runtime result record into renderer-owned state snapshots."""
    if "vehicles" in record:
        return {
            name: _snapshot(value["truth"])
            for name, value in record["vehicles"].items()
        }

    return {"uuv": _snapshot(record["truth"])}


def _snapshot(state):
    """Copy mutable state arrays so rendering cannot modify simulator truth."""
    return {
        key: (
            np.asarray(value, dtype=float).copy()
            if key in {"pose", "body_velocity"}
            else value
        )
        for key, value in state.items()
    }


def panda_rotation_matrix(pose):
    """Return the row-vector Panda transform for a HydroLab body-to-world pose.

    HydroLab matrices act on column vectors. Panda's matrix API stores the
    equivalent row-vector transform, hence the transpose; this keeps roll,
    pitch, and yaw semantics out of Panda's incompatible HPR shorthand.
    """
    pose = np.asarray(pose, dtype=float)

    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite vector with shape (6,)")

    result = np.eye(4)
    result[:3, :3] = rotation_body_to_world(*pose[3:]).T
    result[3, :3] = pose[:3]

    return result


def obstacle_marker_geometry(obstacle):
    """Return a marker centre and scale without eagerly touching box fields."""
    if hasattr(obstacle, "center"):
        return (
            np.asarray(obstacle.center, dtype=float),
            float(obstacle.radius),
        )

    center = (
        np.asarray(obstacle.minimum, dtype=float)
        + np.asarray(obstacle.maximum, dtype=float)
    ) / 2

    return center, 1.0


class TrajectoryHistory:
    """Bounded renderer-only trajectory history."""

    def __init__(self, limit=500):
        self.limit = int(limit)
        self._points = {}

    def append(self, states):
        for name, state in states.items():
            history = self._points.setdefault(
                name,
                deque(maxlen=self.limit),
            )

            history.append(
                np.asarray(
                    state["pose"],
                    dtype=float,
                )[:3].copy()
            )

    def points(self, name):
        return list(self._points.get(name, ()))


class PandaRenderer:
    """Small engineering scene with vehicles, obstacles and telemetry."""

    def __init__(
        self,
        world=None,
        waypoint=None,
        title="Ambu-Parikshan",
    ):
        require_visualization()

        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import (
            AmbientLight,
            CardMaker,
            DirectionalLight,
            LineSegs,
            TextNode,
            Texture,
            TextureStage,
            TransparencyAttrib,
            Vec4,
            WindowProperties,
        )

        try:
            self.base = ShowBase()

            # IMPORTANT:
            # ShowBase enables Panda3D's default mouse camera controller.
            # That controller overwrites manually configured camera transforms
            # during subsequent frames. Ambu-Parikshan owns its engineering
            # camera explicitly, so disable Panda's default controller.
            self.base.disableMouse()

        except Exception as exc:
            raise RuntimeError(
                "Panda3D could not open a graphical display. "
                "Start --render from a desktop/X11/EGL session."
            ) from exc

        self.base.setBackgroundColor(
            0.03,
            0.08,
            0.14,
        )

        properties = WindowProperties()
        properties.setTitle(title)
        self.base.win.requestProperties(properties)

        self.nodes = {}
        self.history = TrajectoryHistory()
        self.world = world
        self._line_segs_type = LineSegs
        self.camera_controller = OrbitCamera()
        self._camera_commands = set()

        light = AmbientLight("ambient")
        light.setColor(
            Vec4(
                0.42,
                0.48,
                0.52,
                1.0,
            )
        )

        light_node = self.base.render.attachNewNode(light)
        self.base.render.setLight(light_node)

        key_light = DirectionalLight("soft-key")
        key_light.setColor(Vec4(0.72, 0.82, 0.88, 1.0))
        key_node = self.base.render.attachNewNode(key_light)
        key_node.setHpr(-35, -55, 0)
        self.base.render.setLight(key_node)

        self._card_maker_type = CardMaker
        self._transparency_mode = TransparencyAttrib.MAlpha
        self._texture_type = Texture
        self._texture_stage_type = TextureStage

        self._build_telemetry_panel(TextNode)

        self._apply_camera()

        self._build_spatial_reference_environment()

        if waypoint is not None:
            self._marker(
                "waypoint",
                waypoint,
                (1.0, 0.8, 0.1, 1.0),
                0.5,
            )

        if world:
            for index, obstacle in enumerate(world.obstacles):
                center, scale = obstacle_marker_geometry(obstacle)

                self._marker(
                    f"obstacle-{index}",
                    center,
                    (0.8, 0.2, 0.2, 1.0),
                    scale,
                )

    def _build_telemetry_panel(self, text_node_type):
        """Create persistent left-side HUD nodes; text content updates in place."""
        panel = self._card_maker_type("telemetry-panel")
        panel.setFrame(-1.32, -0.74, -0.98, 0.98)
        background = self.base.aspect2d.attachNewNode(panel.generate())
        background.setColor(0.025, 0.045, 0.075, 0.78)
        background.setTransparency(self._transparency_mode)
        background.setBin("fixed", 0)
        background.setDepthWrite(False)
        self.nodes["telemetry-panel"] = background

        telemetry_text = text_node_type("telemetry")
        telemetry_text.setTextColor(0.88, 0.92, 0.94, 1.0)
        telemetry_text.setAlign(text_node_type.ALeft)
        self.telemetry = self.base.aspect2d.attachNewNode(telemetry_text)
        self.telemetry.setPos(-1.29, 0, 0.92)
        self.telemetry.setScale(0.032)
        self.telemetry.setBin("fixed", 1)

        legend_text = text_node_type("controls-legend")
        legend_text.setText(CONTROL_LEGEND)
        legend_text.setTextColor(0.58, 0.72, 0.78, 1.0)
        legend_text.setAlign(text_node_type.ALeft)
        self.controls_legend = self.base.aspect2d.attachNewNode(legend_text)
        self.controls_legend.setPos(-1.29, 0, -0.34)
        self.controls_legend.setScale(0.030)
        self.controls_legend.setBin("fixed", 1)

    def _build_spatial_reference_environment(self):
        """Attach nonphysical axes, seabed plane/grid, and water cue."""
        axes = self._line_segs_type("world-axes")
        axes.setThickness(1.35)
        for start, end, color, _label in world_axis_segments(3.5):
            axes.setColor(*color)
            axes.moveTo(*start)
            axes.drawTo(*end)
        self.nodes["world-axes"] = self.base.render.attachNewNode(axes.create())

        grid_spec = seabed_grid_specification(self.world)
        seabed = self._horizontal_card(
            "seabed-plane",
            grid_spec["z"],
            grid_spec["extent"],
            (1.0, 1.0, 1.0, 1.0),
        )
        self._apply_tiled_texture(seabed, "seabed_base.png", grid_spec["extent"], 10.0)
        self.nodes["seabed-plane"] = seabed

        grid = self._line_segs_type("seabed-grid")
        grid.setThickness(1.0)
        grid.setColor(0.12, 0.34, 0.32, 1.0)
        half_extent = grid_spec["extent"] / 2.0
        divisions = grid_spec["divisions"]
        # Keep the reference grid just above the solid card to avoid z-fighting.
        z = grid_spec["z"] + 0.01
        for value in np.linspace(-half_extent, half_extent, divisions + 1):
            grid.moveTo(value, -half_extent, z)
            grid.drawTo(value, half_extent, z)
            grid.moveTo(-half_extent, value, z)
            grid.drawTo(half_extent, value, z)
        self.nodes["seabed-grid"] = self.base.render.attachNewNode(grid.create())

        water_spec = water_surface_specification(self.world)
        water = self._horizontal_card(
            "water-surface",
            water_spec["z"],
            water_spec["extent"],
            # Deliberately visible surface boundary: translucent enough to see
            # the UUV below it, light enough to read as water rather than sky.
            (0.75, 0.90, 1.0, 0.28),
        )
        self._apply_tiled_texture(water, "water_surface.png", water_spec["extent"], 14.0)
        water.setTransparency(self._transparency_mode)
        water.setBin("transparent", 0)
        water.setDepthWrite(False)
        self.nodes["water-surface"] = water

    def _apply_tiled_texture(self, node, filename, extent, metres_per_tile):
        """Apply one repeatable image asset to an existing visual-only card."""
        texture = self.base.loader.loadTexture(str(texture_asset_path(filename)))
        if texture is None:
            raise RuntimeError(f"Panda3D could not load visualization texture: {filename}")
        texture.setWrapU(self._texture_type.WMRepeat)
        texture.setWrapV(self._texture_type.WMRepeat)
        texture.setMinfilter(self._texture_type.FTLinearMipmapLinear)
        stage = self._texture_stage_type.getDefault()
        repeats = texture_repeat_count(extent, metres_per_tile)
        node.setTexture(stage, texture, 1)
        node.setTexScale(stage, repeats, repeats)

    def camera_press(self, command):
        """Mark a presentation-only camera command active."""
        if command not in {"left", "right", "up", "down", "zoom_in", "zoom_out"}:
            raise ValueError(f"unsupported camera command: {command!r}")
        self._camera_commands.add(command)

    def camera_release(self, command):
        """Mark a presentation-only camera command inactive."""
        self._camera_commands.discard(command)

    def reset_camera(self):
        """Restore the default engineering orbit around the current target."""
        self.camera_controller.reset()
        self._apply_camera()

    def update_camera(self, elapsed):
        """Apply held camera commands using elapsed render-frame time only."""
        elapsed = float(elapsed)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("camera elapsed time must be finite and non-negative")
        orbit_rate = 1.1
        zoom_rate = 14.0
        self.camera_controller.orbit(
            azimuth_delta=orbit_rate * elapsed * (
                ("left" in self._camera_commands) - ("right" in self._camera_commands)
            ),
            elevation_delta=orbit_rate * elapsed * (
                ("up" in self._camera_commands) - ("down" in self._camera_commands)
            ),
        )
        self.camera_controller.zoom(zoom_rate * elapsed * (
            ("zoom_out" in self._camera_commands) - ("zoom_in" in self._camera_commands)
        ))
        self._apply_camera()

    def _apply_camera(self):
        self.base.camera.setPos(*self.camera_controller.position())
        self.base.camera.lookAt(*self.camera_controller.target)

    def _horizontal_card(self, name, z, extent, color):
        """Create a horizontal renderer-only card centered at the origin."""
        z, extent = float(z), float(extent)
        if not np.isfinite(z) or not np.isfinite(extent) or extent <= 0:
            raise ValueError("card coordinates must be finite and extent positive")
        card = self._card_maker_type(name)
        half_extent = extent / 2.0
        card.setFrame(-half_extent, half_extent, -half_extent, half_extent)
        node = self.base.render.attachNewNode(card.generate())
        # CardMaker produces an X/Z card; rotate it into the world X/Y plane.
        node.setP(90)
        node.setZ(z)
        node.setColor(*color)
        node.setTwoSided(True)
        return node

    def _marker(
        self,
        name,
        position,
        color,
        scale=0.6,
    ):
        """Create a simple spherical engineering marker."""
        node = self.base.loader.loadModel(
            "models/misc/sphere"
        )

        node.reparentTo(self.base.render)
        node.setPos(*position)
        node.setScale(scale)
        node.setColor(*color)

        self.nodes[name] = node

        return node

    def _uuv_visual(self, name):
        """Build a small procedural UUV whose local nose is HydroLab +X."""
        specification = uuv_visual_specification()
        vehicle = self.base.render.attachNewNode(f"uuv-{name}")

        hull_color = (0.045, 0.10, 0.16, 1.0)
        accent_color = (0.10, 0.55, 0.68, 1.0)
        fin_color = (0.06, 0.18, 0.23, 1.0)

        self._uuv_part(vehicle, "hull", (0.0, 0.0, 0.0), specification["hull_scale"], hull_color)
        self._uuv_part(vehicle, "nose", specification["nose"]["position"], specification["nose"]["scale"], accent_color)
        self._uuv_part(vehicle, "tail", specification["tail"]["position"], specification["tail"]["scale"], fin_color)
        for index, position in enumerate(specification["horizontal_fins"]):
            self._uuv_part(vehicle, f"horizontal-fin-{index}", position, specification["fin_scale"], fin_color)
        self._uuv_part(vehicle, "vertical-fin", specification["vertical_fin"], specification["vertical_fin_scale"], fin_color)
        self._uuv_part(vehicle, "sail", specification["sail"], specification["sail_scale"], accent_color)

        self.nodes[name] = vehicle
        return vehicle

    def _uuv_part(self, parent, name, position, scale, color):
        """Attach a low-poly ellipsoidal component in local body coordinates."""
        node = self.base.loader.loadModel("models/misc/sphere")
        node.setName(name)
        node.reparentTo(parent)
        node.setPos(*np.asarray(position, dtype=float))
        node.setScale(*np.asarray(scale, dtype=float))
        node.setColor(*color)
        return node

    def update(
        self,
        states,
        mode="manual",
        diagnostics=None,
        current_world=None,
    ):
        """Update visible vehicle transforms and telemetry."""
        self.history.append(states)

        primary_state = None
        primary_name = None

        for index, (name, state) in enumerate(states.items()):
            pose = np.asarray(
                state["pose"],
                dtype=float,
            )

            if index == 0:
                # Camera tracking consumes a copied visual pose only.  It does
                # not write to the runtime state or change simulation timing.
                self.camera_controller.track_target(pose[:3])
                primary_state = state
                primary_name = name

            node = self.nodes.get(name)

            if node is None:
                node = self._uuv_visual(name)

            from panda3d.core import Mat4

            node.setMat(
                Mat4(
                    *panda_rotation_matrix(pose).ravel()
                )
            )

        if primary_state is not None:
            snapshot = telemetry_snapshot(
                primary_state,
                diagnostics=diagnostics,
                current_world=current_world,
                mode=mode,
                vehicle_id=primary_name,
            )
            self.telemetry.node().setText(telemetry_panel_text(snapshot))
        self._apply_camera()

    def run(self):
        """Enter Panda3D's render/event loop."""
        self.base.run()
