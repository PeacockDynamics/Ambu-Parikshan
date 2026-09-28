"""Fixed-step interactive application runtime.

Interactive mode is deliberately single-UUV only in 0.1.0. It preserves the
configured action source and adds keyboard wrench input at ``Simulator.step``.
Fleet communication, faults, and artifact logging remain headless-only.
"""
from __future__ import annotations

import numpy as np

from .actions import action_source
from .manual import ManualActionMapper
from .renderer import PandaRenderer
from .runtime import build_simulator
from hydrolab3d.utils.transforms import rotation_body_to_world


CAMERA_KEY_BINDINGS = {
    "j": "left", "l": "right", "i": "up", "k": "down",
    "u": "zoom_in", "h": "zoom_out",
}
SYSTEM_KEY_BINDINGS = {"home": "reset_simulation", "c": "reset_camera"}
SURFACE_GUARD_CLEARANCE = 2.0
COLLISION_GUARD_LOOKAHEAD = 3.0
# The procedural display hull extends a little over 2 m from the simulator
# reference point along +X/-X.  Use a conservative envelope so collision
# guarding is visibly body-to-obstacle rather than point-to-obstacle.
COLLISION_GUARD_RADIUS = 2.5


class FixedStepAccumulator:
    """Bounded wall-time accumulator independent of Panda task execution cost."""

    def __init__(self, simulation_dt: float, maximum_frame_dt: float = 0.25):
        self.simulation_dt = float(simulation_dt)
        self.maximum_frame_dt = float(maximum_frame_dt)
        if self.simulation_dt <= 0 or self.maximum_frame_dt <= 0:
            raise ValueError("timesteps must be strictly positive")
        self.value = 0.0

    def add_frame_time(self, elapsed: float) -> int:
        elapsed = float(elapsed)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("frame time must be finite and non-negative")
        # Excess wall time during a stalled frame is deliberately dropped to
        # avoid an unbounded catch-up/spiral-of-death loop.
        self.value += min(elapsed, self.maximum_frame_dt)
        count = int(self.value // self.simulation_dt)
        self.value -= count * self.simulation_dt
        return count

    def reset(self) -> None:
        self.value = 0.0


def interactive_supported(scenario) -> None:
    if hasattr(scenario, "vehicles"):
        raise ValueError("--render supports single-UUV scenarios only; use --headless for fleet communication and faults")


def guard_manual_surface_heave(action, pose, surface_z=0.0, clearance=SURFACE_GUARD_CLEARANCE):
    """Suppress only manual upward heave near the configured water surface.

    This is an interactive UI safety guard, not a contact model: it does not
    change truth, velocity, sensor behavior, autonomous actions, or HydroLab
    propagation.  The conservative clearance lets existing damping shed upward
    momentum before the depth sensor's modeled surface boundary is reached.
    """
    action = np.asarray(action, dtype=float)
    pose = np.asarray(pose, dtype=float)
    surface_z, clearance = float(surface_z), float(clearance)
    if action.shape != (6,) or pose.shape != (6,) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(pose)):
        raise ValueError("manual action and pose must be finite vectors with shape (6,)")
    if not np.isfinite(surface_z) or not np.isfinite(clearance) or clearance <= 0:
        raise ValueError("surface_z must be finite and clearance strictly positive")
    guarded = action.copy()
    body_z_points_upward = rotation_body_to_world(*pose[3:])[2, 2] > 0.0
    if pose[2] >= surface_z - clearance and body_z_points_upward and guarded[2] > 0.0:
        guarded[2] = 0.0
    return guarded


def guard_manual_collision_translation(
    action,
    pose,
    world,
    *,
    lookahead=COLLISION_GUARD_LOOKAHEAD,
    vehicle_radius=COLLISION_GUARD_RADIUS,
):
    """Suppress manual translation components which would enter nearby geometry.

    This is intentionally an interactive navigation guard, not collision
    physics.  It samples the configured geometry along a short world-frame
    look-ahead for each independently commanded body translation axis.  A
    blocked component is zeroed while the other translation axes and all
    moments remain available, so the operator can steer away.  It never moves
    the vehicle, injects contact forces, or changes ``Simulator.step``.
    """
    action = np.asarray(action, dtype=float)
    pose = np.asarray(pose, dtype=float)
    lookahead, vehicle_radius = float(lookahead), float(vehicle_radius)
    if action.shape != (6,) or pose.shape != (6,) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(pose)):
        raise ValueError("manual action and pose must be finite vectors with shape (6,)")
    if not np.isfinite(lookahead) or lookahead <= 0 or not np.isfinite(vehicle_radius) or vehicle_radius < 0:
        raise ValueError("lookahead must be finite and positive; vehicle_radius must be finite and non-negative")
    if world is None:
        return action.copy()

    guarded = action.copy()
    rotation = rotation_body_to_world(*pose[3:])
    # Sampling, rather than testing only the end point, catches an obstacle
    # whose far side happens to lie beyond the chosen look-ahead endpoint.
    fractions = np.linspace(0.0, 1.0, max(2, int(np.ceil(lookahead / 0.25)) + 1))[1:]
    for axis in range(3):
        if guarded[axis] == 0.0:
            continue
        direction_body = np.zeros(3)
        direction_body[axis] = np.sign(guarded[axis])
        direction_world = rotation @ direction_body
        for fraction in fractions:
            probe = pose[:3] + direction_world * (lookahead * fraction)
            if world.collisions(probe, radius=vehicle_radius):
                guarded[axis] = 0.0
                break
    return guarded


def run_interactive(scenario):
    """Run a single scenario with independent rendering and fixed physics steps."""
    interactive_supported(scenario)
    simulator = build_simulator(scenario)
    source = action_source(scenario.action)
    source.reset(); simulator.reset()
    mapper = ManualActionMapper(); renderer = PandaRenderer(scenario.world, scenario.action.get("waypoint"), scenario.name)
    accumulator = FixedStepAccumulator(scenario.dt)
    current_world = np.asarray(scenario.dynamics_kwargs["current_velocity_world"], dtype=float).copy()
    surface_z = scenario.world.volume.surface_z if scenario.world is not None else 0.0
    last_diagnostics = None

    def reset_run():
        nonlocal last_diagnostics
        simulator.reset(); source.reset(); mapper.reset(); accumulator.reset()
        last_diagnostics = None
        renderer.history = type(renderer.history)()
        renderer.update({"uuv": simulator.state()}, scenario.action["mode"], diagnostics=last_diagnostics, current_world=current_world)

    def step(task):
        nonlocal last_diagnostics
        # globalClock is elapsed frame time; Panda task.dt is callback runtime.
        elapsed = renderer.base.clock.getDt()
        for _ in range(accumulator.add_frame_time(elapsed)):
            state = simulator.state()
            autonomous = source.action(state)
            manual = guard_manual_surface_heave(mapper.action(), state["pose"], surface_z)
            # Geometry guarding applies to the complete action boundary.  This
            # matters for rendered scenarios such as ``basic_navigation``:
            # their configured constant/waypoint action may drive toward an
            # obstacle even while no manual key is pressed.
            action = guard_manual_collision_translation(
                np.asarray(autonomous, dtype=float) + manual,
                state["pose"],
                scenario.world,
            )
            simulator.step(action)
            last_diagnostics = simulator.diagnostics
        renderer.update({"uuv": simulator.state()}, scenario.action["mode"], diagnostics=last_diagnostics, current_world=current_world)
        renderer.update_camera(elapsed)
        return task.cont

    for key in ("w", "s", "a", "d", "r", "f", "q", "e"):
        renderer.base.accept(key, mapper.press, [key]); renderer.base.accept(f"{key}-up", mapper.release, [key])
    for key, command in CAMERA_KEY_BINDINGS.items():
        renderer.base.accept(key, renderer.camera_press, [command]); renderer.base.accept(f"{key}-up", renderer.camera_release, [command])
    renderer.base.accept("space", mapper.reset); renderer.base.accept("escape", renderer.base.userExit); renderer.base.accept("home", reset_run)
    renderer.base.accept("c", renderer.reset_camera)
    renderer.base.taskMgr.add(step, "fixed-step-simulation")
    renderer.update({"uuv": simulator.state()}, scenario.action["mode"], diagnostics=last_diagnostics, current_world=current_world)
    renderer.run()
