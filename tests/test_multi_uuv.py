import numpy as np
import pytest

from hydrolab3d.communication import AcousticChannel
from hydrolab3d.core import MultiUUVSimulator, Simulator


def simulator(offset, seed):
    return Simulator(initial_pose=[offset, 0, -5, 0, 0, 0], initial_body_velocity=[0] * 6, dt=0.02, sensor_seed=seed, sensor_config={}, dynamics_kwargs={
        "mass": 50, "inertia": [2, 3, 4], "added_inertia": [5, 8, 10, .5, .7, .9],
        "linear_damping": [8, 12, 15, 1.5, 2, 2.5], "quadratic_damping": [12, 18, 22, 2, 2.5, 3],
        "fluid_density": 1025, "displaced_volume": 50 / 1025, "center_of_gravity": [0, 0, 0],
        "center_of_buoyancy": [0, 0, .08], "current_velocity_world": [0, 0, 0],
    })


def test_multi_uuv_uses_independent_states_and_actions_deterministically():
    fleet = MultiUUVSimulator({"a": simulator(0, 1), "b": simulator(3, 2)})
    first = fleet.step({"a": [10, 0, 0, 0, 0, 0], "b": [0] * 6})
    assert first["a"]["truth"]["body_velocity"][0] > 0
    assert first["b"]["truth"]["body_velocity"][0] == 0
    snapshot = fleet.state(); snapshot["a"]["pose"][0] = 99
    assert fleet.state()["a"]["pose"][0] != 99
    with pytest.raises(ValueError): fleet.step({"a": [0] * 6})


def test_channel_range_latency_loss_blackout_and_ordering_are_explicit():
    channel = AcousticChannel(10, latency=1, seed=3)
    assert channel.send("a", "b", {"n": 1}, 0, [0, 0, 0], [2, 0, 0])
    assert not channel.send("a", "b", {}, 0, [0, 0, 0], [20, 0, 0])
    assert channel.receive(.9) == []
    assert channel.receive(1)[0].payload == {"n": 1}
    channel.blackouts.add("b")
    assert not channel.send("a", "b", {}, 1, [0, 0, 0], [2, 0, 0])
    assert not AcousticChannel(10, packet_loss=1).send("a", "b", {}, 0, [0, 0, 0], [2, 0, 0])
