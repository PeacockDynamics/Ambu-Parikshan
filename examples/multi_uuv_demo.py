"""Minimal independent multi-UUV and communication demonstration."""

from pathlib import Path

from ambu_parikshan import load_scenario
from ambu_parikshan.runtime import build_simulator
from hydrolab3d.communication import AcousticChannel
from hydrolab3d.core import MultiUUVSimulator


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    scenario = load_scenario(root / "scenarios" / "basic_navigation.yaml")
    fleet = MultiUUVSimulator({"uuv_a": build_simulator(scenario), "uuv_b": build_simulator(scenario)})
    result = fleet.step({"uuv_a": [10, 0, 0, 0, 0, 0], "uuv_b": [0, 0, 0, 0, 0, 0]})
    channel = AcousticChannel(max_range=100, latency=0.1, seed=scenario.seed)
    channel.send("uuv_a", "uuv_b", {"kind": "status"}, fleet.time, result["uuv_a"]["truth"]["pose"][:3], result["uuv_b"]["truth"]["pose"][:3])
    print(f"advanced two vehicles to {fleet.time:.2f}s")
