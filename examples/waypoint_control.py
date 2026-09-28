"""Run the packaged deterministic basic-navigation scenario."""

from pathlib import Path

from ambu_parikshan import load_scenario, run_scenario


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_scenario(load_scenario(root / "scenarios" / "waypoint_navigation.yaml"))
    print(f"{result.scenario_name}: {len(result.records)} steps")
