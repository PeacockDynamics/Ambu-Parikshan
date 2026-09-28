"""Command line interface for headless Ambu-Parikshan runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ScenarioError, load_scenario
from .logging import write_run_artifacts
from .runtime import run_fleet_scenario, run_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ambu-parikshan")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a YAML scenario")
    run.add_argument("scenario", type=Path)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true", help="explicitly select the default headless runtime")
    mode.add_argument("--render", action="store_true", help="run the optional Panda3D interactive renderer")
    run.add_argument("--output", type=Path, help="directory for CSV/JSON run artifacts")
    args = parser.parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        if args.render:
            if args.output:
                raise ScenarioError("--output is not supported with --render; interactive runs do not write artifacts")
            from .interactive import run_interactive
            run_interactive(scenario)
            return 0
        result = run_fleet_scenario(scenario) if hasattr(scenario, "vehicles") else run_scenario(scenario)
        if args.output:
            write_run_artifacts(result, args.output)
    except (ScenarioError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"completed {result.scenario_name}: {len(result.records)} steps, termination={result.termination_reason}")
    return 0
