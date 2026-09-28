"""Portable run artifact writers for the application layer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .runtime import RunResult


def write_run_artifacts(result: RunResult, directory: str | Path) -> dict[str, Path]:
    """Write compact metadata JSON and per-step truth/observation CSV."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / "run_metadata.json"
    trajectory = directory / "trajectory.csv"
    metadata.write_text(json.dumps({
        "scenario_name": result.scenario_name, "seed": result.seed, "dt": result.dt,
        "steps": len(result.records), "termination_reason": result.termination_reason,
        "vehicle_ids": sorted(result.fleet_seeds or {}), "vehicle_seeds": result.fleet_seeds,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["vehicle_id", "step", "time", *[f"pose_{i}" for i in range(6)], *[f"velocity_{i}" for i in range(6)], *[f"action_{i}" for i in range(6)], "depth", "heading"]
    with trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in result.records:
            if "vehicles" in record:
                for vehicle, vehicle_result in record["vehicles"].items():
                    truth = vehicle_result["truth"]; action = record["actions"][vehicle]
                    row = {"vehicle_id": vehicle, "step": record["step"], "time": truth["time"], "depth": vehicle_result["observation"]["depth"], "heading": vehicle_result["observation"]["heading"]}
                    row.update({f"pose_{i}": value for i, value in enumerate(np.asarray(truth["pose"]))}); row.update({f"velocity_{i}": value for i, value in enumerate(np.asarray(truth["body_velocity"]))}); row.update({f"action_{i}": value for i, value in enumerate(action)})
                    writer.writerow(row)
                continue
            truth = record["truth"]
            observation = record["observation"]
            row = {"vehicle_id": "uuv", "step": record["step"], "time": truth["time"], "depth": observation["depth"], "heading": observation["heading"]}
            row.update({f"pose_{i}": value for i, value in enumerate(np.asarray(truth["pose"]))})
            row.update({f"velocity_{i}": value for i, value in enumerate(np.asarray(truth["body_velocity"]))})
            row.update({f"action_{i}": value for i, value in enumerate(np.asarray(record["action"]))})
            writer.writerow(row)
    return {"metadata": metadata, "trajectory": trajectory}
