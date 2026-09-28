"""Manual control maps a user-selected generalized wrench into Simulator.step."""

from pathlib import Path
from ambu_parikshan.cli import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(["run", str(root / "scenarios" / "basic_navigation.yaml"), "--render"])
