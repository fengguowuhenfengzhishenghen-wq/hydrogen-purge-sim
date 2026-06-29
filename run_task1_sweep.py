"""Run Task1 parameter sweep."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2purge.experiments import run_task1_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=10.0, help="Grid spacing in m")
    args = parser.parse_args()
    df = run_task1_sweep(ROOT / "outputs" / "task1", dx=args.dx)
    print(f"Wrote {len(df)} Task1 cases to outputs/task1/task1_summary.csv")


if __name__ == "__main__":
    main()
