"""Run Task2 interruption analysis."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2purge.experiments import run_task2_interrupt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=10.0, help="Grid spacing in m")
    args = parser.parse_args()
    df = run_task2_interrupt(ROOT / "outputs" / "task2", dx=args.dx)
    print(f"Wrote {len(df)} Task2 interruption cases to outputs/task2/interrupt_summary.csv")


if __name__ == "__main__":
    main()
