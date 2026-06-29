"""Run validation checks and write outputs/validation artifacts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2purge.validation import run_all_validations


if __name__ == "__main__":
    summary = run_all_validations(ROOT / "outputs" / "validation")
    print("Validation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
