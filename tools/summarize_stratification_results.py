"""Summarize reduced local stratification result metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for path in sorted((ROOT / "outputs" / "cfd3d").glob("DN1200_u7_p010_stop*_reduced2d/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        case = path.parent.name
        if "stop30" in case:
            stop_fraction = 0.30
        elif "stop60" in case:
            stop_fraction = 0.60
        elif "stop80" in case:
            stop_fraction = 0.80
        else:
            stop_fraction = float("nan")
        rows.append(
            {
                "case": case,
                "interrupt_position_fraction": stop_fraction,
                "stop_duration_s": data["stop_duration_s"],
                "top_bottom_h2_delta": data["top_bottom_h2_delta"],
                "top_h2_max": data["top_h2_max"],
                "flammable_area_ratio": data["flammable_area_ratio"],
                "flammable_volume_ratio": data["flammable_volume_ratio"],
                "mesh_cells": data["mesh_cells"],
                "solver": data["solver"],
            }
        )
    out = ROOT / "outputs" / "cfd3d" / "reduced2d_stratification_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(out)


if __name__ == "__main__":
    main()
