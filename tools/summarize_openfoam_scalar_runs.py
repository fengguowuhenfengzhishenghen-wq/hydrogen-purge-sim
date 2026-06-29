"""Summarize generated OpenFOAM scalarTransportFoam verification runs."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "openfoam_cases"
OUT_PATH = ROOT / "outputs" / "cfd3d" / "openfoam_scalarTransport_summary.csv"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_mesh_cells(text: str) -> int | None:
    match = re.search(r"cells:\s+(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def find_final_vtu(case_dir: Path) -> Path | None:
    vtk_root = case_dir / "VTK"
    if not vtk_root.exists():
        return None
    matches = sorted(vtk_root.glob("*_300/internal.vtu"))
    return matches[-1] if matches else None


def summarize_case(case_dir: Path) -> dict[str, object]:
    check_text = read_text(case_dir / "log.checkMesh")
    solve_text = read_text(case_dir / "log.scalarTransportFoam")
    vtk_text = read_text(case_dir / "log.foamToVTK")
    final_vtu = find_final_vtu(case_dir)

    mesh_ok = "Mesh OK" in check_text
    solver_end = re.search(r"(^|\n)End(\n|$)", solve_text) is not None
    vtk_end = "End" in vtk_text and final_vtu is not None
    fatal = any(
        token in (check_text + solve_text + vtk_text)
        for token in ("FOAM FATAL", "FOAM ERROR", "FatalError", "ERROR")
    )

    name = case_dir.name
    source_case = name.removesuffix("_scalarTransport")

    return {
        "case_id": source_case,
        "openfoam_case_dir": str(case_dir.relative_to(ROOT)),
        "mesh_cells": parse_mesh_cells(check_text),
        "mesh_ok": mesh_ok,
        "solver_reached_300s": "Time = 300" in solve_text,
        "solver_end": solver_end,
        "vtk_exported": vtk_end,
        "fatal_or_error": fatal,
        "final_vtu": str(final_vtu.relative_to(ROOT)) if final_vtu else "",
        "copied_output_vtu": str(
            Path("outputs")
            / "cfd3d"
            / f"{source_case}_reduced2d"
            / "openfoam_scalarTransport_300.vtu"
        ),
        "model_note": (
            "OpenFOAM scalarTransportFoam passive-scalar check mapped from "
            "the 1D H2 mole-fraction field; not a full buoyant multispecies CFD run."
        ),
    }


def main() -> None:
    cases = sorted(CASE_ROOT.glob("*_scalarTransport"))
    rows = [summarize_case(case) for case in cases]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        OUT_PATH.write_text("", encoding="utf-8")
        return
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
