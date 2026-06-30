"""Build offline 3D multi-species CFD case packages.

This script prepares local cylindrical 3D initial fields for external
OpenFOAM/Fluent work.  It does not run a CFD solver.  The Streamlit app reads
completed results from outputs/cfd3d; this package lives under openfoam_cases so
it is not mistaken for solved CFD output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2purge.cfd3d import initial_field_metrics, make_cylindrical_sample_grid, map_profile_to_grid


def load_profile(case_id: str, window: str) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "cfd_cases" / case_id / "windows" / window / "initial_1d_profile.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    x = df["x_local_m"].to_numpy(dtype=float)
    X = df[["x_H2_molfrac", "x_N2_molfrac", "x_Air_molfrac"]].to_numpy(dtype=float)
    return x, X


def write_samples_csv(path: Path, grid, mole: np.ndarray, mass: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "x_m": grid.x_m,
            "y_m": grid.y_m,
            "z_m": grid.z_m,
            "r_m": grid.radial_m,
            "theta_rad": grid.theta_rad,
            "x_H2": mole[:, 0],
            "x_N2": mole[:, 1],
            "x_O2": mole[:, 2],
            "Y_H2": mass[:, 0],
            "Y_N2": mass[:, 1],
            "Y_O2": mass[:, 2],
        }
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_points_vtk(path: Path, grid, mole: np.ndarray, mass: np.ndarray) -> None:
    n = len(grid.x_m)
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Offline 3D cylindrical initial field for H2/N2/O2 CFD\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for x, y, z in zip(grid.x_m, grid.y_m, grid.z_m):
            f.write(f"{x:.8g} {y:.8g} {z:.8g}\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        for name, values in (
            ("x_H2", mole[:, 0]),
            ("x_N2", mole[:, 1]),
            ("x_O2", mole[:, 2]),
            ("Y_H2", mass[:, 0]),
            ("Y_N2", mass[:, 1]),
            ("Y_O2", mass[:, 2]),
        ):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in values:
                f.write(f"{v:.8g}\n")


def write_preview(path: Path, grid, mole: np.ndarray, case_id: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.2, 4.8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, len(grid.x_m) // 6000)
    sc = ax.scatter(
        grid.x_m[::stride] / 1000.0,
        grid.y_m[::stride],
        grid.z_m[::stride],
        c=mole[::stride, 0],
        cmap="viridis",
        s=2.0,
        alpha=0.72,
        vmin=0.0,
        vmax=1.0,
    )
    ax.view_init(elev=18, azim=-64)
    ax.set_proj_type("ortho")
    ax.set_title(f"3D initial H2 mole fraction: {case_id}", pad=16)
    ax.set_box_aspect((8.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, shrink=0.68, label="x_H2")
    fig.subplots_adjust(left=0.02, right=0.90, top=0.88, bottom=0.06)
    fig.savefig(path)
    plt.close(fig)


def write_openfoam_notes(case_dir: Path, case_id: str, window: str, metadata: dict) -> None:
    (case_dir / "README.md").write_text(
        f"""# Offline 3D multi-species CFD package

Source 1D case: `{case_id}`
Source window: `{window}`

This directory is a prepared offline 3D CFD package, not solved output.

Prepared files:

- `initial_3d_samples.csv`: cylindrical 3D sample points with mole fractions `x_*` and mass fractions `Y_*`.
- `initial_3d_points.vtk`: ParaView-readable point cloud for checking the mapped 3D initial field.
- `initial_3d_preview.png`: 3D preview of the H2 mole fraction field.
- `metrics.json`: field consistency and package metadata.
- `openfoam_solver_notes.md`: recommended OpenFOAM setup route.

Species convention:

The 1D model uses H2/N2/Air mole fractions.  For external CFD, Air is split into
O2 and N2:

```text
x_O2 = 0.21 x_Air
x_N2,total = x_N2 + 0.79 x_Air
Y_i = x_i M_i / sum_j(x_j M_j)
```

Current status: `{metadata["status"]}`.

The Streamlit app should show this package as an offline 3D CFD input package
until a real solver writes `outputs/cfd3d/<case>/metrics.json` and result
images/VTK files.
""",
        encoding="utf-8",
    )
    (case_dir / "openfoam_solver_notes.md").write_text(
        """# Recommended offline OpenFOAM route

Use this package as an initialization source for a real 3D circular-pipe CFD
case.  The intended solver family is a compressible transient flow solver with
multi-species transport, for example a `rhoReactingFoam`/`reactingFoam`-style
case with reactions disabled, or an equivalent custom solver.

Minimum physics to include:

- transient compressible or low-Mach variable-density momentum equation;
- gravity `g = (0 0 -9.81)`;
- H2/N2/O2 species transport;
- ideal-gas mixture density;
- no-slip pipe wall;
- closed inlet/outlet for shutdown stratification review;
- initial mass fractions from `initial_3d_samples.csv`.

Recommended workflow:

```bash
# 1. Build/import a circular local pipe mesh in OpenFOAM/Fluent.
# 2. Map initial_3d_samples.csv onto mesh cell centres.
# 3. Run 300 s shutdown stratification simulation.
# 4. Export:
#    outputs/cfd3d/<case>/
#      metrics.json
#      xz_slice_h2.png
#      cross_section_h2.png
#      velocity_magnitude.png
#      flammable_region.png
#      cfd_result.vtu
```

Do not rename this prepared package as solved CFD output until the external
solver has actually run.
""",
        encoding="utf-8",
    )


def build_case(case_id: str, window: str, nx: int, nr: int, ntheta: int, out_root: Path) -> Path:
    x_profile, profile = load_profile(case_id, window)
    length = float(x_profile.max() - x_profile.min())
    diameter = 1.2
    grid = make_cylindrical_sample_grid(length, diameter, nx=nx, nr=nr, ntheta=ntheta)
    mole, mass = map_profile_to_grid(x_profile, profile, grid)
    metrics = initial_field_metrics(mole, mass)
    out = out_root / f"{case_id}_rhoReacting3D"
    out.mkdir(parents=True, exist_ok=True)
    write_samples_csv(out / "initial_3d_samples.csv", grid, mole, mass)
    write_points_vtk(out / "initial_3d_points.vtk", grid, mole, mass)
    write_preview(out / "initial_3d_preview.png", grid, mole, case_id)
    metadata = {
        "status": "prepared_input_not_solved",
        "case_name": out.name,
        "source_case": case_id,
        "source_window": window,
        "target_solver_family": "OpenFOAM rhoReactingFoam/reactingFoam or Fluent species transport",
        "mesh_type": "external circular-pipe 3D mesh required for final solve",
        "local_length_m": length,
        "D_m": diameter,
        "sample_points": int(len(grid.x_m)),
        "nx": int(nx),
        "nr": int(nr),
        "ntheta": int(ntheta),
        **metrics,
        "note": "Prepared offline 3D multi-species CFD initial package; not solved CFD output.",
    }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_openfoam_notes(out, case_id, window, metadata)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="DN1200_u7_p010_stop60")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--window", default="full_isolation_zone_1200m")
    parser.add_argument("--nx", type=int, default=121)
    parser.add_argument("--nr", type=int, default=8)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--out-root", type=Path, default=ROOT / "openfoam_cases")
    args = parser.parse_args()

    case_root = ROOT / "cfd_cases"
    cases = sorted(p.name for p in case_root.iterdir() if p.is_dir()) if args.all else [args.case_id]
    rows = []
    for case_id in cases:
        out = build_case(case_id, args.window, args.nx, args.nr, args.ntheta, args.out_root)
        rows.append(json.loads((out / "metrics.json").read_text(encoding="utf-8")))
        print(out)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_root / "rhoReacting3D_prepared_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
