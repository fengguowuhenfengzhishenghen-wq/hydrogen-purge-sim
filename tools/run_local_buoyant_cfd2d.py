"""Run local 2D buoyant multi-species CFD checks for shutdown cases."""

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

from h2purge.local_cfd2d import LocalCFD2DConfig, LocalCFD2DResult, run_local_cfd2d
from h2purge.plots import configure_matplotlib_cjk
from h2purge.safety import flammable_mask

configure_matplotlib_cjk()

COLORS = {
    "H2": np.array([37, 99, 235]) / 255.0,
    "N2": np.array([31, 157, 85]) / 255.0,
    "Air": np.array([229, 83, 61]) / 255.0,
}


def load_profile(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "windows" / "full_isolation_zone_1200m" / "initial_1d_profile.csv"
    df = pd.read_csv(path)
    x = df["x_local_m"].to_numpy(dtype=float)
    X = df[["x_H2_molfrac", "x_N2_molfrac", "x_Air_molfrac"]].to_numpy(dtype=float)
    return x, X


def write_vtk(path: Path, result: LocalCFD2DResult) -> None:
    x, z, X = result.x_m, result.z_m, result.mole
    nx, nz, _ = X.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Local 2D buoyant multi-species CFD field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {nz} 1\n")
        f.write(f"ORIGIN {x[0]:.8g} {z[0]:.8g} 0\n")
        f.write(f"SPACING {(x[1] - x[0]):.8g} {(z[1] - z[0]):.8g} 1\n")
        f.write(f"POINT_DATA {nx * nz}\n")
        for name, field in (
            ("x_H2", X[:, :, 0]),
            ("x_N2", X[:, :, 1]),
            ("x_Air", X[:, :, 2]),
            ("u_mps", result.u_mps),
            ("w_mps", result.w_mps),
        ):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for iz in range(nz):
                for ix in range(nx):
                    f.write(f"{field[ix, iz]:.8g}\n")


def write_figures(out_dir: Path, result: LocalCFD2DResult, source_case: str) -> None:
    import matplotlib.pyplot as plt

    x_km = result.x_m / 1000.0
    z = result.z_m
    X = result.mole
    rgb = (
        X[:, :, 0].T[:, :, None] * COLORS["H2"]
        + X[:, :, 1].T[:, :, None] * COLORS["N2"]
        + X[:, :, 2].T[:, :, None] * COLORS["Air"]
    )
    speed = np.sqrt(result.u_mps * result.u_mps + result.w_mps * result.w_mps).T
    flammable = flammable_mask(X[:, :, 0], X[:, :, 2]).T

    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=180)
    ax.imshow(rgb, origin="lower", extent=[x_km[0], x_km[-1], z[0], z[-1]], aspect="auto")
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.7)
    ax.set_xlabel("x position (km)")
    ax.set_ylabel("z position (m)")
    ax.set_title(f"Local 2D buoyant CFD mole-fraction field: {source_case}")
    fig.tight_layout()
    fig.savefig(out_dir / "xz_slice_h2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=180)
    im = ax.imshow(speed, origin="lower", extent=[x_km[0], x_km[-1], z[0], z[-1]], aspect="auto", cmap="magma")
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.7)
    ax.set_xlabel("x position (km)")
    ax.set_ylabel("z position (m)")
    ax.set_title("Velocity magnitude from buoyancy-driven circulation")
    fig.colorbar(im, ax=ax, label="speed (m/s)")
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_magnitude.png")
    plt.close(fig)

    mean_h2 = X[:, :, 0].mean(axis=1)
    mid = int(np.argmin(np.abs(mean_h2 - 0.5)))
    yy, zz = np.meshgrid(np.linspace(-1.0, 1.0, 100), np.linspace(-1.0, 1.0, 100))
    rr = yy * yy + zz * zz <= 1.0
    h2_col = np.interp(zz, z / max(abs(z[0]), abs(z[-1])), X[mid, :, 0])
    h2_col = np.where(rr, h2_col, np.nan)
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=180)
    im = ax.imshow(h2_col, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=1, cmap="viridis")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#64748b", lw=1.2))
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.6)
    ax.set_xlabel("y/R")
    ax.set_ylabel("z/R")
    ax.set_title("Circular-section H2 mole-fraction map")
    fig.colorbar(im, ax=ax, label="x_H2")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_section_h2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=180)
    ax.imshow(flammable, origin="lower", extent=[x_km[0], x_km[-1], z[0], z[-1]], aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.7)
    ax.set_xlabel("x position (km)")
    ax.set_ylabel("z position (m)")
    ax.set_title("Conservative flammable-risk cells")
    fig.tight_layout()
    fig.savefig(out_dir / "flammable_region.png")
    plt.close(fig)


def run_case(case_dir: Path, out_root: Path, cfg: LocalCFD2DConfig) -> Path:
    x, X = load_profile(case_dir)
    result = run_local_cfd2d(x, X, cfg)
    out_dir = out_root / f"{case_dir.name}_buoyant2d"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_figures(out_dir, result, case_dir.name)
    write_vtk(out_dir / "cfd_result.vtk", result)
    metrics = dict(result.metrics)
    metrics.update(
        {
            "case_name": out_dir.name,
            "source_case": case_dir.name,
            "D_m": cfg.diameter_m,
            "note": (
                "Solved local 2D low-Mach buoyant multi-species CFD check. "
                "It includes momentum, buoyancy, and H2/N2/Air species transport, "
                "but is not a full 3D compressible pipe CFD."
            ),
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="DN1200_u7_p010_stop60")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--nx", type=int, default=121)
    parser.add_argument("--nz", type=int, default=33)
    args = parser.parse_args()

    cfg = LocalCFD2DConfig(nx=args.nx, nz=args.nz, duration_s=args.duration_s)
    case_root = ROOT / "cfd_cases"
    cases = sorted(p for p in case_root.iterdir() if p.is_dir()) if args.all else [case_root / args.case_id]
    rows = []
    for case_dir in cases:
        out_dir = run_case(case_dir, ROOT / "outputs" / "cfd3d", cfg)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(metrics)
        print(out_dir)
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "cfd3d" / "buoyant2d_cfd_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
