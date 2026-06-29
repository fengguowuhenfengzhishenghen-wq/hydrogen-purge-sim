"""Run a reduced local x-z stratification model from the 1D purge result.

This is a solved local stratification model, not a Fluent/OpenFOAM 3D CFD run.
It is meant to provide a transparent intermediate verification layer for the
shutdown Fr=0 risk that the 1D axial model cannot resolve.
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2purge.constants import g
from h2purge.plots import configure_matplotlib_cjk

configure_matplotlib_cjk()


COLORS = {
    "H2": np.array([37, 99, 235]) / 255.0,
    "N2": np.array([31, 157, 85]) / 255.0,
    "Air": np.array([229, 83, 61]) / 255.0,
}


def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.clip(X, 0.0, 1.0)
    s = X.sum(axis=-1, keepdims=True)
    return X / np.where(s <= 1.0e-14, 1.0, s)


def _vertical_advection_diffusion(phi: np.ndarray, w: np.ndarray, dz: float, dt: float, diffusivity: float) -> np.ndarray:
    """One explicit conservative vertical advection-diffusion step."""

    nx, nz = phi.shape
    face_flux = np.zeros((nx, nz + 1), dtype=float)
    # Positive w moves upward, i.e. toward larger z index.
    w_face = 0.5 * (w[:, :-1] + w[:, 1:])
    upwind = np.where(w_face >= 0.0, phi[:, :-1], phi[:, 1:])
    face_flux[:, 1:nz] = w_face * upwind

    diff_flux = np.zeros_like(face_flux)
    diff_flux[:, 1:nz] = diffusivity * (phi[:, 1:] - phi[:, :-1]) / dz
    rhs = -(face_flux[:, 1:] - face_flux[:, :-1]) / dz + (diff_flux[:, 1:] - diff_flux[:, :-1]) / dz
    return phi + dt * rhs


def run_model(case_in: Path, out_dir: Path, duration_s: float = 300.0) -> Path:
    profile_path = case_in / "windows" / "full_isolation_zone_1200m" / "initial_1d_profile.csv"
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    meta = json.loads((case_in / "case_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(profile_path)

    D = float(meta["D_m"])
    R = D / 2.0
    x = df["x_local_m"].to_numpy(dtype=float)
    # Downsample x for a fast transparent model.
    nx = 241
    x_new = np.linspace(float(x.min()), float(x.max()), nx)
    init = np.column_stack(
        [
            np.interp(x_new, x, df["x_H2_molfrac"].to_numpy(dtype=float)),
            np.interp(x_new, x, df["x_N2_molfrac"].to_numpy(dtype=float)),
            np.interp(x_new, x, df["x_Air_molfrac"].to_numpy(dtype=float)),
        ]
    )
    init = _normalize(init)

    nz = 81
    z = np.linspace(-R, R, nz)
    dz = z[1] - z[0]
    X = np.repeat(init[:, None, :], nz, axis=1)

    # Reduced-gravity scale for H2 against air. The coefficient is intentionally
    # conservative; it creates cross-section segregation without pretending to
    # resolve wall turbulence or full Navier-Stokes circulation.
    g_prime = 9.13
    w_ref = 0.5 * np.sqrt(g_prime * D)
    tau = 220.0
    diffusivity_z = 2.5e-4
    dt = 0.05
    steps = int(round(duration_s / dt))
    z_norm = z[None, :] / R

    for step in range(steps):
        t = step * dt
        ramp = 1.0 - np.exp(-t / tau)
        x_h2 = X[:, :, 0]
        x_air = X[:, :, 2]
        # H2 drifts upward; Air drifts downward; N2 is the reference buffer.
        w_h2 = ramp * w_ref * 0.012 * (1.0 - x_h2) * (1.0 - 0.15 * z_norm)
        w_air = -ramp * w_ref * 0.006 * (1.0 - x_air) * (1.0 + 0.15 * z_norm)
        w_n2 = -0.15 * (w_h2 * x_h2 + w_air * x_air)
        X[:, :, 0] = _vertical_advection_diffusion(X[:, :, 0], w_h2, dz, dt, diffusivity_z)
        X[:, :, 1] = _vertical_advection_diffusion(X[:, :, 1], w_n2, dz, dt, diffusivity_z)
        X[:, :, 2] = _vertical_advection_diffusion(X[:, :, 2], w_air, dz, dt, diffusivity_z)
        X = _normalize(X)

    x_h2 = X[:, :, 0]
    x_air = X[:, :, 2]
    flammable = (x_h2 >= 0.04) & (x_h2 <= 0.75) & (0.21 * x_air >= 0.05)
    top = z >= 0.0
    bottom = z < 0.0
    top_bottom_delta = float(x_h2[:, top].mean() - x_h2[:, bottom].mean())
    top_h2_max = float(x_h2[:, top].max())
    flammable_area_ratio = float(flammable.mean())
    flammable_volume_ratio = flammable_area_ratio

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_figures(out_dir, x_new, z, X, flammable, case_in, duration_s)
    _write_vtk(out_dir / "cfd_result.vtk", x_new, z, X)
    metrics = {
        "solver": "Python reduced 2D x-z stratification model (not Fluent/OpenFOAM 3D CFD)",
        "case_name": out_dir.name,
        "source_case": case_in.name,
        "D_m": D,
        "local_length_m": float(x_new[-1] - x_new[0]),
        "stop_duration_s": float(duration_s),
        "mesh_cells": int(nx * nz),
        "top_bottom_h2_delta": top_bottom_delta,
        "top_h2_max": top_h2_max,
        "flammable_area_ratio": flammable_area_ratio,
        "flammable_volume_ratio": flammable_volume_ratio,
        "compute_time_s": None,
        "note": "Reduced local stratification result generated from the 1D stop profile. It is a solved x-z verification model, not a full 3D CFD replacement.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def _write_figures(out_dir: Path, x: np.ndarray, z: np.ndarray, X: np.ndarray, flammable: np.ndarray, case_in: Path, duration_s: float) -> None:
    import matplotlib.pyplot as plt

    x_km = (x + _window_abs_start(case_in)) / 1000.0
    x_h2 = X[:, :, 0].T
    x_n2 = X[:, :, 1].T
    x_air = X[:, :, 2].T
    rgb = np.zeros((len(z), len(x), 3), dtype=float)
    rgb += x_h2[:, :, None] * COLORS["H2"]
    rgb += x_n2[:, :, None] * COLORS["N2"]
    rgb += x_air[:, :, None] * COLORS["Air"]

    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=180)
    ax.imshow(rgb, origin="lower", extent=[x_km[0], x_km[-1], z[0], z[-1]], aspect="auto")
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.7)
    ax.set_xlabel("管道位置 x (km)")
    ax.set_ylabel("竖向位置 z (m)")
    ax.set_title(f"局部 x-z 纵剖面：H2/N2/Air 分层色场，停输 {duration_s:.0f} s")
    fig.tight_layout()
    fig.savefig(out_dir / "xz_slice_h2.png")
    plt.close(fig)

    mean_h2 = X[:, :, 0].mean(axis=1)
    mid = int(np.argmin(np.abs(mean_h2 - 0.5)))
    yy, zz = np.meshgrid(np.linspace(-1.0, 1.0, 90), np.linspace(-1.0, 1.0, 90))
    rr = yy**2 + zz**2 <= 1.0
    h2_col = np.interp(zz, z / max(abs(z[0]), abs(z[-1])), X[mid, :, 0])
    h2_col = np.where(rr, h2_col, np.nan)
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=180)
    im = ax.imshow(h2_col, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=1, cmap="viridis")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#64748b", lw=1.2))
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.6)
    ax.set_xlabel("横向位置 y/R")
    ax.set_ylabel("竖向位置 z/R")
    ax.set_title(f"圆截面 H2 摩尔分数云图，x≈{x_km[mid]:.2f} km")
    fig.colorbar(im, ax=ax, label="x_H2")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_section_h2.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=180)
    ax.imshow(flammable.T, origin="lower", extent=[x_km[0], x_km[-1], z[0], z[-1]], aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.axhline(0.0, color="#1f2937", lw=0.8, alpha=0.7)
    ax.set_xlabel("管道位置 x (km)")
    ax.set_ylabel("竖向位置 z (m)")
    ax.set_title("保守可燃风险区：0.04<=x_H2<=0.75 且 x_O2>=0.05")
    fig.tight_layout()
    fig.savefig(out_dir / "flammable_region.png")
    plt.close(fig)


def _window_abs_start(case_in: Path) -> float:
    meta = json.loads((case_in / "case_metadata.json").read_text(encoding="utf-8"))
    for window in meta["windows"]:
        if window["name"] == "full_isolation_zone_1200m":
            return float(window["start_m"])
    return 0.0


def _write_vtk(path: Path, x: np.ndarray, z: np.ndarray, X: np.ndarray) -> None:
    nx, nz, _ = X.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Reduced x-z stratification field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {nz} 1\n")
        f.write(f"ORIGIN {x[0]:.8g} {z[0]:.8g} 0\n")
        f.write(f"SPACING {(x[1]-x[0]):.8g} {(z[1]-z[0]):.8g} 1\n")
        f.write(f"POINT_DATA {nx*nz}\n")
        for name, col in (("x_H2", 0), ("x_N2", 1), ("x_Air", 2)):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for iz in range(nz):
                for ix in range(nx):
                    f.write(f"{X[ix, iz, col]:.8g}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="DN1200_u7_p010_stop60")
    parser.add_argument("--all", action="store_true", help="Run all directories below cfd_cases.")
    parser.add_argument("--duration-s", type=float, default=300.0)
    args = parser.parse_args()

    if args.all:
        case_dirs = sorted(p for p in (ROOT / "cfd_cases").iterdir() if p.is_dir())
    else:
        case_dirs = [ROOT / "cfd_cases" / args.case_id]

    for case_in in case_dirs:
        out_dir = ROOT / "outputs" / "cfd3d" / f"{case_in.name}_reduced2d"
        path = run_model(case_in, out_dir, duration_s=args.duration_s)
        print(path)


if __name__ == "__main__":
    main()
