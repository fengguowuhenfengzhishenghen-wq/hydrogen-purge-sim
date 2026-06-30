"""Post-process a solved OpenFOAM 3D pipe case for Streamlit display."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M_H2 = 2.016e-3
M_N2 = 28.0134e-3
M_O2 = 31.998e-3


def latest_numeric_time(case_dir: Path) -> Path:
    times = []
    for path in case_dir.iterdir():
        if path.is_dir():
            try:
                times.append((float(path.name), path))
            except ValueError:
                continue
    if not times:
        raise FileNotFoundError(f"No numeric OpenFOAM time directories in {case_dir}")
    return sorted(times, key=lambda item: item[0])[-1][1]


def parse_scalar_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", text, re.S)
    if not match:
        uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)", text)
        if uniform:
            return np.array([float(uniform.group(1))], dtype=float)
        raise ValueError(f"Could not parse scalar field {path}")
    n = int(match.group(1))
    values = np.fromstring(match.group(2), sep=" ", dtype=float)
    if len(values) != n:
        raise ValueError(f"{path} expected {n} values, parsed {len(values)}")
    return values


def parse_vector_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", text, re.S)
    if not match:
        raise ValueError(f"Could not parse vector field {path}")
    n = int(match.group(1))
    rows = re.findall(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)", match.group(2))
    if len(rows) != n:
        raise ValueError(f"{path} expected {n} vectors, parsed {len(rows)}")
    return np.asarray(rows, dtype=float)


def infer_shape(case_dir: Path, n_cells: int) -> tuple[int, int, int, float]:
    init = pd.read_csv(case_dir / "initial_mass_fraction_cells.csv")
    x_unique = np.unique(np.round(init["cell_x_m"].to_numpy(dtype=float), 10))
    nx = len(x_unique)
    cross = n_cells // nx
    side = int(round(math.sqrt(cross)))
    if nx * side * side != n_cells:
        raise ValueError(f"Cannot infer structured shape from {n_cells} cells and {nx} x positions")
    length = float(x_unique.max() - x_unique.min() + (x_unique[1] - x_unique[0] if nx > 1 else 0.0))
    return nx, side, side, length


def reshape(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nz, ny, nx)).transpose(2, 1, 0)


def mass_to_mole(y_h2: np.ndarray, y_o2: np.ndarray, y_n2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = np.maximum(y_h2 / M_H2 + y_o2 / M_O2 + y_n2 / M_N2, 1.0e-30)
    return y_h2 / M_H2 / denom, y_o2 / M_O2 / denom, y_n2 / M_N2 / denom


def parse_log_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    out: dict[str, float] = {}
    for name in ("H2", "O2", "Ux", "Uy", "Uz", "h", "p"):
        matches = re.findall(
            rf"Solving for {re.escape(name)}, Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations ([0-9]+)",
            text,
        )
        if matches:
            init, final, iters = matches[-1]
            out[f"{name}_initial_residual"] = float(init)
            out[f"{name}_final_residual"] = float(final)
            out[f"{name}_iterations"] = float(iters)
    cont = re.findall(
        r"time step continuity errors : sum local = ([^,]+), global = ([^,]+), cumulative = ([^\n]+)",
        text,
    )
    if cont:
        local, global_, cumulative = cont[-1]
        out["continuity_sum_local"] = float(local)
        out["continuity_global"] = float(global_)
        out["continuity_cumulative"] = float(cumulative)
    return out


def save_field_plot(
    path: Path,
    x: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    title: str,
    cbar: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=180)
    im = ax.pcolormesh(x, z, field.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, label=cbar)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_cross_section(path: Path, y: np.ndarray, z: np.ndarray, field: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=180)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    masked = np.where(yy * yy + zz * zz <= 0.6 * 0.6, field, np.nan)
    im = ax.pcolormesh(y, z, masked.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    circle = plt.Circle((0.0, 0.0), 0.6, fill=False, color="#334155", lw=1.2)
    ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, label="x_H2")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def find_latest_vtu(case_dir: Path) -> Path | None:
    vtk = case_dir / "VTK"
    if not vtk.exists():
        return None
    files = sorted(vtk.glob("*/internal.vtu"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def postprocess(case_dir: Path, out_dir: Path) -> dict[str, float | int | str]:
    time_dir = latest_numeric_time(case_dir)
    y_h2_raw = parse_scalar_field(time_dir / "H2")
    y_o2_raw = parse_scalar_field(time_dir / "O2")
    y_n2_raw = parse_scalar_field(time_dir / "N2")
    p_raw = parse_scalar_field(time_dir / "p")
    u_raw = parse_vector_field(time_dir / "U")
    nx, ny, nz, length = infer_shape(case_dir, len(y_h2_raw))
    y_h2 = reshape(y_h2_raw, nx, ny, nz)
    y_o2 = reshape(y_o2_raw, nx, ny, nz)
    y_n2 = reshape(y_n2_raw, nx, ny, nz)
    p = reshape(p_raw, nx, ny, nz)
    u = u_raw.reshape((nz, ny, nx, 3)).transpose(2, 1, 0, 3)
    x_h2, x_o2, x_n2 = mass_to_mole(y_h2, y_o2, y_n2)
    speed = np.linalg.norm(u, axis=-1)

    x = (np.arange(nx) + 0.5) * length / nx
    y = np.linspace(-0.6, 0.6, ny)
    z = np.linspace(-0.6, 0.6, nz)
    mid_y = ny // 2
    mid_x = nx // 2
    top = z[None, None, :] >= 0.0
    bottom = z[None, None, :] < 0.0
    flammable = (x_h2 >= 0.04) & (x_h2 <= 0.75) & (x_o2 >= 0.05)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_field_plot(out_dir / "xz_slice_h2.png", x, z, x_h2[:, mid_y, :], "OpenFOAM x-z slice: H2 mole fraction", "x_H2")
    save_cross_section(out_dir / "cross_section_h2.png", y, z, x_h2[mid_x, :, :], "OpenFOAM cross-section: H2 mole fraction")
    save_field_plot(out_dir / "velocity_magnitude.png", x, z, speed[:, mid_y, :], "OpenFOAM x-z slice: velocity magnitude", "|U| (m/s)", cmap="magma")
    save_field_plot(
        out_dir / "flammable_region.png",
        x,
        z,
        flammable[:, mid_y, :].astype(float),
        "OpenFOAM x-z slice: flammable mask",
        "mask",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
    )

    vtu = find_latest_vtu(case_dir)
    if vtu is not None:
        shutil.copy2(vtu, out_dir / "cfd_result.vtu")

    metrics: dict[str, float | int | str] = {
        "solver": "OpenFOAM rhoReactingFoam",
        "case_name": out_dir.name,
        "source_case": case_dir.name,
        "latest_time_s": float(time_dir.name),
        "mesh_cells": int(len(y_h2_raw)),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "local_length_m": float(length),
        "D_m": 1.2,
        "species": "H2/O2/N2 mass fractions; Air interpreted from O2 for risk criterion",
        "top_bottom_h2_delta": float(x_h2[:, :, top[0, 0, :]].mean() - x_h2[:, :, bottom[0, 0, :]].mean()),
        "top_bottom_o2_delta": float(x_o2[:, :, top[0, 0, :]].mean() - x_o2[:, :, bottom[0, 0, :]].mean()),
        "top_h2_max": float(x_h2[:, :, top[0, 0, :]].max()),
        "flammable_volume_ratio": float(flammable.mean()),
        "max_speed_mps": float(speed.max()),
        "mean_speed_mps": float(speed.mean()),
        "p_min_Pa": float(p.min()),
        "p_max_Pa": float(p.max()),
        "mass_fraction_sum_max_error": float(np.max(np.abs(y_h2 + y_o2 + y_n2 - 1.0))),
        "note": "Imported solved OpenFOAM 3D case. Full 300 s industrial CFD remains too expensive for Streamlit; this is a local shutdown smoke solve.",
    }
    metrics.update(parse_log_metrics(case_dir / "log.rhoReactingFoam"))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=ROOT / "openfoam_cases" / "DN1200_u7_p010_stop60_rhoReactingPipe3D")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cfd3d" / "DN1200_u7_p010_stop60_openfoam3d")
    args = parser.parse_args()
    metrics = postprocess(args.case_dir, args.out_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
