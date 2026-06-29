"""Build external CFD input data from the verified 1D purge model.

This script does not create CFD results. It prepares the concentration fields
and metadata needed to initialize a separate Fluent/OpenFOAM local CFD case.
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

from h2purge.config import SimulationParams
from h2purge.constants import L_PIPE
from h2purge.metrics import front_position
from h2purge.plots import configure_matplotlib_cjk
from h2purge.solver_fvm import run_simulation

configure_matplotlib_cjk()


def _interp_profile(x_src: np.ndarray, profile: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    out = np.column_stack(
        [
            np.interp(x_new, x_src, profile[:, 0]),
            np.interp(x_new, x_src, profile[:, 1]),
            np.interp(x_new, x_src, profile[:, 2]),
        ]
    )
    out = np.clip(out, 0.0, 1.0)
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s <= 1.0e-15, 1.0, s)


def _rising_front_position(x_species: np.ndarray, x_grid: np.ndarray, level: float = 0.5) -> float:
    """Return the leftmost position where a left-low profile rises through level."""

    x_species = np.asarray(x_species, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    idxs = np.where((x_species[:-1] < level) & (x_species[1:] >= level))[0]
    if len(idxs) == 0:
        idx = int(np.argmin(np.abs(x_species - level)))
        return float(x_grid[idx])
    i = int(idxs[0])
    y0, y1 = x_species[i], x_species[i + 1]
    if abs(y1 - y0) < 1.0e-15:
        return float(x_grid[i])
    frac = (level - y0) / (y1 - y0)
    return float(x_grid[i] + frac * (x_grid[i + 1] - x_grid[i]))


def _write_profile_csv(path: Path, x_abs: np.ndarray, x_local: np.ndarray, profile: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "x_abs_m": x_abs,
            "x_local_m": x_local,
            "x_H2_molfrac": profile[:, 0],
            "x_N2_molfrac": profile[:, 1],
            "x_Air_molfrac": profile[:, 2],
            "x_O2_molfrac": 0.21 * profile[:, 2],
            "x_inert_molfrac": profile[:, 1] + 0.79 * profile[:, 2],
        }
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_point_vtk(path: Path, x_local: np.ndarray, profile: np.ndarray, diameter: float) -> None:
    radius = diameter / 2.0
    yz = np.linspace(-radius, radius, 17)
    points: list[tuple[float, float, float, int]] = []
    for i, x in enumerate(x_local):
        for y in yz:
            for z in yz:
                if y * y + z * z <= radius * radius + 1.0e-12:
                    points.append((float(x), float(y), float(z), i))

    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Initial 1D mole-fraction field mapped to a cylindrical local CFD window\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for x, y, z, _ in points:
            f.write(f"{x:.8g} {y:.8g} {z:.8g}\n")
        f.write(f"VERTICES {len(points)} {2 * len(points)}\n")
        for idx in range(len(points)):
            f.write(f"1 {idx}\n")
        f.write(f"POINT_DATA {len(points)}\n")
        for name, col in (("x_H2", 0), ("x_N2", 1), ("x_Air", 2)):
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for *_, i in points:
                f.write(f"{profile[i, col]:.8g}\n")


def _write_window_preview(path: Path, x_local: np.ndarray, profile: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=180)
    ax.plot(x_local, profile[:, 0], label="H2", color="#2563eb", linewidth=2.0)
    ax.plot(x_local, profile[:, 1], label="N2", color="#1f9d55", linewidth=2.0)
    ax.plot(x_local, profile[:, 2], label="Air", color="#e5533d", linewidth=2.0)
    ax.set_xlabel("局部窗口位置 x_local (m)")
    ax.set_ylabel("摩尔分数 / 体积分数")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _make_window(
    case_dir: Path,
    name: str,
    center_m: float,
    length_m: float,
    x_grid: np.ndarray,
    profile: np.ndarray,
    diameter: float,
) -> dict:
    start = max(0.0, center_m - length_m / 2.0)
    end = min(L_PIPE, center_m + length_m / 2.0)
    x_abs = np.linspace(start, end, int(round(end - start)) + 1)
    x_local = x_abs - start
    local = _interp_profile(x_grid, profile, x_abs)

    window_dir = case_dir / "windows" / name
    window_dir.mkdir(parents=True, exist_ok=True)
    _write_profile_csv(window_dir / "initial_1d_profile.csv", x_abs, x_local, local)
    _write_point_vtk(window_dir / "initial_field_points.vtk", x_local, local, diameter)
    _write_window_preview(window_dir / "initial_profile_preview.png", x_local, local, name)

    return {
        "name": name,
        "center_m": float(center_m),
        "start_m": float(start),
        "end_m": float(end),
        "length_m": float(end - start),
        "profile_csv": str((window_dir / "initial_1d_profile.csv").relative_to(case_dir)),
        "initial_point_vtk": str((window_dir / "initial_field_points.vtk").relative_to(case_dir)),
        "initial_preview_png": str((window_dir / "initial_profile_preview.png").relative_to(case_dir)),
        "mean_x_H2": float(local[:, 0].mean()),
        "mean_x_N2": float(local[:, 1].mean()),
        "mean_x_Air": float(local[:, 2].mean()),
        "max_x_H2": float(local[:, 0].max()),
        "max_x_N2": float(local[:, 1].max()),
        "max_x_Air": float(local[:, 2].max()),
    }


def _write_readme(case_dir: Path, metadata: dict) -> None:
    (case_dir / "README.md").write_text(
        "\n".join(
            [
                "# 外部 CFD 输入数据包",
                "",
                "本目录由已经验证的一维 H2/N2/Air 置换模型生成，用于给 Fluent、OpenFOAM 等外部 CFD 工程提供初始浓度场。",
                "注意：这里不是 CFD 求解结果，不能把这些文件当成 CFD 云图导入 Streamlit 的“外部三维 CFD 复核结果”区域。",
                "",
                "文件含义：",
                "",
                "- `case_metadata.json`：工况、停输位置、前缘位置和局部窗口信息。",
                "- `full_pipe_stop_profile.csv`：60% 中断瞬间 12 km 全管一维摩尔分数场。",
                "- `windows/*/initial_1d_profile.csv`：局部 CFD 窗口的一维轴向初始浓度。",
                "- `windows/*/initial_field_points.vtk`：把一维初始场映射到圆截面点云，方便 ParaView 检查或二次映射。",
                "- `windows/*/initial_profile_preview.png`：局部窗口初始浓度预览图，不是 CFD 结果图。",
                "",
                "建议的 CFD 复核目标：",
                "",
                "验证一维模型无法解析的停输截面分层风险，包括 H2 上浮富集、Air/N2 下沉、",
                "以及停输 300 s 后横截面内是否形成局部可燃区域。",
                "",
                "只有当外部求解器真正输出 `metrics.json`、`xz_slice_h2.png`、`cross_section_h2.png`、",
                "`flammable_region.png` 或 VTK/VTU 结果文件后，才能称为外部 CFD 复核结果。",
                "",
                f"工况：{metadata['case_id']}",
                f"中断位置：{metadata['stop_position_fraction']:.0%} L",
                f"停输时刻：{metadata['stop_time_s']:.3f} s",
                f"H2 前缘：{metadata['h2_front_m']:.3f} m",
                f"N2/Air 前缘：{metadata['n2_air_front_m']:.3f} m",
            ]
        ),
        encoding="utf-8",
    )


def build_case(case_id: str, stop_fraction: float, out_root: Path) -> Path:
    D = 1.2
    u = 7.0
    stop_time = stop_fraction * L_PIPE / u
    params = SimulationParams(
        D=D,
        u_nominal=u,
        p_back_abs=0.10e6,
        roughness=0.05e-3,
        dx=10.0,
        t_end=stop_time,
        output_times=[0.0, stop_time],
        pressure_mode="friction_profile",
    )
    result = run_simulation(params)
    profile = result.profiles[-1]
    x_grid = result.x_grid
    metrics = result.metrics[-1]
    h2_front = float(metrics["h2_front_m"])
    n2_air_front = float(_rising_front_position(profile[:, 2], x_grid, level=0.5))

    case_dir = out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_profile_csv(case_dir / "full_pipe_stop_profile.csv", x_grid, x_grid, profile)

    windows = [
        _make_window(case_dir, "h2_n2_front_200m", h2_front, 200.0, x_grid, profile, D),
        _make_window(case_dir, "n2_air_front_200m", n2_air_front, 200.0, x_grid, profile, D),
        _make_window(case_dir, "full_isolation_zone_1200m", 0.5 * (h2_front + n2_air_front), 1200.0, x_grid, profile, D),
    ]

    metadata = {
        "status": "prepared_input_not_solved",
        "case_id": case_id,
        "source_model": "1D H2/N2/Air mole-fraction FVM",
        "D_m": D,
        "u_mps": u,
        "p_back_MPa_abs": 0.10,
        "dx_m": 10.0,
        "beta_K": 0.5,
        "stop_position_fraction": stop_fraction,
        "stop_position_m": stop_fraction * L_PIPE,
        "stop_time_s": stop_time,
        "h2_front_m": h2_front,
        "n2_air_front_m": n2_air_front,
        "mixed_length_m": float(metrics["mixed_length_m"]),
        "effective_n2_length_m": float(metrics["effective_n2_length_m"]),
        "flammable_length_m": float(metrics["flammable_length_m"]),
        "Fr_before_shutdown": float(metrics["Fr"]),
        "Fr_shutdown": 0.0,
        "windows": windows,
        "important_note": "This is an external CFD input package, not solved CFD output.",
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(case_dir, metadata)
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="DN1200_u7_p010_stop60")
    parser.add_argument("--stop-fraction", type=float, default=0.60)
    parser.add_argument("--out-root", type=Path, default=ROOT / "cfd_cases")
    args = parser.parse_args()
    case_dir = build_case(args.case_id, args.stop_fraction, args.out_root)
    print(case_dir)


if __name__ == "__main__":
    main()
