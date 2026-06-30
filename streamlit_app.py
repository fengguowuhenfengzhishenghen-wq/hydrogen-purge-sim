"""Streamlit UI for the hydrogen purge model.

The source is kept ASCII-safe. Chinese UI strings are written with unicode
escapes to avoid Windows console encoding damage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2purge.cfd_import import (
    find_cfd_volume_file,
    list_cfd_cases,
    list_prepared_3d_cases,
    load_cfd_images,
    load_cfd_metrics,
)
from h2purge.config import SimulationParams
from h2purge.solver_fvm import run_simulation
from h2purge.ui_pipe_animation import SPECIES_COLORS as COL, render_pipe


def font_setup() -> None:
    for name in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        p = Path(name)
        if p.exists():
            fm.fontManager.addfont(str(p))
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [fm.FontProperties(fname=str(p)).get_name(), "DejaVu Sans"]
            break
    mpl.rcParams["axes.unicode_minus"] = False


font_setup()

C = {
    "title": "\u8f93\u6c22\u7ba1\u9053\u6295\u4ea7\u6df7\u6c14\u89c4\u5f8b\u6a21\u62df",
    "note": "\u7ed3\u679c\u6f14\u793a\u754c\u9762\uff1b\u6a21\u578b\u5047\u8bbe\u548c\u9002\u7528\u8fb9\u754c\u89c1\u62a5\u544a\u3002",
    "input": "\u8f93\u5165\u53c2\u6570",
    "diam": "\u7ba1\u5f84 D (m)",
    "vel": "\u7f6e\u6362\u901f\u5ea6 u (m/s)",
    "back": "\u51fa\u53e3\u80cc\u538b p_back (MPa, \u7edd\u538b)",
    "dx": "\u7f51\u683c dx (m)",
    "pressure": "\u538b\u529b\u6a21\u5f0f",
    "fric": "\u6469\u963b\u538b\u964d\u5256\u9762",
    "refp": "\u5168\u7ba1\u7b49\u538b",
    "run": "\u8fd0\u884c\u6a21\u62df",
    "params": "\u53c2\u6570\u8868",
    "time": "\u52a8\u6001\u663e\u793a\u65f6\u523b",
    "pipe": "\u52a8\u6001\u7f6e\u6362\u6a21\u62df",
    "local": "\u5c40\u90e8\u8f74\u5411\u653e\u5927\uff08\u4e00\u7ef4\u6a21\u578b\u7ed3\u679c\uff09",
    "profiles": "\u6d53\u5ea6\u66f2\u7ebf\u4e0e\u65f6\u5e8f\u6307\u6807",
    "metrics": "\u65f6\u5e8f\u6307\u6807",
    "task2": "\u4efb\u52a12\u4e2d\u65ad\u5de5\u51b5\u8bf4\u660e",
    "cfd_input": "\u5206\u5c42\u590d\u6838\u8f93\u5165\u5305\uff08\u6765\u81ea\u4e00\u7ef4\u6a21\u578b\uff09",
    "cfd_result": "\u5c40\u90e8\u5206\u5c42\u590d\u6838\u7ed3\u679c",
    "no_cfd": "\u672a\u5bfc\u5165\u5c40\u90e8\u5206\u5c42\u590d\u6838\u7ed3\u679c\u3002\u5f53\u524d\u4e0d\u663e\u793a\u590d\u6838\u56fe\uff0c\u907f\u514d\u5c06\u4e00\u7ef4\u53ef\u89c6\u5316\u8bef\u8ba4\u4e3a\u590d\u6838\u7ed3\u679c\u3002",
    "mol": "Mole fraction / volume fraction",
    "xpos": "Pipe position x (km)",
    "length": "Length (m)",
    "download": "\u4e0b\u8f7d\u5f53\u524d\u7b97\u4f8b CSV",
}

PRESSURE_MODES = {C["refp"]: "reference_pressure", C["fric"]: "friction_profile"}


@st.cache_data(show_spinner=False)
def run_cached(D, u, p_back, dx, beta, pressure_mode, t_factor):
    t_end = float(t_factor) * 12000.0 / max(float(u), 1.0e-9)
    params = SimulationParams(
        D=float(D),
        u_nominal=float(u),
        p_back_abs=float(p_back) * 1e6,
        dx=float(dx),
        beta_K=float(beta),
        t_end=t_end,
        output_times=np.linspace(0.0, t_end, 81),
        pressure_mode=pressure_mode,
        roughness=0.05e-3,
    )
    return run_simulation(params)


def render_local_zoom(result, idx):
    prof = result.profiles[idx]
    metric = result.metrics[idx]
    center = metric.get("h2_front_m", result.params.L * 0.5)
    if not np.isfinite(center):
        center = result.params.L * 0.5
    mask = (result.x_grid >= center - 900) & (result.x_grid <= center + 900)
    x = result.x_grid[mask] / 1000
    y = prof[mask]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y[:, 0], color=COL["H2"], label="H2", lw=2)
    ax.plot(x, y[:, 1], color=COL["N2"], label="N2", lw=2)
    ax.plot(x, y[:, 2], color=COL["Air"], label="Air", lw=2)
    ax.set_xlabel(C["xpos"])
    ax.set_ylabel(C["mol"])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3)
    st.pyplot(fig)
    st.caption("\u8be5\u56fe\u6765\u81ea\u4e00\u7ef4\u8f74\u5411\u5bf9\u6d41-\u5f25\u6563\u6a21\u578b\uff0c\u7528\u4e8e\u653e\u5927\u67e5\u770b H2/N2/Air \u754c\u9762\uff0c\u4e0d\u662f CFD \u7ed3\u679c\u3002")


def render_curves(result, metrics_df):
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for idx in np.linspace(0, len(result.times) - 1, 5, dtype=int):
            ax.plot(result.x_grid / 1000, result.profiles[idx, :, 0], label=f"H2 {result.times[idx]:.0f}s")
            ax.plot(result.x_grid / 1000, result.profiles[idx, :, 1], "--", label=f"N2 {result.times[idx]:.0f}s")
            ax.plot(result.x_grid / 1000, result.profiles[idx, :, 2], ":", label=f"Air {result.times[idx]:.0f}s")
        ax.set_xlabel(C["xpos"])
        ax.set_ylabel(C["mol"])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(metrics_df["time_s"] / 60, metrics_df["mixed_length_m"], label="Mixed length")
        ax.plot(metrics_df["time_s"] / 60, metrics_df["flammable_length_m"], label="Flammable length")
        ax.plot(metrics_df["time_s"] / 60, metrics_df["effective_n2_length_m"], label="Effective N2 length")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(C["length"])
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)


def render_task2():
    st.subheader(C["task2"])
    path = ROOT / "outputs" / "task2" / "interrupt_summary.csv"
    if not path.exists():
        st.warning("\u672a\u627e\u5230 outputs/task2/interrupt_summary.csv\uff0c\u8bf7\u5148\u8fd0\u884c python run_task2_interrupt.py")
        return
    df = pd.read_csv(path)
    view = pd.DataFrame({
        "\u4e2d\u65ad\u4f4d\u7f6e": [f"{v:.0%}" for v in df["interrupt_position_fraction"]],
        "\u505c\u8f93\u65f6\u95f4 (s)": df["stop_time_s"].round(1),
        "\u505c\u524d\u6df7\u6c14\u6bb5 (m)": df["mixed_length_before_m"].round(0).astype(int),
        "\u505c\u540e\u6df7\u6c14\u6bb5 (m)": df["mixed_length_after_300s_m"].round(0).astype(int),
        "\u8f74\u5411\u589e\u957f (m)": df["mixed_length_growth_m"].round(0).astype(int),
        "\u505c\u540e\u6709\u6548 N2 (m)": df["effective_n2_after_300s_m"].round(0).astype(int),
        "Fr": df["Fr_before"].round(2),
        "\u91cd\u529b\u6d41\u4fb5\u5165\u4f30\u7b97 (m)": df["shutdown_gravity_intrusion_estimate_m"].round(0).astype(int),
    })
    st.dataframe(view, width="stretch", hide_index=True)
    st.warning("5 min \u5185\u8f74\u5411\u6df7\u6c14\u6bb5\u589e\u957f\u4e3a 0 m \u4e0d\u4ee3\u8868\u505c\u8f93\u5b89\u5168\uff1b\u505c\u8f93\u65f6 Fr=0\uff0c\u6c34\u5e73\u7ba1\u5185\u5206\u5c42\u98ce\u9669\u589e\u5f3a\u3002")
    cols = st.columns(3)
    for col, (_, row) in zip(cols, df.iterrows()):
        label = f"\u4e2d\u65ad {row['interrupt_position_fraction']:.0%}"
        col.metric(label, f"{row['effective_n2_after_300s_m']:.0f} m", help="\u505c\u8f93 300 s \u540e\u7684\u6709\u6548 N2 \u9694\u79bb\u6bb5\u957f\u5ea6")
        col.caption(
            f"\u8f74\u5411\u589e\u957f {row['mixed_length_growth_m']:.0f} m\uff1b"
            f"\u91cd\u529b\u6d41\u4f30\u7b97 {row['shutdown_gravity_intrusion_estimate_m']:.0f} m"
        )
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = [f"{v:.0%}" for v in df["interrupt_position_fraction"]]
    ax.bar(labels, df["effective_n2_after_300s_m"], color="#2563eb", label="Effective N2 after 300 s")
    ax.plot(labels, df["shutdown_gravity_intrusion_estimate_m"], color="#ef4444", marker="o", label="Gravity-current estimate")
    ax.set_xlabel("Shutdown position")
    ax.set_ylabel("Length (m)")
    ax.set_title("Task 2 shutdown risk indicators")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    st.caption("\u4e0a\u56fe\u4e3a\u7f51\u9875\u6839\u636e CSV \u6570\u636e\u91cd\u7ed8\uff0c\u4e0d\u518d\u5c55\u793a\u542b\u5b57\u4f53\u635f\u574f\u98ce\u9669\u7684\u9759\u6001 PNG\u3002")


def render_cfd_input():
    st.subheader(C["cfd_input"])
    st.caption("\u8fd9\u91cc\u662f\u4ece\u4e00\u7ef4\u6a21\u578b\u5bfc\u51fa\u7684\u5c40\u90e8\u521d\u59cb\u573a\uff0c\u7528\u4e8e OpenFOAM/Fluent \u6216\u5c40\u90e8\u5206\u5c42\u590d\u6838\u6d41\u7a0b\uff1b\u8f93\u5165\u5305\u672c\u8eab\u4e0d\u662f\u6c42\u89e3\u7ed3\u679c\u3002")
    base = ROOT / "cfd_cases"
    cases = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    if not cases:
        st.info("\u6682\u65e0\u5916\u90e8\u590d\u6838\u8f93\u5165\u5305\u3002")
        return
    case_labels = [p.name for p in cases]
    case = base / st.selectbox(
        "\u9009\u62e9\u8f93\u5165\u5305",
        case_labels,
        index=default_cfd_label_index([""] + case_labels) - 1 if any("stop60" in v for v in case_labels) else 0,
    )
    meta = json.loads((case / "case_metadata.json").read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\u505c\u8f93\u65f6\u523b", f"{meta['stop_time_s']:.1f} s")
    c2.metric("H2 \u524d\u7f18", f"{meta['h2_front_m']/1000:.2f} km")
    c3.metric("N2/Air \u524d\u7f18", f"{meta['n2_air_front_m']/1000:.2f} km")
    c4.metric("\u6709\u6548 N2", f"{meta['effective_n2_length_m']:.0f} m")
    previews = []
    for w in meta.get("windows", []):
        p = case / w.get("initial_preview_png", "")
        if p.exists():
            previews.append((w["name"], p))
    if previews:
        cols = st.columns(min(3, len(previews)))
        for i, (name, p) in enumerate(previews):
            cols[i % len(cols)].image(str(p), caption=name)


def render_prepared_3d_cases():
    st.subheader("\u79bb\u7ebf\u4e09\u7ef4\u590d\u6838\u5de5\u7a0b\u5305")
    cases = list_prepared_3d_cases(ROOT / "openfoam_cases")
    if not cases:
        st.info("\u6682\u65e0\u79bb\u7ebf\u4e09\u7ef4\u590d\u6838\u5de5\u7a0b\u5305\u3002")
        return
    case = cases[0]
    if len(cases) > 1:
        labels = [p.name for p in cases]
        chosen = st.selectbox("\u9009\u62e9\u4e09\u7ef4\u590d\u6838\u5de5\u7a0b\u5305", labels, index=default_cfd_label_index([""] + labels) - 1)
        case = cases[labels.index(chosen)]
    metrics_path = case / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    st.warning("\u8fd9\u662f\u79bb\u7ebf\u4e09\u7ef4\u521d\u59cb\u573a/\u5de5\u7a0b\u5305\uff0c\u4e0d\u662f\u5df2\u6c42\u89e3\u7ed3\u679c\u3002\u8dd1\u5b8c Fluent/OpenFOAM \u540e\uff0c\u518d\u5c06\u7ed3\u679c\u5bfc\u5165 outputs/cfd3d\u3002")
    st.caption(f"case: {case.name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\u4e09\u7ef4\u91c7\u6837\u70b9", f"{int(metrics.get('sample_points', 0))}")
    c2.metric("\u5c40\u90e8\u7a97\u53e3", f"{metrics.get('local_length_m', 0):.0f} m")
    c3.metric("\u6469\u5c14\u548c\u8bef\u5dee", f"{metrics.get('mole_sum_max_error', 0):.1e}")
    c4.metric("\u8d28\u91cf\u548c\u8bef\u5dee", f"{metrics.get('mass_sum_max_error', 0):.1e}")
    c5, c6, c7 = st.columns(3)
    c5.metric("max x_H2", f"{metrics.get('x_h2_max', 0):.3f}")
    c6.metric("max Y_H2", f"{metrics.get('y_h2_max', 0):.3f}")
    c7.metric("max x_O2", f"{metrics.get('x_o2_max', 0):.3f}")
    preview = case / "initial_3d_preview.png"
    if preview.exists():
        st.image(str(preview), caption="\u4e09\u7ef4\u521d\u59cb H2 \u6469\u5c14\u5206\u6570\u9884\u89c8")
    downloads = [
        ("initial_3d_samples.csv", "text/csv"),
        ("initial_3d_points.vtk", "application/octet-stream"),
        ("openfoam_solver_notes.md", "text/markdown"),
    ]
    cols = st.columns(len(downloads))
    for col, (name, mime) in zip(cols, downloads):
        path = case / name
        if path.exists():
            col.download_button(name, path.read_bytes(), file_name=name, mime=mime)


def render_cfd_results(case_dir):
    st.subheader(C["cfd_result"])
    if case_dir is None:
        st.warning(C["no_cfd"])
        return
    metrics = load_cfd_metrics(case_dir)
    images = load_cfd_images(case_dir)
    missing = [k for k in ("xz_slice_h2", "cross_section_h2", "flammable_region") if k not in images]
    if not metrics or missing:
        st.warning(C["no_cfd"])
        return
    solver = str(metrics.get("solver", ""))
    is_openfoam = "OpenFOAM" in solver
    if "low-Mach buoyant" in solver:
        st.success("\u5df2\u8bfb\u53d6\u5c40\u90e8\u4e8c\u7ef4\u5206\u5c42\u590d\u6838\u7ed3\u679c\uff1a\u6a21\u578b\u5305\u542b\u6d6e\u529b\u9a71\u52a8\u548c H2/N2/Air \u7ec4\u5206\u8f93\u8fd0\uff0c\u7528\u4e8e\u63d0\u793a\u505c\u8f93\u5206\u5c42\u98ce\u9669\u3002")
    elif is_openfoam:
        st.success("\u5df2\u8bfb\u53d6 OpenFOAM rhoReactingFoam \u4e09\u7ef4\u5706\u7ba1\u590d\u6838\u7ed3\u679c\uff1a\u5305\u542b H2/O2/N2 \u7ec4\u5206\u3001\u901f\u5ea6\u573a\u548c\u538b\u529b\u573a\u3002")
    else:
        st.success("\u5df2\u8bfb\u53d6\u5c40\u90e8\u590d\u6838\u7ed3\u679c\uff1a\u56fe\u50cf\u6765\u81ea\u5916\u90e8\u6216\u5c40\u90e8\u590d\u6838\u6d41\u7a0b\uff0c\u4e0d\u662f\u7f51\u9875\u4e34\u65f6\u4f2a\u9020\u3002")
    st.caption(f"case: {Path(case_dir).name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\u4e0a/\u4e0b\u534a\u7ba1 H2 \u5dee\u503c", f"{metrics.get('top_bottom_h2_delta', 0):.3e}" if is_openfoam else f"{metrics.get('top_bottom_h2_delta', 0):.3f}")
    if is_openfoam:
        c2.metric("\u4e0a/\u4e0b\u534a\u7ba1 O2 \u5dee\u503c", f"{metrics.get('top_bottom_o2_delta', 0):.3e}")
        c3.metric("\u53ef\u71c3\u4f53\u79ef\u6bd4\u4f8b", f"{100*metrics.get('flammable_volume_ratio', 0):.1f}%")
        c4.metric("\u6700\u5927\u901f\u5ea6", f"{metrics.get('max_speed_mps', 0):.2e} m/s")
    else:
        c2.metric("\u4e0a/\u4e0b\u534a\u7ba1 N2 \u5dee\u503c", f"{metrics.get('top_bottom_n2_delta', 0):.3f}")
        c3.metric("\u4e0a/\u4e0b\u534a\u7ba1 Air \u5dee\u503c", f"{metrics.get('top_bottom_air_delta', 0):.3f}")
        c4.metric("H2-Air \u5782\u5411\u5206\u79bb", f"{metrics.get('h2_air_vertical_separation_m', 0):.3f} m")
    extra_cols = st.columns(3)
    extra_cols[0].metric("\u4e0a\u534a\u7ba1 H2 \u6700\u5927\u503c", f"{metrics.get('top_h2_max', 0):.3f}")
    extra_cols[1].metric("\u538b\u529b\u6700\u7ec8\u6b8b\u5dee" if is_openfoam else "\u53ef\u71c3\u9762\u79ef\u6bd4\u4f8b", f"{metrics.get('p_final_residual', 0):.2e}" if is_openfoam else f"{100*metrics.get('flammable_area_ratio', 0):.1f}%")
    extra_cols[2].metric("\u8fde\u7eed\u6027\u7d2f\u8ba1\u8bef\u5dee" if is_openfoam else "\u6700\u5927\u5c40\u90e8\u901f\u5ea6", f"{metrics.get('continuity_cumulative', 0):.2e}" if is_openfoam else f"{metrics.get('max_speed_mps', 0):.3f} m/s")
    if is_openfoam:
        detail = pd.DataFrame([{
            "\u6c42\u89e3\u5668": solver,
            "\u7f51\u683c": f"{int(metrics.get('nx', 0))} x {int(metrics.get('ny', 0))} x {int(metrics.get('nz', 0))}",
            "\u5355\u5143\u6570": int(metrics.get("mesh_cells", 0)),
            "\u6700\u65b0\u65f6\u95f4 (s)": float(metrics.get("latest_time_s", 0)),
            "\u5c40\u90e8\u957f\u5ea6 (m)": float(metrics.get("local_length_m", 0)),
            "\u8d28\u91cf\u5206\u6570\u548c\u8bef\u5dee": f"{metrics.get('mass_fraction_sum_max_error', 0):.1e}",
        }])
    else:
        detail = pd.DataFrame([{
            "\u7f51\u683c": f"{int(metrics.get('nx', 0))} x {int(metrics.get('nz', 0))}",
            "\u5355\u5143\u6570": int(metrics.get("mesh_cells", 0)),
            "\u65f6\u95f4\u6b65\u957f (s)": float(metrics.get("dt_s", 0)),
            "\u6d6e\u529b\u7cfb\u6570": float(metrics.get("buoyancy_scale", 0)),
            "\u901f\u5ea6\u88c1\u526a (m/s)": float(metrics.get("velocity_clip_mps", 0)),
            "\u7ec4\u5206\u548c\u8bef\u5dee": f"{metrics.get('species_sum_max_error', 0):.1e}",
        }])
    st.dataframe(detail, width="stretch", hide_index=True)
    display_solver = solver
    if "low-Mach buoyant" in solver:
        display_solver = "\u5c40\u90e8\u4e8c\u7ef4\u4f4e\u9a6c\u8d6b\u6d6e\u529b\u591a\u7ec4\u5206\u8f93\u8fd0\u590d\u6838\uff08Python \u79bb\u7ebf\u6c42\u89e3\uff09"
    st.caption(display_solver)
    st.image(str(images["xz_slice_h2"]), caption="x-z \u7eb5\u5256\u9762 H2 \u5206\u5e03")
    cols = st.columns(2)
    cols[0].image(str(images["cross_section_h2"]), caption="\u622a\u9762 H2 \u5206\u5e03")
    cols[1].image(str(images["flammable_region"]), caption="\u53ef\u71c3\u98ce\u9669\u533a\u5206\u5e03")
    if "velocity_magnitude" in images:
        st.image(str(images["velocity_magnitude"]), caption="\u6d6e\u529b\u9a71\u52a8\u5c40\u90e8\u901f\u5ea6\u573a")
    vol = find_cfd_volume_file(case_dir)
    if vol:
        st.caption(f"VTK/VTU: {vol.relative_to(ROOT)}")
        st.download_button(
            "VTU \u4f53\u6570\u636e\u4e0b\u8f7d",
            vol.read_bytes(),
            file_name=vol.name,
            mime="application/octet-stream",
        )
    if metrics.get("note"):
        if "low-Mach buoyant" in solver:
            st.info("\u8be5\u7ed3\u679c\u7528\u4e8e\u8865\u5145\u8bc4\u4f30\u505c\u8f93\u540e\u7684\u622a\u9762\u5206\u5c42\u98ce\u9669\uff1b\u6b63\u5f0f\u4e09\u7ef4\u5de5\u7a0b\u6821\u6838\u5e94\u4ee5 Fluent/OpenFOAM \u5b8c\u6574\u7b97\u4f8b\u4e3a\u51c6\u3002")
        else:
            st.info(str(metrics["note"]))


def default_cfd_label_index(labels):
    for i, label in enumerate(labels):
        if "stop60" in label and "openfoam3d" in label:
            return i
    for i, label in enumerate(labels):
        if "stop60" in label and "buoyant2d" in label:
            return i
    for i, label in enumerate(labels):
        if "stop60" in label:
            return i
    return 1 if len(labels) > 1 else 0


def main():
    st.set_page_config(page_title=C["title"], layout="wide")
    st.title(C["title"])
    st.caption(C["note"])
    st.sidebar.header(C["input"])
    D = st.sidebar.number_input(C["diam"], min_value=0.7, max_value=1.4, value=1.2, step=0.1, format="%.2f")
    u = st.sidebar.number_input(C["vel"], min_value=1.0, max_value=15.0, value=7.0, step=1.0, format="%.2f")
    p_back = st.sidebar.selectbox(C["back"], [0.02, 0.05, 0.10], index=2)
    dx = st.sidebar.number_input(C["dx"], min_value=5.0, max_value=50.0, value=10.0, step=5.0, format="%.2f")
    beta = st.sidebar.slider("beta_K", 0.2, 0.8, 0.5, 0.05)
    pressure_label = st.sidebar.selectbox(C["pressure"], list(PRESSURE_MODES), index=1)
    t_factor = st.sidebar.slider("\u6a21\u62df\u7ec8\u6b62\u500d\u6570 L/u", 0.4, 1.2, 1.0, 0.1)
    particles = st.sidebar.slider("3D \u4ee3\u8868\u6027\u7c92\u5b50\u6570\uff08\u4ec5\u53ef\u89c6\u5316\uff09", 0, 800, 360, 40)
    cfd_cases = list_cfd_cases(ROOT / "outputs" / "cfd3d")
    if cfd_cases:
        labels = ["\u4e0d\u663e\u793a\u5916\u90e8\u590d\u6838\u7ed3\u679c"] + [p.name for p in cfd_cases]
        chosen = st.sidebar.selectbox(
            "\u5c40\u90e8\u5206\u5c42\u590d\u6838 case",
            labels,
            index=default_cfd_label_index(labels),
        )
        cfd_case = None if chosen == labels[0] else cfd_cases[labels.index(chosen) - 1]
    else:
        st.sidebar.caption("\u6682\u65e0\u5c40\u90e8\u5206\u5c42\u590d\u6838\u7ed3\u679c")
        cfd_case = None
    run_clicked = st.sidebar.button(C["run"], type="primary")

    if run_clicked or "result" not in st.session_state:
        with st.spinner("\u6b63\u5728\u8fd0\u884c\u4e00\u7ef4\u5bf9\u6d41-\u5f25\u6563\u6a21\u578b..."):
            st.session_state["result"] = run_cached(D, u, p_back, dx, beta, PRESSURE_MODES[pressure_label], t_factor)
    result = st.session_state["result"]
    metrics_df = pd.DataFrame(result.metrics)

    st.dataframe(pd.DataFrame([{
        C["diam"]: D,
        C["vel"]: u,
        C["back"]: p_back,
        C["dx"]: dx,
        "beta_K": beta,
        C["pressure"]: pressure_label,
        "\u5e73\u5747 Fr": float(metrics_df["Fr"].mean()),
    }]), width="stretch", hide_index=True)

    page = st.radio(
        "\u6a21\u5757\u5207\u6362",
        [
            "\u52a8\u6001\u6a21\u62df",
            "\u66f2\u7ebf\u6307\u6807",
            "\u4efb\u52a12\u4e2d\u65ad",
            "\u5206\u5c42\u590d\u6838",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "\u52a8\u6001\u6a21\u62df":
        idx = st.slider(C["time"], 0, len(result.times) - 1, max(0, len(result.times) // 2))
        st.caption(f"{result.times[idx]/60:.1f} min")
        st.subheader(C["pipe"])
        shown_idx = render_pipe(result, idx, particles)
        st.subheader(C["local"])
        render_local_zoom(result, shown_idx)
    elif page == "\u66f2\u7ebf\u6307\u6807":
        st.subheader(C["profiles"])
        render_curves(result, metrics_df)
        st.subheader(C["metrics"])
        display = metrics_df.rename(columns={
            "time_s": "\u65f6\u95f4 (s)",
            "mixed_length_m": "\u6df7\u6c14\u6bb5\u957f\u5ea6 (m)",
            "flammable_length_m": "\u53ef\u71c3\u98ce\u9669\u6bb5\u957f\u5ea6 (m)",
            "effective_n2_length_m": "\u6709\u6548 N2 \u9694\u79bb\u6bb5\u957f\u5ea6 (m)",
            "h2_front_m": "H2 \u524d\u7f18\u4f4d\u7f6e (m)",
        })
        st.dataframe(display, width="stretch")
        st.download_button(C["download"], display.to_csv(index=False).encode("utf-8-sig"), "purge_metrics_cn.csv", "text/csv")
    elif page == "\u4efb\u52a12\u4e2d\u65ad":
        render_task2()
    elif page == "\u5206\u5c42\u590d\u6838":
        render_cfd_input()
        render_prepared_3d_cases()
        render_cfd_results(cfd_case)


if __name__ == "__main__":
    main()
