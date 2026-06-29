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
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2purge.cfd_import import find_cfd_volume_file, list_cfd_cases, load_cfd_images, load_cfd_metrics
from h2purge.config import SimulationParams
from h2purge.solver_fvm import run_simulation


def font_setup() -> None:
    for name in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
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
    "note": "\u672c\u7f51\u9875\u5c55\u793a\u4e00\u7ef4\u8f74\u5411\u5bf9\u6d41-\u5f25\u6563\u6a21\u578b\u7ed3\u679c\u30023D \u7ba1\u9053\u4e3a\u6469\u5c14\u5206\u6570\u573a\u53ef\u89c6\u5316\u6620\u5c04\uff0c\u4e0d\u4ee3\u8868\u4e09\u7ef4 CFD \u6c42\u89e3\u3002",
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
    "cfd_input": "\u5916\u90e8\u590d\u6838\u8f93\u5165\u5305\uff081D \u2192 CFD/\u5c40\u90e8\u6a21\u578b\uff09",
    "cfd_result": "\u5c40\u90e8\u5206\u5c42\u590d\u6838\u7ed3\u679c\uff08OpenFOAM \u6807\u91cf\u8f93\u8fd0 + reduced x-z\uff09",
    "no_cfd": "\u672a\u5bfc\u5165\u5c40\u90e8\u5206\u5c42\u590d\u6838\u7ed3\u679c\u3002\u5f53\u524d\u4e0d\u663e\u793a\u4e91\u56fe\uff0c\u907f\u514d\u5c06\u4e00\u7ef4\u53ef\u89c6\u5316\u8bef\u8ba4\u4e3a CFD\u3002",
    "mol": "\u6469\u5c14\u5206\u6570 / \u4f53\u79ef\u5206\u6570",
    "xpos": "\u7ba1\u9053\u4f4d\u7f6e x (km)",
    "length": "\u957f\u5ea6 (m)",
    "download": "\u4e0b\u8f7d\u5f53\u524d\u7b97\u4f8b CSV",
}

COL = {"H2": "#2563eb", "N2": "#1f9d55", "Air": "#e5533d"}
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


def mix_color(h2, n2, air):
    r = int(37 * h2 + 31 * n2 + 229 * air)
    g = int(99 * h2 + 157 * n2 + 83 * air)
    b = int(235 * h2 + 85 * n2 + 61 * air)
    return f"rgb({r},{g},{b})"


def pipe_frame_payload(result, particles):
    cell_ids = np.linspace(0, len(result.x_grid) - 1, 160, dtype=int)
    n_particles = min(int(particles), 520)
    frames = []
    for idx, prof in enumerate(result.profiles):
        metric = result.metrics[idx]
        cell_colors = [mix_color(*prof[k]) for k in cell_ids]
        rng = np.random.default_rng(2026 + idx)
        picks = rng.choice(len(prof), size=n_particles, replace=True)
        fr = float(metric.get("Fr", 3.0))
        strat = max(0.0, min(1.0, (3.0 - fr) / 3.0))
        dots = []
        for k in picks:
            h2, n2, air = prof[k]
            probs = np.array([h2, n2, air], dtype=float)
            probs = probs / max(float(probs.sum()), 1.0e-12)
            comp = rng.choice(["H2", "N2", "Air"], p=probs)
            dots.append(
                {
                    "x": float(0.05 + 0.90 * result.x_grid[k] / result.params.L),
                    "y": float(0.50 + rng.normal(0, 0.055) + (comp == "Air") * 0.070 * strat - (comp == "H2") * 0.105 * strat),
                    "c": COL[comp],
                }
            )
        frames.append(
            {
                "time": float(result.times[idx] / 60.0),
                "cells": cell_colors,
                "dots": dots,
                "mixed": float(metric["mixed_length_m"]),
                "flammable": float(metric["flammable_length_m"]),
                "n2": float(metric["effective_n2_length_m"]),
                "fr": float(metric["Fr"]),
            }
        )
    return {"initial": int(len(result.times) // 2), "frames": frames}


def render_pipe(result, idx, particles):
    payload = pipe_frame_payload(result, particles)
    payload["initial"] = int(idx)
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""
    <div id="app" style="font-family:Arial,'Microsoft YaHei',sans-serif;border:1px solid #d4deea;border-radius:12px;background:#eef5fc;padding:16px;color:#344054;">
      <div style="display:flex;justify-content:space-between;font-weight:700;font-size:18px;">
        <div>\u5165\u53e3<br><span style="font-weight:400;font-size:15px;">0 km</span></div>
        <div style="text-align:right;">\u51fa\u53e3<br><span style="font-weight:400;font-size:15px;">12 km</span></div>
      </div>
      <canvas id="pipeCanvas" width="1200" height="330" style="width:100%;height:330px;display:block;"></canvas>
      <div style="display:flex;gap:12px;align-items:center;margin-top:8px;flex-wrap:wrap;">
        <button id="playBtn" style="border:1px solid #b8c4d4;border-radius:7px;background:white;padding:7px 18px;cursor:pointer;">\u64ad\u653e</button>
        <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" style="flex:1;min-width:240px;">
        <span id="timeLabel" style="font-weight:700;min-width:92px;"></span>
      </div>
      <div id="legend" style="display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:15px;">
        <span><b style="color:{COL['H2']}">\u25a0</b> H2</span>
        <span><b style="color:{COL['N2']}">\u25a0</b> N2</span>
        <span><b style="color:{COL['Air']}">\u25a0</b> Air</span>
        <span id="metrics"></span>
      </div>
      <div style="font-size:13px;margin-top:8px;color:#667085;">3D \u7ba1\u9053\u4e3a\u4e00\u7ef4\u6469\u5c14\u5206\u6570\u573a\u7684\u53ef\u89c6\u5316\u6620\u5c04\uff0c\u4e0d\u662f CFD \u6c42\u89e3\u7ed3\u679c\u3002</div>
    </div>
    <script>
    const payload = {data};
    const frames = payload.frames;
    const canvas = document.getElementById('pipeCanvas');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('frameSlider');
    const playBtn = document.getElementById('playBtn');
    const timeLabel = document.getElementById('timeLabel');
    const metrics = document.getElementById('metrics');
    let frame = Math.max(0, Math.min(payload.initial, frames.length - 1));
    let timer = null;
    slider.max = frames.length - 1;
    slider.value = frame;

    function roundedRect(x, y, w, h, r) {{
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }}

    function draw(i) {{
      const f = frames[i];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#eef5fc';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const x0 = 70, y0 = 122, w = 1060, h = 74, r = 37;
      ctx.save();
      roundedRect(x0, y0, w, h, r);
      ctx.clip();
      const cw = w / f.cells.length;
      for (let j = 0; j < f.cells.length; j++) {{
        ctx.fillStyle = f.cells[j];
        ctx.globalAlpha = 0.78;
        ctx.fillRect(x0 + j * cw, y0, cw + 1, h);
      }}
      ctx.globalAlpha = 1.0;
      ctx.restore();
      ctx.strokeStyle = '#6b7d92';
      ctx.lineWidth = 3;
      roundedRect(x0, y0, w, h, r);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(107,125,146,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + 20, y0 + h + 22);
      ctx.lineTo(x0 + w - 20, y0 + h + 22);
      ctx.stroke();
      for (const p of f.dots) {{
        ctx.beginPath();
        ctx.fillStyle = p.c;
        ctx.globalAlpha = 0.58;
        ctx.arc(70 + p.x * 1060, 122 + p.y * 74, 4.2, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalAlpha = 1.0;
      timeLabel.textContent = f.time.toFixed(1) + ' min';
      metrics.textContent = `\u6df7\u6c14\u6bb5 ${{f.mixed.toFixed(0)}} m | \u53ef\u71c3\u98ce\u9669\u6bb5 ${{f.flammable.toFixed(0)}} m | \u6709\u6548 N2 ${{f.n2.toFixed(0)}} m | Fr ${{f.fr.toFixed(2)}}`;
      slider.value = i;
    }}

    function stop() {{
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = '\u64ad\u653e';
    }}

    function play() {{
      stop();
      playBtn.textContent = '\u6682\u505c';
      timer = setInterval(() => {{
        frame += 1;
        if (frame >= frames.length) frame = 0;
        draw(frame);
      }}, 110);
    }}

    playBtn.onclick = () => timer ? stop() : play();
    slider.oninput = (e) => {{
      stop();
      frame = parseInt(e.target.value, 10);
      draw(frame);
    }};
    draw(frame);
    </script>
    """
    components.html(html, height=475)
    return idx


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
        ax.plot(metrics_df["time_s"] / 60, metrics_df["mixed_length_m"], label="\u6df7\u6c14\u6bb5\u957f\u5ea6")
        ax.plot(metrics_df["time_s"] / 60, metrics_df["flammable_length_m"], label="\u53ef\u71c3\u98ce\u9669\u6bb5")
        ax.plot(metrics_df["time_s"] / 60, metrics_df["effective_n2_length_m"], label="\u6709\u6548 N2 \u9694\u79bb\u6bb5")
        ax.set_xlabel("\u65f6\u95f4 (min)")
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
    cols = st.columns(2)
    for col, img, cap in [
        (cols[0], ROOT / "outputs" / "task2" / "effective_n2_after_interrupt.png", "\u6709\u6548 N2 \u9694\u79bb\u6bb5"),
        (cols[1], ROOT / "outputs" / "task2" / "mixed_length_growth_interrupt.png", "\u8f74\u5411\u6df7\u6c14\u6bb5\u589e\u957f\u89e3\u91ca"),
    ]:
        if img.exists():
            col.image(str(img), caption=cap)


def render_cfd_input():
    st.subheader(C["cfd_input"])
    st.caption("\u8fd9\u91cc\u662f\u4ece\u4e00\u7ef4\u6a21\u578b\u5bfc\u51fa\u7684\u5c40\u90e8\u521d\u59cb\u573a\uff0c\u7528\u4e8e OpenFOAM/Fluent \u6216\u5c40\u90e8\u5206\u5c42\u6a21\u578b\u590d\u6838\uff0c\u4e0d\u662f CFD \u6c42\u89e3\u7ed3\u679c\u3002")
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
    st.success("\u5df2\u8bfb\u53d6\u5c40\u90e8\u590d\u6838\u7ed3\u679c\uff1a\u4e91\u56fe\u6765\u81ea reduced x-z \u5206\u5c42\u6a21\u578b\uff0cVTU \u6765\u81ea OpenFOAM scalarTransportFoam \u5bfc\u51fa\uff0c\u4e0d\u662f\u7f51\u9875\u4e34\u65f6\u4f2a\u9020\u3002")
    st.caption(f"case: {Path(case_dir).name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\u4e0a/\u4e0b\u534a\u7ba1 H2 \u5dee\u503c", f"{metrics.get('top_bottom_h2_delta', 0):.3f}")
    c2.metric("\u4e0a\u534a\u7ba1 H2 \u6700\u5927\u503c", f"{metrics.get('top_h2_max', 0):.3f}")
    c3.metric("\u53ef\u71c3\u9762\u79ef\u6bd4\u4f8b", f"{100*metrics.get('flammable_area_ratio', 0):.1f}%")
    c4.metric("\u53ef\u71c3\u4f53\u79ef\u6bd4\u4f8b", f"{100*metrics.get('flammable_volume_ratio', 0):.1f}%")
    st.caption(str(metrics.get("solver", "")))
    summary_path = ROOT / "outputs" / "cfd3d" / "openfoam_scalarTransport_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        case_id = Path(case_dir).name.removesuffix("_reduced2d")
        row = summary[summary["case_id"] == case_id]
        if not row.empty:
            status = row.iloc[0]
            st.success(
                "\u68c0\u67e5\u72b6\u6001\uff1a"
                f"OpenFOAM mesh={int(status['mesh_cells'])} cells\uff1b"
                f"Mesh OK={bool(status['mesh_ok'])}\uff1b"
                f"300s={bool(status['solver_reached_300s'])}\uff1b"
                f"End={bool(status['solver_end'])}\uff1b"
                f"VTK={bool(status['vtk_exported'])}\uff1b"
                f"Fatal/Error={bool(status['fatal_or_error'])}"
            )
            check = row[[
                "case_id",
                "mesh_cells",
                "mesh_ok",
                "solver_reached_300s",
                "solver_end",
                "vtk_exported",
                "fatal_or_error",
            ]].rename(columns={
                "case_id": "case",
                "mesh_cells": "\u7f51\u683c\u5355\u5143\u6570",
                "mesh_ok": "\u7f51\u683c\u68c0\u67e5",
                "solver_reached_300s": "\u8ba1\u7b97\u5230 300s",
                "solver_end": "\u6c42\u89e3\u6b63\u5e38\u7ed3\u675f",
                "vtk_exported": "VTK \u5df2\u5bfc\u51fa",
                "fatal_or_error": "\u662f\u5426\u6709\u81f4\u547d\u9519\u8bef",
            })
            st.dataframe(check, width="stretch", hide_index=True)
    st.image(str(images["xz_slice_h2"]), caption="x-z \u7eb5\u5256\u9762\u4e91\u56fe")
    cols = st.columns(2)
    cols[0].image(str(images["cross_section_h2"]), caption="\u5706\u622a\u9762 H2 \u4e91\u56fe")
    cols[1].image(str(images["flammable_region"]), caption="\u53ef\u71c3\u98ce\u9669\u533a\u4e91\u56fe")
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
        st.info(str(metrics["note"]))


def default_cfd_label_index(labels):
    for i, label in enumerate(labels):
        if "stop60" in label:
            return i
    return 1 if len(labels) > 1 else 0


def main():
    st.set_page_config(page_title=C["title"], layout="wide")
    st.title(C["title"])
    st.info(C["note"])
    with st.expander("\u4eff\u771f\u67b6\u6784\u8bf4\u660e", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- **1D \u4e3b\u6a21\u578b**\uff1a\u8d1f\u8d23 12 km \u5168\u7ba1\u7f6e\u6362\u3001\u6df7\u6c14\u6bb5\u3001\u53ef\u71c3\u98ce\u9669\u6bb5\u548c\u6709\u6548 N2 \u9694\u79bb\u6bb5\u3002",
                    "- **\u5c40\u90e8\u653e\u5927**\uff1a\u6765\u81ea 1D \u8ba1\u7b97\u7ed3\u679c\uff0c\u7528\u4e8e\u67e5\u770b H2/N2/Air \u754c\u9762\u3002",
                    "- **\u5c40\u90e8\u5206\u5c42\u590d\u6838**\uff1a\u8bfb\u53d6 reduced x-z \u5206\u5c42\u6a21\u578b\u548c OpenFOAM scalarTransportFoam \u6807\u91cf\u8f93\u8fd0\u7ed3\u679c\uff0c\u4e0d\u7b49\u540c\u4e8e\u5b8c\u6574\u4e09\u7ef4\u591a\u7ec4\u5206\u91cd\u529b CFD\u3002",
                    "- **3D \u7ba1\u9053\u52a8\u753b**\uff1a\u53ea\u662f\u5c06 1D \u6469\u5c14\u5206\u6570\u6620\u5c04\u5230\u7ba1\u9053\u753b\u9762\uff0c\u4fbf\u4e8e\u7b54\u8fa9\u5c55\u793a\u3002",
                ]
            )
        )
    st.sidebar.header(C["input"])
    D = st.sidebar.number_input(C["diam"], min_value=0.7, max_value=1.4, value=1.2, step=0.1, format="%.2f")
    u = st.sidebar.number_input(C["vel"], min_value=1.0, max_value=15.0, value=7.0, step=1.0, format="%.2f")
    p_back = st.sidebar.selectbox(C["back"], [0.02, 0.05, 0.10], index=2)
    dx = st.sidebar.number_input(C["dx"], min_value=5.0, max_value=50.0, value=10.0, step=5.0, format="%.2f")
    beta = st.sidebar.slider("beta_K", 0.2, 0.8, 0.5, 0.05)
    pressure_label = st.sidebar.selectbox(C["pressure"], list(PRESSURE_MODES), index=1)
    t_factor = st.sidebar.slider("\u6a21\u62df\u7ec8\u6b62\u500d\u6570 L/u", 0.4, 1.2, 1.0, 0.1)
    particles = st.sidebar.slider("3D \u4ee3\u8868\u6027\u7c92\u5b50\u6570\uff08\u4ec5\u53ef\u89c6\u5316\uff09", 0, 800, 360, 40)
    show_task2 = st.sidebar.checkbox("\u663e\u793a\u4efb\u52a12\u4e2d\u65ad\u8bf4\u660e", value=True)
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

    st.subheader(C["params"])
    st.dataframe(pd.DataFrame([{
        C["diam"]: D,
        C["vel"]: u,
        C["back"]: p_back,
        C["dx"]: dx,
        "beta_K": beta,
        C["pressure"]: pressure_label,
        "\u5e73\u5747 Fr": float(metrics_df["Fr"].mean()),
    }]), width="stretch", hide_index=True)

    idx = st.slider(C["time"], 0, len(result.times) - 1, max(0, len(result.times) // 2))
    st.caption(f"{result.times[idx]/60:.1f} min")
    st.subheader(C["pipe"])
    shown_idx = render_pipe(result, idx, particles)
    st.subheader(C["local"])
    render_local_zoom(result, shown_idx)
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
    if show_task2:
        render_task2()
    render_cfd_input()
    render_cfd_results(cfd_case)


if __name__ == "__main__":
    main()
