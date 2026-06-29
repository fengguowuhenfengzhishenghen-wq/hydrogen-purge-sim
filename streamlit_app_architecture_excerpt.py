"""Streamlit web architecture excerpt for review.

This file is a readable architecture extract from streamlit_app.py.
The full runnable app is streamlit_app.py. The long 3D HTML/JS renderer is
omitted here on purpose; it lives in render_pipeline_animation_3d().
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from h2purge.config import SimulationParams
from h2purge.solver_fvm import run_simulation


PRESSURE_MODE_LABELS = {
    "reference_pressure": "全管等压",
    "friction_profile": "摩阻压降剖面",
}

PARAM_COLUMN_LABELS = {
    "D_m": "管径 D (m)",
    "u_mps": "置换速度 u (m/s)",
    "p_back_MPa_abs": "出口背压 (MPa, 绝压)",
    "dx_m": "网格 dx (m)",
    "beta_K": "轴向弥散系数 beta",
    "pressure_mode": "压力模式",
    "Fr_mean": "平均 Fr",
}

METRIC_COLUMN_LABELS = {
    "time_s": "时间 (s)",
    "mixed_length_m": "混气段长度 (m)",
    "flammable_length_m": "可燃风险段长度 (m)",
    "effective_n2_length_m": "有效 N2 隔离段长度 (m)",
    "h2_front_m": "H2 前缘位置 (m)",
    "Re_min": "最小 Re",
    "Re_max": "最大 Re",
    "K_min": "最小 K (m2/s)",
    "K_max": "最大 K (m2/s)",
    "Fr": "Fr",
    "dp_estimated_Pa": "估算压降 (Pa)",
    "rho_mean_kg_m3": "平均密度 (kg/m3)",
    "p_mean_Pa": "平均压力 (Pa)",
}


def localized_dataframe(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Use readable Chinese column labels in the web UI."""

    return df.rename(columns={key: value for key, value in labels.items() if key in df.columns})


def _animation_payload(result, max_cells: int = 180) -> dict:
    """Downsample simulation profiles for browser-side animation."""

    n_cells = len(result.x_grid)
    cell_idx = np.unique(np.linspace(0, n_cells - 1, min(max_cells, n_cells), dtype=int))
    frames = []
    for k, t in enumerate(result.times):
        prof = result.profiles[k, cell_idx, :]
        frames.append(
            {
                "time_s": float(t),
                "h2": np.round(prof[:, 0], 4).tolist(),
                "n2": np.round(prof[:, 1], 4).tolist(),
                "air": np.round(prof[:, 2], 4).tolist(),
                "mixed_m": float(result.metrics[k]["mixed_length_m"]),
                "flammable_m": float(result.metrics[k]["flammable_length_m"]),
                "n2_effective_m": float(result.metrics[k]["effective_n2_length_m"]),
                "fr": float(result.metrics[k]["Fr"]),
            }
        )
    return {"x_km": np.round(result.x_grid[cell_idx] / 1000.0, 3).tolist(), "frames": frames}


def render_pipeline_animation_3d(result, molecule_count: int = 360) -> None:
    """Full implementation is in streamlit_app.py.

    It converts _animation_payload(result) into a browser-side 3D pipe,
    molecule particles, local zoom panel, playback button, and progress slider.
    """

    st.info("3D 动态置换模拟组件在 streamlit_app.py 的 render_pipeline_animation_3d() 中实现。")


st.set_page_config(page_title="Hydrogen purge simulator", layout="wide")
st.title("输氢管道投产混气规律模拟")

with st.sidebar:
    D = st.number_input("管径 D (m)", value=1.2, min_value=0.1, step=0.1)
    u = st.number_input("置换速度 u (m/s)", value=7.0, min_value=0.0, step=0.5)
    p_back = st.number_input("出口背压 p_back (MPa, 绝压)", value=0.10, min_value=0.01, step=0.01)
    dx = st.number_input("网格 dx (m)", value=20.0, min_value=2.0, step=5.0)
    beta_K = st.slider("beta_K", min_value=0.1, max_value=1.2, value=0.5, step=0.05)
    pressure_mode = st.selectbox(
        "压力模式",
        ["reference_pressure", "friction_profile"],
        index=1,
        format_func=lambda value: PRESSURE_MODE_LABELS.get(value, value),
    )
    enable_interrupt = st.checkbox("启用中断工况", value=False)
    interrupt_position = st.selectbox("中断位置", [0.30, 0.60, 0.80], index=1)
    interrupt_time = st.number_input("中断时间 (s)", value=300.0, min_value=0.0, step=60.0)
    molecule_count = st.slider("3D 分子粒子数", min_value=0, max_value=900, value=360, step=60)
    run_btn = st.button("运行模拟", type="primary")

if run_btn:
    t_end = 1.2 * 12000.0 / max(u, 1.0e-9) if u > 0 else interrupt_time
    params = SimulationParams(
        D=D,
        u_nominal=u,
        p_back_abs=p_back * 1.0e6,
        dx=dx,
        beta_K=beta_K,
        t_end=t_end,
        output_times=np.linspace(0.0, t_end, 61),
        pressure_mode=pressure_mode,
    )
    result = run_simulation(params)
    metrics_df = pd.DataFrame(result.metrics)

    st.subheader("参数表")
    param_df = pd.DataFrame(
        [
            {
                "D_m": D,
                "u_mps": u,
                "p_back_MPa_abs": p_back,
                "dx_m": result.dx,
                "beta_K": beta_K,
                "pressure_mode": PRESSURE_MODE_LABELS.get(pressure_mode, pressure_mode),
                "Fr_mean": metrics_df["Fr"].mean(),
            }
        ]
    )
    st.dataframe(localized_dataframe(param_df, PARAM_COLUMN_LABELS), width="stretch")

    st.subheader("动态置换模拟")
    render_pipeline_animation_3d(result, molecule_count=molecule_count)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for idx in np.linspace(0, len(result.times) - 1, 5, dtype=int):
            ax.plot(result.x_grid / 1000.0, result.profiles[idx, :, 0], label=f"H2 {result.times[idx]:.0f}s")
            ax.plot(result.x_grid / 1000.0, result.profiles[idx, :, 1], "--", label=f"N2 {result.times[idx]:.0f}s")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("摩尔分数")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(metrics_df["time_s"] / 60.0, metrics_df["mixed_length_m"], label="混气段长度")
        ax.plot(metrics_df["time_s"] / 60.0, metrics_df["flammable_length_m"], label="可燃风险长度")
        ax.plot(metrics_df["time_s"] / 60.0, metrics_df["effective_n2_length_m"], label="有效 N2 隔离段")
        ax.set_xlabel("时间 (min)")
        ax.set_ylabel("长度 (m)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)

    st.subheader("时序指标")
    metrics_display = localized_dataframe(metrics_df, METRIC_COLUMN_LABELS)
    st.dataframe(metrics_display, width="stretch")
    st.download_button(
        "下载中文 CSV",
        data=metrics_display.to_csv(index=False).encode("utf-8-sig"),
        file_name="purge_metrics_cn.csv",
        mime="text/csv",
    )

    if enable_interrupt:
        st.info("中断专题脚本默认计算 30%、60%、80% 三个位置；GUI 当前显示主算例结果。")
