"""Task1 sweep and Task2 interruption workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationParams
from .constants import D_MOL_DEFAULT, L_PIPE
from .io_utils import ensure_dir, output_dir
from .metrics import summarize_metric_series
from .plots import plot_bar, plot_concentration_profiles, plot_heatmap_table, plot_metric_time
from .safety import gravity_current_speed, stratification_risk_level
from .solver_fvm import run_simulation


def _breakthrough_index(result, threshold=0.01):
    outlet_h2 = result.profiles[:, -1, 0]
    idx = np.where(outlet_h2 >= threshold)[0]
    return int(idx[0]) if len(idx) else None


def _breakthrough_time(result, threshold=0.01):
    idx = _breakthrough_index(result, threshold=threshold)
    return float(result.times[idx]) if idx is not None else float("nan")


def _metric_at_index(result, idx: int | None, key: str):
    if idx is None:
        return float("nan")
    return float(result.metrics[idx][key])


def _front_reach_index(result, target_position_m: float):
    fronts = np.array([row["h2_front_m"] for row in result.metrics], dtype=float)
    valid = np.isfinite(fronts)
    idx = np.where(valid & (fronts >= target_position_m))[0]
    return int(idx[0]) if len(idx) else None


def run_task1_case(D: float, u: float, p_back_MPa: float, dx: float = 10.0, pressure_mode: str = "friction_profile"):
    t_end = 1.2 * L_PIPE / u
    params = SimulationParams(
        L=L_PIPE,
        D=D,
        u_nominal=u,
        p_back_abs=p_back_MPa * 1.0e6,
        roughness=0.05e-3,
        dx=dx,
        t_end=t_end,
        output_times=np.linspace(0.0, t_end, 81),
        pressure_mode=pressure_mode,
    )
    result = run_simulation(params)
    breakthrough_idx = _breakthrough_index(result)
    front_80_idx = _front_reach_index(result, 0.80 * L_PIPE)
    summary = {
        "D_m": D,
        "u_mps": u,
        "p_back_MPa": p_back_MPa,
        "dx_m": result.dx,
        "t_end_s": t_end,
        "breakthrough_time_s": _breakthrough_time(result),
        "mixed_length_at_breakthrough_m": _metric_at_index(result, breakthrough_idx, "mixed_length_m"),
        "h2_n2_interface_at_breakthrough_m": _metric_at_index(result, breakthrough_idx, "h2_n2_interface_length_m"),
        "n2_air_interface_at_breakthrough_m": _metric_at_index(result, breakthrough_idx, "n2_air_interface_length_m"),
        "effective_n2_at_breakthrough_m": _metric_at_index(result, breakthrough_idx, "effective_n2_length_m"),
        "flammable_length_at_breakthrough_m": _metric_at_index(result, breakthrough_idx, "flammable_length_m"),
        "time_front_80pct_s": _metric_at_index(result, front_80_idx, "time_s"),
        "mixed_length_at_80pct_m": _metric_at_index(result, front_80_idx, "mixed_length_m"),
        "h2_n2_interface_at_80pct_m": _metric_at_index(result, front_80_idx, "h2_n2_interface_length_m"),
        "n2_air_interface_at_80pct_m": _metric_at_index(result, front_80_idx, "n2_air_interface_length_m"),
        "effective_n2_at_80pct_m": _metric_at_index(result, front_80_idx, "effective_n2_length_m"),
        "flammable_length_at_80pct_m": _metric_at_index(result, front_80_idx, "flammable_length_m"),
    }
    summary.update(summarize_metric_series(result.metrics))
    metric_df = pd.DataFrame(result.metrics)
    summary.update(
        {
            "Re_min": float(metric_df["Re_min"].min()),
            "Re_max": float(metric_df["Re_max"].max()),
            "K_min": float(metric_df["K_min"].min()),
            "K_max": float(metric_df["K_max"].max()),
            "Fr": float(metric_df["Fr"].mean()),
            "stratification_risk_level": stratification_risk_level(float(metric_df["Fr"].mean())),
            "dp_estimated_Pa": float(metric_df["dp_estimated_Pa"].max()),
            "dp_over_p_back": float(metric_df["dp_estimated_Pa"].max() / (p_back_MPa * 1.0e6)),
            "recommended_flag": False,
        }
    )
    return result, summary


def _normalize(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if abs(span) < 1.0e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def make_recommendations(summary_df: pd.DataFrame, dp_cap: float = 0.10) -> pd.DataFrame:
    """Recommend a replacement speed per diameter at p_back=0.10 MPa."""

    rows = []
    for D, group in summary_df[summary_df["p_back_MPa"] == 0.10].groupby("D_m"):
        g = group.copy().sort_values("u_mps").reset_index(drop=True)
        feasible = g[g["dp_over_p_back"] < dp_cap]
        if len(feasible):
            best = feasible.sort_values(["Fr", "breakthrough_time_s"], ascending=[False, True]).iloc[0]
            basis = f"max Fr subject to dp/p_back < {dp_cap:g}"
        else:
            best = g.sort_values("dp_over_p_back").iloc[0]
            basis = f"no speed meets dp/p_back < {dp_cap:g}; fell back to lowest dp"
        best_u = float(best["u_mps"])
        for _, row in g.iterrows():
            rows.append(
                {
                    "D_m": float(D),
                    "u_mps": float(row["u_mps"]),
                    "p_back_MPa": float(row["p_back_MPa"]),
                    "Fr": float(row["Fr"]),
                    "dp_over_p_back": float(row["dp_over_p_back"]),
                    "breakthrough_time_s": float(row["breakthrough_time_s"]),
                    "dp_feasible": bool(row["dp_over_p_back"] < dp_cap),
                    "recommended_flag": bool(abs(float(row["u_mps"]) - best_u) < 1.0e-9),
                    "recommendation_basis": basis,
                }
            )
    return pd.DataFrame(rows)


def write_task1_interpretation(out: Path, df: pd.DataFrame, rec: pd.DataFrame) -> Path:
    path = out / "model_interpretation.md"
    max_dp = float(df["dp_over_p_back"].max())
    recommended = rec[rec["recommended_flag"]][["D_m", "u_mps"]].to_dict("records")
    path.write_text(
        "# Task1 model interpretation\n\n"
        "The transported variables are mole fractions. The dispersion model is K = beta * u * D + D_mol. "
        "At equal replacement progress, the interface variance scales as 2 K L / u, so velocity cancels when K is proportional to uD. "
        "Therefore mixed length is controlled mainly by pipe diameter and beta, not by displacement velocity or outlet back pressure.\n\n"
        "Outlet back pressure affects density, Reynolds number, density Froude number diagnostics, and estimated frictional pressure drop. "
        "It does not change the mole-fraction profiles in this V1 constant-u model; this limitation must be stated in the report.\n\n"
        "Recommended speed is selected at p_back=0.10 MPa by applying a hard dp/p_back < 0.10 feasibility constraint, then choosing the highest Fr among feasible speeds. "
        "This treats pressure drop as an engineering constraint and Fr as the primary stratification-risk screen.\n\n"
        "The final mixed length and minimum effective N2 at 1.2 L/u may be zero because the pipe has already been flushed by nearly pure H2. "
        "For comparison, the summary CSV also reports values when the H2 front reaches 80% of pipe length and when H2 first breaks through the outlet.\n\n"
        f"Maximum dp/p_back in the sweep: {max_dp:.3f}. Recommended speeds at p_back=0.10 MPa: {recommended}.\n",
        encoding="utf-8",
    )
    return path


def run_task1_sweep(outdir=None, dx: float = 10.0, save_case_plots: bool = True) -> pd.DataFrame:
    out = ensure_dir(outdir or output_dir("task1"))
    rows = []
    sample_result = None
    for D in [0.7, 1.0, 1.2, 1.4]:
        for u in [5.0, 7.0, 9.0]:
            for p_back in [0.02, 0.05, 0.10]:
                result, summary = run_task1_case(D, u, p_back, dx=dx)
                rows.append(summary)
                if sample_result is None or (D == 1.2 and u == 7.0 and abs(p_back - 0.10) < 1.0e-9):
                    sample_result = result
    df = pd.DataFrame(rows)
    rec = make_recommendations(df)
    for _, row in rec[rec["recommended_flag"]].iterrows():
        mask = (df["D_m"] == row["D_m"]) & (df["u_mps"] == row["u_mps"]) & (df["p_back_MPa"] == row["p_back_MPa"])
        df.loc[mask, "recommended_flag"] = True
    df.to_csv(out / "task1_summary.csv", index=False)
    rec.to_csv(out / "recommendation_table.csv", index=False)
    write_task1_interpretation(out, df, rec)
    if save_case_plots and sample_result is not None:
        pd.DataFrame(sample_result.metrics).to_csv(out / "sample_case_metrics.csv", index=False)
        plot_concentration_profiles(sample_result, out / "sample_concentration_profiles.png", title="任务1代表工况 H2/N2/Air 沿程浓度分布")
        plot_metric_time(sample_result, "mixed_length_m", out / "sample_mixed_length_time.png", "混气段长度 (m)")
        plot_metric_time(sample_result, "flammable_length_m", out / "sample_flammable_length_time.png", "可燃风险段长度 (m)")
    plot_heatmap_table(
        df[df["p_back_MPa"] == 0.10],
        index="D_m",
        columns="u_mps",
        values="mixed_length_max_m",
        path=out / "mixed_length_max_heatmap_p010.png",
        title="出口背压 0.10 MPa 时全过程最大混气段长度",
        xlabel="置换速度 u (m/s)",
        ylabel="管径 D (m)",
        colorbar_label="最大混气段长度 (m)",
    )
    plot_bar(
        df[df["p_back_MPa"] == 0.10],
        x_col="D_m",
        y_col="Fr",
        hue_col="u_mps",
        path=out / "fr_by_diameter_velocity_p010.png",
        title="出口背压 0.10 MPa 时 Fr 分层风险指标",
        xlabel="管径 D (m)",
        ylabel="密度 Froude 数 Fr",
        hue_label="u={value} m/s",
    )
    return df


def _nearest_stop_index(result, target_position_m: float) -> int:
    fronts = np.array([row["h2_front_m"] for row in result.metrics], dtype=float)
    valid = np.isfinite(fronts)
    idxs = np.where(valid & (fronts >= target_position_m))[0]
    if len(idxs):
        return int(idxs[0])
    return int(np.nanargmin(np.abs(fronts - target_position_m)))


def _plot_task2_bar_cn(df: pd.DataFrame, y_col: str, path, ylabel: str, note: str | None = None):
    import matplotlib.pyplot as plt

    labels = [f"{int(float(v) * 100)}%" for v in df["interrupt_position_fraction"]]
    values = df[y_col].astype(float).to_numpy()
    vmax = float(np.nanmax(values)) if len(values) else 0.0
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=220)
    bars = ax.bar(labels, values, width=0.48, color="#2f6db3")
    ax.set_xlabel("中断位置（H2 前缘占管长比例）", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0.0, 10.0 if vmax <= 0.0 else vmax * 1.18)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars:
        value = bar.get_height()
        y = value + (0.35 if vmax <= 0.0 else vmax * 0.025)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.0f} m", ha="center", va="bottom", fontsize=10)
    if note:
        ax.text(0.5, 0.90, note, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout(pad=1.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _plot_task2_zero_growth_cn(df: pd.DataFrame, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    labels = [f"{int(float(v) * 100)}%" for v in df["interrupt_position_fraction"]]
    growth = df["mixed_length_growth_m"].astype(float).to_numpy()
    dx = float(df["dx_m"].iloc[0]) if "dx_m" in df else 10.0
    erfc_width = float(df["shutdown_erfc_1_99_width_estimate_m"].iloc[0])
    gravity = float(df["shutdown_gravity_intrusion_estimate_m"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=220)
    ax.set_axis_off()
    ax.text(0.5, 0.94, "5 min 停输后轴向混气段增长", ha="center", va="center", fontsize=12, weight="bold", transform=ax.transAxes)
    ax.text(
        0.5,
        0.84,
        f"一维轴向模型：分子扩散展宽约 {erfc_width:.1f} m，低于 dx={dx:.0f} m 网格分辨率",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        transform=ax.transAxes,
    )

    card_w = 0.24
    card_h = 0.34
    y0 = 0.38
    xs = [0.12, 0.38, 0.64]
    fills = ["#e9f1fb", "#eaf5ef", "#fcefea"]
    edges = ["#4e79a7", "#59a14f", "#e15759"]
    for x0, label, value, fill, edge in zip(xs, labels, growth, fills, edges):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                card_w,
                card_h,
                boxstyle="round,pad=0.014,rounding_size=0.025",
                linewidth=1.1,
                edgecolor=edge,
                facecolor=fill,
                transform=ax.transAxes,
            )
        )
        ax.text(x0 + card_w / 2, y0 + card_h * 0.72, f"中断 {label}", ha="center", va="center", fontsize=10, weight="bold", color="#333333", transform=ax.transAxes)
        ax.text(x0 + card_w / 2, y0 + card_h * 0.43, f"{value:.0f} m", ha="center", va="center", fontsize=16, weight="bold", color=edge, transform=ax.transAxes)
        ax.text(x0 + card_w / 2, y0 + card_h * 0.18, "按网格统计增长量", ha="center", va="center", fontsize=8, color="#555555", transform=ax.transAxes)

    ax.text(
        0.5,
        0.18,
        f"解释：0 m 不是没有风险，而是 300 s 内轴向分子扩散没有跨过一个网格。\n停输时 Fr=0，重力流侵入量级约 {gravity:.0f} m，仍需关注截面分层风险。",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
        transform=ax.transAxes,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def run_task2_interrupt(outdir=None, dx: float = 10.0) -> pd.DataFrame:
    out = ensure_dir(outdir or output_dir("task2"))
    rows = []
    profile_pairs = []
    shutdown_duration_s = 300.0
    sqrt_dt = float(np.sqrt(D_MOL_DEFAULT * shutdown_duration_s))
    erfc_1_99_width = float(7.29 * sqrt_dt)
    gravity_speed = gravity_current_speed(1.2)
    gravity_intrusion = gravity_speed * shutdown_duration_s
    for frac in [0.30, 0.60, 0.80]:
        target = frac * L_PIPE
        approx_t = target / 7.0
        normal_params = SimulationParams(
            D=1.2,
            u_nominal=7.0,
            p_back_abs=0.10e6,
            roughness=0.05e-3,
            dx=dx,
            t_end=1.25 * approx_t,
            output_times=np.linspace(0.0, 1.25 * approx_t, 101),
            pressure_mode="friction_profile",
        )
        normal = run_simulation(normal_params)
        stop_idx = _nearest_stop_index(normal, target)
        before_profile = normal.profiles[stop_idx]
        before_metrics = normal.metrics[stop_idx]
        shutdown_params = SimulationParams(
            D=1.2,
            u_nominal=0.0,
            p_back_abs=0.10e6,
            roughness=0.05e-3,
            dx=dx,
            t_end=shutdown_duration_s,
            output_times=[0.0, shutdown_duration_s],
            K_model="molecular",
            D_mol=D_MOL_DEFAULT,
            pressure_mode="reference_pressure",
            initial_profiles={"H2": before_profile[:, 0], "N2": before_profile[:, 1], "Air": before_profile[:, 2]},
        )
        shutdown = run_simulation(shutdown_params)
        after_metrics = shutdown.metrics[-1]
        rows.append(
            {
                "interrupt_position_fraction": frac,
                "interrupt_position_m": target,
                "stop_time_s": float(normal.times[stop_idx]),
                "mixed_length_before_m": before_metrics["mixed_length_m"],
                "mixed_length_after_300s_m": after_metrics["mixed_length_m"],
                "mixed_length_growth_m": after_metrics["mixed_length_m"] - before_metrics["mixed_length_m"],
                "effective_n2_before_m": before_metrics["effective_n2_length_m"],
                "effective_n2_after_300s_m": after_metrics["effective_n2_length_m"],
                "effective_n2_loss_m": before_metrics["effective_n2_length_m"] - after_metrics["effective_n2_length_m"],
                "flammable_length_before_m": before_metrics["flammable_length_m"],
                "flammable_length_after_300s_m": after_metrics["flammable_length_m"],
                "Fr_before": before_metrics["Fr"],
                "Fr_shutdown": 0.0,
                "shutdown_sqrt_Dt_m": sqrt_dt,
                "shutdown_erfc_1_99_width_estimate_m": erfc_1_99_width,
                "shutdown_gravity_current_speed_mps": gravity_speed,
                "shutdown_gravity_intrusion_estimate_m": gravity_intrusion,
                "dx_m": dx,
                "limitation_note": "停输时 Fr=0；300 s 内一维轴向分子扩散低于 10 m 网格分辨率，但重力流侵入可达数百米量级。一维轴向模型不能替代水平管截面分层分析。",
            }
        )
        profile_pairs.append((frac, normal, stop_idx, shutdown))
    df = pd.DataFrame(rows)
    df.to_csv(out / "interrupt_summary.csv", index=False)
    for frac, normal, stop_idx, shutdown in profile_pairs:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.8))
        labels = [("停输前", normal.x_grid, normal.profiles[stop_idx]), ("停输300s后", shutdown.x_grid, shutdown.profiles[-1])]
        for label, x, prof in labels:
            ls = "-" if label == "停输前" else "--"
            ax.plot(x / 1000.0, prof[:, 0], ls, label=f"H2 {label}")
            ax.plot(x / 1000.0, prof[:, 1], ls, label=f"N2 {label}")
            ax.plot(x / 1000.0, prof[:, 2], ls, label=f"Air {label}")
        ax.set_xlabel("管道位置 x (km)")
        ax.set_ylabel("摩尔分数 / 体积分数")
        ax.set_title(f"{frac:.0%} 管长中断前后浓度剖面对比")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"interrupt_profiles_{int(frac*100)}pct.png", dpi=180)
        plt.close(fig)
    _plot_task2_bar_cn(df, "effective_n2_after_300s_m", out / "effective_n2_after_interrupt.png", "有效 N2 隔离段剩余长度 (m)")
    _plot_task2_zero_growth_cn(df, out / "mixed_length_growth_interrupt.png")
    _plot_task2_zero_growth_cn(df, out / "mixed_length_growth_interrupt_cn.png")
    fr_df = df.assign(interrupt_position_label=[f"{int(v * 100)}%" for v in df["interrupt_position_fraction"]])
    plot_bar(
        fr_df,
        "interrupt_position_label",
        "Fr_before",
        out / "fr_before_interrupt.png",
        title="不同中断位置的停输前 Fr 指标",
        xlabel="中断位置（H2 前缘占管长比例）",
        ylabel="停输前 Fr",
    )
    return df
