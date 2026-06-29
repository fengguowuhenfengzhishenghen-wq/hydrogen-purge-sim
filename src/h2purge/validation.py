"""Validation cases for the one-dimensional purge solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfc

from .config import SimulationParams
from .constants import D_MOL_DEFAULT, L_PIPE
from .dispersion import numerical_dispersion_upper_bound
from .grid import create_grid
from .io_utils import ensure_dir, output_dir
from .metrics import summarize_metric_series
from .plots import plot_concentration_profiles
from .solver_fvm import run_simulation


def erfc_reference(x, x0: float, u: float, K: float, t: float):
    return 0.5 * erfc((x - x0 - u * t) / (2.0 * np.sqrt(K * t)))


def run_erfc_validation(outdir=None, make_plot: bool = True) -> dict:
    out = ensure_dir(outdir or output_dir("validation"))
    L = 3000.0
    dx = 10.0
    u = 1.0
    K = 5.0
    t_end = 300.0
    x0 = 900.0
    grid = create_grid(L, dx)
    x_h2 = np.where(grid.x <= x0, 1.0, 0.0)
    params = SimulationParams(
        L=L,
        D=1.2,
        u_nominal=u,
        p_back_abs=0.10e6,
        dx=dx,
        t_end=t_end,
        output_times=[0.0, t_end],
        K_override=K,
        initial_profiles={"H2": x_h2, "N2": 1.0 - x_h2, "Air": np.zeros_like(x_h2)},
    )
    result = run_simulation(params)
    numerical = result.species_profile("H2")
    analytical = erfc_reference(result.x_grid, x0, u, K, t_end)
    mask = (result.x_grid > 200.0) & (result.x_grid < 2200.0)
    err = numerical[mask] - analytical[mask]
    row = {
        "case": "single_interface_erfc",
        "L2_error": float(np.sqrt(np.mean(err * err))),
        "Linf_error": float(np.max(np.abs(err))),
        "dx_m": dx,
        "u_mps": u,
        "K_m2_s": K,
        "t_s": t_end,
    }
    pd.DataFrame([row]).to_csv(out / "erfc_validation.csv", index=False)
    if make_plot:
        import matplotlib.pyplot as plt

        fig, (ax, ax_err) = plt.subplots(
            2,
            1,
            figsize=(8.0, 5.2),
            gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.10},
            sharex=True,
        )
        ax.plot(result.x_grid, numerical, label="数值解 TVD-FVM", lw=2.0, color="#1f77b4")
        ax.plot(result.x_grid, analytical, "--", label="erfc 解析解", lw=1.8, color="#ff7f0e")
        ax.set_ylabel("H2 摩尔分数")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        ax.text(
            0.03,
            0.08,
            f"L2={row['L2_error']:.4f}, Linf={row['Linf_error']:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.90},
        )

        full_err = numerical - analytical
        ax_err.plot(result.x_grid, full_err, color="#444444", lw=1.3)
        ax_err.axhline(0.0, color="#999999", lw=0.8)
        ax_err.set_xlabel("位置 x (m)")
        ax_err.set_ylabel("数值-解析")
        ax_err.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "erfc_validation.png", dpi=180)
        plt.close(fig)
    return row


def species_sum_error(result) -> float:
    return float(np.max(np.abs(np.sum(result.profiles, axis=2) - 1.0)))


def boundary_inventory_balance_error(result) -> dict:
    """Compare species inventory change with net advective boundary flux."""

    inventory_delta = result.inventory[-1] - result.inventory[0]
    flux_delta = result.boundary_flux_integral[-1]
    err = inventory_delta - flux_delta
    return {
        "balance_error_H2_m": float(err[0]),
        "balance_error_N2_m": float(err[1]),
        "balance_error_Air_m": float(err[2]),
        "balance_error_max_abs_m": float(np.max(np.abs(err))),
    }


def run_grid_independence(outdir=None) -> pd.DataFrame:
    out = ensure_dir(outdir or output_dir("validation"))
    rows = []
    for dx in [20.0, 10.0, 5.0]:
        params = SimulationParams(
            D=1.2,
            u_nominal=7.0,
            p_back_abs=0.10e6,
            dx=dx,
            t_end=900.0,
            output_times=np.linspace(0.0, 900.0, 31),
            pressure_mode="friction_profile",
        )
        result = run_simulation(params)
        row = {"dx_m": dx, "species_sum_max_error": species_sum_error(result)}
        row.update(boundary_inventory_balance_error(result))
        row.update(summarize_metric_series(result.metrics))
        rows.append(row)
    df = pd.DataFrame(rows)
    fine = df[df["dx_m"] == 5.0].iloc[0]
    for col in ["mixed_length_final_m", "mixed_length_max_m", "flammable_length_max_m", "effective_n2_min_m"]:
        denom = abs(fine[col]) if abs(fine[col]) > 1.0e-12 else 1.0
        df[f"{col}_relerr_vs_5m"] = (df[col] - fine[col]).abs() / denom
    df.to_csv(out / "grid_independence.csv", index=False)
    return df


def run_numerical_dispersion_check(outdir=None, u: float = 7.0, dx: float = 10.0, CFL: float = 0.45, K: float = 4.2) -> dict:
    out = ensure_dir(outdir or output_dir("validation"))
    D_num = numerical_dispersion_upper_bound(u, dx, CFL)
    row = {
        "u_mps": u,
        "dx_m": dx,
        "CFL": CFL,
        "D_num_upper_m2_s": D_num,
        "K_physical_m2_s": K,
        "D_num_over_K": D_num / K if K > 0 else np.inf,
        "note": "Use smaller dx or higher-order settings if this ratio approaches or exceeds 1.",
    }
    pd.DataFrame([row]).to_csv(out / "numerical_dispersion_check.csv", index=False)
    return row


def run_beta_sensitivity(outdir=None) -> pd.DataFrame:
    """Check how the empirical beta_uD dispersion factor affects interface width."""

    out = ensure_dir(outdir or output_dir("validation"))
    rows = []
    D = 1.2
    u = 7.0
    t_end = 1.2 * L_PIPE / u
    for beta in [0.2, 0.5, 0.8]:
        params = SimulationParams(
            D=D,
            u_nominal=u,
            p_back_abs=0.10e6,
            dx=10.0,
            beta_K=beta,
            t_end=t_end,
            output_times=np.linspace(0.0, t_end, 81),
            pressure_mode="friction_profile",
        )
        result = run_simulation(params)
        series = summarize_metric_series(result.metrics)
        sigma = float(np.sqrt(2.0 * beta * D * L_PIPE))
        estimate = 9.3 * sigma
        rows.append(
            {
                "beta_K": beta,
                "D_m": D,
                "u_mps": u,
                "K_m2_s": float(beta * u * D + D_MOL_DEFAULT),
                "sigma_m": sigma,
                "two_interface_1_99_estimate_m": estimate,
                "mixed_length_max_m": series["mixed_length_max_m"],
                "relative_error_vs_9p3sigma": abs(series["mixed_length_max_m"] - estimate) / estimate,
                "note": "beta sensitivity for Taylor-Aris/Austin-Palfrey style engineering calibration",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "beta_sensitivity.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(df["beta_K"], df["mixed_length_max_m"], "o-", label="数值最大混气段")
    ax.plot(df["beta_K"], df["two_interface_1_99_estimate_m"], "s--", label="9.3σ 估算")
    ax.set_xlabel("弥散系数 β")
    ax.set_ylabel("混气段长度 (m)")
    ax.set_title("β 对轴向弥散与混气段尺度的影响")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "beta_sensitivity.png", dpi=180)
    plt.close(fig)
    return df


def write_austin_palfrey_scaling_note(outdir=None) -> Path:
    out = ensure_dir(outdir or output_dir("validation"))
    path = out / "scaling_check.md"
    path.write_text(
        "# Austin-Palfrey scaling check\n\n"
        "Austin-Palfrey style relations are treated only as an engineering order-of-magnitude cross-check here. "
        "They were developed for liquid sequential transport interface growth, while this project models weakly "
        "compressible gases with strong H2/air density contrast and possible horizontal-pipe stratification. "
        "Therefore the constants are not directly transferred into the gas purge model or used as strict validation.\n\n"
        "The implemented beta_uD model should be read as a Taylor-Aris/Austin-Palfrey inspired engineering closure. "
        "The default beta=0.5 is deliberately conservative; outputs include beta=0.2/0.5/0.8 sensitivity so the report "
        "does not rely on a single uncalibrated knob.\n",
        encoding="utf-8",
    )
    return path


def run_representative_case(outdir=None):
    out = ensure_dir(outdir or output_dir("validation"))
    params = SimulationParams(
        D=1.2,
        u_nominal=7.0,
        p_back_abs=0.10e6,
        dx=10.0,
        t_end=900.0,
        output_times=np.linspace(0.0, 900.0, 31),
        pressure_mode="friction_profile",
    )
    result = run_simulation(params)
    pd.DataFrame(result.metrics).to_csv(out / "representative_metrics.csv", index=False)
    plot_concentration_profiles(result, out / "representative_profiles.png", title="DN1200, 7 m/s, 0.10 MPa")
    return result


def run_all_validations(outdir=None) -> dict:
    out = ensure_dir(outdir or output_dir("validation"))
    rep = run_representative_case(out)
    erfc_row = run_erfc_validation(out)
    grid_df = run_grid_independence(out)
    beta_df = run_beta_sensitivity(out)
    disp_row = run_numerical_dispersion_check(out, K=0.5 * 7.0 * 1.2 + D_MOL_DEFAULT)
    scaling_path = write_austin_palfrey_scaling_note(out)
    summary = {
        "species_sum_max_error_representative": species_sum_error(rep),
        **boundary_inventory_balance_error(rep),
        "erfc_L2_error": erfc_row["L2_error"],
        "erfc_Linf_error": erfc_row["Linf_error"],
        "grid_rows": len(grid_df),
        "beta_sensitivity_rows": len(beta_df),
        "scaling_note": str(scaling_path),
    }
    pd.DataFrame([summary]).to_csv(out / "validation_summary.csv", index=False)
    return summary
