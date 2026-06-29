"""Simplified pressure profiles used by the one-dimensional model."""

from __future__ import annotations

import numpy as np

from .gas_props import (
    friction_factor_haaland,
    mixture_density,
    mixture_viscosity_simple,
    reynolds_number,
)


def reference_pressure(x_grid, p_back_abs: float) -> tuple[np.ndarray, float]:
    """Uniform pressure equal to outlet absolute backpressure."""

    p = np.full_like(np.asarray(x_grid, dtype=float), float(p_back_abs), dtype=float)
    return p, 0.0


def friction_profile(
    x_grid,
    L: float,
    p_back_abs: float,
    u: float,
    D: float,
    roughness: float,
    T: float,
    x_mean: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Estimate a linear Darcy-Weisbach pressure profile."""

    x_h2, x_n2, x_air = x_mean
    rho = float(mixture_density(p_back_abs, T, x_h2, x_n2, x_air))
    mu = float(mixture_viscosity_simple(x_h2, x_n2, x_air))
    Re = float(reynolds_number(rho, u, D, mu))
    f = float(friction_factor_haaland(Re, roughness, D))
    dp = f * (L / D) * 0.5 * rho * u * u
    p_in = p_back_abs + max(dp, 0.0)
    x_arr = np.asarray(x_grid, dtype=float)
    p = p_in - dp * x_arr / max(L, 1.0)
    if np.any(p <= 0.0):
        raise ValueError("Computed pressure profile contains non-positive pressure")
    return p, dp


def compute_pressure_profile(
    mode: str,
    x_grid,
    L: float,
    p_back_abs: float,
    u: float,
    D: float,
    roughness: float,
    T: float,
    x_mean: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Dispatch pressure profile modes."""

    if mode == "reference_pressure":
        return reference_pressure(x_grid, p_back_abs)
    if mode == "friction_profile":
        return friction_profile(x_grid, L, p_back_abs, u, D, roughness, T, x_mean)
    raise ValueError(f"Unknown pressure mode: {mode}")
