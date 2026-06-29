"""Safety criteria and density Froude-number screening."""

from __future__ import annotations

import numpy as np

from .constants import M_AIR, M_H2, R, g


def compute_o2_fraction(x_air):
    return 0.21 * np.asarray(x_air)


def compute_inert_fraction(x_n2, x_air):
    return np.asarray(x_n2) + 0.79 * np.asarray(x_air)


def flammable_mask(x_h2, x_air, h2_low=0.04, h2_high=0.75, loc_o2=0.05):
    x_h2 = np.asarray(x_h2)
    x_o2 = compute_o2_fraction(x_air)
    return (x_h2 >= h2_low) & (x_h2 <= h2_high) & (x_o2 >= loc_o2)


def compute_density_froude(u, D, p, T):
    """Return density Froude number using H2 and air density contrast."""

    p_arr = np.asarray(p, dtype=float)
    rho_light = p_arr * M_H2 / (R * T)
    rho_heavy = p_arr * M_AIR / (R * T)
    g_prime = g * np.maximum(rho_heavy - rho_light, 0.0) / np.maximum(rho_heavy, 1.0e-12)
    denom = np.sqrt(np.maximum(g_prime * D, 1.0e-30))
    return abs(u) / denom


def reduced_gravity_h2_air():
    """Return ideal-gas reduced gravity for H2 displacing air."""

    return g * max(M_AIR - M_H2, 0.0) / M_AIR


def gravity_current_speed(D: float, coefficient: float = 0.5) -> float:
    """Estimate buoyant gravity-current speed in a horizontal pipe.

    This is an order-of-magnitude shutdown risk screen, not a 3D CFD model.
    """

    return float(coefficient * np.sqrt(reduced_gravity_h2_air() * D))


def stratification_risk_level(Fr: float) -> str:
    if Fr < 1.0:
        return "strong_stratification_risk"
    if Fr < 3.0:
        return "moderate_stratification_risk"
    return "lower_stratification_risk"
