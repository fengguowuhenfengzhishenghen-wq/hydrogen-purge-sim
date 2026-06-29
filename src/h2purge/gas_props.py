"""Gas mixture properties based on mole fractions."""

from __future__ import annotations

import numpy as np

from .constants import M_AIR, M_H2, M_N2, R, mu_AIR, mu_H2, mu_N2


def mixture_molar_mass(x_h2, x_n2, x_air):
    """Return mixture molar mass in kg/mol from mole fractions."""

    return x_h2 * M_H2 + x_n2 * M_N2 + x_air * M_AIR


def molar_concentration(p, T):
    """Return total molar concentration C = p / (R T), mol/m3."""

    return np.asarray(p) / (R * T)


def mixture_density(p, T, x_h2, x_n2, x_air):
    """Return ideal-gas mixture density in kg/m3."""

    return np.asarray(p) * mixture_molar_mass(x_h2, x_n2, x_air) / (R * T)


def mixture_viscosity_simple(x_h2, x_n2, x_air):
    """Return a simple mole-fraction linear viscosity mixture, Pa*s."""

    return x_h2 * mu_H2 + x_n2 * mu_N2 + x_air * mu_AIR


def reynolds_number(rho, u, D, mu):
    """Return Reynolds number with zero-viscosity protection."""

    mu_safe = np.maximum(np.asarray(mu), 1.0e-12)
    return np.asarray(rho) * abs(u) * D / mu_safe


def friction_factor_haaland(Re, roughness, D):
    """Darcy friction factor using laminar and Haaland turbulent formulas."""

    Re_arr = np.asarray(Re, dtype=float)
    Re_safe = np.maximum(Re_arr, 1.0)
    laminar = 64.0 / Re_safe
    turbulent = (
        -1.8
        * np.log10((roughness / (3.7 * D)) ** 1.11 + 6.9 / np.maximum(Re_safe, 1.0))
    ) ** -2
    return np.where(Re_safe < 2300.0, laminar, turbulent)
