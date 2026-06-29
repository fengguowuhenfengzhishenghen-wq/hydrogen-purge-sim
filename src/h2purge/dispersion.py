"""Axial dispersion coefficient models."""

from __future__ import annotations

import numpy as np


def compute_K(u, D, Re=None, model: str = "beta_uD", beta: float = 0.5, D_mol: float = 1.0e-4):
    """Return axial dispersion coefficient in m2/s.

    The default engineering model is K = D_mol + beta * |u| * D.
    """

    if model == "constant":
        return np.asarray(D_mol, dtype=float)
    if model == "molecular":
        return np.asarray(D_mol, dtype=float)
    if model == "alpha_L":
        return np.asarray(D_mol + beta * abs(u), dtype=float)
    if model == "beta_uD":
        return np.asarray(D_mol + beta * abs(u) * D, dtype=float)
    raise ValueError(f"Unknown dispersion model: {model}")


def numerical_dispersion_upper_bound(u: float, dx: float, CFL: float) -> float:
    """Conservative first-order upwind numerical dispersion estimate."""

    return abs(u) * dx * max(0.0, 1.0 - CFL) / 2.0
