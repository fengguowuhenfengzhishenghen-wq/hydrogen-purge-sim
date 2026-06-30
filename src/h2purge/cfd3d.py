"""Utilities for offline 3D CFD case preparation.

The functions here prepare local cylindrical 3D initial fields for external
CFD solvers.  They do not solve CFD equations and are intentionally separate
from the Streamlit UI and from the 1D purge solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import M_H2, M_N2

M_O2 = 31.998e-3


@dataclass(frozen=True)
class CylindricalSampleGrid:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    radial_m: np.ndarray
    theta_rad: np.ndarray
    axial_index: np.ndarray


def air_split_mole_fractions(x_h2: np.ndarray, x_n2: np.ndarray, x_air: np.ndarray) -> np.ndarray:
    """Convert H2/N2/Air mole fractions to H2/N2/O2 mole fractions."""

    h2 = np.asarray(x_h2, dtype=float)
    n2 = np.asarray(x_n2, dtype=float) + 0.79 * np.asarray(x_air, dtype=float)
    o2 = 0.21 * np.asarray(x_air, dtype=float)
    mole = np.column_stack([h2, n2, o2])
    total = mole.sum(axis=1, keepdims=True)
    return mole / np.where(total <= 1.0e-15, 1.0, total)


def mole_to_mass_fractions(mole_h2_n2_o2: np.ndarray) -> np.ndarray:
    """Convert H2/N2/O2 mole fractions to mass fractions."""

    mole = np.clip(np.asarray(mole_h2_n2_o2, dtype=float), 0.0, 1.0)
    molar_mass = np.array([M_H2, M_N2, M_O2], dtype=float)
    mass = mole * molar_mass[None, :]
    total = mass.sum(axis=1, keepdims=True)
    return mass / np.where(total <= 1.0e-15, 1.0, total)


def make_cylindrical_sample_grid(length_m: float, diameter_m: float, nx: int, nr: int, ntheta: int) -> CylindricalSampleGrid:
    """Create cell-centre-like sample points inside a cylindrical pipe."""

    radius = 0.5 * float(diameter_m)
    x_centres = (np.arange(nx, dtype=float) + 0.5) * float(length_m) / float(nx)
    r_edges = np.linspace(0.0, radius, nr + 1)
    # Area-weighted radial sample location in each annulus.
    r_centres = np.sqrt(0.5 * (r_edges[:-1] ** 2 + r_edges[1:] ** 2))
    theta = (np.arange(ntheta, dtype=float) + 0.5) * 2.0 * np.pi / float(ntheta)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    rs: list[float] = []
    ts: list[float] = []
    axial: list[int] = []
    for ix, x in enumerate(x_centres):
        for r in r_centres:
            for th in theta:
                xs.append(float(x))
                ys.append(float(r * np.cos(th)))
                zs.append(float(r * np.sin(th)))
                rs.append(float(r))
                ts.append(float(th))
                axial.append(ix)
    return CylindricalSampleGrid(
        x_m=np.asarray(xs),
        y_m=np.asarray(ys),
        z_m=np.asarray(zs),
        radial_m=np.asarray(rs),
        theta_rad=np.asarray(ts),
        axial_index=np.asarray(axial, dtype=int),
    )


def map_profile_to_grid(profile_x_m: np.ndarray, profile_mole_h2_n2_air: np.ndarray, grid: CylindricalSampleGrid) -> tuple[np.ndarray, np.ndarray]:
    """Map a 1D H2/N2/Air mole-fraction profile to 3D cylindrical samples."""

    x_src = np.asarray(profile_x_m, dtype=float)
    p = np.asarray(profile_mole_h2_n2_air, dtype=float)
    h2 = np.interp(grid.x_m, x_src, p[:, 0])
    n2 = np.interp(grid.x_m, x_src, p[:, 1])
    air = np.interp(grid.x_m, x_src, p[:, 2])
    mole = air_split_mole_fractions(h2, n2, air)
    mass = mole_to_mass_fractions(mole)
    return mole, mass


def initial_field_metrics(mole_h2_n2_o2: np.ndarray, mass_h2_n2_o2: np.ndarray) -> dict[str, float]:
    """Return consistency metrics for a prepared 3D initial field."""

    mole = np.asarray(mole_h2_n2_o2, dtype=float)
    mass = np.asarray(mass_h2_n2_o2, dtype=float)
    return {
        "mole_sum_max_error": float(np.max(np.abs(mole.sum(axis=1) - 1.0))),
        "mass_sum_max_error": float(np.max(np.abs(mass.sum(axis=1) - 1.0))),
        "x_h2_max": float(mole[:, 0].max()),
        "x_o2_max": float(mole[:, 2].max()),
        "y_h2_max": float(mass[:, 0].max()),
        "y_o2_max": float(mass[:, 2].max()),
    }
