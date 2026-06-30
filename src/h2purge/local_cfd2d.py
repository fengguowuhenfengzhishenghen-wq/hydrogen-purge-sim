"""Local 2D low-Mach buoyant multi-species CFD check.

This module is intentionally separate from the 1D purge solver.  It solves a
small x-z shutdown window with a vorticity-streamfunction formulation:

    d omega/dt + u d omega/dx + w d omega/dz = nu Lap(omega) + d b/dx
    Lap(psi) = -omega
    u = d psi/dz, w = -d psi/dx
    d x_i/dt + u d x_i/dx + w d x_i/dz = D_i Lap(x_i)

The density field is computed from H2/N2/Air mole fractions and enters the
momentum equation through the buoyancy term b.  This is a local 2D low-Mach CFD
screen for shutdown stratification, not a full 3D compressible pipe CFD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import D_MOL_DEFAULT, M_AIR, M_H2, M_N2, g
from .safety import flammable_mask


@dataclass(frozen=True)
class LocalCFD2DConfig:
    length_m: float = 1200.0
    diameter_m: float = 1.2
    nx: int = 121
    nz: int = 33
    duration_s: float = 300.0
    dt_s: float = 1.0
    nu_m2_s: float = 1.5e-4
    diffusivity_m2_s: float = D_MOL_DEFAULT
    poisson_iterations: int = 180
    poisson_omega: float = 1.0
    buoyancy_scale: float = 0.08
    velocity_clip_mps: float = 0.4
    vorticity_clip_s: float = 5.0


@dataclass
class LocalCFD2DResult:
    x_m: np.ndarray
    z_m: np.ndarray
    mole: np.ndarray
    u_mps: np.ndarray
    w_mps: np.ndarray
    omega: np.ndarray
    time_s: float
    metrics: dict[str, float | int | str]


MOLAR_MASS = np.array([M_H2, M_N2, M_AIR], dtype=float)


def normalize_mole(X: np.ndarray) -> np.ndarray:
    X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
    total = X.sum(axis=-1, keepdims=True)
    return X / np.where(total <= 1.0e-14, 1.0, total)


def mixture_molar_mass_field(X: np.ndarray) -> np.ndarray:
    return np.tensordot(X, MOLAR_MASS, axes=([-1], [0]))


def build_initial_field(profile_x: np.ndarray, profile_mole: np.ndarray, cfg: LocalCFD2DConfig):
    """Map a 1D mole-fraction profile into a local x-z CFD window."""

    x_src = np.asarray(profile_x, dtype=float)
    X_src = normalize_mole(np.asarray(profile_mole, dtype=float))
    center = 0.5 * (float(x_src.min()) + float(x_src.max()))
    half = 0.5 * cfg.length_m
    x = np.linspace(center - half, center + half, cfg.nx)
    z = np.linspace(-0.5 * cfg.diameter_m, 0.5 * cfg.diameter_m, cfg.nz)
    init_1d = np.column_stack([np.interp(x, x_src, X_src[:, i]) for i in range(3)])
    init_1d = normalize_mole(init_1d)
    X = np.repeat(init_1d[:, None, :], cfg.nz, axis=1)
    return x, z, normalize_mole(X)


def _laplacian(A: np.ndarray, dx: float, dz: float) -> np.ndarray:
    lap = np.zeros_like(A)
    lap[1:-1, 1:-1] = (
        (A[2:, 1:-1] - 2.0 * A[1:-1, 1:-1] + A[:-2, 1:-1]) / dx**2
        + (A[1:-1, 2:] - 2.0 * A[1:-1, 1:-1] + A[1:-1, :-2]) / dz**2
    )
    return lap


def _advect_upwind(A: np.ndarray, u: np.ndarray, w: np.ndarray, dx: float, dz: float) -> np.ndarray:
    d_x_back = (A - np.roll(A, 1, axis=0)) / dx
    d_x_forw = (np.roll(A, -1, axis=0) - A) / dx
    d_z_back = (A - np.roll(A, 1, axis=1)) / dz
    d_z_forw = (np.roll(A, -1, axis=1) - A) / dz
    d_x_back[0, :] = d_x_back[1, :]
    d_x_forw[-1, :] = d_x_forw[-2, :]
    d_z_back[:, 0] = d_z_back[:, 1]
    d_z_forw[:, -1] = d_z_forw[:, -2]
    return u * np.where(u >= 0.0, d_x_back, d_x_forw) + w * np.where(w >= 0.0, d_z_back, d_z_forw)


def _solve_streamfunction(omega: np.ndarray, dx: float, dz: float, iterations: int, relax: float) -> np.ndarray:
    psi = np.zeros_like(omega)
    dx2 = dx * dx
    dz2 = dz * dz
    denom = 2.0 * (dx2 + dz2)
    for _ in range(iterations):
        old = psi[1:-1, 1:-1]
        new = (
            dz2 * (psi[2:, 1:-1] + psi[:-2, 1:-1])
            + dx2 * (psi[1:-1, 2:] + psi[1:-1, :-2])
            + dx2 * dz2 * omega[1:-1, 1:-1]
        ) / denom
        psi[1:-1, 1:-1] = (1.0 - relax) * old + relax * new
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0
    return psi


def _velocity_from_psi(psi: np.ndarray, dx: float, dz: float, clip: float) -> tuple[np.ndarray, np.ndarray]:
    u = np.zeros_like(psi)
    w = np.zeros_like(psi)
    u[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dz)
    w[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    u = np.clip(u, -clip, clip)
    w = np.clip(w, -clip, clip)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = 0.0
    return u, w


def run_local_cfd2d(profile_x: np.ndarray, profile_mole: np.ndarray, cfg: LocalCFD2DConfig) -> LocalCFD2DResult:
    x, z, X = build_initial_field(profile_x, profile_mole, cfg)
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    omega = np.zeros((cfg.nx, cfg.nz), dtype=float)
    steps = int(round(cfg.duration_s / cfg.dt_s))

    for _ in range(steps):
        psi = _solve_streamfunction(omega, dx, dz, cfg.poisson_iterations, cfg.poisson_omega)
        u, w = _velocity_from_psi(psi, dx, dz, cfg.velocity_clip_mps)
        molar_mass = mixture_molar_mass_field(X)
        rho_ref = float(np.mean(molar_mass))
        buoyancy = -cfg.buoyancy_scale * g * (molar_mass - rho_ref) / max(rho_ref, 1.0e-12)
        dbdx = np.zeros_like(buoyancy)
        dbdx[1:-1, :] = (buoyancy[2:, :] - buoyancy[:-2, :]) / (2.0 * dx)

        omega_rhs = -_advect_upwind(omega, u, w, dx, dz) + cfg.nu_m2_s * _laplacian(omega, dx, dz) + dbdx
        omega = omega + cfg.dt_s * omega_rhs
        omega = np.clip(omega, -cfg.vorticity_clip_s, cfg.vorticity_clip_s)
        omega[0, :] = omega[-1, :] = omega[:, 0] = omega[:, -1] = 0.0

        for i in range(3):
            rhs = -_advect_upwind(X[:, :, i], u, w, dx, dz) + cfg.diffusivity_m2_s * _laplacian(X[:, :, i], dx, dz)
            X[:, :, i] = X[:, :, i] + cfg.dt_s * rhs
        X = normalize_mole(X)
        if not (np.isfinite(omega).all() and np.isfinite(X).all()):
            raise FloatingPointError("Local 2D CFD diverged; reduce dt_s, buoyancy_scale, or mesh aspect ratio.")

    psi = _solve_streamfunction(omega, dx, dz, cfg.poisson_iterations, cfg.poisson_omega)
    u, w = _velocity_from_psi(psi, dx, dz, cfg.velocity_clip_mps)
    if not (np.isfinite(u).all() and np.isfinite(w).all()):
        raise FloatingPointError("Local 2D CFD velocity field is not finite.")
    metrics = compute_local_cfd2d_metrics(x, z, X, u, w, cfg)
    return LocalCFD2DResult(x, z, X, u, w, omega, cfg.duration_s, metrics)


def compute_local_cfd2d_metrics(x: np.ndarray, z: np.ndarray, X: np.ndarray, u: np.ndarray, w: np.ndarray, cfg: LocalCFD2DConfig) -> dict[str, float | int | str]:
    x_h2 = X[:, :, 0]
    x_n2 = X[:, :, 1]
    x_air = X[:, :, 2]
    top = z >= 0.0
    bottom = z < 0.0
    flam = flammable_mask(x_h2, x_air)
    speed = np.sqrt(u * u + w * w)

    def z_centroid(field: np.ndarray) -> float:
        weights = np.maximum(field, 0.0)
        total = float(weights.sum())
        if total <= 1.0e-14:
            return 0.0
        return float((weights * z[None, :]).sum() / total)

    h2_top_mean = float(x_h2[:, top].mean())
    h2_bottom_mean = float(x_h2[:, bottom].mean())
    n2_top_mean = float(x_n2[:, top].mean())
    n2_bottom_mean = float(x_n2[:, bottom].mean())
    air_top_mean = float(x_air[:, top].mean())
    air_bottom_mean = float(x_air[:, bottom].mean())
    h2_z = z_centroid(x_h2)
    n2_z = z_centroid(x_n2)
    air_z = z_centroid(x_air)
    return {
        "solver": "Python 2D low-Mach buoyant multi-species transport check (vorticity-streamfunction)",
        "model_scope": "local x-z shutdown window; not full 3D compressible pipe CFD",
        "mesh_cells": int(cfg.nx * cfg.nz),
        "nx": int(cfg.nx),
        "nz": int(cfg.nz),
        "dt_s": float(cfg.dt_s),
        "stop_duration_s": float(cfg.duration_s),
        "local_length_m": float(x[-1] - x[0]),
        "diameter_m": float(cfg.diameter_m),
        "nu_m2_s": float(cfg.nu_m2_s),
        "diffusivity_m2_s": float(cfg.diffusivity_m2_s),
        "buoyancy_scale": float(cfg.buoyancy_scale),
        "velocity_clip_mps": float(cfg.velocity_clip_mps),
        "poisson_iterations": int(cfg.poisson_iterations),
        "h2_top_mean": h2_top_mean,
        "h2_bottom_mean": h2_bottom_mean,
        "n2_top_mean": n2_top_mean,
        "n2_bottom_mean": n2_bottom_mean,
        "air_top_mean": air_top_mean,
        "air_bottom_mean": air_bottom_mean,
        "top_bottom_h2_delta": h2_top_mean - h2_bottom_mean,
        "top_bottom_n2_delta": n2_top_mean - n2_bottom_mean,
        "top_bottom_air_delta": air_top_mean - air_bottom_mean,
        "h2_vertical_centroid_m": h2_z,
        "n2_vertical_centroid_m": n2_z,
        "air_vertical_centroid_m": air_z,
        "h2_air_vertical_separation_m": h2_z - air_z,
        "h2_n2_vertical_separation_m": h2_z - n2_z,
        "top_h2_max": float(x_h2[:, top].max()),
        "bottom_h2_mean": h2_bottom_mean,
        "flammable_area_ratio": float(flam.mean()),
        "flammable_volume_ratio": float(flam.mean()),
        "max_speed_mps": float(speed.max()),
        "mean_speed_mps": float(speed.mean()),
        "species_sum_max_error": float(np.max(np.abs(X.sum(axis=-1) - 1.0))),
    }
