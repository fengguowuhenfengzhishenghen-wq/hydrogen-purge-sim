"""Explicit finite-volume solver for H2/N2/Air purge simulations."""

from __future__ import annotations

import numpy as np

from .config import SimulationParams, SimulationResult
from .constants import N2_PLUG_FRACTION
from .dispersion import compute_K
from .gas_props import mixture_density, mixture_viscosity_simple, reynolds_number
from .grid import create_grid
from .metrics import effective_n2_length, flammable_length, front_position, mixed_length, pair_transition_length
from .pressure import compute_pressure_profile
from .safety import compute_density_froude


SPECIES = ("H2", "N2", "Air")


def van_leer_limiter(r):
    r = np.asarray(r, dtype=float)
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def initial_profile(params: SimulationParams, x_grid: np.ndarray) -> np.ndarray:
    """Create or validate initial mole-fraction profiles."""

    N = len(x_grid)
    X = np.zeros((N, 3), dtype=float)
    if params.initial_profiles is not None:
        for j, name in enumerate(SPECIES):
            if name not in params.initial_profiles:
                raise ValueError(f"initial_profiles missing {name}")
            X[:, j] = np.asarray(params.initial_profiles[name], dtype=float)
    else:
        plug_len = params.n2_plug_fraction * params.L
        n2_mask = x_grid <= plug_len
        X[n2_mask, 1] = 1.0
        X[~n2_mask, 2] = 1.0
    return normalize_species(X)


def normalize_species(X: np.ndarray) -> np.ndarray:
    """Clip negative values and renormalize each cell to sum to one."""

    X = np.maximum(X, 0.0)
    row_sum = np.sum(X, axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1.0e-15
    if np.any(bad):
        X[bad, :] = np.array([0.0, 0.0, 1.0])
        row_sum = np.sum(X, axis=1, keepdims=True)
    return X / row_sum


def _tvd_face_states_positive(X: np.ndarray, inlet: np.ndarray, slope_scale: float) -> np.ndarray:
    N, n_species = X.shape
    slopes = np.zeros_like(X)
    if N > 2:
        delta_up = X[1:-1] - X[:-2]
        delta_dn = X[2:] - X[1:-1]
        r = delta_up / np.where(np.abs(delta_dn) < 1.0e-14, 1.0e-14, delta_dn)
        slopes[1:-1] = van_leer_limiter(r) * delta_dn
    face = np.zeros((N + 1, n_species), dtype=float)
    face[0] = inlet
    face[1:N] = X[:-1] + 0.5 * slope_scale * slopes[:-1]
    face[N] = X[-1]
    return _normalize_face_states(face)


def _tvd_face_states_negative(X: np.ndarray, outlet_state: np.ndarray, slope_scale: float) -> np.ndarray:
    N, n_species = X.shape
    slopes = np.zeros_like(X)
    if N > 2:
        delta_up = X[2:] - X[1:-1]
        delta_dn = X[1:-1] - X[:-2]
        r = delta_up / np.where(np.abs(delta_dn) < 1.0e-14, 1.0e-14, delta_dn)
        slopes[1:-1] = van_leer_limiter(r) * delta_dn
    face = np.zeros((N + 1, n_species), dtype=float)
    face[0] = X[0]
    face[1:N] = X[1:] - 0.5 * slope_scale * slopes[1:]
    face[N] = outlet_state
    return _normalize_face_states(face)


def _normalize_face_states(face: np.ndarray) -> np.ndarray:
    face = np.clip(face, 0.0, 1.0)
    row_sum = np.sum(face, axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1.0e-15
    if np.any(bad):
        face[bad, :] = np.array([0.0, 0.0, 1.0])
        row_sum = np.sum(face, axis=1, keepdims=True)
    return face / row_sum


def advective_flux_tvd(X: np.ndarray, u: float, inlet: np.ndarray, courant: float = 0.0) -> np.ndarray:
    if abs(u) < 1.0e-15:
        return np.zeros((X.shape[0] + 1, X.shape[1]), dtype=float)
    slope_scale = float(np.clip(1.0 - courant, 0.0, 1.0))
    if u > 0.0:
        return u * _tvd_face_states_positive(X, inlet, slope_scale)
    return u * _tvd_face_states_negative(X, X[-1], slope_scale)


def diffusion_flux_centered(X: np.ndarray, K, dx: float) -> np.ndarray:
    N, n_species = X.shape
    flux = np.zeros((N + 1, n_species), dtype=float)
    K_arr = np.asarray(K, dtype=float)
    if K_arr.ndim == 0:
        K_face = float(K_arr)
        flux[1:N] = K_face * (X[1:] - X[:-1]) / dx
    else:
        K_faces = 0.5 * (K_arr[:-1] + K_arr[1:])
        flux[1:N] = K_faces[:, None] * (X[1:] - X[:-1]) / dx
    return flux


def _output_times(params: SimulationParams) -> np.ndarray:
    if params.t_end is None and params.output_times is None:
        t_end = 1.2 * params.L / max(abs(params.u_nominal), 1.0e-9)
    elif params.t_end is None:
        raw_times = np.asarray(params.output_times, dtype=float)
        t_end = float(np.max(raw_times)) if raw_times.size else 0.0
    else:
        t_end = float(params.t_end)
    if params.output_times is None:
        times = np.linspace(0.0, t_end, 101)
    else:
        times = np.asarray(params.output_times, dtype=float)
        times = times[(times >= 0.0) & (times <= t_end + 1.0e-9)]
        times = np.unique(np.concatenate(([0.0, t_end], times)))
    return np.sort(times)


def _diagnostics(params: SimulationParams, x_grid: np.ndarray, X: np.ndarray, u: float):
    x_mean = tuple(float(v) for v in np.mean(X, axis=0))
    p, dp = compute_pressure_profile(
        params.pressure_mode,
        x_grid,
        params.L,
        params.effective_back_pressure(),
        u,
        params.D,
        params.roughness,
        params.T,
        x_mean,
    )
    rho = mixture_density(p, params.T, X[:, 0], X[:, 1], X[:, 2])
    mu = mixture_viscosity_simple(X[:, 0], X[:, 1], X[:, 2])
    Re = reynolds_number(rho, u, params.D, mu)
    if params.K_override is not None:
        K = np.full_like(x_grid, float(params.K_override), dtype=float)
    else:
        K_scalar = float(
            compute_K(
                u,
                params.D,
                Re=np.nanmean(Re),
                model=params.K_model,
                beta=params.beta_K,
                D_mol=params.D_mol,
            )
        )
        K = np.full_like(x_grid, K_scalar, dtype=float)
    Fr = compute_density_froude(u, params.D, p, params.T)
    return p, rho, Re, K, Fr, dp


def _record_metrics(t: float, params: SimulationParams, x_grid, X, p, rho, Re, K, Fr, dp):
    return {
        "time_s": float(t),
        "mixed_length_m": mixed_length(X[:, 0], X[:, 1], X[:, 2], params.L / len(x_grid)),
        "mixed_length_095_m": mixed_length(X[:, 0], X[:, 1], X[:, 2], params.L / len(x_grid), threshold=0.95),
        "h2_n2_interface_length_m": pair_transition_length(X[:, 0], X[:, 1], params.L / len(x_grid)),
        "n2_air_interface_length_m": pair_transition_length(X[:, 1], X[:, 2], params.L / len(x_grid)),
        "flammable_length_m": flammable_length(X[:, 0], X[:, 2], params.L / len(x_grid)),
        "effective_n2_length_m": effective_n2_length(X[:, 0], X[:, 1], X[:, 2], params.L / len(x_grid)),
        "h2_front_m": front_position(X[:, 0], x_grid, level=0.5),
        "Re_min": float(np.nanmin(Re)),
        "Re_max": float(np.nanmax(Re)),
        "K_min": float(np.nanmin(K)),
        "K_max": float(np.nanmax(K)),
        "Fr": float(np.nanmean(Fr)),
        "dp_estimated_Pa": float(dp),
        "rho_mean_kg_m3": float(np.nanmean(rho)),
        "p_mean_Pa": float(np.nanmean(p)),
    }


def run_simulation(params: SimulationParams) -> SimulationResult:
    """Run one explicit FVM simulation and return recorded profiles."""

    grid = create_grid(params.L, params.dx)
    X = initial_profile(params, grid.x)
    inlet = np.asarray(params.inlet_composition, dtype=float)
    inlet = inlet / np.sum(inlet)
    times = _output_times(params)
    t_final = float(times[-1])
    t = 0.0
    output_i = 0
    profiles = []
    pressures = []
    densities = []
    reynolds_list = []
    dispersions = []
    froudes = []
    metric_rows = []
    flux_integrals = []
    inventories = []
    cumulative_flux = np.zeros(3, dtype=float)

    def record(now: float):
        p, rho, Re, K, Fr, dp = _diagnostics(params, grid.x, X, params.u_nominal)
        profiles.append(X.copy())
        pressures.append(p.copy())
        densities.append(rho.copy())
        reynolds_list.append(Re.copy())
        dispersions.append(K.copy())
        froudes.append(Fr.copy())
        metric_rows.append(_record_metrics(now, params, grid.x, X, p, rho, Re, K, Fr, dp))
        flux_integrals.append(cumulative_flux.copy())
        inventories.append(np.sum(X, axis=0) * grid.dx)

    record(0.0)
    output_i = 1
    step = 0
    while t < t_final - 1.0e-12:
        if step >= params.max_steps:
            raise RuntimeError(f"Exceeded max_steps={params.max_steps}; reduce t_end or increase dx")
        p, rho, Re, K, _Fr, _dp = _diagnostics(params, grid.x, X, params.u_nominal)
        K_max = float(np.nanmax(K))
        if abs(params.u_nominal) > 1.0e-15:
            dt_adv = params.CFL * grid.dx / abs(params.u_nominal)
        else:
            dt_adv = np.inf
        dt_diff = params.cfl_diff_safety * grid.dx * grid.dx / max(2.0 * K_max, 1.0e-30)
        dt = min(dt_adv, dt_diff)
        next_output = float(times[output_i]) if output_i < len(times) else t_final
        dt = min(dt, t_final - t, next_output - t)
        if dt <= 1.0e-14:
            record(next_output)
            t = next_output
            output_i += 1
            continue

        courant = abs(params.u_nominal) * dt / grid.dx if abs(params.u_nominal) > 1.0e-15 else 0.0
        adv = advective_flux_tvd(X, params.u_nominal, inlet, courant=courant)
        diff = diffusion_flux_centered(X, K, grid.dx)
        rhs = -(adv[1:] - adv[:-1]) / grid.dx + (diff[1:] - diff[:-1]) / grid.dx
        X = normalize_species(X + dt * rhs)
        cumulative_flux += (adv[0] - adv[-1]) * dt
        t += dt
        step += 1

        if output_i < len(times) and t >= times[output_i] - 1.0e-10:
            record(float(times[output_i]))
            output_i += 1

    return SimulationResult(
        params=params,
        x_grid=grid.x,
        x_edges=grid.edges,
        dx=grid.dx,
        times=np.asarray(times, dtype=float),
        profiles=np.asarray(profiles),
        pressure=np.asarray(pressures),
        density=np.asarray(densities),
        reynolds=np.asarray(reynolds_list),
        dispersion=np.asarray(dispersions),
        froude=np.asarray(froudes),
        metrics=metric_rows,
        boundary_flux_integral=np.asarray(flux_integrals),
        inventory=np.asarray(inventories),
    )
