"""Engineering metrics for purge concentration profiles."""

from __future__ import annotations

import numpy as np

from .safety import flammable_mask


def mixed_length(x_h2, x_n2, x_air, dx, threshold=0.99):
    arr = np.vstack([x_h2, x_n2, x_air])
    mask = np.max(arr, axis=0) < threshold
    return float(np.count_nonzero(mask) * dx)


def pair_transition_length(x_a, x_b, dx, eps=0.01):
    """Length where two named components coexist across one interface."""

    mask = (np.asarray(x_a) > eps) & (np.asarray(x_b) > eps)
    return float(np.count_nonzero(mask) * dx)


def flammable_length(x_h2, x_air, dx):
    return float(np.count_nonzero(flammable_mask(x_h2, x_air)) * dx)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def effective_n2_length(x_h2, x_n2, x_air, dx, n2_threshold=0.95):
    """Length of the longest continuous high-N2 isolation segment."""

    del x_h2, x_air
    mask = np.asarray(x_n2) >= n2_threshold
    runs = _contiguous_runs(mask)
    if not runs:
        return 0.0
    longest = max(end - start for start, end in runs)
    return float(longest * dx)


def front_position(x_species, x_grid, level=0.5):
    """Return rightmost position where a left-high front crosses a level."""

    x_species = np.asarray(x_species, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    above = x_species >= level
    if not np.any(above):
        return float("nan")
    if np.all(above):
        return float(x_grid[-1])
    idxs = np.where(above[:-1] & ~above[1:])[0]
    if len(idxs) == 0:
        idx = int(np.argmax(np.abs(x_species - level)))
        return float(x_grid[idx])
    i = int(idxs[-1])
    y0, y1 = x_species[i], x_species[i + 1]
    if abs(y1 - y0) < 1.0e-15:
        return float(x_grid[i])
    frac = (level - y0) / (y1 - y0)
    return float(x_grid[i] + frac * (x_grid[i + 1] - x_grid[i]))


def risk_time_integral(time_series, flammable_length_series):
    return float(np.trapz(flammable_length_series, time_series))


def summarize_metric_series(metric_rows: list[dict]) -> dict:
    times = np.array([row["time_s"] for row in metric_rows], dtype=float)
    flam = np.array([row["flammable_length_m"] for row in metric_rows], dtype=float)
    mixed = np.array([row["mixed_length_m"] for row in metric_rows], dtype=float)
    eff_n2 = np.array([row["effective_n2_length_m"] for row in metric_rows], dtype=float)
    return {
        "mixed_length_final_m": float(mixed[-1]),
        "mixed_length_max_m": float(np.max(mixed)),
        "flammable_length_max_m": float(np.max(flam)),
        "flammable_time_integral_m_s": risk_time_integral(times, flam),
        "effective_n2_min_m": float(np.min(eff_n2)),
    }
