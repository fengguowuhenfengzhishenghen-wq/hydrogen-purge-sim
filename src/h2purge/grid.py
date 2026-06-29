"""One-dimensional cell-centered grid utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid1D:
    x: np.ndarray
    edges: np.ndarray
    dx: float
    N: int


def create_grid(L: float, dx: float) -> Grid1D:
    """Create a uniform cell-centered grid over [0, L]."""

    if L <= 0:
        raise ValueError("Pipe length L must be positive")
    if dx <= 0:
        raise ValueError("Grid spacing dx must be positive")
    N = int(round(L / dx))
    if N < 2:
        raise ValueError("Grid needs at least two cells")
    dx_eff = L / N
    edges = np.linspace(0.0, L, N + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    return Grid1D(x=x, edges=edges, dx=dx_eff, N=N)
