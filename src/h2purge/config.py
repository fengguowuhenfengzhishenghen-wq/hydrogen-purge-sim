"""Configuration and result containers for the purge simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import D_MOL_DEFAULT, L_PIPE, T_DEFAULT


SpeciesTuple = Tuple[float, float, float]


@dataclass
class SimulationParams:
    """Inputs for one one-dimensional purge simulation.

    All pressures are absolute Pa inside the solver. User-facing scripts convert
    MPa to Pa before constructing this object.
    """

    L: float = L_PIPE
    D: float = 1.2
    u_nominal: float = 7.0
    p_back_abs: float = 0.10e6
    T: float = T_DEFAULT
    roughness: float = 5.0e-5
    dx: float = 10.0
    CFL: float = 0.45
    t_end: Optional[float] = None
    output_times: Optional[Sequence[float]] = None
    K_model: str = "beta_uD"
    beta_K: float = 0.5
    D_mol: float = D_MOL_DEFAULT
    K_override: Optional[float] = None
    pressure_mode: str = "reference_pressure"
    pressure_is_gauge: bool = False
    initial_profiles: Optional[Mapping[str, np.ndarray]] = None
    inlet_composition: SpeciesTuple = (1.0, 0.0, 0.0)
    n2_plug_fraction: float = 0.08
    cfl_diff_safety: float = 0.45
    max_steps: int = 2_000_000

    def effective_back_pressure(self) -> float:
        from .constants import ATM_PRESSURE

        return self.p_back_abs + ATM_PRESSURE if self.pressure_is_gauge else self.p_back_abs


@dataclass
class SimulationResult:
    """Numerical profiles and diagnostics recorded at output times."""

    params: SimulationParams
    x_grid: np.ndarray
    x_edges: np.ndarray
    dx: float
    times: np.ndarray
    profiles: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    reynolds: np.ndarray
    dispersion: np.ndarray
    froude: np.ndarray
    metrics: list[dict] = field(default_factory=list)
    boundary_flux_integral: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    inventory: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    species: Tuple[str, str, str] = ("H2", "N2", "Air")

    def species_index(self, name: str) -> int:
        key = name.strip().lower()
        lookup = {"h2": 0, "n2": 1, "air": 2}
        if key not in lookup:
            raise KeyError(f"Unknown species {name!r}; expected H2, N2, or Air")
        return lookup[key]

    def profile(self, time_index: int = -1) -> Dict[str, np.ndarray]:
        arr = self.profiles[time_index]
        return {"H2": arr[:, 0], "N2": arr[:, 1], "Air": arr[:, 2]}

    def species_profile(self, name: str, time_index: int = -1) -> np.ndarray:
        return self.profiles[time_index, :, self.species_index(name)]

    @property
    def final_time(self) -> float:
        return float(self.times[-1])
