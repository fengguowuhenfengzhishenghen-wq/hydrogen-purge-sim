import numpy as np

from h2purge.config import SimulationParams
from h2purge.solver_fvm import run_simulation


def test_h2_inventory_matches_boundary_flux_integral():
    params = SimulationParams(
        L=1000.0,
        D=1.0,
        u_nominal=1.0,
        p_back_abs=0.10e6,
        dx=10.0,
        t_end=100.0,
        output_times=[0.0, 100.0],
        K_override=0.0,
    )
    result = run_simulation(params)
    inventory_delta = result.inventory[-1, 0] - result.inventory[0, 0]
    flux_delta = result.boundary_flux_integral[-1, 0]
    assert abs(inventory_delta - flux_delta) < 1.0e-6
