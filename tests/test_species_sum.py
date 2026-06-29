import numpy as np

from h2purge.config import SimulationParams
from h2purge.solver_fvm import run_simulation


def test_species_sum_stays_one():
    params = SimulationParams(
        L=1000.0,
        D=1.2,
        u_nominal=5.0,
        p_back_abs=0.10e6,
        dx=20.0,
        t_end=120.0,
        output_times=np.linspace(0.0, 120.0, 7),
    )
    result = run_simulation(params)
    err = np.max(np.abs(np.sum(result.profiles, axis=2) - 1.0))
    assert err < 1.0e-6
