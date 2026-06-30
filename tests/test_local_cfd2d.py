import numpy as np

from h2purge.local_cfd2d import LocalCFD2DConfig, build_initial_field, compute_local_cfd2d_metrics


def test_build_initial_field_preserves_mole_fraction_sum():
    x_src = np.linspace(0.0, 100.0, 6)
    profile = np.column_stack(
        [
            np.linspace(1.0, 0.0, 6),
            np.linspace(0.0, 1.0, 6),
            np.zeros(6),
        ]
    )
    cfg = LocalCFD2DConfig(length_m=100.0, diameter_m=1.0, nx=11, nz=7)
    _x, _z, field = build_initial_field(x_src, profile, cfg)
    assert np.max(np.abs(field.sum(axis=-1) - 1.0)) < 1.0e-12


def test_local_cfd2d_metrics_capture_vertical_stratification():
    x = np.linspace(0.0, 100.0, 5)
    z = np.linspace(-0.5, 0.5, 5)
    X = np.zeros((len(x), len(z), 3))
    top = z >= 0.0
    bottom = z < 0.0
    X[:, top, 0] = 0.70
    X[:, top, 1] = 0.25
    X[:, top, 2] = 0.05
    X[:, bottom, 0] = 0.10
    X[:, bottom, 1] = 0.30
    X[:, bottom, 2] = 0.60
    u = np.zeros((len(x), len(z)))
    w = np.zeros_like(u)
    metrics = compute_local_cfd2d_metrics(x, z, X, u, w, LocalCFD2DConfig(nx=len(x), nz=len(z)))
    assert metrics["top_bottom_h2_delta"] > 0.0
    assert metrics["top_bottom_air_delta"] < 0.0
    assert metrics["h2_air_vertical_separation_m"] > 0.0
    assert metrics["species_sum_max_error"] < 1.0e-12
