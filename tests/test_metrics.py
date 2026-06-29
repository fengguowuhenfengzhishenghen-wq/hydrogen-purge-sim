import numpy as np

from h2purge.metrics import effective_n2_length, flammable_length, front_position, mixed_length, pair_transition_length


def test_mixed_length_counts_cells_below_threshold():
    x_h2 = np.array([1.0, 0.98, 0.50, 0.02])
    x_n2 = np.array([0.0, 0.02, 0.45, 0.96])
    x_air = np.array([0.0, 0.00, 0.05, 0.02])
    assert mixed_length(x_h2, x_n2, x_air, dx=10.0, threshold=0.99) == 30.0


def test_flammable_length_uses_o2_loc():
    x_h2 = np.array([0.00, 0.05, 0.50, 0.80])
    x_air = np.array([1.00, 0.50, 0.20, 0.20])
    assert flammable_length(x_h2, x_air, dx=10.0) == 10.0


def test_pair_transition_length_counts_component_overlap():
    x_h2 = np.array([0.99, 0.50, 0.02, 0.00])
    x_n2 = np.array([0.01, 0.50, 0.90, 0.99])
    assert pair_transition_length(x_h2, x_n2, dx=10.0, eps=0.01) == 20.0


def test_effective_n2_length_longest_high_n2_run():
    x_h2 = np.zeros(5)
    x_n2 = np.array([0.1, 0.96, 0.97, 0.2, 0.96])
    x_air = 1.0 - x_n2
    assert effective_n2_length(x_h2, x_n2, x_air, dx=10.0) == 20.0


def test_front_position_interpolates_rightmost_crossing():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([1.0, 0.6, 0.2])
    assert front_position(y, x, level=0.5) == 12.5
