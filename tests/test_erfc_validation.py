from h2purge.validation import run_erfc_validation


def test_erfc_validation_l2():
    row = run_erfc_validation(make_plot=False)
    assert row["L2_error"] < 0.03
