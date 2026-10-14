import numpy as np
import pytest


def test_decomposition_reconciles_actual_squared_error_change(load_experiment):
    mod = load_experiment('time_baseline_diagnostics')
    y = np.array([1., 2., 4., 8., 10.])
    rf = np.array([-1., 0., 3., 6., 7.])
    gin = np.array([4., 3., 7., 6., 9.])
    parts = mod.decomposition(y, rf, gin)
    expected = np.mean((y-rf)**2 - (y-gin)**2) / y.var()
    assert sum(parts[k] for k in ('level', 'scale', 'correlation')) == pytest.approx(expected)
    reverse = mod.decomposition(y, gin, rf)
    for k in ('level', 'scale', 'correlation'):
        assert parts[k] == pytest.approx(-reverse[k])


def test_baseline_comparison_identity_and_perfect_predictions(load_experiment):
    mod = load_experiment('time_baseline_diagnostics')
    y = np.arange(10, dtype=float)
    point, lo, hi, valid = mod.constant_comparison(y, [np.full(10, 3.), y], 3., n_boot=200)
    assert point[0] == lo[0] == hi[0] == 0
    assert point[1] == pytest.approx(np.mean((y-3)**2) / y.var())
    assert lo[1] > 0 and valid == 200
