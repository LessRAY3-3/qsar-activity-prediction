"""Boundary and degenerate-resample checks for clustered scaffold inference."""
import numpy as np
import pytest


@pytest.mark.parametrize("y,groups,n_boot", [([], [], 5), ([1, 2], [0, 0], 5),
                                            ([[1], [2]], [0, 1], 5), ([1, 2], [0, 1], 0),
                                            ([1, 2], [0, 1], 1.5), ([1, np.nan], [0, 1], 5)])
def test_cluster_rejects_invalid_inputs(load_experiment, y, groups, n_boot):
    mod = load_experiment("scaffold_bootstrap")
    with pytest.raises(ValueError):
        mod.cluster_bootstrap(y, y, y, groups, n_boot=n_boot)


def test_cluster_records_and_excludes_degenerate_draws(load_experiment):
    mod = load_experiment("scaffold_bootstrap")
    y = np.array([1., 1., 3., 3.])
    result = mod.cluster_bootstrap(y, y, y + 1, [0, 0, 1, 1], n_boot=200, seed=7)
    assert 0 < result["n_valid_boot"] < result["n_boot"] == 200
    assert result["ci_lo"] == result["ci_hi"] == 1.
    assert result["winner"] == "gnn"
    assert result == mod.cluster_bootstrap(y, y, y + 1, [0, 0, 1, 1], n_boot=200, seed=7)
