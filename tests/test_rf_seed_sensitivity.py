"""RF supplement uses versioned matched predictions and explicit SD conventions."""
import numpy as np
import pandas as pd
import pytest


def test_rf_summary_distinguishes_population_sample_and_fixed_gin(load_experiment):
    mod = load_experiment("rf_seed_sensitivity")
    rf = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    frame = pd.DataFrame({"tag": ["tiny"] * 5, "r2": rf, "rmse": 1 - rf,
                          "mae": 0.5 - rf, "delta_fixed_gin_minus_rf": 0.6 - rf})
    result = mod.summarize(frame)[0]
    assert result["n_seeds"] == 5
    assert result["r2_mean"] == pytest.approx(0.3)
    assert result["r2_std_ddof0"] == pytest.approx(np.sqrt(0.02))
    assert result["r2_std_ddof1"] == pytest.approx(np.sqrt(0.025))
    assert result["delta_fixed_gin_minus_rf_mean"] == pytest.approx(0.3)
    assert result["delta_fixed_gin_minus_rf_std_ddof0"] == pytest.approx(result["r2_std_ddof0"])


@pytest.mark.parametrize("corruption", ["label_id", "test_idx", "y_true", "shape", "finite"])
def test_rf_supplement_refuses_incompatible_predictions(load_experiment, corruption):
    mod = load_experiment("rf_seed_sensitivity")
    split = {"label_id": "abc", "test_idx": np.array([1, 2]), "y": np.array([4., 5., 6.])}
    pred = {"label_id": "abc", "test_idx": np.array([1, 2]),
            "y_true": np.array([5., 6.]), "y_pred": np.array([5.1, 6.2])}
    mod.assert_aligned(pred, split)
    changes = {"label_id": ("label_id", "old"), "test_idx": ("test_idx", np.array([2, 1])),
               "y_true": ("y_true", np.array([5., 7.])),
               "shape": ("y_pred", np.array([[5.1, 6.2]])),
               "finite": ("y_pred", np.array([5.1, np.nan]))}
    key, value = changes[corruption]
    pred[key] = value
    with pytest.raises(ValueError):
        mod.assert_aligned(pred, split)
