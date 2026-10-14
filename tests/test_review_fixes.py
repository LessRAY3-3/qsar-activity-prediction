"""Regression tests for the PR follow-up: data boundaries and side effects."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from time_labels import LABEL_POLICY, dataset_signature, label_signature, load_time_split


def test_asof_labels_ignore_future_censored_and_undated_records(load_script):
    mod = load_script("gnn_08_time_splits")
    # First molecule reproduces the reported 1999 / 2024 median contamination.
    acts = pd.DataFrame({
        "canonical_smiles": ["CCO", "CCO.Cl", "CCO", "CCO", "CCO", "CCN", "CCC", "CCCC"],
        "document_chembl_id": ["old", "old", "future", "old", "unknown", "valid", "future", "unknown"],
        "standard_value": [8, 2000, 8, 0.01, 0.01, 100, 1000, 100],
        "standard_units": ["nM"] * 8,
        "standard_relation": [" = ", "=", "=", "<", "=", "=", "=", "="],
    })
    years = pd.DataFrame({"document_chembl_id": ["old", "valid", "future"],
                          "document_year": [1999, 2019, 2024]})
    pools = [np.array([0, 3]), np.array([1]), np.array([2])]
    y, kept, excluded = mod.asof_labels(["CCO", "CCN", "CCC", "CCCC"], acts, years, pools, 2019, 2021)
    np.testing.assert_allclose(y[:3], [6.8979400087, 7, 6], atol=1e-6)
    assert np.isnan(y[3])
    assert excluded.tolist() == [3]
    assert kept[0].tolist() == [0]
    # Changes to a future training-compound measurement cannot change its label.
    acts.loc[2, "standard_value"] = 1e9
    changed, _, _ = mod.asof_labels(["CCO", "CCN", "CCC", "CCCC"], acts, years, pools, 2019, 2021)
    np.testing.assert_array_equal(changed, y)


def test_time_na_wins_over_stale_npz(tmp_path):
    np.savez(tmp_path / "tiny_time.npz", train_idx=[0], test_idx=[1])
    (tmp_path / "tiny_time_NA.json").write_text(json.dumps({"reason": "invalidated"}))
    assert load_time_split(tmp_path, "tiny") is None
    (tmp_path / "tiny_time_NA.json").unlink()
    with pytest.raises(ValueError, match="legacy"):
        load_time_split(tmp_path, "tiny")


def test_plot_only_time_never_loads_or_trains_model(tmp_path, monkeypatch, load_script):
    mod = load_script("04_train_and_evaluate")
    np.savez(tmp_path / f"rf_preds_{mod.TAG}_time.npz", test_idx=[0, 1], y_true=[1., 2.], y_pred=[1.1, 1.9])
    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "FIG_DIR", str(tmp_path / "figs"))
    monkeypatch.setattr(mod, "SPLIT_DIR", str(tmp_path / "splits"))
    def forbidden(*args, **kwargs):
        pytest.fail("plot-only must not train, load models or resolve training data")
    monkeypatch.setattr(mod, "run_single", forbidden)
    monkeypatch.setattr(mod, "resolve_time_split", forbidden)
    monkeypatch.setattr(mod.joblib, "load", forbidden)
    before = (tmp_path / f"rf_preds_{mod.TAG}_time.npz").read_bytes()
    monkeypatch.setattr(sys, "argv", ["rf", "--plot-only", "--split", "time"])
    mod.main()
    assert (tmp_path / "figs" / f"pred_vs_actual_time_{mod.TAG}.png").exists()
    assert len(list((tmp_path / "figs").iterdir())) == 1
    assert (tmp_path / f"rf_preds_{mod.TAG}_time.npz").read_bytes() == before


def test_model_specific_ad_includes_validation_for_rf_only(load_experiment):
    mod = load_experiment("uncertainty_ad")
    # The test molecule is identical to a validation molecule, unlike train.
    smiles = np.array(["c1ccccc1", "CCO", "CCO"])
    sp = {"train_idx": np.array([0]), "valid_idx": np.array([1]), "test_idx": np.array([2])}
    sims = mod.model_ad_similarities(smiles, sp, split_name="random")
    assert sims["rf"][0] == 1.0
    assert sims["gin"][0] < 0.4


def test_qc_counts_padded_exact_relations(tmp_path, monkeypatch, load_script, capsys):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"standard_value": [0.1, 100., 0.01], "canonical_smiles": ["CCO"] * 3,
                  "standard_units": ["nM"] * 3, "standard_relation": [" = ", "=", " < "]}).to_csv(path, index=False)
    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(path))
    mod.check_ultra_potent()
    assert "1 / 2 labelled rows" in capsys.readouterr().out


def test_cluster_bootstrap_matches_explicit_whole_group_resampling(load_experiment):
    mod = load_experiment("scaffold_bootstrap")
    y = np.array([0., 1., 3., 5., 7.])
    pg, pr = y + [1., 1., 2., 2., 2.], y + [2., 2., 1., 1., 1.]
    groups = np.array([0, 0, 1, 1, 1])
    result = mod.cluster_bootstrap(y, pg, pr, groups, n_boot=100, seed=7)
    rng = np.random.default_rng(7)
    deltas = []
    for chosen in rng.integers(2, size=(100, 2)):
        idx = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        deltas.append(mod.r2(y[idx], pg[idx]) - mod.r2(y[idx], pr[idx]))
    np.testing.assert_allclose([result["ci_lo"], result["ci_hi"]], np.percentile(deltas, [2.5, 97.5]))
    assert result["n_groups"] == 2


def test_label_version_detects_changed_rows_and_tampered_labels(tmp_path):
    smiles = np.array(["CC", "CCC", "CCCC", "CCO", "CCN", "CCCl"])
    y = np.arange(6, dtype=np.float32)
    sp = dict(smiles=smiles, y=y + 1, original_y=y, year=np.arange(len(y), dtype=float), train_idx=np.array([0, 1]), valid_idx=np.array([2, 3]),
              test_idx=np.array([4, 5]), excluded_idx=np.array([], dtype=int),
              label_policy=LABEL_POLICY, dataset_signature=dataset_signature(smiles, y),
              source_sha256="fixture", cutoff_valid=2019, cutoff_test=2021)
    sp["label_id"] = label_signature(sp)
    np.savez(tmp_path / "tiny_time.npz", **sp)
    assert load_time_split(tmp_path, "tiny", smiles, y)["label_id"] == sp["label_id"]
    with pytest.raises(ValueError, match="row order"):
        load_time_split(tmp_path, "tiny", smiles[::-1], y)
    with pytest.raises(ValueError, match="data version"):
        load_time_split(tmp_path, "tiny", smiles, y + 1)
    sp["y"][0] = 100
    np.savez(tmp_path / "tiny_time.npz", **sp)
    with pytest.raises(ValueError, match="signature"):
        load_time_split(tmp_path, "tiny")
