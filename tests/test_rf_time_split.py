"""--split time for 04_train_and_evaluate: synthetic data only, all inside tmp_path.

Covers the three behaviours added on top of the untouched default path:
  1. --split time trains one GridSearchCV RF on {TAG}_time.npz's train_idx,
     evaluates on test_idx, saves model + rf_preds npz (schema identical to
     the existing rf_preds_{TAG}_{split}.npz files), and merges ONLY the
     "time" key into metrics_{TAG}.json (random/scaffold byte-preserved).
  2. Missing npz but a {TAG}_time_NA.json marker -> its reason is printed,
     exit 0, nothing written.
  3. Neither file -> error exit.

The default (no --split) behaviour is covered by test_pipeline_smoke.
"""
import json
import sys

import joblib
import numpy as np
import pytest

N = 120
TRAIN_N, VALID_N, TEST_N = 80, 12, 28
# distinctive values so the "unchanged" assertion can't pass by accident
RANDOM_R2 = 0.123456
SCAFFOLD_R2 = 0.654321
NA_REASON = "no publication years recorded for 92% of the documents"


def _setup(tmp_path, monkeypatch, load_script):
    """Synthetic fingerprints + metrics json + path constants -> tmp_path."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(N, 32)).astype(np.uint8)
    # y follows part of the fingerprint + noise -> RF can learn it -> r2 in [0, 1]
    y = (X[:, :8].sum(axis=1) * 0.4 + rng.normal(0, 0.5, N) + 5.0).astype(np.float32)
    smiles = np.array([f"mol_{i:03d}" for i in range(N)])  # invalid on purpose; fallback path

    proc = tmp_path / "data" / "processed"
    split_dir = proc / "splits"
    split_dir.mkdir(parents=True)
    fp = proc / "fingerprints.npz"
    np.savez(fp, X=X, y=y, smiles=smiles, target="Synthetic kinase")

    results = tmp_path / "results"
    results.mkdir()
    metrics_path = results / "metrics.json"
    backup = {
        "target": "Synthetic kinase",
        "tag": "tiny",
        "random": {"split": "random", "r2": RANDOM_R2, "rmse": 1.5, "mae": 1.1},
        "scaffold": {"split": "scaffold", "r2": SCAFFOLD_R2, "rmse": 2.5, "mae": 2.1},
        "best_params_random": {"n_estimators": 300, "max_depth": None},
    }
    metrics_path.write_text(json.dumps(backup, indent=2))

    mod = load_script("04_train_and_evaluate")
    monkeypatch.setattr(mod, "IN_NPZ", str(fp))
    monkeypatch.setattr(mod, "SPLIT_DIR", str(split_dir))
    monkeypatch.setattr(mod, "FIG_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(mod, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(mod, "RESULTS_DIR", str(results))
    monkeypatch.setattr(mod, "METRICS_JSON", str(metrics_path))
    # shrink the grid to 1 config and 2 folds -> the GridSearch finishes in ms
    monkeypatch.setattr(
        mod,
        "PARAM_GRID",
        {"n_estimators": [3], "max_depth": [None], "min_samples_split": [2]},
    )
    monkeypatch.setattr(mod, "CV_FOLDS", 2)
    return mod, backup, metrics_path, split_dir, y


def _write_time_split(split_dir, tag):
    train_idx = np.arange(0, TRAIN_N, dtype=np.int64)
    valid_idx = np.arange(TRAIN_N, TRAIN_N + VALID_N, dtype=np.int64)
    test_idx = np.arange(TRAIN_N + VALID_N, N, dtype=np.int64)
    np.savez(
        split_dir / f"{tag}_time.npz",
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        scaffold_smiles=np.array([f"scaf_{i % 5}" for i in range(N)]),
    )
    return train_idx, test_idx


def test_time_split_trains_and_merges_only_time_key(tmp_path, monkeypatch, load_script):
    mod, backup, metrics_path, split_dir, y = _setup(tmp_path, monkeypatch, load_script)
    _, test_idx = _write_time_split(split_dir, mod.TAG)

    monkeypatch.setattr(sys, "argv", ["04_train_and_evaluate.py", "--split", "time"])
    assert mod.main() is None  # exit 0

    # model + preds npz written, schema matches the existing rf_preds files
    model_path = tmp_path / "models" / f"rf_time_split_{mod.TAG}.joblib"
    preds_path = tmp_path / "results" / f"rf_preds_{mod.TAG}_time.npz"
    assert model_path.exists()
    assert joblib.load(model_path).predict  # loads and is a usable estimator
    assert preds_path.exists()
    preds = np.load(preds_path)
    assert set(preds.files) == {"test_idx", "y_true", "y_pred"}
    np.testing.assert_array_equal(preds["test_idx"], test_idx)
    assert preds["test_idx"].dtype == np.int64
    np.testing.assert_array_equal(preds["y_true"], y[test_idx])
    assert preds["y_pred"].dtype == np.float64
    assert preds["y_pred"].shape == (TEST_N,)
    assert (tmp_path / "figures" / f"pred_vs_actual_time_{mod.TAG}.png").exists()

    # metrics: random/scaffold bit-identical, "time" added, r2 inside [-1, 1]
    payload = json.loads(metrics_path.read_text())
    assert payload["random"] == backup["random"]
    assert payload["scaffold"] == backup["scaffold"]
    assert payload["random"]["r2"] == RANDOM_R2
    assert payload["scaffold"]["r2"] == SCAFFOLD_R2
    assert payload["target"] == backup["target"]
    assert payload["best_params_random"] == backup["best_params_random"]
    assert "time" in payload
    assert payload["time"]["split"] == "time"
    assert -1.0 <= payload["time"]["r2"] <= 1.0
    assert np.isfinite(payload["time"]["rmse"])
    assert np.isfinite(payload["time"]["mae"])


def test_time_na_marker_skips_with_exit_0(tmp_path, monkeypatch, load_script, capsys):
    mod, backup, metrics_path, split_dir, _ = _setup(tmp_path, monkeypatch, load_script)
    # ONLY the NA marker: no {tag}_time.npz
    (split_dir / f"{mod.TAG}_time_NA.json").write_text(json.dumps({"reason": NA_REASON}))
    before = metrics_path.read_bytes()

    monkeypatch.setattr(sys, "argv", ["04_train_and_evaluate.py", "--split", "time"])
    assert mod.main() is None  # exit 0

    assert NA_REASON in capsys.readouterr().out
    # nothing at all was written: no model, no preds, no figure, metrics untouched
    assert not (tmp_path / "models" / f"rf_time_split_{mod.TAG}.joblib").exists()
    assert not (tmp_path / "results" / f"rf_preds_{mod.TAG}_time.npz").exists()
    assert not (tmp_path / "figures" / f"pred_vs_actual_time_{mod.TAG}.png").exists()
    assert metrics_path.read_bytes() == before


def test_time_split_missing_both_errors(tmp_path, monkeypatch, load_script):
    mod, _, metrics_path, split_dir, _ = _setup(tmp_path, monkeypatch, load_script)
    before = metrics_path.read_bytes()

    monkeypatch.setattr(sys, "argv", ["04_train_and_evaluate.py", "--split", "time"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code  # non-zero / message -> error exit

    assert not (tmp_path / "models" / f"rf_time_split_{mod.TAG}.joblib").exists()
    assert metrics_path.read_bytes() == before
