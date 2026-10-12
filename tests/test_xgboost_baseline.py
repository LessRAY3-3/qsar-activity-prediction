"""XGBoost baseline: pool semantics and the same-pool control guard.

``load_rf_control`` must never read the learning-curve RF (its ``n_train``
can sit far below the pool and its rows are not the headline control) and
must claim ``same_pool`` only when the XGB fit pool holds exactly the
headline RF's training rows (every non-test molecule). ``fit_pool`` must
return train + valid, not the GNN's 10%-carved ``train_idx``.
"""
import json

import numpy as np
import pytest


def _write_metrics(tmp_path, r2_random=0.71, r2_scaffold=0.65):
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    metrics = {
        "random": {"split": "random", "r2": r2_random, "rmse": 0.7, "mae": 0.5},
        "scaffold": {"split": "scaffold", "r2": r2_scaffold, "rmse": 1.0, "mae": 0.8},
    }
    (results / "metrics_egfr.json").write_text(json.dumps(metrics))
    return metrics


def _write_learning_curve(tmp_path, r2=0.55, n_max=3000):
    lc_dir = tmp_path / "results" / "learning_curve"
    lc_dir.mkdir(parents=True, exist_ok=True)
    (lc_dir / "learning_curve_egfr.csv").write_text(
        "split,n_train,seed,model,r2\n"
        f"random,{n_max},42,rf,{r2}\n"
        f"scaffold,{n_max},42,rf,{r2}\n"
    )


def test_control_never_reads_learning_curve(tmp_path, monkeypatch, load_experiment):
    """A present learning-curve CSV (even with different R2) is ignored."""
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path, r2_random=0.71, r2_scaffold=0.65)
    _write_learning_curve(tmp_path, r2=0.55, n_max=3000)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    control = mod.load_rf_control(
        "egfr",
        pool_sizes={"random": 3648, "scaffold": 3648},
        test_sizes={"random": 912, "scaffold": 912},
        n_total=4560,
    )
    for split, r2 in (("random", 0.71), ("scaffold", 0.65)):
        assert control[split]["r2_mean"] == pytest.approx(r2)
        assert "learning_curve" not in control[split]["source"]
        assert control[split]["same_pool"] is True
        assert control[split]["xgb_n_train"] == 3648
        assert control[split]["r2_std"] is None


def test_same_pool_false_when_fit_pool_is_train_only(tmp_path, monkeypatch, load_experiment):
    """train_idx-only pool (3283) vs headline RF pool (4560-912=3648)."""
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    control = mod.load_rf_control(
        "egfr",
        pool_sizes={"random": 3283, "scaffold": 3283},
        test_sizes={"random": 912, "scaffold": 912},
        n_total=4560,
    )
    for split in ("random", "scaffold"):
        assert control[split]["same_pool"] is False
        assert "NOT same-pool" in control[split]["source"]
        # the control value itself is still the headline number
        assert control[split]["r2_mean"] == pytest.approx(0.71 if split == "random" else 0.65)


def test_control_missing_split_raises(tmp_path, monkeypatch, load_experiment):
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    with pytest.raises(SystemExit, match="time"):
        mod.load_rf_control(
            "egfr",
            pool_sizes={"time": 4000},
            test_sizes={"time": 560},
            n_total=4560,
        )


def test_fit_pool_is_train_plus_valid(tmp_path, load_experiment):
    """XGB must train on the full non-test pool, not the GNN train_idx."""
    mod = load_experiment("xgboost_baseline")
    path = tmp_path / "split.npz"
    np.savez(path,
             train_idx=np.arange(10),
             valid_idx=np.arange(10, 15),
             test_idx=np.arange(15, 20))
    got = mod.fit_pool(np.load(path))
    assert got.tolist() == list(range(15))


def test_fit_pool_without_valid_idx(tmp_path, load_experiment):
    """Split files without a valid_idx (legacy shape) fall back to train."""
    mod = load_experiment("xgboost_baseline")
    path = tmp_path / "split.npz"
    np.savez(path, train_idx=np.arange(10), test_idx=np.arange(10, 20))
    got = mod.fit_pool(np.load(path))
    assert got.tolist() == list(range(10))
