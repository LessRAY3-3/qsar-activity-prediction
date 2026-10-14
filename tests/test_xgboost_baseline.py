"""XGBoost baseline: pool semantics and the fingerprint-based same-pool guard.

``load_rf_control`` must never read the learning-curve RF (its ``n_train``
can sit far below the pool and its rows are not the headline control) and
must claim ``same_pool`` only when the XGB fit pool and the RF's training
pool hold exactly the same row indices - decided by comparing
``pool_fingerprint`` hashes, not row counts, so a pool of the right size
with different composition is flagged. When a fingerprint cannot be
computed the check falls back to row counts and prints a warning.
``fit_pool`` must return train + valid, not the GNN's 10%-carved
``train_idx``. ``rf_train_pool`` rebuilds the RF's training rows from the
split npz (complement of test for random/scaffold; ``train_idx`` only for
time).
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

    pools = {"random": np.arange(3648), "scaffold": np.arange(3648)}
    control = mod.load_rf_control(
        "egfr",
        xgb_pools=pools,
        rf_pools=pools,
        test_sizes={"random": 912, "scaffold": 912},
        n_total=4560,
        xgb_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
        rf_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
    )
    for split, r2 in (("random", 0.71), ("scaffold", 0.65)):
        assert control[split]["r2_mean"] == pytest.approx(r2)
        assert "learning_curve" not in control[split]["source"]
        assert control[split]["same_pool"] is True
        assert control[split]["xgb_n_train"] == 3648
        assert control[split]["r2_std"] is None
        # both fingerprints are recorded and match
        assert control[split]["xgb_pool_sha256"] == control[split]["rf_pool_sha256"]
        assert control[split]["xgb_pool_sha256"] == mod.pool_fingerprint(np.arange(3648))


def test_same_pool_false_when_fit_pool_is_train_only(tmp_path, monkeypatch, load_experiment):
    """train_idx-only pool (3283) vs headline RF pool (4560-912=3648)."""
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    control = mod.load_rf_control(
        "egfr",
        xgb_pools={"random": np.arange(3283), "scaffold": np.arange(3283)},
        rf_pools={"random": np.arange(3648), "scaffold": np.arange(3648)},
        test_sizes={"random": 912, "scaffold": 912},
        n_total=4560,
        xgb_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
        rf_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
    )
    for split in ("random", "scaffold"):
        assert control[split]["same_pool"] is False
        assert "NOT same-pool" in control[split]["source"]
        assert control[split]["xgb_pool_sha256"] != control[split]["rf_pool_sha256"]
        # the control value itself is still the headline number
        assert control[split]["r2_mean"] == pytest.approx(0.71 if split == "random" else 0.65)


def test_same_pool_false_when_same_size_different_composition(tmp_path, monkeypatch, load_experiment):
    """Row counts alone must not decide same_pool: two 3648-row pools with
    different membership must be flagged as NOT same-pool."""
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    xgb_pool = np.arange(3648)  # rows 0..3647
    rf_pool = np.concatenate([np.arange(1800), np.arange(5000, 6848)])  # 3648 rows, other rows
    assert len(xgb_pool) == len(rf_pool)
    control = mod.load_rf_control(
        "egfr",
        xgb_pools={"random": xgb_pool, "scaffold": xgb_pool},
        rf_pools={"random": rf_pool, "scaffold": rf_pool},
        test_sizes={"random": 912, "scaffold": 912},
        n_total=9000,
    )
    for split in ("random", "scaffold"):
        assert control[split]["same_pool"] is False
        assert "NOT same-pool" in control[split]["source"]
        assert "sha256" in control[split]["source"] or "fingerprint" in control[split]["source"]
        assert control[split]["xgb_pool_sha256"] != control[split]["rf_pool_sha256"]


def test_same_pool_fallback_when_fingerprint_unavailable(tmp_path, monkeypatch, capsys, load_experiment):
    """rf_pools[split] = None -> size comparison + a printed warning."""
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    control = mod.load_rf_control(
        "egfr",
        xgb_pools={"random": np.arange(3648), "scaffold": np.arange(3283)},
        rf_pools={"random": None, "scaffold": None},
        test_sizes={"random": 912, "scaffold": 912},
        n_total=4560,
        xgb_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
        rf_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
    )
    out = capsys.readouterr().out
    assert "fingerprint unavailable" in out
    assert control["random"]["same_pool"] is False      # sizes alone cannot verify
    assert control["scaffold"]["same_pool"] is False    # 3283 != 3648
    assert control["random"]["rf_pool_sha256"] is None
    assert control["random"]["xgb_pool_sha256"] is not None
    assert "NOT same-pool" in control["scaffold"]["source"]


def test_pool_fingerprint_is_order_invariant_and_content_sensitive():
    """Hash covers sorted membership: any row order hashes equal, any
    membership change hashes different."""
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "experiments",
                        "xgboost_baseline.py")
    spec = importlib.util.spec_from_file_location("xgb_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    a = np.array([5, 1, 9, 3])
    b = np.array([9, 5, 3, 1])
    c = np.array([5, 1, 9, 4])
    assert mod.pool_fingerprint(a) == mod.pool_fingerprint(b)
    assert mod.pool_fingerprint(a) != mod.pool_fingerprint(c)
    assert mod.pool_fingerprint(None) is None


def test_control_missing_split_raises(tmp_path, monkeypatch, load_experiment):
    mod = load_experiment("xgboost_baseline")
    _write_metrics(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    with pytest.raises(SystemExit, match="time"):
        mod.load_rf_control(
            "egfr",
            xgb_pools={"time": np.arange(4000)},
            rf_pools={"time": np.arange(3440)},
            test_sizes={"time": 560},
            n_total=4560,
        xgb_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
        rf_tests={s: np.arange(3648, 4560) for s in ("random", "scaffold")},
        )


def test_rf_train_pool_random_is_complement_of_test(tmp_path, load_experiment):
    """random/scaffold: the RF trains on every non-test row."""
    mod = load_experiment("xgboost_baseline")
    path = tmp_path / "split.npz"
    np.savez(path,
             train_idx=np.arange(0, 10),
             valid_idx=np.arange(10, 15),
             test_idx=np.arange(15, 20))
    got = mod.rf_train_pool("random", np.load(path), n_total=20)
    assert got.tolist() == list(range(15))


def test_rf_train_pool_time_is_train_idx_only(tmp_path, load_experiment):
    """time: the RF fits train_idx only - train+valid is a different pool,
    so same_pool against the XGB train+valid pool must be False."""
    mod = load_experiment("xgboost_baseline")
    path = tmp_path / "split.npz"
    np.savez(path,
             train_idx=np.arange(0, 10),
             valid_idx=np.arange(10, 15),
             test_idx=np.arange(15, 20))
    sp = np.load(path)
    rf_pool = mod.rf_train_pool("time", sp, n_total=20)
    xgb_pool = mod.fit_pool(sp)
    assert rf_pool.tolist() == list(range(10))
    assert len(xgb_pool) == 15
    assert mod.pool_fingerprint(xgb_pool) != mod.pool_fingerprint(rf_pool)


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
