"""gnn_04_compare.comparison_table effect-size columns + summary.csv verifier."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# RF metrics (schema of 04_train_and_evaluate's metrics_{TAG}.json)
RF_METRICS = {
    "target": "Test Kinase",
    "tag": "demo",
    "random": {"split": "random", "r2": 0.50, "rmse": 1.20, "mae": 0.90},
    "scaffold": {"split": "scaffold", "r2": 0.30, "rmse": 1.40, "mae": 1.00},
}

# GIN metrics: base file (= seed 42) + two seed siblings -> 3-seed collection
GIN_BASE = {
    "random": {"r2": 0.55, "rmse": 1.10, "mae": 0.85},
    "scaffold": {"r2": 0.42, "rmse": 1.30, "mae": 0.95},
}
GIN_SEED1 = {
    "random": {"r2": 0.60, "rmse": 1.05, "mae": 0.80},
    "scaffold": {"r2": 0.44, "rmse": 1.25, "mae": 0.90},
}
GIN_SEED2 = {
    "random": {"r2": 0.70, "rmse": 1.00, "mae": 0.75},
    "scaffold": {"r2": 0.46, "rmse": 1.20, "mae": 0.88},
}

# Fixtures carry the same seed provenance required of real metric files.
for _payload, _seed in ((GIN_BASE, 42), (GIN_SEED1, 1), (GIN_SEED2, 2)):
    for _split in _payload.values():
        _split["params"] = {"seed": _seed}

RANDOM_R2_MEAN = float(np.mean([0.55, 0.60, 0.70]))
SCAFFOLD_R2_MEAN = float(np.mean([0.42, 0.44, 0.46]))


@pytest.fixture
def gnn_compare(tmp_path, monkeypatch, load_script):
    """gnn_04_compare with every path constant pointed into tmp_path."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "metrics_demo.json").write_text(json.dumps(RF_METRICS))
    (results / "gnn_metrics_demo.json").write_text(json.dumps(GIN_BASE))
    (results / "gnn_metrics_demo_seed1.json").write_text(json.dumps(GIN_SEED1))
    (results / "gnn_metrics_demo_seed2.json").write_text(json.dumps(GIN_SEED2))

    mod = load_script("gnn_04_compare")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(mod, "TAG", "demo")
    monkeypatch.setattr(mod, "RF_METRICS", str(results / "metrics_demo.json"))
    monkeypatch.setattr(mod, "GIN_METRICS", str(results / "gnn_metrics_demo.json"))
    monkeypatch.setattr(mod, "FP_NPZ", str(tmp_path / "unused_fingerprints.npz"))
    return mod


def test_comparison_table_effect_sizes(gnn_compare):
    rows = gnn_compare.comparison_table()
    assert [r["split"] for r in rows] == ["random", "scaffold"]
    for r in rows:
        assert "d_r2_mean_gin_minus_rf" in r
        assert "d_r2_seed42_gin_minus_rf" in r
        assert r["gin_n_seeds"] == 3

    by = {r["split"]: r for r in rows}

    rnd = by["random"]
    assert rnd["d_r2_mean_gin_minus_rf"] == pytest.approx(
        RANDOM_R2_MEAN - RF_METRICS["random"]["r2"], rel=0, abs=1e-12
    )
    assert rnd["d_r2_seed42_gin_minus_rf"] == pytest.approx(
        GIN_BASE["random"]["r2"] - RF_METRICS["random"]["r2"], rel=0, abs=1e-12
    )
    # the deltas must also be consistent with the table's own columns
    assert rnd["d_r2_mean_gin_minus_rf"] == pytest.approx(
        rnd["gin_r2_mean"] - rnd["rf_r2"], rel=0, abs=1e-12
    )
    assert rnd["d_r2_seed42_gin_minus_rf"] == pytest.approx(
        rnd["gin_r2"] - rnd["rf_r2"], rel=0, abs=1e-12
    )

    scaf = by["scaffold"]
    assert scaf["d_r2_mean_gin_minus_rf"] == pytest.approx(
        SCAFFOLD_R2_MEAN - RF_METRICS["scaffold"]["r2"], rel=0, abs=1e-12
    )
    assert scaf["d_r2_seed42_gin_minus_rf"] == pytest.approx(
        GIN_BASE["scaffold"]["r2"] - RF_METRICS["scaffold"]["r2"], rel=0, abs=1e-12
    )
    # the two conventions genuinely differ on this fixture (mean vs single seed)
    assert rnd["d_r2_mean_gin_minus_rf"] != pytest.approx(
        rnd["d_r2_seed42_gin_minus_rf"], abs=1e-12
    )


def test_gin_n_seeds_counts_only_seeds_present_per_split(gnn_compare,
                                                         tmp_path, capsys):
    """A seed file missing one split must not inflate that split's n:
    the row reports the seeds that actually contributed (and warns)."""
    seed2 = tmp_path / "results" / "gnn_metrics_demo_seed2.json"
    seed2.write_text(json.dumps({"random": GIN_SEED2["random"]}))

    rows = gnn_compare.comparison_table()
    by = {r["split"]: r for r in rows}
    assert by["random"]["gin_n_seeds"] == 3
    assert by["scaffold"]["gin_n_seeds"] == 2
    # the scaffold mean/std use the same two seeds the count reports
    assert by["scaffold"]["gin_r2_mean"] == pytest.approx(
        float(np.mean([GIN_BASE["scaffold"]["r2"], GIN_SEED1["scaffold"]["r2"]]))
    )
    assert "gin_n_seeds reports" in capsys.readouterr().out


def _summary_rows():
    """Rows shaped like multi_target_summary.load_tag's output."""
    return [
        {
            "tag": "egfr",
            "family": "kinase",
            "n_molecules": 1000,
            "rf_r2_random": 0.50,
            "gin_r2_mean_random": 0.62,
            "gin_r2_std_random": 0.01,
            "d_r2_random": 0.62 - 0.50,
            "rf_r2_scaffold": 0.30,
            "gin_r2_mean_scaffold": 0.35,
            "gin_r2_std_scaffold": 0.02,
            "d_r2_scaffold": 0.35 - 0.30,
            "d_r2_seed42_random": 0.05,
            "d_r2_seed42_scaffold": 0.12,
        },
        {
            "tag": "a2a",
            "family": "gpcr",
            "n_molecules": 500,
            "rf_r2_random": 0.44,
            "gin_r2_mean_random": 0.41,
            "gin_r2_std_random": 0.03,
            "d_r2_random": 0.41 - 0.44,
            "rf_r2_scaffold": 0.25,
            "gin_r2_mean_scaffold": 0.20,
            "gin_r2_std_scaffold": 0.03,
            "d_r2_scaffold": 0.20 - 0.25,
            "d_r2_seed42_random": -0.02,
            "d_r2_seed42_scaffold": 0.01,
        },
    ]


def test_verify_summary_csv_passes_then_fails_on_tamper(tmp_path, load_experiment):
    mod = load_experiment("multi_target_summary")

    path = tmp_path / "summary.csv"
    mod.summary_frame(_summary_rows()).to_csv(path, index=False)
    assert mod.verify_summary_csv(path) is True

    # tamper with one d_r2 cell: the full-precision round-trip must catch it
    df = pd.read_csv(path)
    df.loc[0, "d_r2_random (mean)"] = df.loc[0, "d_r2_random (mean)"] + 0.01
    df.to_csv(path, index=False)
    assert mod.verify_summary_csv(path) is False


def test_verify_summary_csv_missing_split_round_trips(tmp_path, load_experiment):
    """A tag with one absent split stays NaN on disk; the round-trip must
    treat NaN == NaN as equal instead of failing the whole check."""
    mod = load_experiment("multi_target_summary")

    rows = _summary_rows()
    for col in ("rf_r2_scaffold", "gin_r2_mean_scaffold", "gin_r2_std_scaffold",
                "d_r2_scaffold", "d_r2_seed42_scaffold"):
        rows[1][col] = float("nan")

    path = tmp_path / "summary.csv"
    mod.summary_frame(rows).to_csv(path, index=False)
    assert mod.verify_summary_csv(path) is True


# ---------------------------------------------------------------- RF-vs-split check

@pytest.fixture
def rf_split_tree(tmp_path, monkeypatch, load_script):
    """gnn_04_compare pointed at a tiny synthetic split with a real RF.

    Every path constant is redirected into tmp_path, and the metrics json
    records the R2/RMSE the saved model actually scores on the persisted
    test_idx -- the non-circular reference load_split_data verifies against
    before overwriting results/rf_preds_*.
    """
    rng = np.random.default_rng(0)
    n, n_test = 60, 20
    X = rng.integers(0, 2, size=(n, 32)).astype(np.uint8)
    y = rng.normal(6.0, 1.0, n)
    test_idx = np.arange(n_test)          # head of the array is the test set
    smiles = np.array([f"CCO{i % 5}" for i in range(n)])

    proc = tmp_path / "data" / "processed"
    (proc / "splits").mkdir(parents=True)
    np.savez(proc / "demo_fingerprints.npz", X=X, y=y, smiles=smiles)
    np.savez(proc / "splits" / "demo_random.npz", test_idx=test_idx,
             scaffold_smiles=smiles, smiles=smiles)

    rf = RandomForestRegressor(n_estimators=8, random_state=0).fit(
        X[n_test:], y[n_test:])
    (tmp_path / "models").mkdir()
    joblib.dump(rf, tmp_path / "models" / "rf_random_split_demo.joblib")
    pred = rf.predict(X[test_idx])

    results = tmp_path / "results"
    results.mkdir()
    np.savez(results / "gnn_preds_demo_random.npz", test_idx=test_idx,
             y_true=y[test_idx], y_pred=pred + 0.1)

    r2 = r2_score(y[test_idx], pred)
    rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))
    metrics_path = results / "metrics_demo.json"
    metrics_path.write_text(json.dumps(
        {"random": {"split": "random", "r2": r2, "rmse": rmse}}))

    mod = load_script("gnn_04_compare")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(mod, "TAG", "demo")
    monkeypatch.setattr(mod, "RF_METRICS", str(metrics_path))
    monkeypatch.setattr(mod, "FP_NPZ", str(proc / "demo_fingerprints.npz"))
    monkeypatch.setattr(mod, "SPLIT_NPZ", str(proc / "splits" / "demo_{}.npz"))
    return mod, metrics_path, pred


def test_rf_repredict_mismatch_aborts_before_writing_preds(rf_split_tree):
    """A regenerated split whose re-prediction misses metrics must abort
    BEFORE results/rf_preds_* is overwritten (the non-circular check)."""
    mod, metrics_path, _ = rf_split_tree
    mj = json.loads(metrics_path.read_text())
    mj["random"]["r2"] += 0.1               # r2 alone must already trip it
    metrics_path.write_text(json.dumps(mj))

    with pytest.raises(SystemExit):
        mod.load_split_data("random")
    assert not (Path(mod.BASE) / "results" / "rf_preds_demo_random.npz").exists()


def test_rf_repredict_match_writes_verified_preds(rf_split_tree):
    mod, _, pred = rf_split_tree
    y_te, rf_pred, g_pred, scaf = mod.load_split_data("random")

    written = np.load(Path(mod.BASE) / "results" / "rf_preds_demo_random.npz")
    assert np.array_equal(written["test_idx"], np.arange(20))
    assert np.allclose(written["y_pred"], pred)
    assert np.allclose(rf_pred, pred)
    assert len(y_te) == len(g_pred) == len(scaf) == 20


def test_skip_model_check_writes_unverified_preds(rf_split_tree):
    """verify=False is the announced bypass: it must skip the comparison,
    not keep enforcing it (mutant: flag ignored -> SystemExit here)."""
    mod, metrics_path, pred = rf_split_tree
    mj = json.loads(metrics_path.read_text())
    mj["random"]["r2"] += 0.1
    metrics_path.write_text(json.dumps(mj))

    mod.load_split_data("random", verify=False)
    written = np.load(Path(mod.BASE) / "results" / "rf_preds_demo_random.npz")
    assert np.allclose(written["y_pred"], pred)
