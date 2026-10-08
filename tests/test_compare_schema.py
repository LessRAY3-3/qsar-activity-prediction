"""gnn_04_compare.comparison_table effect-size columns + summary.csv verifier."""
import json

import numpy as np
import pandas as pd
import pytest

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
