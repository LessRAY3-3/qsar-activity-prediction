"""P8 phase-1 GINE panel: pure row/aggregation helpers.

Loads experiments/gine_panel.py by path (import-safe: matplotlib is
imported inside the figure functions only) and checks the panel-row
contract on synthetic inputs with a known ground truth.
"""
import numpy as np
import pandas as pd
import pytest


def test_aggregate_seed_r2_mean_population_std(load_experiment):
    mod = load_experiment("gine_panel")
    mean, std, n = mod.aggregate_seed_r2([0.7, 0.8, 0.9])
    assert n == 3
    assert mean == pytest.approx(0.8)
    assert std == pytest.approx(np.std([0.7, 0.8, 0.9], ddof=0))


def test_aggregate_seed_r2_rejects_empty(load_experiment):
    mod = load_experiment("gine_panel")
    with pytest.raises(ValueError):
        mod.aggregate_seed_r2([])


def test_make_row_delta_and_bootstrap_note(load_experiment):
    mod = load_experiment("gine_panel")
    row = mod.make_row("a2a", "random", 0.70, [0.60, 0.62, 0.64],
                       ci=(-0.05, -0.01, True))
    assert row["gine_r2_mean"] == pytest.approx(0.62)
    assert row["gine_n_seeds"] == 3
    assert row["d_r2_mean"] == pytest.approx(0.62 - 0.70)
    assert row["ci_lo"] == -0.05 and row["ci_hi"] == -0.01
    assert row["significant"] is True
    assert "mean3" in row["note"]
    assert row["note"] != mod.NOTE_POINT_ONLY


def test_make_row_without_ci_is_point_estimate_only(load_experiment):
    mod = load_experiment("gine_panel")
    row = mod.make_row("vegfr2", "scaffold", 0.60, [0.55, 0.58, 0.61])
    assert row["d_r2_mean"] == pytest.approx(np.mean([0.55, 0.58, 0.61]) - 0.60)
    assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
    assert row["significant"] is None
    assert row["note"] == mod.NOTE_POINT_ONLY


def test_summary_csv_carries_delta_convention(tmp_path, monkeypatch, load_experiment):
    """build_panel -> summary.csv keeps every column and appends delta_convention."""
    mod = load_experiment("gine_panel")
    monkeypatch.setattr(mod, "load_rf_r2", lambda tag, split: 0.70)
    monkeypatch.setattr(mod, "load_gine_seed_r2",
                        lambda tag, split: [0.60, 0.62, 0.64])
    monkeypatch.setattr(mod, "load_gine_ci", lambda tag, split, sig_dir: None)

    panel = mod.build_panel("unused")
    out = tmp_path / "summary.csv"
    panel.to_csv(out, index=False)

    got = pd.read_csv(out)
    assert list(got.columns) == mod.PANEL_COLS
    assert got.columns[-1] == "delta_convention"
    assert len(got) == len(mod.TAGS) * len(mod.SPLITS)
    assert got["delta_convention"].tolist() == [mod.DELTA_CONVENTION] * len(got)
    assert mod.DELTA_CONVENTION == "point=per-seed-mean_R2;ci=ensemble_mean-pred_R2"
