"""time_split_summary: summary.csv carries the delta_convention column.

build_summary's four data sources are stubbed (hermetic: no split npz or
significance csv needed); the real SUMMARY_COLS / row plumbing runs, the
frame goes through a tmp csv and comes back with the self-describing
column appended after every pre-existing column.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SIG_ROW = {"d_r2": 0.1, "ci_lo": 0.0, "ci_hi": 0.2, "significant": True,
           "winner": "gin", "frac_boot_gin_better": 0.95}


def test_summary_csv_carries_delta_convention(tmp_path, monkeypatch, load_experiment):
    mod = load_experiment("time_split_summary")
    monkeypatch.setattr(mod, "time_split_stats", lambda tag: {
        "test_year_start": 2020, "test_year_end": 2021,
        "n_total": 10, "n_train": 6, "n_valid": 2, "n_test": 2,
        "test_frac": 0.2})
    # one canned payload serves both metrics_{tag}.json and gnn_metrics_{tag}suf
    monkeypatch.setattr(mod, "_load_json",
                        lambda name: {"time": {"r2": 0.5, "rmse": 1.0}})
    monkeypatch.setattr(mod, "comparison_reference",
                        lambda tag: {"random": (0.7, 0.6), "scaffold": (0.6, 0.55)})
    sig = pd.DataFrame([SIG_ROW] * len(mod.TAGS), index=list(mod.TAGS))
    monkeypatch.setattr(mod, "time_significance", lambda: sig)

    df = mod.build_summary()
    out = tmp_path / "summary.csv"
    df.to_csv(out, index=False)

    got = pd.read_csv(out)
    assert list(got.columns) == mod.SUMMARY_COLS
    assert got.columns[-1] == "delta_convention"
    assert len(got) == len(mod.TAGS)
    assert got["delta_convention"].tolist() == [mod.DELTA_CONVENTION] * len(got)
    assert mod.DELTA_CONVENTION == "point=ensemble_mean-pred_R2;ci=ensemble_mean-pred_R2"


# Independently verified herg/time ground truth (six decimals), recomputed
# from the committed per-seed prediction npz files:
#   RF time R2        = -0.033775  (pinned constant, see the test below)
#   per-seed GIN R2s  = -0.019990 / -0.045609 / -0.162938
#                       (mean -0.076179 -> per-seed-mean delta -0.042404)
#   R2(mean GIN pred) = +0.041903  ->  ensemble delta +0.075677
# The gnn_preds_* npz files are GNN campaign artifacts; an RF re-run never
# touches them, while it does rewrite results/metrics_*.json / rf_preds_*.
HERG_RF_TIME_R2 = -0.033775
HERG_GIN_ENSEMBLE_R2 = 0.041903
HERG_GIN_ENSEMBLE_DELTA = 0.075677
HERG_GIN_PER_SEED_DELTA = -0.042404


def _r2(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    sst = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - p) ** 2).sum() / sst


def test_delta_convention_matches_independent_ensemble_recomputation(load_experiment):
    """DELTA_CONVENTION must describe the delta d_r2_mean3 actually stores.

    The M4 review caught the old label (point=per-seed-mean_R2) describing
    a quantity the summary never stored: for herg the two displayed columns
    subtract to -0.042 while the stored ensemble delta is +0.076. The
    convention string is therefore pinned against an independent
    recomputation from the committed herg/time per-seed GIN predictions;
    the RF R2 stays a pinned constant so an RF re-run cannot invalidate
    the test.
    """
    mod = load_experiment("time_split_summary")
    assert (mod.DELTA_CONVENTION
            == "point=ensemble_mean-pred_R2;ci=ensemble_mean-pred_R2")

    results = Path(__file__).resolve().parent.parent / "results"
    seeds = [np.load(results / f"gnn_preds_herg_time{suf}.npz")
             for suf in ("", "_seed1", "_seed2")]
    assert all(np.array_equal(s["test_idx"], seeds[0]["test_idx"])
               for s in seeds[1:])
    y = np.asarray(seeds[0]["y_true"], dtype=np.float64)
    seed_preds = [np.asarray(s["y_pred"], dtype=np.float64) for s in seeds]

    ensemble_r2 = _r2(y, np.mean(np.stack(seed_preds), axis=0))
    per_seed_delta = (float(np.mean([_r2(y, p) for p in seed_preds]))
                      - HERG_RF_TIME_R2)

    assert ensemble_r2 == pytest.approx(HERG_GIN_ENSEMBLE_R2, abs=5e-4)
    assert ensemble_r2 - HERG_RF_TIME_R2 == pytest.approx(
        HERG_GIN_ENSEMBLE_DELTA, abs=5e-4)
    # the difference of the two displayed per-seed columns is a DIFFERENT
    # quantity - negative here while the ensemble delta is positive
    assert per_seed_delta == pytest.approx(HERG_GIN_PER_SEED_DELTA, abs=5e-4)
