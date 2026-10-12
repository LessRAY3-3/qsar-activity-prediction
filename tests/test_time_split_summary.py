"""time_split_summary: delta_convention plumbing + the d_r2_mean3 regression.

Three tests:
1. summary.csv carries the self-describing delta_convention column (the real
   SUMMARY_COLS / row plumbing runs against stubbed inputs).
2. for EVERY tag, results/time_split/summary.csv's stored d_r2_mean3 equals
   the ensemble delta recomputed from the committed prediction npz files --
   the chain is never trusted to pinned constants (the old regression test
   pinned herg values from mixed-vintage test sets and never read the file).
3. behavioral: build_summary on synthetic predictions where the ensemble
   delta and the naive column difference are both known by construction and
   differ (even in sign), so a mutant storing
   gin_time_r2_mean - rf_time_r2 instead of the ensemble delta is killed.

Columns are always selected by NAME: a concurrent change appends columns
(certain_r2, gin_best_epochs, ...) to time_split_summary.py.
"""
import json
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
    assert "delta_convention" in got.columns          # by name, never by position
    assert len(got) == len(mod.TAGS)
    assert got["delta_convention"].tolist() == [mod.DELTA_CONVENTION] * len(got)
    assert mod.DELTA_CONVENTION == "point=ensemble_mean-pred_R2;ci=ensemble_mean-pred_R2"


def _r2(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    sst = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - p) ** 2).sum() / sst


def _load_time_cell(results, tag):
    """Aligned (y, rf_pred, gin_ens_pred) from the committed time-split npz."""
    rf = np.load(results / f"rf_preds_{tag}_time.npz")
    gins = [np.load(results / f"gnn_preds_{tag}_time{suf}.npz")
            for suf in ("", "_seed1", "_seed2")]
    for g in gins:
        assert np.array_equal(g["test_idx"], rf["test_idx"])
        assert np.allclose(g["y_true"], rf["y_true"], atol=1e-4)
    y = np.asarray(rf["y_true"], dtype=np.float64)
    rf_pred = np.asarray(rf["y_pred"], dtype=np.float64)
    gin_ens = np.mean(np.stack([np.asarray(g["y_pred"], dtype=np.float64)
                                for g in gins]), axis=0)
    return y, rf_pred, gin_ens


def test_stored_d_r2_mean3_matches_recomputed_ensemble_delta(load_experiment):
    """results/time_split/summary.csv: d_r2_mean3 = ensemble delta, every tag.

    d_r2_mean3 is paired_bootstrap's mean3 point estimate
    R2(mean of the 3 GIN seed predictions) - R2(RF), i.e. what
    DELTA_CONVENTION calls "ensemble_mean-pred_R2". It is NOT
    gin_time_r2_mean - rf_time_r2: that displayed-column difference
    subtracts R2 values of different prediction sets (mean of per-seed R2
    vs RF R2) and can even flip sign (egfr: column diff -0.035, stored
    ensemble delta +0.039; the real herg stored value is +0.032304, not
    the old test's +0.007734).
    """
    mod = load_experiment("time_split_summary")
    results = Path(__file__).resolve().parent.parent / "results"

    stored = pd.read_csv(results / "time_split" / "summary.csv")
    assert set(stored["tag"]) == set(mod.TAGS)
    stored = stored.set_index("tag")

    sig = pd.read_csv(results / "significance" / "summary_time.csv")
    sig = sig.loc[sig["pair"] == "mean3"].set_index("tag")
    assert set(sig.index) == set(mod.TAGS)

    for tag in mod.TAGS:
        y, rf_pred, gin_ens = _load_time_cell(results, tag)
        ens_delta = _r2(y, gin_ens) - _r2(y, rf_pred)

        # independent recomputation from the committed prediction files
        assert float(stored.loc[tag, "d_r2_mean3"]) == pytest.approx(
            ens_delta, abs=1e-9)
        # build_summary copies the mean3 bootstrap point estimate verbatim
        assert float(stored.loc[tag, "d_r2_mean3"]) == pytest.approx(
            float(sig.loc[tag, "d_r2"]), abs=1e-12)
        # ...which is NOT the naive difference of the two displayed columns
        col_diff = (float(stored.loc[tag, "gin_time_r2_mean"])
                    - float(stored.loc[tag, "rf_time_r2"]))
        assert not np.isclose(ens_delta, col_diff, rtol=0, atol=1e-9)


def test_build_summary_keeps_ensemble_delta_not_column_difference(
        tmp_path, monkeypatch, load_experiment):
    """Behavioral: build_summary carries the ensemble delta, not the column diff.

    Synthetic predictions fix both quantities by construction and make them
    disagree (with this seed: ensemble delta ~= +0.283, column difference
    ~= -0.050). The ensemble delta is written into the synthetic
    significance summary exactly where paired_bootstrap's mean3
    delta_r2_point would put it; build_summary must pass it through into
    d_r2_mean3. A mutant storing gin_time_r2_mean - rf_time_r2 instead
    fails both assertions.
    """
    mod = load_experiment("time_split_summary")

    rng = np.random.default_rng(0)
    n_test = 60
    y = rng.normal(6.0, 1.5, size=n_test)
    rf_pred = y + rng.normal(0.0, 1.0, size=n_test)
    gin_preds = [y + rng.normal(0.0, 1.0, size=n_test) for _ in range(3)]

    rf_r2 = _r2(y, rf_pred)
    seed_r2s = [_r2(y, p) for p in gin_preds]
    gin_col_r2 = float(np.mean(seed_r2s))
    ens_delta = _r2(y, np.mean(np.stack(gin_preds), axis=0)) - rf_r2
    col_diff = gin_col_r2 - rf_r2
    # ground truths known by construction and clearly distinct
    assert abs(ens_delta - col_diff) > 0.1
    assert ens_delta > 0 > col_diff

    # synthetic on-disk inputs for every file build_summary reads
    n_total = 100
    years = rng.integers(1990, 2026, size=n_total).astype(np.float64)
    train_idx = np.arange(0, 20)
    valid_idx = np.arange(20, 40)
    test_idx = np.arange(40, n_total)
    assert len(test_idx) == n_test
    rmse = float(np.sqrt(np.mean((y - rf_pred) ** 2)))
    # y for the constant (no-skill) baseline; test rows are the same
    # targets the predictions above are scored against
    y_all = rng.normal(6.0, 1.5, size=n_total)
    y_all[test_idx] = y

    (tmp_path / "data" / "processed" / "splits").mkdir(parents=True)
    (tmp_path / "results" / "significance").mkdir(parents=True)
    sig_rows = []
    for tag in mod.TAGS:
        np.savez(tmp_path / "data" / "processed" / "splits" / f"{tag}_time.npz",
                 year=years, train_idx=train_idx, valid_idx=valid_idx,
                 test_idx=test_idx)
        np.savez(tmp_path / "data" / "processed" / f"{tag}_fingerprints.npz",
                 y=y_all)
        (tmp_path / "results" / f"metrics_{tag}.json").write_text(
            json.dumps({"time": {"r2": rf_r2, "rmse": rmse}}))
        for i, (suf, r2i) in enumerate(zip(("", "_seed1", "_seed2"), seed_r2s)):
            (tmp_path / "results" / f"gnn_metrics_{tag}{suf}.json").write_text(
                json.dumps({"time": {"r2": r2i, "best_epoch": 5 * i + 1}}))
        pd.DataFrame([
            {"split": "random", "rf_r2": 0.70, "gin_r2_mean": 0.60},
            {"split": "scaffold", "rf_r2": 0.65, "gin_r2_mean": 0.55},
        ]).to_csv(tmp_path / "results" / f"comparison_{tag}.csv", index=False)
        # seed42 row deliberately carries a different d_r2 so a mutant that
        # drops the mean3 filter cannot produce the expected value either
        sig_rows += [
            {"tag": tag, "split": "time", "pair": "seed42",
             "d_r2": ens_delta + 0.5, "ci_lo": ens_delta + 0.4,
             "ci_hi": ens_delta + 0.6, "significant": True,
             "winner": "gin", "frac_boot_gin_better": 1.0},
            {"tag": tag, "split": "time", "pair": "mean3",
             "d_r2": ens_delta, "ci_lo": ens_delta - 0.01,
             "ci_hi": ens_delta + 0.01, "significant": True,
             "winner": "gin", "frac_boot_gin_better": 0.99},
        ]
    pd.DataFrame(sig_rows).to_csv(
        tmp_path / "results" / "significance" / "summary_time.csv", index=False)

    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    df = mod.build_summary()

    got_d = df["d_r2_mean3"].to_numpy(dtype=np.float64)
    naive = (df["gin_time_r2_mean"] - df["rf_time_r2"]).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(got_d, ens_delta, rtol=0, atol=1e-12)
    assert not np.allclose(got_d, naive, rtol=0, atol=1e-9)
