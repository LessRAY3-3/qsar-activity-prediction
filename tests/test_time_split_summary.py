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
    # constant_r2 reads data/processed/*.npz, which CI does not check out
    monkeypatch.setattr(mod, "constant_r2", lambda tag: 0.0)
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


# ------------------------------------------------- constant_r2 / gin_best_epochs (N4)

def _write_constant_tree(tmp_path, monkeypatch, load_experiment):
    """Synthetic y where every plausible baseline pool gives a different R2.

    train rows = 1s, valid rows = 2s, test rows = 9/9/10: the non-test pool
    mean (9/7), the all-rows mean (3.7), the train-only mean (1.0) and the
    test mean (28/3) all differ, so a wrong-pool mutant cannot land on the
    expected value by accident.
    """
    mod = load_experiment("time_split_summary")
    y = np.array([1, 1, 1, 1, 1, 2, 2, 9, 9, 10], dtype=np.float32)
    proc = tmp_path / "data" / "processed"
    (proc / "splits").mkdir(parents=True)
    np.savez(proc / "t1_fingerprints.npz", y=y)
    np.savez(proc / "splits" / "t1_time.npz",
             year=np.arange(len(y), dtype=np.float64),
             train_idx=np.arange(0, 5), valid_idx=np.arange(5, 7),
             test_idx=np.arange(7, 10))
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    return mod, np.asarray(y, dtype=np.float64)


def _baseline_r2(y, test_idx, pool_idx):
    """Test R2 of predicting pool_idx's mean pIC50 on every test row."""
    yt = y[test_idx]
    pred = np.full(len(test_idx), y[pool_idx].mean())
    return float(1.0 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())


def test_constant_r2_is_never_hardcoded_zero(tmp_path, monkeypatch,
                                             load_experiment):
    """Mutant killed: constant_r2 returning 0.0 for every input.

    On a test set far from the training pool the no-skill baseline R2 is
    strongly negative; a constant 0 is exactly what the wrong "predict the
    test mean" baseline would give and must not be reported.
    """
    mod, y = _write_constant_tree(tmp_path, monkeypatch, load_experiment)
    got = mod.constant_r2("t1")
    assert not np.isclose(got, 0.0, atol=1e-9)
    assert got == pytest.approx(
        _baseline_r2(y, np.arange(7, 10), np.arange(0, 7)), abs=1e-12)


def test_constant_r2_predicts_from_non_test_pool(tmp_path, monkeypatch,
                                                 load_experiment):
    """Mutant killed: computing the constant from the wrong pool.

    The baseline is the mean of every NON-test row (train+valid); all-rows,
    train-only and test-mean pools each give a different R2 here and none
    of them may match.
    """
    mod, y = _write_constant_tree(tmp_path, monkeypatch, load_experiment)
    test_idx = np.arange(7, 10)
    got = mod.constant_r2("t1")
    assert got == pytest.approx(_baseline_r2(y, test_idx, np.arange(0, 7)),
                                abs=1e-12)
    wrong_pools = {
        "all rows": np.arange(0, 10),
        "train only": np.arange(0, 5),
        "test rows": test_idx,
    }
    for name, pool in wrong_pools.items():
        assert not np.isclose(got, _baseline_r2(y, test_idx, pool), atol=1e-6), \
            f"constant_r2 matches the {name} pool"


def test_gin_best_epochs_reads_time_metric(load_experiment):
    """Mutant killed: best_epoch taken from the wrong split metric.

    gin_best_epochs must join the three seeds' ``time`` early-stop epochs;
    on the committed files every tag's random/scaffold best epochs differ
    from its time epochs, and the stored summary column must equal the
    time values exactly.
    """
    mod = load_experiment("time_split_summary")
    results = Path(__file__).resolve().parent.parent / "results"
    stored = pd.read_csv(results / "time_split" / "summary.csv").set_index("tag")

    def _be(tag, suf, split):
        with open(results / f"gnn_metrics_{tag}{suf}.json") as fh:
            return int(json.load(fh)[split]["best_epoch"])

    for tag in mod.TAGS:
        expected = ",".join(str(_be(tag, suf, "time"))
                            for suf in ("", "_seed1", "_seed2"))
        got = mod.gin_best_epochs(tag)
        assert got == expected
        assert str(stored.loc[tag, "gin_best_epochs"]) == expected
        for split in ("random", "scaffold"):
            wrong = ",".join(str(_be(tag, suf, split))
                             for suf in ("", "_seed1", "_seed2"))
            assert got != wrong, f"{tag}: time epochs equal {split} epochs"


def _write_per_tag_tree(tmp_path, mod):
    """Per-tag synthetic tree: every tag's y differs, so one tag's
    constant stored for all tags cannot pass."""
    rng = np.random.default_rng(0)
    test_idx = np.arange(40, 100)
    (tmp_path / "data" / "processed" / "splits").mkdir(parents=True)
    (tmp_path / "results").mkdir(exist_ok=True)
    per_tag_y = {}
    for i, tag in enumerate(mod.TAGS):
        y_tag = rng.normal(6.0 + 0.7 * i, 1.5, size=100)
        y_tag[test_idx] = rng.normal(9.0 + 0.5 * i, 1.0, size=len(test_idx))
        per_tag_y[tag] = y_tag
        np.savez(tmp_path / "data" / "processed" / "splits" / f"{tag}_time.npz",
                 year=rng.integers(1990, 2026, size=100).astype(np.float64),
                 train_idx=np.arange(0, 20), valid_idx=np.arange(20, 40),
                 test_idx=test_idx)
        np.savez(tmp_path / "data" / "processed" / f"{tag}_fingerprints.npz",
                 y=y_tag)
        epochs = (i * 10 + 1, i * 10 + 2, i * 10 + 3)
        for suf, e in zip(("", "_seed1", "_seed2"), epochs):
            (tmp_path / "results" / f"gnn_metrics_{tag}{suf}.json").write_text(
                json.dumps({"time": {"best_epoch": e, "r2": 0.5},
                            "random": {"best_epoch": 99, "r2": 0.9}}))
    return per_tag_y, test_idx


def _expected_constants(per_tag_y, test_idx):
    """Per-tag expected constant_r2 (pool = all non-test rows)."""
    pool_mask = np.ones(len(next(iter(per_tag_y.values()))), dtype=bool)
    pool_mask[test_idx] = False
    pool_idx = np.where(pool_mask)[0]
    return {tag: _baseline_r2(np.asarray(y, dtype=np.float64), test_idx,
                              pool_idx)
            for tag, y in per_tag_y.items()}


def test_build_summary_writes_correct_baseline_columns(tmp_path, monkeypatch,
                                                       load_experiment):
    """Mutant killed: column present but filled with the wrong value.

    build_summary runs the REAL constant_r2 / gin_best_epochs (nothing is
    stubbed for them): constant_r2 must equal each tag's own non-test-pool
    R2. The tree carries a different y per tag and the expected constants
    are pairwise distinct, so storing ONE tag's constant for every tag --
    the shared-fixture gap the 2026-10-13 review found here -- fails.
    gin_best_epochs must join the time best_epoch of each seed -- a decoy
    random.best_epoch sits in every json to catch a wrong-metric read at
    plumbing level too.
    """
    mod = load_experiment("time_split_summary")
    per_tag_y, test_idx = _write_per_tag_tree(tmp_path, mod)
    expected_const = _expected_constants(per_tag_y, test_idx)
    # the fixture must discriminate per-tag constants at all
    assert len({round(v, 9) for v in expected_const.values()}) == len(mod.TAGS)

    # everything else build_summary reads is canned (see test 1)
    monkeypatch.setattr(mod, "time_split_stats", lambda tag: {
        "test_year_start": 2020, "test_year_end": 2021,
        "n_total": 100, "n_train": 20, "n_valid": 20, "n_test": 60,
        "test_frac": 0.6})
    monkeypatch.setattr(mod, "_load_json",
                        lambda name: {"time": {"r2": 0.5, "rmse": 1.0}})
    monkeypatch.setattr(mod, "comparison_reference",
                        lambda tag: {"random": (0.7, 0.6),
                                     "scaffold": (0.6, 0.55)})
    monkeypatch.setattr(mod, "time_significance", lambda: pd.DataFrame(
        [SIG_ROW] * len(mod.TAGS), index=list(mod.TAGS)))
    monkeypatch.setattr(mod, "BASE", str(tmp_path))

    df = mod.build_summary().set_index("tag")
    assert "constant_r2" in df.columns and "gin_best_epochs" in df.columns

    for tag in mod.TAGS:
        assert not np.isclose(expected_const[tag], 0.0, atol=1e-9)
        assert float(df.loc[tag, "constant_r2"]) == pytest.approx(
            expected_const[tag], abs=1e-12)
        # ...and NOT any other tag's constant (shared-fixture mutant)
        others = [expected_const[t] for t in mod.TAGS if t != tag]
        assert not any(
            np.isclose(float(df.loc[tag, "constant_r2"]), o, atol=1e-9)
            for o in others), f"{tag}: constant_r2 matches another tag's pool"

    expected_epochs = {
        tag: ",".join(str(i * 10 + k) for k in (1, 2, 3))
        for i, tag in enumerate(mod.TAGS)}
    assert df["gin_best_epochs"].to_dict() == expected_epochs


def test_stored_constant_r2_matches_recomputed_baseline(tmp_path, monkeypatch,
                                                        load_experiment):
    """results/time_split/summary.csv: constant_r2 = recomputed no-skill R2.

    Dual path. When the committed data/processed npz files are checked out
    (local full repo; CI does not check them out), every stored tag value
    must equal an INDEPENDENT recomputation from
    {tag}_fingerprints.npz + {tag}_time.npz (pool = all non-test rows).
    Without the npz (CI) the same recomputation contract is enforced on a
    synthetic committed-style tree through the real build_summary, and the
    committed summary.csv must still carry a finite constant_r2 for every
    tag.
    """
    mod = load_experiment("time_split_summary")
    repo = Path(__file__).resolve().parent.parent
    results = repo / "results"
    proc = repo / "data" / "processed"
    stored = pd.read_csv(results / "time_split" / "summary.csv").set_index("tag")
    assert set(stored.index) == set(mod.TAGS)
    assert stored["constant_r2"].notna().all()

    def _recompute(tag, root):
        y = np.asarray(
            np.load(root / "data" / "processed" / f"{tag}_fingerprints.npz")["y"],
            dtype=np.float64)
        sp = np.load(root / "data" / "processed" / "splits" / f"{tag}_time.npz")
        test = np.asarray(sp["test_idx"])
        pool = np.ones(len(y), dtype=bool)
        pool[test] = False
        return _baseline_r2(y, test, np.where(pool)[0])

    have_npz = all(
        (proc / f"{tag}_fingerprints.npz").exists()
        and (proc / "splits" / f"{tag}_time.npz").exists()
        for tag in mod.TAGS)
    if have_npz:
        for tag in mod.TAGS:
            assert float(stored.loc[tag, "constant_r2"]) == pytest.approx(
                _recompute(tag, repo), abs=1e-9)
        return

    # CI path: no committed npz -- enforce the same contract on a synthetic
    # committed-style tree through the real build_summary
    per_tag_y, test_idx = _write_per_tag_tree(tmp_path, mod)
    expected_const = _expected_constants(per_tag_y, test_idx)
    monkeypatch.setattr(mod, "time_split_stats", lambda tag: {
        "test_year_start": 2020, "test_year_end": 2021,
        "n_total": 100, "n_train": 20, "n_valid": 20, "n_test": 60,
        "test_frac": 0.6})
    monkeypatch.setattr(mod, "_load_json",
                        lambda name: {"time": {"r2": 0.5, "rmse": 1.0}})
    monkeypatch.setattr(mod, "comparison_reference",
                        lambda tag: {"random": (0.7, 0.6),
                                     "scaffold": (0.6, 0.55)})
    monkeypatch.setattr(mod, "time_significance", lambda: pd.DataFrame(
        [SIG_ROW] * len(mod.TAGS), index=list(mod.TAGS)))
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    df = mod.build_summary().set_index("tag")
    for tag in mod.TAGS:
        assert float(df.loc[tag, "constant_r2"]) == pytest.approx(
            expected_const[tag], abs=1e-12)
        assert float(df.loc[tag, "constant_r2"]) == pytest.approx(
            _recompute(tag, tmp_path), abs=1e-12)


def test_print_conclusion_labels_drift_conditionally(load_experiment, capsys):
    """Conditional drift wording (6c1e3bb; condition corrected per PR #6).

    Mutant killed: the unconditional label -- every negative RF time R2
    announced as "genuine temporal drift" regardless of the constant
    baseline. Final polarity: 'holder' (negative RF that holds at/above a
    negative constant) reads as genuine temporal drift; 'undercut'
    (RF below the constant) is below no-skill, not drift. The two must
    land in different note lines, never in one shared drift line.
    """
    mod = load_experiment("time_split_summary")
    df = pd.DataFrame([
        {"tag": "undercut", "rf_time_r2": -0.30, "constant_r2": -0.05,
         "gin_time_r2_mean": -0.10, "rf_random_r2": 0.70,
         "gin_random_r2_mean": 0.65, "significant": False, "winner": "tie",
         "test_frac": 0.2},
        {"tag": "holder", "rf_time_r2": -0.05, "constant_r2": -0.40,
         "gin_time_r2_mean": 0.10, "rf_random_r2": 0.70,
         "gin_random_r2_mean": 0.65, "significant": False, "winner": "tie",
         "test_frac": 0.2},
        {"tag": "positive", "rf_time_r2": 0.40, "constant_r2": -0.01,
         "gin_time_r2_mean": 0.35, "rf_random_r2": 0.70,
         "gin_random_r2_mean": 0.65, "significant": False, "winner": "tie",
         "test_frac": 0.2},
    ])
    mod.print_conclusion(df)
    out = capsys.readouterr().out

    note_lines = [ln for ln in out.splitlines() if "note:" in ln]
    drift = [ln for ln in note_lines
             if "genuine temporal drift" in ln and "not claimed" not in ln]
    non_drift = [ln for ln in note_lines
                 if "not claimed as genuine temporal drift" in ln]
    assert len(drift) == 1, f"expected one drift note, got {drift}"
    assert "holder" in drift[0] and "undercut" not in drift[0]
    assert len(non_drift) == 1, f"expected one non-drift note, got {non_drift}"
    assert "undercut" in non_drift[0] and "holder" not in non_drift[0]
