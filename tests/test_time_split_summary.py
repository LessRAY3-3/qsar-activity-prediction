"""time_split_summary: summary.csv carries the delta_convention column.

build_summary's four data sources are stubbed (hermetic: no split npz or
significance csv needed); the real SUMMARY_COLS / row plumbing runs, the
frame goes through a tmp csv and comes back with the self-describing
column appended after every pre-existing column.
"""
import pandas as pd

SIG_ROW = {"d_r2": 0.1, "ci_lo": 0.0, "ci_hi": 0.2, "significant": True,
           "winner": "gin", "p_one_sided": 0.95}


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
    assert mod.DELTA_CONVENTION == "point=per-seed-mean_R2;ci=ensemble_mean-pred_R2"
