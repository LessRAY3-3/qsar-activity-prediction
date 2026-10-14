"""L5 latent-robustness sub-items 3-5 (per codex-verification-2026-10-10).

Synthetic inputs assert the behaviours the fixes doc claims:

3. ``uncertainty_ad.process_cell`` treats a PARTIAL checkpoint set like an
   absent one: the GNN rows are skipped (no warn-then-raise on the missing
   .pt) and the summary records the true ``n_gnn_seeds``.
4. ``gine_panel`` / ``afp_panel`` colour bars by the CI (``ci_lo > 0`` /
   CI excludes 0), never by the sign of the point estimate.
5. ``multi_target_summary`` / ``time_split_summary`` console verdicts are
   generated from the data, not hard-coded campaign text.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BLUE, RED, GRAY = "tab:blue", "tab:red", "0.75"


# ---------------------------------------------------------------- L5.3

def _write_uncertainty_tree(base):
    rng = np.random.default_rng(0)
    n, train_n = 70, 40
    proc = base / "data" / "processed"
    split_dir = proc / "splits"
    split_dir.mkdir(parents=True)
    results = base / "results"
    (results / "gnn_models").mkdir(parents=True)
    models = base / "models"
    models.mkdir()

    pool = ["CCO", "c1ccncc1", "CC(=O)O", "c1ccoc1", "C1CCCCC1",
            "c1ccsc1", "C1CCNCC1", "O=C(c1ccccc1)Nc1ccccc1"]
    smiles = np.array([pool[i % len(pool)] for i in range(n)])
    X = rng.normal(size=(n, 16)).astype(np.float32)
    np.savez(proc / "tiny_fingerprints.npz", X=X, smiles=smiles)

    train_idx = np.arange(train_n)
    test_idx = np.arange(train_n, n)
    np.savez(split_dir / "tiny_random.npz", train_idx=train_idx,
             valid_idx=np.array([], dtype=int), test_idx=test_idx)

    y_all = rng.normal(6.0, 1.0, n)
    forest = RandomForestRegressor(n_estimators=12, random_state=0).fit(
        X[train_idx], y_all[train_idx])
    joblib.dump(forest, models / "rf_random_split_tiny.joblib")
    np.savez(results / "rf_preds_tiny_random.npz", test_idx=test_idx,
             y_true=y_all[test_idx], y_pred=forest.predict(X[test_idx]))

    # only 2 of the 3 checkpoints exist -> partial set for model_kind=gin
    gnn_dir = results / "gnn_models"
    (gnn_dir / "tiny_gin_random.pt").write_bytes(b"")
    (gnn_dir / "tiny_gin_random_seed1.pt").write_bytes(b"")
    # seed2 deliberately absent
    return len(test_idx)


def test_partial_checkpoint_set_skips_gnn_rows(load_experiment, monkeypatch,
                                               tmp_path, capsys):
    mod = load_experiment("uncertainty_ad")
    n_test = _write_uncertainty_tree(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out_dir = tmp_path / "results" / "uncertainty"
    fig_dir = tmp_path / "figures" / "uncertainty"
    out_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)

    res = mod.process_cell("tiny", "random", "gin", str(out_dir), str(fig_dir))
    out = capsys.readouterr().out

    # the partial set is reported, not mixed into a 3-seed ensemble, and
    # nothing raises (the old path warned here, then died in
    # gnn_ensemble_predict on the missing checkpoint)
    assert "only 2/3 gin checkpoints" in out
    assert "GNN rows omitted" in out
    assert res["n_gnn_seeds"] == 2
    assert res["gnn_ensemble_max_diff"] is None
    assert set(res["calibration"]) == {"rf"}
    assert "gin" not in res["ad"]
    assert res["n_test"] == n_test

    csv = pd.read_csv(out_dir / "tiny_random.csv")
    assert set(csv["model"]) == {"rf"}
    assert len(csv) == n_test
    # the summary dict survives the jsonable round trip used for summary_*.json
    assert mod.jsonable(res)["gnn_ensemble_max_diff"] is None


# ---------------------------------------------------------------- L5.4

def _panel_rows(mod, nan_row=False):
    """Synthetic panel frame with known colour decisions per cell."""
    rows = []
    for split in mod.SPLITS:
        for i, tag in enumerate(mod.TAGS):
            d, lo, hi, sig = 0.05, 0.01, 0.10, True     # default: blue
            if split == "random":
                if i == 0:            # negative point, all-positive CI -> blue
                    d, lo, hi = -0.01, 0.01, 0.05
                elif i == 1:          # significant RF win -> red
                    d, lo, hi = -0.20, -0.30, -0.10
                elif i == 2:          # CI includes 0 -> grey
                    if nan_row:
                        d, lo, hi, sig = 0.10, np.nan, np.nan, None
                    else:
                        sig = False
                elif i == 3:          # positive point, negative CI -> red
                    d, lo, hi = 0.15, -0.25, -0.05
                elif i == 4 and nan_row:   # finite CI but not significant
                    sig = False
            rows.append({"tag": tag, "split": split, "d_r2_mean": d,
                         "ci_lo": lo, "ci_hi": hi, "significant": sig})
    return pd.DataFrame(rows)


def _bar_colors(monkeypatch, make_figure, df, fig_path):
    """Run a panel figure and return the color list of each ax.bar call."""
    import matplotlib.axes

    orig_bar = matplotlib.axes.Axes.bar
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("color"))
        return orig_bar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy)
    make_figure(df, str(fig_path))
    return calls


def test_gine_panel_bar_colors_follow_ci_not_point_sign(load_experiment,
                                                        monkeypatch, tmp_path):
    mod = load_experiment("gine_panel")
    df = _panel_rows(mod)
    calls = _bar_colors(monkeypatch, mod.make_panel_figure, df,
                        tmp_path / "panel.png")
    assert len(calls) == len(mod.SPLITS)
    # random: conflict row (negative point, positive CI) is still blue,
    # positive point with negative CI is red, non-significant is grey
    assert calls[0] == [BLUE, RED, GRAY, RED, BLUE, BLUE, BLUE, BLUE]
    assert calls[1] == [BLUE] * len(mod.TAGS)


def test_afp_panel_bar_colors_follow_ci_not_point_sign(load_experiment,
                                                       monkeypatch, tmp_path):
    mod = load_experiment("afp_panel")
    df = _panel_rows(mod, nan_row=True)
    calls = _bar_colors(monkeypatch, mod.make_panel_figure, df,
                        tmp_path / "panel.png")
    assert len(calls) == len(mod.SPLITS)
    # i==2 has no bootstrap (NaN CI) -> grey even with a positive point
    assert calls[0] == [BLUE, RED, GRAY, RED, GRAY, BLUE, BLUE, BLUE]
    assert calls[1] == [BLUE] * len(mod.TAGS)


# ---------------------------------------------------------------- L5.5

def test_multi_target_verdict_follows_the_data(load_experiment, capsys):
    mod = load_experiment("multi_target_summary")

    mod.print_conclusion([{"tag": "t1", "d_r2_random": 0.05,
                           "d_r2_scaffold": 0.0}])
    out = capsys.readouterr().out
    assert "the GIN has a positive point estimate on 1 cell(s): t1/random" in out
    assert "no target gives the GIN a positive rounded point estimate" not in out

    # everything rounds to 0.000 -> the verdict flips with the data
    mod.print_conclusion([{"tag": "t1", "d_r2_random": 0.0004,
                           "d_r2_scaffold": -0.0004}])
    out = capsys.readouterr().out
    assert "no target gives the GIN a positive rounded point estimate" in out
    assert "has a positive point estimate" not in out


def test_time_split_conclusion_follows_the_data(load_experiment, capsys):
    mod = load_experiment("time_split_summary")
    healthy = pd.DataFrame([
        {"tag": "t1", "rf_time_r2": 0.3, "rf_random_r2": 0.6,
         "gin_time_r2_mean": 0.35, "gin_random_r2_mean": 0.6,
         "constant_r2": 0.1, "significant": False, "winner": "rf",
         "test_frac": 0.2},
        {"tag": "t2", "rf_time_r2": 0.4, "rf_random_r2": 0.6,
         "gin_time_r2_mean": 0.45, "gin_random_r2_mean": 0.6,
         "constant_r2": 0.1, "significant": False, "winner": "rf",
         "test_frac": 0.2},
        {"tag": "t3", "rf_time_r2": 0.5, "rf_random_r2": 0.6,
         "gin_time_r2_mean": 0.55, "gin_random_r2_mean": 0.6,
         "constant_r2": 0.1, "significant": False, "winner": "rf",
         "test_frac": 0.2},
    ])
    mod.print_conclusion(healthy)
    out = capsys.readouterr().out
    assert "negative time R²: RF none; GIN none" in out
    assert "RF below constant baseline: none" in out
    assert "note: negative time-split" not in out
    # The label-policy caveat applies even to positive R2. Only the data-driven
    # large-test-pool warning should disappear for these small test fractions.
    assert "test_frac=" not in out
    assert "random/scaffold use all-year labels; time uses as-of labels" in out

    troubled = pd.DataFrame([
        {"tag": "t1", "rf_time_r2": -0.4, "rf_random_r2": 0.6,
         "gin_time_r2_mean": -0.2, "gin_random_r2_mean": 0.6,
         "constant_r2": 0.1, "significant": False, "winner": "rf",
         "test_frac": 0.6},
        {"tag": "t2", "rf_time_r2": 0.5, "rf_random_r2": 0.7,
         "gin_time_r2_mean": 0.55, "gin_random_r2_mean": 0.7,
         "constant_r2": 0.1, "significant": True, "winner": "gin",
         "test_frac": 0.2},
    ])
    mod.print_conclusion(troubled)
    out = capsys.readouterr().out
    assert "descriptive protocol ΔR² (time − random): RF -0.600, GIN -0.475" in out
    assert "negative time R²: RF ['t1']; GIN ['t1']" in out
    assert "RF below constant baseline: t1 -0.400 < +0.100" in out
    assert "GIN ahead: ['t2']; RF ahead: none" in out
    assert "caveat: t1 test_frac=0.60 > 0.5" in out
    assert "t1 RF -0.400, constant +0.100: below the constant baseline" in out
    assert "do not establish temporal drift" in out
