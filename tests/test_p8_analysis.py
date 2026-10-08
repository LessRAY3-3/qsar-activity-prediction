"""P8 phase-0 analysis: paired-bootstrap CI signs + uncertainty/AD sanity.

Loads the two experiment modules by path (they must be import-safe: no
main() side effects, no torch/matplotlib at import time) and checks the
pure functions on small synthetic inputs with a known ground truth.
"""
import numpy as np
import pandas as pd
import pytest


def _synthetic_pair(n=30, seed=0, gin_noise=1.0, rf_noise=0.15):
    """y with a bad GIN (gin_noise) and a good RF (rf_noise)."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(3.0, 9.0, n)
    gin = y + rng.normal(0.0, gin_noise, n)
    rf = y + rng.normal(0.0, rf_noise, n)
    return y, gin, rf


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_rf_wins_ci_excludes_zero(load_experiment):
    mod = load_experiment("paired_bootstrap")
    y, gin, rf = _synthetic_pair()
    res = mod.paired_bootstrap(y, gin, rf, n_boot=2000, seed=42)

    assert res["delta_r2_point"] < 0            # GIN is the worse model
    assert res["ci_hi"] < 0                     # CI keeps the known sign
    assert res["p_one_sided"] < 0.01            # never beats RF
    assert res["delta_rmse_point"] > 0          # GIN RMSE larger
    assert res["delta_rmse_ci_lo"] > 0
    assert res["n_boot"] == 2000


def test_bootstrap_gin_wins_ci_excludes_zero(load_experiment):
    mod = load_experiment("paired_bootstrap")
    y, gin, rf = _synthetic_pair(gin_noise=0.15, rf_noise=1.0)
    res = mod.paired_bootstrap(y, gin, rf, n_boot=2000, seed=42)

    assert res["delta_r2_point"] > 0
    assert res["ci_lo"] > 0
    assert res["p_one_sided"] > 0.99


def test_bootstrap_reproducible_and_bounded(load_experiment):
    mod = load_experiment("paired_bootstrap")
    y, gin, rf = _synthetic_pair()
    a = mod.paired_bootstrap(y, gin, rf, n_boot=1000, seed=7)
    b = mod.paired_bootstrap(y, gin, rf, n_boot=1000, seed=7)
    assert a == b                              # fixed seed -> identical run

    assert a["ci_lo"] <= a["delta_r2_point"] <= a["ci_hi"]
    assert a["ci_lo"] < a["ci_hi"]
    assert 0.0 <= a["p_one_sided"] <= 1.0
    assert a["delta_rmse_ci_lo"] <= a["delta_rmse_point"] <= a["delta_rmse_ci_hi"]


def test_bootstrap_rejects_length_mismatch(load_experiment):
    mod = load_experiment("paired_bootstrap")
    with pytest.raises(ValueError):
        mod.paired_bootstrap(np.arange(10.0), np.arange(9.0), np.arange(10.0))


# ---------------------------------------------------------------- --model gine

def test_model_naming_gin_defaults_unchanged(load_experiment):
    """gin keeps the campaign's artifact names byte-for-byte."""
    mod = load_experiment("paired_bootstrap")
    assert mod.gnn_preds_name("a2a", "random", "") == "gnn_preds_a2a_random.npz"
    assert mod.gnn_preds_name("a2a", "random", "_seed1") == \
        "gnn_preds_a2a_random_seed1.npz"
    assert mod.out_stem("a2a", "random") == "a2a_random"
    assert mod.summary_name() == "summary.csv"
    assert mod.default_fig_name() == "ci_panel.png"


def test_model_naming_gine_distinguishes_outputs(load_experiment):
    """gine artifacts carry the _gine infix/suffix so gin files are safe."""
    mod = load_experiment("paired_bootstrap")
    assert mod.gnn_preds_name("a2a", "random", "", "gine") == \
        "gnn_preds_a2a_gine_random.npz"
    assert mod.gnn_preds_name("a2a", "scaffold", "_seed2", "gine") == \
        "gnn_preds_a2a_gine_scaffold_seed2.npz"
    assert mod.out_stem("a2a", "random", "gine") == "a2a_random_gine"
    assert mod.summary_name("gine") == "summary_gine.csv"
    assert mod.default_fig_name("gine") == "ci_panel_gine.png"


def test_parse_args_defaults_to_gin(load_experiment):
    mod = load_experiment("paired_bootstrap")
    args = mod.parse_args([])
    assert args.model == "gin"
    assert args.n_boot == 10_000
    assert args.fig is None                    # resolved per --model in main
    assert mod.parse_args(["--model", "gine"]).model == "gine"
    assert mod.parse_args(["--model", "attentivefp"]).model == "attentivefp"


# ---------------------------------------------------------------- --model attentivefp

def test_model_naming_attentivefp_distinguishes_outputs(load_experiment):
    """attentivefp artifacts carry the _attentivefp infix/suffix, and the
    gin/gine names stay byte-for-byte what they were."""
    mod = load_experiment("paired_bootstrap")
    assert "attentivefp" in mod.MODELS
    assert mod.gnn_preds_name("a2a", "random", "", "attentivefp") == \
        "gnn_preds_a2a_attentivefp_random.npz"
    assert mod.gnn_preds_name("a2a", "scaffold", "_seed2", "attentivefp") == \
        "gnn_preds_a2a_attentivefp_scaffold_seed2.npz"
    assert mod.out_stem("a2a", "random", "attentivefp") == "a2a_random_attentivefp"
    assert mod.summary_name("attentivefp") == "summary_attentivefp.csv"
    assert mod.default_fig_name("attentivefp") == "ci_panel_attentivefp.png"
    # gin / gine defaults unchanged
    assert mod.out_stem("a2a", "random") == "a2a_random"
    assert mod.out_stem("a2a", "random", "gine") == "a2a_random_gine"
    assert mod.summary_name() == "summary.csv"
    assert mod.summary_name("gine") == "summary_gine.csv"
    assert mod.default_fig_name() == "ci_panel.png"
    assert mod.default_fig_name("gine") == "ci_panel_gine.png"


def test_main_attentivefp_run_never_touches_gin_gine_significance(load_experiment,
                                                                  tmp_path, monkeypatch):
    """--model attentivefp writes only *_attentivefp artifacts; sentinel
    gin/gine cell CSVs, summaries and figures stay byte-for-byte intact."""
    mod = load_experiment("paired_bootstrap")
    _write_cell(tmp_path, "tiny", "random", model="attentivefp")
    _write_cell(tmp_path, "tiny", "scaffold", model="attentivefp")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    sentinels = {
        out / "summary.csv": "gin summary,do not clobber\n",
        out / "summary_gine.csv": "gine summary,do not clobber\n",
        out / "tiny_random.csv": "gin cell,do not clobber\n",
        out / "tiny_scaffold_gine.csv": "gine cell,do not clobber\n",
    }
    for path, text in sentinels.items():
        path.write_text(text)
    fig_dir = tmp_path / "figs"
    fig_sentinels = {}
    for name, text in (("ci_panel.png", "gin fig"),
                       ("ci_panel_gine.png", "gine fig")):
        p = fig_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        fig_sentinels[p] = text

    mod.main(["--model", "attentivefp", "--tags", "tiny",
              "--splits", "random,scaffold", "--n-boot", "300",
              "--out-dir", str(out),
              "--fig", str(fig_dir / "ci_panel_attentivefp.png")])

    for path, text in {**sentinels, **fig_sentinels}.items():
        assert path.read_text() == text, f"{path.name} was clobbered"
    # the attentivefp run's own outputs exist, gin/gine names never chosen
    for split in ("random", "scaffold"):
        cell = out / f"tiny_{split}_attentivefp.csv"
        assert cell.exists() and cell.stat().st_size > 0
    summary = pd.read_csv(out / "summary_attentivefp.csv")
    assert set(summary["split"]) == {"random", "scaffold"}
    assert set(summary["pair"]) == {"seed42", "mean3"}
    assert (fig_dir / "ci_panel_attentivefp.png").stat().st_size > 0
    assert mod.out_stem("tiny", "random", "attentivefp") != "tiny_random"
    assert mod.summary_name("attentivefp") != "summary.csv"
    assert mod.summary_name("attentivefp") != "summary_gine.csv"


# ---------------------------------------------------------------- --splits time

def test_split_suffix_protects_campaign_outputs(load_experiment):
    """Default runs keep legacy names; any foreign split gets a suffix."""
    mod = load_experiment("paired_bootstrap")
    assert mod.split_suffix(["random", "scaffold"]) == ""
    assert mod.split_suffix(["scaffold", "random"]) == ""
    assert mod.split_suffix(["time"]) == "_time"
    assert mod.split_suffix(["random", "time"]) == "_time"
    assert mod.split_suffix(["time", "year"]) == "_time_year"


def test_shared_names_with_split_suffix(load_experiment):
    """summary/fig names: byte-for-byte legacy for default runs, suffixed
    for time so results/significance/{summary,ci_panel} survive."""
    mod = load_experiment("paired_bootstrap")
    assert mod.summary_name() == "summary.csv"
    assert mod.default_fig_name() == "ci_panel.png"
    assert mod.summary_name(suffix="_time") == "summary_time.csv"
    assert mod.default_fig_name(suffix="_time") == "ci_panel_time.png"
    # split suffix composes with the existing gine suffix
    assert mod.summary_name("gine", "_time") == "summary_time_gine.csv"
    assert mod.default_fig_name("gine", "_time") == "ci_panel_time_gine.png"


def test_figure_cells_follow_rows_not_default_grid(load_experiment):
    """A time-only run plots its own cells, not the TAGS x SPLITS grid."""
    mod = load_experiment("paired_bootstrap")
    rows = [{"tag": t, "split": "time", "model_pair": pair}
            for t in ("t1", "t2") for pair in mod.PAIRS]
    assert mod.figure_cells(rows) == [("t1", "time"), ("t2", "time")]
    mixed = [{"tag": "t1", "split": "random", "model_pair": "seed42"},
             {"tag": "t1", "split": "time", "model_pair": "seed42"},
             {"tag": "t1", "split": "time", "model_pair": "mean3"}]
    assert mod.figure_cells(mixed) == [("t1", "random"), ("t1", "time")]


def _write_cell(tmp_path, tag, split, n=60, seed=0, model=""):
    """Synthetic split npz + rf/gnn preds npz for one (tag, split) cell.

    ``model`` adds the preds infix ("" -> gin campaign names, "gine" ->
    gnn_preds_{tag}_gine_{split}..., "attentivefp" -> ..._attentivefp_...).
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(3.0, 9.0, n)
    test_idx = np.arange(n // 3, n)
    split_dir = tmp_path / "data" / "processed" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / f"{tag}_{split}.npz", test_idx=test_idx)
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    np.savez(results / f"rf_preds_{tag}_{split}.npz", test_idx=test_idx,
             y_true=y, y_pred=y + rng.normal(0, 0.5, n))
    infix = f"_{model}" if model else ""
    for suf in ("", "_seed1", "_seed2"):
        np.savez(results / f"gnn_preds_{tag}{infix}_{split}{suf}.npz",
                 test_idx=test_idx, y_true=y,
                 y_pred=y + rng.normal(0, 0.7, n))
    return results, split_dir


def test_main_time_run_never_touches_campaign_summary(load_experiment,
                                                      tmp_path, monkeypatch):
    """--splits time writes summary_time.csv and leaves a sentinel
    summary.csv byte-for-byte untouched (suffix-protection integration)."""
    mod = load_experiment("paired_bootstrap")
    _write_cell(tmp_path, "tiny", "time")
    _write_cell(tmp_path, "tiny", "random")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "summary.csv"
    sentinel.write_text("sentinel,do not clobber\n")
    fig = tmp_path / "figs" / "ci_time.png"

    mod.main(["--tags", "tiny", "--splits", "time", "--n-boot", "300",
              "--out-dir", str(out), "--fig", str(fig)])

    assert sentinel.read_text() == "sentinel,do not clobber\n"
    assert fig.exists() and fig.stat().st_size > 0
    cell = out / "tiny_time.csv"
    assert cell.exists() and not (out / "tiny_random.csv").exists()
    summary = pd.read_csv(out / "summary_time.csv")
    assert set(summary["split"]) == {"time"}
    assert set(summary["pair"]) == {"seed42", "mean3"}
    assert {"ci_lo", "ci_hi", "significant"} <= set(summary.columns)

    # a default-split run still uses the legacy shared name and replaces
    # the sentinel in summary.csv (that is its documented behaviour)
    mod.main(["--tags", "tiny", "--splits", "random", "--n-boot", "300",
              "--out-dir", str(out),
              "--fig", str(tmp_path / "figs" / "ci_default.png")])
    default = pd.read_csv(out / "summary.csv")
    assert set(default["split"]) == {"random"}
    # the time run's outputs were never part of that overwrite
    summary = pd.read_csv(out / "summary_time.csv")
    assert set(summary["split"]) == {"time"}


# ---------------------------------------------------------------- uncertainty / AD

def test_max_similarities_in_unit_interval_with_self_match(load_experiment,
                                                           valid_smiles):
    mod = load_experiment("uncertainty_ad")
    fps = mod.morgan_fingerprints(valid_smiles)
    assert len(fps) == len(valid_smiles) and all(f is not None for f in fps)

    sims = mod.max_similarities(fps, fps)
    assert sims.shape == (len(fps),)
    assert np.all(np.isfinite(sims))
    assert np.all(sims >= 0.0) and np.all(sims <= 1.0)
    assert np.allclose(sims, 1.0)              # every query is in the ref set


def test_max_similarities_nan_for_invalid_smiles(load_experiment,
                                                 valid_smiles):
    mod = load_experiment("uncertainty_ad")
    refs = mod.morgan_fingerprints(valid_smiles)
    bad = mod.morgan_fingerprints(["INVALID_SMILES_XYZ"])
    assert bad == [None]
    sims = mod.max_similarities(bad, refs)
    assert sims.shape == (1,)
    assert np.isnan(sims[0])


def test_rf_tree_predict_mean_matches_sklearn_and_std_nonnegative(load_experiment):
    from sklearn.ensemble import RandomForestRegressor

    mod = load_experiment("uncertainty_ad")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 12))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    forest = RandomForestRegressor(n_estimators=25, random_state=0).fit(X, y)
    X_test = rng.normal(size=(20, 12))

    pred, std = mod.rf_tree_predict(forest, X_test)
    assert pred.shape == (20,) and std.shape == (20,)
    assert np.all(std >= 0.0)
    # the tree-mean IS sklearn's forest prediction
    assert np.allclose(pred, forest.predict(X_test), atol=1e-9)


def test_quintile_calibration_bins_and_positive_slope(load_experiment):
    mod = load_experiment("uncertainty_ad")
    rng = np.random.default_rng(1)
    n = 100
    y = rng.uniform(4.0, 8.0, n)
    std_true = np.linspace(0.05, 0.8, n)       # heteroscedastic noise
    pred = y + rng.normal(0.0, 1.0, n) * std_true

    cal = mod.quintile_calibration(y, pred, std_true, n_bins=5)
    assert sum(cal["bin_count"]) == n
    assert len(cal["bin_mean_std"]) == 5
    assert all(r >= 0.0 for r in cal["bin_rmse"])
    assert cal["slope"] > 0                    # bigger std -> bigger RMSE
    assert cal["spearman_std_vs_abs_err"] > 0.3
    assert np.isfinite(cal["intercept"])


def test_screening_precision_recall_known_answer(load_experiment):
    mod = load_experiment("uncertainty_ad")
    n = 20
    y_true = np.array([8.0, 7.5, 7.2, 7.0] + [6.0] * 16)   # 4 hits
    pred = np.array([9.0, 8.5, 8.0, 7.5] + [5.0] * 16)     # perfect ranking
    std = np.arange(n, dtype=float)                          # low std = hits
    max_sim = np.full(n, 0.6)                               # inside AD 0.5

    scr = mod.screening_summary(pred, std, max_sim, y_true,
                                ad_thresholds=(0.5, 0.9),
                                hit_threshold=7.0, topk=(4, 10),
                                top_frac=0.25)
    top = scr["strategies"]["rf_top"]
    assert scr["n_hits"] == 4
    assert top["coverage"] == 1.0
    assert top["precision_at"]["4"] == 1.0
    assert top["recall_at"]["4"] == 1.0
    assert top["precision_at"]["10"] == pytest.approx(0.4)   # 4 hits / 10 picks
    assert top["recall_at"]["10"] == 1.0

    # AD 0.5 keeps everything (identical to rf_top); AD 0.9 empties it
    assert scr["strategies"]["ad_t0.50"]["precision_at"]["4"] == 1.0
    assert scr["strategies"]["ad_t0.90"]["coverage"] == 0.0
    assert np.isnan(scr["strategies"]["ad_t0.90"]["precision_at"]["4"])

    # uncertainty ranking: ascending std = ascending index = hits first
    assert scr["strategies"]["unc_lowstd"]["precision_at"]["4"] == 1.0

    flagged = scr["flagged_pred_ge_threshold"]["all"]
    assert flagged["n_flagged"] == 4          # only the 4 preds >= 7.0
    assert flagged["precision"] == 1.0
    assert flagged["recall"] == 1.0
    # top-5% truth = ceil(0.25*20) = 5 best y -> one non-hit inside it
    assert flagged["precision_top5"] == 1.0    # flagged set is all hits
    assert flagged["recall_top5"] == pytest.approx(0.8)  # 4 of 5 top truth


def test_jsonable_nan_becomes_none(load_experiment):
    mod = load_experiment("uncertainty_ad")
    out = mod.jsonable({"a": float("nan"), "b": np.float64(np.inf),
                        "c": np.int64(3), "d": [np.bool_(True), 1.5]})
    assert out == {"a": None, "b": None, "c": 3, "d": [True, 1.5]}
