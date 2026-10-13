"""AD-only reruns must preserve predictions and reject stale input artifacts."""
import json

import numpy as np
import pandas as pd
import pytest


def _cell(tmp_path, monkeypatch, mod):
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    processed = tmp_path / "data" / "processed"
    (processed / "splits").mkdir(parents=True)
    smiles = np.array(["c1ccccc1", "CCO", "CCO", "CCC"])
    y = np.array([5., 6., 7., 8.])
    idx = np.array([2, 3])
    np.savez(processed / "tiny_fingerprints.npz", smiles=smiles, y=y)
    np.savez(processed / "splits" / "tiny_random.npz", train_idx=[0], valid_idx=[1], test_idx=idx)
    results = tmp_path / "results"
    results.mkdir()
    np.savez(results / "rf_preds_tiny_random.npz", y_true=y[idx], y_pred=[7.1, 7.8], test_idx=idx)
    stack = np.array([[7., 7.5], [7.3, 8.], [6.9, 7.6]])
    for suf, pred in zip(mod.SEED_SUFFIXES, stack):
        np.savez(results / f"gnn_preds_tiny_random{suf}.npz", y_true=y[idx], y_pred=pred, test_idx=idx)
    out = tmp_path / "out"
    out.mkdir()
    frames = []
    for model, pred, std in (("rf", np.array([7.1, 7.8]), np.array([.5, .4])),
                             ("gin", stack.mean(0), stack.std(0))):
        frames.append(pd.DataFrame(dict(model=model, test_idx=idx, smiles=smiles[idx], y_true=y[idx],
                                        pred=pred, std=std, max_sim=0., abs_err=abs(y[idx] - pred))))
    path = out / "tiny_random.csv"
    pd.concat(frames).to_csv(path, index=False)
    summary = dict(tag="tiny", model_kind="gin", ad_threshold=.4,
                   splits={"random": {"ad": {}, "calibration": {}}, "scaffold": {"sentinel": "keep me"}})
    (out / "summary_tiny.json").write_text(json.dumps(summary))
    monkeypatch.setattr(mod, "plot_ad_error", lambda *a, **kw: None)
    return out, path, results


def test_refresh_changes_only_ad_and_matches_summary(load_experiment, tmp_path, monkeypatch):
    mod = load_experiment("uncertainty_ad")
    out, path, _ = _cell(tmp_path, monkeypatch, mod)
    before = pd.read_csv(path, dtype=str)
    cell = mod.refresh_stored_ad("tiny", "random", "gin", str(out), str(tmp_path / "fig"))
    after_text = pd.read_csv(path, dtype=str)
    pd.testing.assert_frame_equal(before.drop(columns="max_sim"), after_text.drop(columns="max_sim"))
    after = pd.read_csv(path)
    rf, gin = [after[after.model == model] for model in ("rf", "gin")]
    assert rf.iloc[0].max_sim == 1
    assert gin.iloc[0].max_sim < .4
    for model, rows in (("rf", rf), ("gin", gin)):
        expected = mod.ad_metrics(rows.y_true.to_numpy(), rows.pred.to_numpy(), rows.max_sim.to_numpy())
        assert mod.jsonable(cell["ad"][model]) == mod.jsonable(expected)
    assert mod.jsonable(cell["screening"]) == mod.jsonable(mod.screening_summary(
        rf.pred.to_numpy(), rf["std"].to_numpy(), rf.max_sim.to_numpy(), rf.y_true.to_numpy()))


@pytest.mark.parametrize("fault", ["rf_index", "rf_label", "rf_prediction", "gnn_prediction", "gnn_std", "smiles"])
def test_refresh_rejects_stale_inputs_without_writing(load_experiment, tmp_path, monkeypatch, fault):
    mod = load_experiment("uncertainty_ad")
    out, path, results = _cell(tmp_path, monkeypatch, mod)
    if fault.startswith("rf_"):
        p = results / "rf_preds_tiny_random.npz"
        with np.load(p) as npz:
            values = dict(npz)
        key = {"rf_index": "test_idx", "rf_label": "y_true", "rf_prediction": "y_pred"}[fault]
        values[key] = values[key] + 1
        np.savez(p, **values)
    elif fault == "gnn_prediction":
        p = results / "gnn_preds_tiny_random_seed1.npz"
        with np.load(p) as npz:
            values = dict(npz)
        values["y_pred"] = values["y_pred"] + .1
        np.savez(p, **values)
    else:
        frame = pd.read_csv(path)
        if fault == "gnn_std":
            frame.loc[frame.model == "gin", "std"] += .1
        else:
            frame.loc[0, "smiles"] = "CCN"
        frame.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mod.refresh_stored_ad("tiny", "random", "gin", str(out), str(tmp_path / "fig"))
    assert path.read_bytes() == before


def test_partial_refresh_preserves_other_split_and_full_panel(load_experiment, tmp_path, monkeypatch):
    mod = load_experiment("uncertainty_ad")
    out, _, _ = _cell(tmp_path, monkeypatch, mod)
    figs = tmp_path / "fig"
    figs.mkdir()
    panel = figs / "vs_screen_panel_random.png"
    panel.write_bytes(b"full campaign")
    def plot(split, cells, path):
        from pathlib import Path
        Path(path).write_bytes(b"partial campaign")
    monkeypatch.setattr(mod, "plot_vs_screen_panel", plot)
    monkeypatch.setattr(mod, "print_conclusion", lambda *args: None)
    mod.main(["--tags", "tiny", "--splits", "random", "--refresh-ad-only",
              "--out-dir", str(out), "--fig-dir", str(figs)])
    summary = json.loads((out / "summary_tiny.json").read_text())
    assert summary["splits"]["scaffold"] == {"sentinel": "keep me"}
    assert summary["splits"]["random"]["ad_reference"] == {"rf": "train+valid", "gin": "train"}
    assert panel.read_bytes() == b"full campaign"
    assert (figs / "vs_screen_panel_random_partial.png").exists()


@pytest.mark.parametrize("args", [["--splits", "time"], ["--splits", ""], ["--tags", ""],
                                   ["--splits", "random,random"], ["--batch-size", "0"]])
def test_reject_invalid_selection_before_writes(load_experiment, tmp_path, args):
    mod = load_experiment("uncertainty_ad")
    out = tmp_path / "not-created"
    with pytest.raises(ValueError):
        mod.main(args + ["--out-dir", str(out)])
    assert not out.exists()


def test_plot_uses_separate_model_similarities(load_experiment, tmp_path, monkeypatch):
    mod = load_experiment("uncertainty_ad")
    import matplotlib.axes
    actual = []
    original = matplotlib.axes.Axes.scatter
    def capture(self, x, y, *args, **kwargs):
        actual.append(np.array(x))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", capture)
    sims = {"rf": np.array([1., .8]), "gin": np.array([.2, .3])}
    mod.plot_ad_error("tiny", "random", [7., 8.], [("rf", [7., 7.5], "red"),
                     ("gin", [7.1, 7.8], "green")], sims, .4, str(tmp_path / "plot.png"))
    np.testing.assert_array_equal(actual[0], sims["rf"])
    np.testing.assert_array_equal(actual[1], sims["gin"])
