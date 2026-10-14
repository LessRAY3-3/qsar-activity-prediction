"""P8 phase-3 AttentiveFP panel: pure row/aggregation helpers.

Loads experiments/afp_panel.py by path (import-safe: matplotlib is
imported inside the figure functions only) and checks the panel-row
contract on synthetic inputs with a known ground truth.
"""
import numpy as np
import pandas as pd
import pytest


def test_aggregate_seed_r2_mean_population_std(load_experiment):
    mod = load_experiment("afp_panel")
    mean, std, n = mod.aggregate_seed_r2([0.7, 0.8, 0.9])
    assert n == 3
    assert mean == pytest.approx(0.8)
    assert std == pytest.approx(np.std([0.7, 0.8, 0.9], ddof=0))


def test_aggregate_seed_r2_rejects_empty(load_experiment):
    mod = load_experiment("afp_panel")
    with pytest.raises(ValueError):
        mod.aggregate_seed_r2([])


def test_make_row_delta_and_bootstrap_note(load_experiment):
    mod = load_experiment("afp_panel")
    row = mod.make_row("a2a", "random", 0.70, [0.60, 0.62, 0.64],
                       ci=(-0.05, -0.01, True))
    assert row["afp_r2_mean"] == pytest.approx(0.62)
    assert row["afp_n_seeds"] == 3
    assert row["d_r2_mean"] == pytest.approx(0.62 - 0.70)
    assert row["ci_lo"] == -0.05 and row["ci_hi"] == -0.01
    assert row["significant"] is True
    assert "mean3" in row["note"]
    assert row["note"] != mod.NOTE_NO_CI


def test_make_row_without_ci_flags_missing_bootstrap(load_experiment):
    mod = load_experiment("afp_panel")
    row = mod.make_row("vegfr2", "scaffold", 0.60, [0.55, 0.58, 0.61])
    assert row["d_r2_mean"] == pytest.approx(np.mean([0.55, 0.58, 0.61]) - 0.60)
    assert np.isnan(row["ci_lo"]) and np.isnan(row["ci_hi"])
    assert row["significant"] is None
    assert row["note"] == mod.NOTE_NO_CI


def test_output_paths_never_collide_with_gine_panel(load_experiment):
    """afp panel defaults are distinct from every gine panel / phase-0 path."""
    afp = load_experiment("afp_panel")
    gine = load_experiment("gine_panel")
    a, g = afp.parse_args([]), gine.parse_args([])
    assert a.out_csv != g.out_csv                 # afp_panel/ vs gine_panel/
    assert a.fig != g.fig
    assert a.combined_fig.endswith("ci_panel_with_afp.png")
    assert g.combined_fig.endswith("ci_panel_with_gine.png")
    assert not a.combined_fig.endswith("ci_panel.png")


def test_summary_csv_carries_delta_convention(tmp_path, monkeypatch, load_experiment):
    """build_panel -> summary.csv keeps every column and appends delta_convention."""
    mod = load_experiment("afp_panel")
    monkeypatch.setattr(mod, "load_rf_r2", lambda tag, split: 0.70)
    monkeypatch.setattr(mod, "load_afp_seed_r2",
                        lambda tag, split: [0.60, 0.62, 0.64])
    monkeypatch.setattr(mod, "load_afp_ci", lambda tag, split, sig_dir: None)

    panel = mod.build_panel("unused")
    out = tmp_path / "summary.csv"
    panel.to_csv(out, index=False)

    got = pd.read_csv(out)
    assert list(got.columns) == mod.PANEL_COLS
    assert got.columns[-1] == "delta_convention"
    assert len(got) == len(mod.TAGS) * len(mod.SPLITS)
    assert got["delta_convention"].tolist() == [mod.DELTA_CONVENTION] * len(got)
    assert mod.DELTA_CONVENTION == "point=per-seed-mean_R2;ci=ensemble_mean-pred_R2"


# ------------------------------------------- combined forest: fallback whisker

_SUMMARY_COLS = ["tag", "split", "pair", "d_r2", "ci_lo", "ci_hi", "significant"]


def _all_cells(mod):
    return [(t, s) for t in mod.TAGS for s in mod.SPLITS]


def _write_mean3(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_SUMMARY_COLS).to_csv(path, index=False)


def _record_forest(monkeypatch):
    """Capture every Axes.hlines / Axes.plot call the forest makes."""
    import matplotlib.axes

    hlines, plots = [], []
    orig_h = matplotlib.axes.Axes.hlines
    orig_p = matplotlib.axes.Axes.plot

    def rec_h(self, y, xmin, xmax, *args, **kwargs):
        hlines.append((float(np.ravel(y)[0]), float(np.ravel(xmin)[0]),
                       float(np.ravel(xmax)[0])))
        return orig_h(self, y, xmin, xmax, *args, **kwargs)

    def rec_p(self, *args, **kwargs):
        if len(args) >= 2:
            try:
                plots.append((float(np.ravel(args[0])[0]),
                              float(np.ravel(args[1])[0]), dict(kwargs)))
            except (TypeError, ValueError):
                pass
        return orig_p(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hlines", rec_h)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", rec_p)
    return hlines, plots


def test_combined_forest_fallback_draws_whisker_and_centre_from_panel(
        tmp_path, monkeypatch, load_experiment
):
    """A fairness cell absent from summary_gine draws whisker AND centre
    from results/gine_panel/summary.csv, together.

    Mutant killed: the pre-6c1e3bb fallback mixed files (whisker from the
    per-cell bootstrap CSVs / summary_gine, centre from the panel) -- the
    decoy per-cell CSVs and the decoy panel rows for PRESENT cells carry
    deliberately different values, so neither mix can pass.
    """
    mod = load_experiment("afp_panel")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    sig_dir = tmp_path / "results" / "significance"
    fallback = ("abl1", "random")          # missing from summary_gine.csv
    present = ("a2a", "scaffold")          # carried by summary_gine.csv
    cells = _all_cells(mod)

    base = lambda t, s: {"tag": t, "split": s, "pair": "mean3", "significant": True}  # noqa: E731
    _write_mean3(sig_dir / "summary.csv",
                 [{**base(t, s), "d_r2": 0.01, "ci_lo": -0.01, "ci_hi": 0.03}
                  for t, s in cells])
    _write_mean3(sig_dir / "summary_attentivefp.csv",
                 [{**base(t, s), "d_r2": 0.02, "ci_lo": -0.02, "ci_hi": 0.06}
                  for t, s in cells])
    _write_mean3(sig_dir / "summary_gine.csv",
                 [{**base(t, s), "d_r2": 0.05, "ci_lo": 0.01, "ci_hi": 0.09}
                  for t, s in cells if (t, s) != fallback])

    # per-cell bootstrap CSVs: the old mixed-source whisker values
    for t, s in cells:
        pd.DataFrame([{"model_pair": "mean3", "d_r2": -0.9, "ci_lo": -0.95,
                       "ci_hi": -0.85, "significant": True}]).to_csv(
            sig_dir / f"{t}_{s}_gine.csv", index=False)

    # panel summary under BASE: fallback row's real values; the present
    # cell also carries decoy panel values that must NOT be drawn for it
    panel_rows = []
    for t, s in cells:
        if (t, s) == fallback:
            panel_rows.append({"tag": t, "split": s, "d_r2_mean": 0.35,
                               "ci_lo": 0.30, "ci_hi": 0.40,
                               "significant": True})
        elif (t, s) == present:
            panel_rows.append({"tag": t, "split": s, "d_r2_mean": 0.55,
                               "ci_lo": 0.50, "ci_hi": 0.60,
                               "significant": True})
        else:
            panel_rows.append({"tag": t, "split": s, "d_r2_mean": -0.7,
                               "ci_lo": -0.75, "ci_hi": -0.65,
                               "significant": True})
    panel_path = tmp_path / "results" / "gine_panel" / "summary.csv"
    panel_path.parent.mkdir(parents=True)
    pd.DataFrame(panel_rows).to_csv(panel_path, index=False)

    hlines, plots = _record_forest(monkeypatch)
    mod.make_combined_forest(str(sig_dir), str(tmp_path / "fig.png"))

    # gine offset is 0.0 in the afp forest; y of cells[i] is n-1-i
    n = len(cells)
    y_fb = (n - 1) - cells.index(fallback)
    fb_whiskers = [h for h in hlines if abs(h[0] - y_fb) < 1e-9]
    assert len(fb_whiskers) == 1
    _, lo, hi = fb_whiskers[0]
    assert (lo, hi) == (0.30, 0.40)        # panel summary, not the cell CSV
    fb_markers = [p for p in plots if abs(p[1] - y_fb) < 1e-9]
    assert len(fb_markers) == 1
    assert fb_markers[0][2]["markerfacecolor"] == "white"
    assert fb_markers[0][0] == pytest.approx(0.35)   # same file as whisker

    # a cell summary_gine carries keeps summary_gine for whisker AND centre,
    # despite the decoy panel row
    y_pr = (n - 1) - cells.index(present)
    pr_whiskers = [h for h in hlines if abs(h[0] - y_pr) < 1e-9]
    assert len(pr_whiskers) == 1
    _, lo, hi = pr_whiskers[0]
    assert (lo, hi) == (0.01, 0.09)
    pr_markers = [p for p in plots if abs(p[1] - y_pr) < 1e-9]
    assert len(pr_markers) == 1
    assert pr_markers[0][0] == pytest.approx(0.05)
