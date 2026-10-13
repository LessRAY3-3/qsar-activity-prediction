"""P8 phase-1 GINE panel: pure row/aggregation helpers.

Loads experiments/gine_panel.py by path (import-safe: matplotlib is
imported inside the figure functions only) and checks the panel-row
contract on synthetic inputs with a known ground truth. The all-8-tags
bootstrap invariant is checked against the committed significance files:
every cell must carry a mean3 CI, no point-only rows.
"""
import os

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
    assert row["note"] == mod.NOTE_BOOTSTRAP


def test_make_row_rejects_missing_ci(load_experiment):
    """A cell without a bootstrap CI is an error, never a point-only row."""
    mod = load_experiment("gine_panel")
    with pytest.raises(ValueError, match="missing bootstrap CI"):
        mod.make_row("vegfr2", "scaffold", 0.60, [0.55, 0.58, 0.61], ci=None)


def test_all_eight_tags_carry_bootstrap_cis(load_experiment):
    """build_panel over the committed files: all 16 cells have a mean3 CI."""
    mod = load_experiment("gine_panel")
    sig_dir = os.path.join(mod.BASE, "results", "significance")
    panel = mod.build_panel(sig_dir)

    assert len(panel) == len(mod.TAGS) * len(mod.SPLITS)
    assert set(panel["tag"]) == set(mod.TAGS)
    assert (panel["gine_n_seeds"] == 3).all()
    assert panel["ci_lo"].notna().all() and panel["ci_hi"].notna().all()
    assert (panel["ci_lo"] < panel["ci_hi"]).all()
    assert (panel["note"] == mod.NOTE_BOOTSTRAP).all()
    assert panel["significant"].map(
        lambda v: isinstance(v, (bool, np.bool_))).all()


def test_summary_csv_carries_delta_convention(tmp_path, monkeypatch, load_experiment):
    """build_panel -> summary.csv keeps every column and appends delta_convention."""
    mod = load_experiment("gine_panel")
    monkeypatch.setattr(mod, "load_rf_r2", lambda tag, split: 0.70)
    monkeypatch.setattr(mod, "load_gine_seed_r2",
                        lambda tag, split: [0.60, 0.62, 0.64])
    monkeypatch.setattr(mod, "load_gine_ci",
                        lambda tag, split, sig_dir: (0.01, 0.03, True))

    panel = mod.build_panel("unused")
    out = tmp_path / "summary.csv"
    panel.to_csv(out, index=False)

    got = pd.read_csv(out)
    assert list(got.columns) == mod.PANEL_COLS
    assert got.columns[-1] == "delta_convention"
    assert len(got) == len(mod.TAGS) * len(mod.SPLITS)
    assert got["delta_convention"].tolist() == [mod.DELTA_CONVENTION] * len(got)
    assert mod.DELTA_CONVENTION == "point=per-seed-mean_R2;ci=ensemble_mean-pred_R2"


# ------------------------------------------- combined forest: whisker source

_SUMMARY_COLS = ["tag", "split", "pair", "d_r2", "ci_lo", "ci_hi", "significant"]


def _all_cells(mod):
    return [(t, s) for t in mod.TAGS for s in mod.SPLITS]


def _write_sig_summaries(sig_dir, gin_rows, gine_rows):
    sig_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gin_rows, columns=_SUMMARY_COLS).to_csv(
        sig_dir / "summary.csv", index=False)
    pd.DataFrame(gine_rows, columns=_SUMMARY_COLS).to_csv(
        sig_dir / "summary_gine.csv", index=False)


def _per_cell_csvs(mod, sig_dir, value=-0.9, lo=-0.95, hi=-0.85):
    """Per-cell bootstrap CSVs with values that must NEVER be drawn."""
    for tag, split in _all_cells(mod):
        pd.DataFrame([{"model_pair": "mean3", "d_r2": value,
                       "ci_lo": lo, "ci_hi": hi, "significant": True}]).to_csv(
            sig_dir / f"{tag}_{split}_gine.csv", index=False)


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
                              float(np.ravel(args[1])[0])))
            except (TypeError, ValueError):
                pass
        return orig_p(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hlines", rec_h)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", rec_p)
    return hlines, plots


def _gine_row_y(cell_index, n_cells, off):
    """y coordinate of the gine whisker/marker for cells[cell_index]."""
    return (n_cells - 1 - cell_index) - off


def test_combined_forest_whisker_and_centre_share_summary_gine(
        tmp_path, monkeypatch, load_experiment
):
    """A summary_gine cell draws BOTH whisker and centre from summary_gine.

    Mutant killed: the pre-6c1e3bb forest drew the whisker from the
    per-cell bootstrap CSVs ({tag}_{split}_gine.csv) and the centre from
    summary_gine -- the per-cell CSVs below carry deliberately different
    values, so a mixed-source draw cannot pass.
    """
    mod = load_experiment("gine_panel")
    sig_dir = tmp_path / "significance"
    target = ("a2a", "random")
    cells = _all_cells(mod)
    gin_rows, gine_rows = [], []
    for tag, split in cells:
        base = dict(tag=tag, split=split, pair="mean3", significant=True)
        gin_rows.append({**base, "d_r2": 0.01, "ci_lo": -0.01, "ci_hi": 0.03})
        if (tag, split) == target:
            gine_rows.append({**base, "d_r2": 0.25, "ci_lo": 0.20, "ci_hi": 0.30})
        else:
            gine_rows.append({**base, "d_r2": 0.05, "ci_lo": 0.01, "ci_hi": 0.09})
    _write_sig_summaries(sig_dir, gin_rows, gine_rows)
    _per_cell_csvs(mod, sig_dir)

    panel_df = pd.DataFrame([
        {"tag": t, "split": s, "d_r2_mean": -0.7, "ci_lo": -0.75,
         "ci_hi": -0.65}
        for t, s in cells
    ])
    hlines, plots = _record_forest(monkeypatch)
    mod.make_combined_forest(panel_df, str(sig_dir), str(tmp_path / "fig.png"))

    off = 0.19
    y = _gine_row_y(cells.index(target), len(cells), off)
    whiskers = [h for h in hlines if abs(h[0] - y) < 1e-9]
    assert len(whiskers) == 1
    _, lo, hi = whiskers[0]
    assert (lo, hi) == (0.20, 0.30)          # summary_gine, not the cell CSV
    markers = [p for p in plots if abs(p[1] - y) < 1e-9]
    assert len(markers) == 1
    assert markers[0][0] == pytest.approx(0.25)   # same file as the whisker


def test_combined_forest_fallback_row_uses_panel_summary_for_both(
        tmp_path, monkeypatch, load_experiment
):
    """A cell absent from summary_gine falls back to the PANEL summary for
    whisker AND centre together -- never a panel centre on a per-cell-CSV
    whisker (the "same file" contract from 6c1e3bb)."""
    mod = load_experiment("gine_panel")
    sig_dir = tmp_path / "significance"
    target = ("abl1", "scaffold")
    cells = _all_cells(mod)
    gin_rows, gine_rows = [], []
    for tag, split in cells:
        base = dict(tag=tag, split=split, pair="mean3", significant=True)
        gin_rows.append({**base, "d_r2": 0.01, "ci_lo": -0.01, "ci_hi": 0.03})
        if (tag, split) != target:                # target missing from gine
            gine_rows.append({**base, "d_r2": 0.05, "ci_lo": 0.01,
                              "ci_hi": 0.09})
    _write_sig_summaries(sig_dir, gin_rows, gine_rows)
    _per_cell_csvs(mod, sig_dir)

    panel_df = pd.DataFrame([
        {"tag": t, "split": s,
         "d_r2_mean": 0.35 if (t, s) == target else -0.7,
         "ci_lo": 0.30 if (t, s) == target else -0.75,
         "ci_hi": 0.40 if (t, s) == target else -0.65}
        for t, s in cells
    ])
    hlines, plots = _record_forest(monkeypatch)
    mod.make_combined_forest(panel_df, str(sig_dir), str(tmp_path / "fig.png"))

    off = 0.19
    y = _gine_row_y(cells.index(target), len(cells), off)
    whiskers = [h for h in hlines if abs(h[0] - y) < 1e-9]
    assert len(whiskers) == 1
    _, lo, hi = whiskers[0]
    assert (lo, hi) == (0.30, 0.40)          # panel summary, not the cell CSV
    markers = [p for p in plots if abs(p[1] - y) < 1e-9]
    assert len(markers) == 1
    assert markers[0][0] == pytest.approx(0.35)
