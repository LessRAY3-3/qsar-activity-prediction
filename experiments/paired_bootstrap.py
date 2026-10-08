"""P8 phase 0: paired bootstrap significance for the RF-vs-GIN deltas.

Question: the campaign's effect size (3-seed mean GIN - RF) is reported per
(tag, split) cell, but is any single cell's delta separable from resampling
noise? This script resamples test MOLECULES with replacement (paired: the
same resample index matrix is applied to both models' predictions) and
turns each cell's delta into a 95% CI plus a one-sided empirical p-value.
Analysis only -- no model is trained, no existing file is modified.

`--model gine` (P8 phase 1) reruns the identical procedure against the
GINE predictions; outputs then carry a `_gine` suffix ({tag}_{split}_gine.csv,
summary_gine.csv, ci_panel_gine.png) so the phase-0 gin artifacts are never
overwritten. Default behaviour (--model gin) is unchanged.

`--model attentivefp` (P8 phase 3) follows the same contract against the
AttentiveFP campaign's predictions: {tag}_{split}_attentivefp.csv,
summary_attentivefp.csv, ci_panel_attentivefp.png -- gin and gine
artifacts stay untouched byte-for-byte.

`--splits time` (P8 time-split analysis) gets the same protection for the
shared outputs: cell files already carry the split in their name
({tag}_time.csv), and any run touching a split outside the default
random/scaffold set auto-suffixes the shared summary/figure with that
split name -> summary_time.csv, ci_panel_time.png (see split_suffix).
The campaign's summary.csv and ci_panel.png are therefore never
overwritten by a time run.

Inputs (per tag x split; test_idx alignment is asserted, never assumed):
  results/rf_preds_{tag}_{split}.npz          RF test preds (original pIC50)
  results/gnn_preds_{tag}_{split}.npz         GIN seed 42   (--model gine /
  results/gnn_preds_{tag}_{split}_seed1.npz   GIN seed 1     attentivefp:
  results/gnn_preds_{tag}_{split}_seed2.npz   GIN seed 2     infix _gine /
                                                             _attentivefp)

Two conventions ("model_pair" rows, both written for every cell):
  seed42  single seed-42 GIN vs RF            (reference; single-seed
                                               deltas are seed noise)
  mean3   per-molecule mean of the 3 GIN seed
          predictions vs RF                    (primary; matches the
                                               campaign's mean convention)

Effect-size signs (documented so the CSV cannot be misread):
  delta_r2   = R2_GIN - R2_RF    > 0 favours GIN
  delta_rmse = RMSE_GIN - RMSE_RF  > 0 favours RF (GIN worse)

Both pairs of one cell draw from an identically seeded Generator
(seed=42, fresh per call), so seed42 and mean3 are evaluated on the exact
same resample matrices and the run is order-independent.

Outputs (suffix `_gine` / `_attentivefp` on every path when --model gine /
attentivefp; gin is unchanged; the shared summary/figure additionally carry
the split suffix when --splits touches a split outside random/scaffold,
e.g. summary_time.csv / ci_panel_time.png -- see split_suffix):
  results/significance/{tag}_{split}.csv   one row per model_pair with the
                                           point deltas, the 95% CI bounds,
                                           p_one_sided = #(dR2_boot > 0)/B
  results/significance/summary{suffix}.csv cells x 2 pairs;
                                           significant = CI excludes 0
  figures/significance/ci_panel{suffix}.png forest plot, one row per
                                           (tag, split), faceted by pair
Console: per-pair tally of cells whose CI excludes 0 and the list of
cells where RF (or GIN/GINE) wins significantly.
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
SPLITS = ("random", "scaffold")
MODELS = ("gin", "gine", "attentivefp")
SEED_SUFFIXES = ("", "_seed1", "_seed2")
PAIRS = ("seed42", "mean3")
N_BOOT = 10_000
BOOT_SEED = 42
CI_PCT = (2.5, 97.5)
CHUNK = 500          # resamples per vectorised block (bounds peak memory)

SUMMARY_COLS = ["tag", "split", "pair", "d_r2", "ci_lo", "ci_hi",
                "significant", "winner", "p_one_sided",
                "d_rmse", "d_rmse_ci_lo", "d_rmse_ci_hi"]


# ---------------------------------------------------------------- naming

def gnn_preds_name(tag, split, seed_suffix, model="gin"):
    """Preds npz filename; gin keeps the campaign's infix-free names."""
    infix = "" if model == "gin" else f"_{model}"
    return f"gnn_preds_{tag}{infix}_{split}{seed_suffix}.npz"


def out_stem(tag, split, model="gin"):
    """Output stem per cell: gine artifacts carry a distinguishing suffix."""
    return f"{tag}_{split}" if model == "gin" else f"{tag}_{split}_{model}"


def summary_name(model="gin", suffix=""):
    """Shared summary filename; suffix separates split groups (e.g. _time)."""
    base = f"summary{suffix}"
    return f"{base}.csv" if model == "gin" else f"{base}_{model}.csv"


def default_fig_name(model="gin", suffix=""):
    """Shared forest-plot filename; suffix separates split groups."""
    base = f"ci_panel{suffix}"
    return f"{base}.png" if model == "gin" else f"{base}_{model}.png"


def split_suffix(splits):
    """Suffix protecting the shared summary/figure from a foreign split.

    Cell outputs already carry the split ({tag}_{split}.csv), but
    summary.csv / ci_panel.png are shared across splits: a run touching
    any split outside the campaign's default random/scaffold set must not
    clobber them. Returns "" for default-only runs (legacy names kept
    byte-for-byte) and e.g. "_time" the moment time is requested.
    """
    extra = [s for s in splits if s not in SPLITS]
    if not extra:
        return ""
    return "_" + "_".join(extra)



# ---------------------------------------------------------------- metrics

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / sst


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------- loading

def load_cell(tag, split, model="gin"):
    """Aligned (y_true, rf_pred, gnn42_pred, gnn_mean3_pred) for one cell.

    Alignment is checked against the split npz test_idx (and the GNN files
    against the RF file) -- predictions are never zipped by position alone.
    """
    def _npz(name):
        path = os.path.join(BASE, "results", name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {os.path.relpath(path, BASE)}")
        return np.load(path)

    rf = _npz(f"rf_preds_{tag}_{split}.npz")
    sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                              f"{tag}_{split}.npz"))
    if not np.array_equal(rf["test_idx"], sp["test_idx"]):
        raise ValueError(f"{tag}/{split}: rf test_idx != split test_idx")

    gins = []
    for suf in SEED_SUFFIXES:
        g = _npz(gnn_preds_name(tag, split, suf, model))
        if not np.array_equal(g["test_idx"], rf["test_idx"]):
            raise ValueError(f"{tag}/{split}: gnn{suffix_label(suf)} test_idx "
                             f"!= rf test_idx")
        if not np.allclose(g["y_true"], rf["y_true"], atol=1e-4):
            raise ValueError(f"{tag}/{split}: gnn{suffix_label(suf)} y_true "
                             f"!= rf y_true")
        gins.append(np.asarray(g["y_pred"], dtype=np.float64))

    return {
        "y": np.asarray(rf["y_true"], dtype=np.float64),
        "rf": np.asarray(rf["y_pred"], dtype=np.float64),
        "gin42": gins[0],
        "gin_mean3": np.mean(np.stack(gins), axis=0),
        "n_test": int(len(rf["test_idx"])),
    }


def suffix_label(suf):
    return suf or " (seed42)"


# ---------------------------------------------------------------- bootstrap

def paired_bootstrap(y_true, pred_gin, pred_rf, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap over test molecules for dR2 = R2_GIN - R2_RF.

    Vectorised: resamples are drawn in CHUNK blocks of shape (chunk, n) and
    the sums for both models are computed per block, so peak memory stays
    at a few tens of MB even for the ~2.7k-molecule cells. A fresh
    Generator(seed) per call makes the result independent of call order.
    Degenerate resamples (zero variance in y) are dropped from the
    percentiles rather than producing inf.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p_gin = np.asarray(pred_gin, dtype=np.float64)
    p_rf = np.asarray(pred_rf, dtype=np.float64)
    if not (len(y) == len(p_gin) == len(p_rf)):
        raise ValueError("y_true / pred_gin / pred_rf length mismatch")
    n = len(y)
    rng = np.random.default_rng(seed)

    d_r2 = np.empty(n_boot, dtype=np.float64)
    d_rmse = np.empty(n_boot, dtype=np.float64)
    pos = 0
    while pos < n_boot:
        b = min(CHUNK, n_boot - pos)
        idx = rng.integers(0, n, size=(b, n))
        ys = y[idx]
        sst = ((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        r2s, rmses = [], []
        for p in (p_gin, p_rf):
            sse = ((ys - p[idx]) ** 2).sum(axis=1)
            r2s.append(1.0 - sse / sst)
            rmses.append(np.sqrt(sse / n))
        d_r2[pos:pos + b] = r2s[0] - r2s[1]
        d_rmse[pos:pos + b] = rmses[0] - rmses[1]
        pos += b

    ok = np.isfinite(d_r2) & np.isfinite(d_rmse)
    if not ok.all():
        d_r2, d_rmse = d_r2[ok], d_rmse[ok]
        if len(d_r2) == 0:
            raise ValueError("every bootstrap resample was degenerate")
    lo, hi = np.percentile(d_r2, CI_PCT)
    rlo, rhi = np.percentile(d_rmse, CI_PCT)
    return {
        "delta_r2_point": r2(y, p_gin) - r2(y, p_rf),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_one_sided": float(np.mean(d_r2 > 0)),
        "delta_rmse_point": rmse(y, p_gin) - rmse(y, p_rf),
        "delta_rmse_ci_lo": float(rlo),
        "delta_rmse_ci_hi": float(rhi),
        "n_boot": int(len(d_r2)),
    }


def cell_rows(tag, split, n_boot=N_BOOT, seed=BOOT_SEED, model="gin"):
    """The two model_pair rows for one (tag, split) cell."""
    cell = load_cell(tag, split, model)
    rows = []
    for pair in PAIRS:
        pred_gin = cell["gin42"] if pair == "seed42" else cell["gin_mean3"]
        res = paired_bootstrap(cell["y"], pred_gin, cell["rf"],
                               n_boot=n_boot, seed=seed)
        lo, hi = res["ci_lo"], res["ci_hi"]
        rows.append({
            "tag": tag,
            "split": split,
            "model_pair": pair,
            "n_test": cell["n_test"],
            "r2_gin": r2(cell["y"], pred_gin),
            "r2_rf": r2(cell["y"], cell["rf"]),
            "delta_r2_point": res["delta_r2_point"],
            "ci_lo": lo,
            "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0),
            "winner": model if lo > 0 else ("rf" if hi < 0 else "tie"),
            "p_one_sided": res["p_one_sided"],
            "delta_rmse_point": res["delta_rmse_point"],
            "delta_rmse_ci_lo": res["delta_rmse_ci_lo"],
            "delta_rmse_ci_hi": res["delta_rmse_ci_hi"],
            "n_boot": res["n_boot"],
            "boot_seed": seed,
        })
    return rows


# ---------------------------------------------------------------- figure

def figure_cells(rows):
    """(tag, split) cells to plot, in row order.

    Derived from the rows rather than the default TAGS x SPLITS grid so a
    --splits time run plots its own cells instead of an all-missing grid.
    """
    return [(r["tag"], r["split"]) for r in rows
            if r["model_pair"] == PAIRS[0]]


def make_figure(rows, fig_path, n_boot=N_BOOT, model="gin"):
    """Forest plot: one row per (tag, split), one panel per model_pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    cells = figure_cells(rows)
    y_pos = np.arange(len(cells))[::-1]   # first cell at the top

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.0), sharey=True)
    colors = {"rf": "tab:red", model: "tab:blue", "tie": "0.55"}
    label = model.upper()
    for ax, pair in zip(axes, PAIRS):
        lookup = {(r["tag"], r["split"]): r for r in rows
                  if r["model_pair"] == pair}
        for y, cell in zip(y_pos, cells):
            r = lookup.get(cell)
            if r is None:
                continue
            c = colors[r["winner"]]
            ax.errorbar(r["delta_r2_point"], y,
                        xerr=[[r["delta_r2_point"] - r["ci_lo"]],
                              [r["ci_hi"] - r["delta_r2_point"]]],
                        fmt="o", color=c, capsize=3, markersize=5.5,
                        linewidth=1.4)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"ΔR² ({label} − RF), 95% CI")
        ax.set_title(f"pair = {pair}" + ("  (primary)" if pair == "mean3" else ""))
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{t}/{s}" for t, s in cells], fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=lab)
               for c, lab in ((colors["rf"], "RF wins (CI < 0)"),
                              (colors[model], f"{label} wins (CI > 0)"),
                              (colors["tie"], "CI includes 0"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle(f"Paired bootstrap of ΔR² ({label} − RF), B = "
                 f"{n_boot}, test molecules resampled", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def print_conclusion(summary, model="gin"):
    """Console verdict: which cells' CIs exclude 0, per convention."""
    label = model.upper()
    print(f"\n=== Paired bootstrap significance: {label} vs RF ===")
    print(f"  dR2 = R2_{label} - R2_RF (positive favours {label}); "
          "significant = 95% CI excludes 0")
    for pair in PAIRS:
        sub = summary[summary["pair"] == pair]
        sig = sub[sub["significant"]]
        rf_wins = [f"{r.tag}/{r.split}" for r in sig.itertuples()
                   if r.winner == "rf"]
        gin_wins = [f"{r.tag}/{r.split}" for r in sig.itertuples()
                    if r.winner == model]
        print(f"\n  [{pair}] {len(sig)}/{len(sub)} cells with CI excluding 0")
        if rf_wins:
            print(f"    RF significantly ahead ({len(rf_wins)}): "
                  f"{', '.join(rf_wins)}")
        if gin_wins:
            print(f"    {label} significantly ahead ({len(gin_wins)}): "
                  f"{', '.join(gin_wins)}")
        if not rf_wins and not gin_wins:
            print("    no cell separates from zero")
        d = sub["d_r2"]
        print(f"    dR2 point range: [{d.min():+.4f}, {d.max():+.4f}]")
    m3 = summary[summary["pair"] == "mean3"]
    n = len(m3)
    n_rf = int(((m3["ci_hi"] < 0)).sum())
    print(f"\n  mean3 (primary): RF significantly wins in {n_rf}/{n} cells; "
          f"{label} in {int((m3['ci_lo'] > 0).sum())}/{n}; "
          f"{int((~m3['significant']).sum())}/{n} inconclusive")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="gin", choices=list(MODELS),
                   help="GNN variant to compare against RF (default gin; "
                        "gine/attentivefp write *_gine / *_attentivefp "
                        "outputs)")
    p.add_argument("--tags", default=",".join(TAGS),
                   help="comma-separated dataset tags")
    p.add_argument("--splits", default=",".join(SPLITS),
                   help="comma-separated splits (random, scaffold, time); "
                        "non-default splits suffix the shared summary/fig "
                        "(e.g. summary_time.csv)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=BOOT_SEED,
                   help="bootstrap Generator seed (fixed for reproducibility)")
    p.add_argument("--out-dir", default=os.path.join(BASE, "results",
                                                     "significance"))
    p.add_argument("--fig", default=None,
                   help="forest-plot path (default figures/significance/"
                        "{ci_panel|ci_panel_gine}.png per --model)")
    return p.parse_args(argv)


def main(args=None):
    args = parse_args(args)
    model = args.model
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for tag in tags:
        for split in splits:
            cell = cell_rows(tag, split, n_boot=args.n_boot, seed=args.seed,
                             model=model)
            path = os.path.join(out_dir, f"{out_stem(tag, split, model)}.csv")
            pd.DataFrame(cell).to_csv(path, index=False)
            for r in cell:
                rows.append(r)
            print(f"[{tag}/{split}] n={cell[0]['n_test']}  "
                  f"mean3 dR2={cell[1]['delta_r2_point']:+.4f} "
                  f"CI[{cell[1]['ci_lo']:+.4f}, {cell[1]['ci_hi']:+.4f}]  "
                  f"p={cell[1]['p_one_sided']:.4f}")
    if not rows:
        print("no cells processed")
        return

    summary_rows = [{**r,
                     "pair": r["model_pair"],
                     "d_r2": r["delta_r2_point"],
                     "d_rmse": r["delta_rmse_point"],
                     "d_rmse_ci_lo": r["delta_rmse_ci_lo"],
                     "d_rmse_ci_hi": r["delta_rmse_ci_hi"]}
                    for r in rows]
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLS)
    suffix = split_suffix(splits)
    summary_path = os.path.join(out_dir, summary_name(model, suffix))
    summary.to_csv(summary_path, index=False)

    fig_path = args.fig or os.path.join(BASE, "figures", "significance",
                                        default_fig_name(model, suffix))
    fig_path = os.path.abspath(fig_path)
    make_figure(rows, fig_path, n_boot=args.n_boot, model=model)
    print_conclusion(summary, model=model)
    if suffix:
        print(f"\nsplit suffix '{suffix}': shared outputs kept separate "
              "from the random/scaffold campaign files")
    print(f"\nsummary -> {os.path.relpath(summary_path, BASE)} "
          f"({len(summary)} rows)")
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")


if __name__ == "__main__":
    main()
