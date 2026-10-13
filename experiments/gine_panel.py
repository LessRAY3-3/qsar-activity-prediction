"""P8 phase 1: GINE-vs-RF delta panel across 8 targets x 2 splits.

Question: with the frozen gnn_03 recipe, does GGINEConv (GINE + bond
features) beat fingerprint+RF anywhere, and is any single cell's delta
separable from resampling noise? Analysis only -- no model is trained,
no existing file is modified (matplotlib Agg, single-process numpy).

Data sources (all 8 tags, one frozen-recipe vintage):
  * per-seed test R2 in results/gnn_metrics_{tag}_gine{,_seed1,_seed2}.json
    (128 hidden / 4 layers / 0.2 dropout, seeds 42/1/2) and per-seed test
    predictions in results/gnn_preds_{tag}_gine_{split}[_seedN].npz for
    every tag, including abl1/herg/vegfr2 (committed in 23e202a, aligned
    with the refreshed RF test sets).
  * every cell gets a paired-bootstrap 95% CI (mean3 convention,
    B=10000) from results/significance/{tag}_{split}_gine.csv -- run
    `paired_bootstrap.py --model gine` first; a missing cell file is an
    error, never a silent point estimate.

Conventions (identical to gnn_fairness.py / gnn_04_compare.py):
  gine_r2_mean/std = 3-seed mean +- population std (ddof=0) of test R2
  d_r2_mean        = gine_r2_mean - rf_r2        (> 0 favours GINE)
  ci_lo/ci_hi      = mean3 paired-bootstrap CI of R2(mean of the 3 GINE
                     seed predictions) - R2(RF). The bootstrap point is
                     an *ensemble* delta, so it can differ a little from
                     d_r2_mean (per-seed-mean convention); both are
                     reported per their own convention, never mixed.

Outputs:
  results/gine_panel/summary.csv          16 rows (tag, split) x the
                                          documented columns
  figures/gine_panel/panel_dR2.png        8 targets x 2 splits bar chart
                                          of d_r2_mean with a CI whisker
                                          on every cell, y = 0 line
  figures/significance/ci_panel_with_gine.png
                                          forest plot, gin vs gine mean3
                                          CI side by side (does NOT
                                          overwrite ci_panel.png)
Console: the full 16-cell delta table, how many CIs exclude 0, and a
one-line verdict.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
SPLITS = ("random", "scaffold")
SEED_SUFFIXES = ("", "_seed1", "_seed2")
NOTE_BOOTSTRAP = "paired bootstrap mean3, B=10000"
DELTA_CONVENTION = "point=per-seed-mean_R2;ci=ensemble_mean-pred_R2"

PANEL_COLS = ["tag", "split", "rf_r2", "gine_r2_mean", "gine_r2_std",
              "gine_n_seeds", "d_r2_mean", "ci_lo", "ci_hi",
              "significant", "note", "delta_convention"]


# ---------------------------------------------------------------- loading

def aggregate_seed_r2(r2_values):
    """(mean, population std, n) over seeds -- gnn_fairness convention."""
    v = np.asarray(list(r2_values), dtype=np.float64)
    if v.size == 0:
        raise ValueError("no per-seed R2 values")
    return float(v.mean()), float(v.std(ddof=0)), int(v.size)


def load_rf_r2(tag, split):
    path = os.path.join(BASE, "results", f"metrics_{tag}.json")
    with open(path) as fh:
        return float(json.load(fh)[split]["r2"])


def load_gine_seed_r2(tag, split):
    """Per-seed GINE test R2 for one (tag, split) cell -- all 8 tags."""
    vals = []
    for suf in SEED_SUFFIXES:
        path = os.path.join(BASE, "results",
                            f"gnn_metrics_{tag}_gine{suf}.json")
        with open(path) as fh:
            vals.append(float(json.load(fh)[split]["r2"]))
    return vals


def load_gine_ci(tag, split, sig_dir):
    """(ci_lo, ci_hi, significant) from the mean3 row of the gine bootstrap.

    Every cell must be bootstrapped; a missing cell file is an error, never
    a silent point estimate (run paired_bootstrap.py --model gine first).
    """
    path = os.path.join(sig_dir, f"{tag}_{split}_gine.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {path} -- run paired_bootstrap.py --model gine "
            f"--n-boot 10000 first (every cell needs a bootstrap CI)")
    df = pd.read_csv(path)
    rows = df[df["model_pair"] == "mean3"]
    if rows.empty:
        raise ValueError(f"{path}: no mean3 row")
    row = rows.iloc[0]
    return float(row["ci_lo"]), float(row["ci_hi"]), bool(row["significant"])


# ---------------------------------------------------------------- panel

def make_row(tag, split, rf_r2, gine_r2s, ci):
    """One (tag, split) panel row; ci is (ci_lo, ci_hi, significant)."""
    mean, std, n = aggregate_seed_r2(gine_r2s)
    if ci is None:
        raise ValueError(
            f"{tag}/{split}: missing bootstrap CI -- every cell needs one")
    lo, hi, sig = ci
    return {
        "tag": tag,
        "split": split,
        "rf_r2": float(rf_r2),
        "gine_r2_mean": mean,
        "gine_r2_std": std,
        "gine_n_seeds": n,
        "d_r2_mean": mean - float(rf_r2),
        "ci_lo": lo,
        "ci_hi": hi,
        "significant": sig,
        "note": NOTE_BOOTSTRAP,
        "delta_convention": DELTA_CONVENTION,
    }


def build_panel(sig_dir):
    rows = []
    for tag in TAGS:
        for split in SPLITS:
            rows.append(make_row(tag, split, load_rf_r2(tag, split),
                                 load_gine_seed_r2(tag, split),
                                 load_gine_ci(tag, split, sig_dir)))
    return pd.DataFrame(rows, columns=PANEL_COLS)


# ---------------------------------------------------------------- figures

def _ci_whiskers(ax, x, point, lo, hi, color):
    """Vertical CI whisker that never requires point inside [lo, hi]."""
    cap = 0.12
    ax.vlines(x, lo, hi, color=color, linewidth=1.6, zorder=3)
    ax.hlines(hi, x - cap, x + cap, color=color, linewidth=1.6, zorder=3)
    ax.hlines(lo, x - cap, x + cap, color=color, linewidth=1.6, zorder=3)
    if not (lo <= point <= hi):       # mark the convention mismatch
        ax.plot([x], [point], marker="x", color=color, markersize=6,
                zorder=4)


def _is_true(v):
    """True only for an actual bool True (non-bool / None -> False)."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def make_panel_figure(df, fig_path):
    """Bar chart: d_r2_mean per cell with the mean3 bootstrap CI whisker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    x = np.arange(len(TAGS))

    for ax, split in zip(axes, SPLITS):
        sub = df[df["split"] == split].set_index("tag").loc[list(TAGS)]
        colors = []
        for r in sub.itertuples():
            # colour follows the CI, never the point estimate: a slightly
            # negative per-seed mean with an all-positive CI is still
            # "GINE ahead", and vice versa
            if not _is_true(r.significant):
                colors.append("0.75")
            elif r.ci_lo > 0:
                colors.append("tab:blue")
            else:
                colors.append("tab:red")
        ax.bar(x, sub["d_r2_mean"], color=colors, width=0.68,
               edgecolor="white", zorder=2)
        for i, r in enumerate(sub.itertuples()):
            _ci_whiskers(ax, i, r.d_r2_mean, r.ci_lo, r.ci_hi,
                         color="black")
        ax.axhline(0, color="black", linewidth=1.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(list(TAGS), rotation=40, ha="right", fontsize=9)
        ax.set_title(f"split = {split}")
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_ylabel("ΔR² (GINE − RF), 3-seed mean")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:red"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.75"),
        plt.Line2D([], [], color="black", linewidth=1.6,
                   label="95% CI (paired bootstrap, mean3)"),
    ]
    labels = ["GINE significantly ahead (CI > 0)",
              "RF significantly ahead (CI < 0)",
              "CI includes 0",
              "95% CI (paired bootstrap, mean3)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle("GINE vs RF per cell: ΔR² = mean(3-seed R²) − RF R², "
                 "whiskers = mean3 paired-bootstrap 95% CI", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_combined_forest(panel_df, sig_dir, fig_path):
    """gin vs gine mean3 CIs side by side, one row per (tag, split).

    Reads the phase-0 gin summary (results/significance/summary.csv) plus
    the phase-1 gine summary (summary_gine.csv). Whisker AND centre of a
    row always come from the SAME campaign summary file: summary_gine.csv
    when it carries the cell, otherwise both fall back to the panel
    summary (results/gine_panel/summary.csv) -- the old code took the
    whisker from the per-cell bootstrap CSVs and the centre from
    summary_gine. Writes a NEW file -- ci_panel.png is never touched.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    gin_path = os.path.join(sig_dir, "summary.csv")
    gin = pd.read_csv(gin_path)
    gin = gin[gin["pair"] == "mean3"].set_index(["tag", "split"])
    gine_path = os.path.join(sig_dir, "summary_gine.csv")
    gine = (pd.read_csv(gine_path).query("pair == 'mean3'")
            .set_index(["tag", "split"])
            if os.path.exists(gine_path) else None)

    cells = [(t, s) for t in TAGS for s in SPLITS]
    y_pos = np.arange(len(cells))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    off = 0.19
    colors = {"gin": "tab:orange", "gine": "tab:green"}

    for y, cell in zip(y_pos, cells):
        g = gin.loc[cell]
        ax.hlines(y + off, g["ci_lo"], g["ci_hi"], color=colors["gin"],
                  linewidth=1.6)
        ax.plot([g["d_r2"]], [y + off], "o", color=colors["gin"],
                markersize=5)

        # one source per row: campaign gine summary carries both the
        # whisker and the centre; the panel summary is the same-file
        # fallback when the cell is absent from summary_gine
        if gine is not None and cell in gine.index:
            r = gine.loc[cell]
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            point = float(r["d_r2"])
        else:
            row = panel_df[(panel_df["tag"] == cell[0])
                           & (panel_df["split"] == cell[1])].iloc[0]
            lo, hi = row["ci_lo"], row["ci_hi"]
            point = row["d_r2_mean"]
        ax.hlines(y - off, lo, hi, color=colors["gine"], linewidth=1.6)
        ax.plot([point], [y - off], "o", color=colors["gine"],
                markersize=5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t}/{s}" for t, s in cells], fontsize=8)
    ax.set_xlabel("ΔR², 95% CI (mean3 convention)")
    ax.grid(axis="x", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=colors["gin"],
                   label="GIN mean3 (phase 0)"),
        plt.Line2D([], [], marker="o", ls="", color=colors["gine"],
                   label="GINE mean3 (phase 1)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Paired bootstrap ΔR² (model − RF): GIN vs GINE, "
                 "mean3 convention", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def print_table(df):
    """The 8x2 delta table with CIs, plus the significance tally."""
    print("\n=== GINE vs RF panel (8 targets x 2 splits) ===")
    print(f"  {'tag':<10} {'split':<9} {'rf_r2':>7} {'gine_r2':>17} "
          f"{'d_r2':>8}  {'95% CI (mean3)':<26} sig")
    n_sig = 0
    for r in df.itertuples():
        gine = f"{r.gine_r2_mean:.4f}±{r.gine_r2_std:.4f}"
        ci = f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        sig = "yes" if _is_true(r.significant) else "no"
        n_sig += int(_is_true(r.significant))
        print(f"  {r.tag:<10} {r.split:<9} {r.rf_r2:>7.4f} {gine:>17} "
              f"{r.d_r2_mean:>+8.4f}  {ci:<26} {sig}")
    n_ci = len(df)
    rf_wins = [f"{r.tag}/{r.split}" for r in df.itertuples()
               if _is_true(r.significant) and r.ci_hi < 0]
    gine_wins = [f"{r.tag}/{r.split}" for r in df.itertuples()
                 if _is_true(r.significant) and r.ci_lo > 0]
    print(f"\n  {n_ci}/16 cells have a bootstrap CI; {n_sig}/{n_ci} exclude 0")
    if gine_wins:
        print(f"  GINE significantly ahead ({len(gine_wins)}): "
              f"{', '.join(gine_wins)}")
    else:
        print("  GINE significantly ahead: none")
    if rf_wins:
        print(f"  RF significantly ahead ({len(rf_wins)}): "
              f"{', '.join(rf_wins)}")
    if gine_wins:
        verdict = (f"GINE beats RF significantly in {len(gine_wins)}/{n_ci} "
                   f"cells, RF in {len(rf_wins)}/{n_ci} -- an isolated win "
                   "does not flip the panel verdict.")
    else:
        verdict = (f"GINE never beats RF significantly: 0/{n_ci} bootstrapped "
                   "cells have a CI above 0 -- bond features do not flip the "
                   "verdict.")
    print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sig-dir", default=os.path.join(BASE, "results",
                                                     "significance"),
                   help="directory holding {tag}_{split}_gine.csv")
    p.add_argument("--out-csv", default=os.path.join(BASE, "results",
                                                     "gine_panel",
                                                     "summary.csv"))
    p.add_argument("--fig", default=os.path.join(BASE, "figures",
                                                 "gine_panel",
                                                 "panel_dR2.png"))
    p.add_argument("--combined-fig",
                   default=os.path.join(BASE, "figures", "significance",
                                        "ci_panel_with_gine.png"))
    return p.parse_args(argv)


def main(args=None):
    args = parse_args(args)
    panel = build_panel(os.path.abspath(args.sig_dir))

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    panel.to_csv(out_csv, index=False)

    make_panel_figure(panel, os.path.abspath(args.fig))
    make_combined_forest(panel, os.path.abspath(args.sig_dir),
                         os.path.abspath(args.combined_fig))

    print_table(panel)
    print(f"\nsummary -> {os.path.relpath(out_csv, BASE)} ({len(panel)} rows)")
    print(f"figure  -> {os.path.relpath(os.path.abspath(args.fig), BASE)}")
    print(f"combined -> {os.path.relpath(os.path.abspath(args.combined_fig), BASE)}")


if __name__ == "__main__":
    main()
