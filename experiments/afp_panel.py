"""P8 phase 3: AttentiveFP-vs-RF delta panel across 8 targets x 2 splits.

Question: with the frozen recipe, does the attention-based readout
(AttentiveFP) beat fingerprint+RF anywhere, and is any single cell's delta
separable from resampling noise? Analysis only -- no model is trained, no
existing file is modified (matplotlib Agg, single-process numpy).

Data sources (all 48 campaign runs, commit 20980df):
  * per-seed test R2 in results/gnn_metrics_{tag}_attentivefp{,_seed1,
    _seed2}.json (top-level key = split name)
  * per-seed test predictions in
    results/gnn_preds_{tag}_attentivefp_{split}[_seedN].npz
  * paired-bootstrap 95% CI (mean3 convention, B=10000) in
    results/significance/{tag}_{split}_attentivefp.csv -- run
    `paired_bootstrap.py --model attentivefp` first.
Unlike the GINE panel (fairness tags had no per-seed preds), every one of
the 16 cells is bootstrappable, so no point-estimate-only rows exist.

Conventions (identical to gine_panel.py / gnn_fairness.py):
  afp_r2_mean/std = 3-seed mean +- population std (ddof=0) of test R2
  d_r2_mean       = afp_r2_mean - rf_r2        (> 0 favours AttentiveFP)
  ci_lo/ci_hi     = mean3 paired-bootstrap CI of R2(mean of the 3
                    AttentiveFP seed predictions) - R2(RF). The bootstrap
                    point is an *ensemble* delta, so it can differ a
                    little from d_r2_mean (per-seed-mean convention);
                    both are reported per their own convention.

Outputs (never overwrite gin/gine artifacts):
  results/afp_panel/summary.csv          16 rows (tag, split) x the
                                         documented columns
  figures/afp_panel/panel_dR2.png        8 targets x 2 splits bar chart
                                         of d_r2_mean with CI whiskers,
                                         y = 0 line
  figures/significance/ci_panel_with_afp.png
                                         forest plot, gin / gine / afp
                                         mean3 CI side by side (does NOT
                                         overwrite ci_panel*.png)
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
NOTE_NO_CI = "no bootstrap CSV (run paired_bootstrap.py --model attentivefp first)"
DELTA_CONVENTION = "point=per-seed-mean_R2;ci=ensemble_mean-pred_R2"

PANEL_COLS = ["tag", "split", "rf_r2", "afp_r2_mean", "afp_r2_std",
              "afp_n_seeds", "d_r2_mean", "ci_lo", "ci_hi",
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


def load_afp_seed_r2(tag, split):
    """Per-seed AttentiveFP test R2 for one (tag, split) cell."""
    vals = []
    for suf in SEED_SUFFIXES:
        path = os.path.join(BASE, "results",
                            f"gnn_metrics_{tag}_attentivefp{suf}.json")
        with open(path) as fh:
            vals.append(float(json.load(fh)[split]["r2"]))
    return vals


def load_afp_ci(tag, split, sig_dir):
    """(ci_lo, ci_hi, significant) from the mean3 row of the AFP bootstrap.

    Returns None when the cell has no bootstrap artifact yet.
    """
    path = os.path.join(sig_dir, f"{tag}_{split}_attentivefp.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    rows = df[df["model_pair"] == "mean3"]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["ci_lo"]), float(row["ci_hi"]), bool(row["significant"])


# ---------------------------------------------------------------- panel

def make_row(tag, split, rf_r2, afp_r2s, ci=None):
    """One (tag, split) panel row; ci=None -> bootstrap not run yet."""
    mean, std, n = aggregate_seed_r2(afp_r2s)
    row = {
        "tag": tag,
        "split": split,
        "rf_r2": float(rf_r2),
        "afp_r2_mean": mean,
        "afp_r2_std": std,
        "afp_n_seeds": n,
        "d_r2_mean": mean - float(rf_r2),
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "significant": None,
        "note": NOTE_NO_CI,
        "delta_convention": DELTA_CONVENTION,
    }
    if ci is not None:
        lo, hi, sig = ci
        row["ci_lo"], row["ci_hi"], row["significant"] = lo, hi, sig
        row["note"] = NOTE_BOOTSTRAP
    return row


def build_panel(sig_dir):
    rows = []
    for tag in TAGS:
        for split in SPLITS:
            rows.append(make_row(tag, split, load_rf_r2(tag, split),
                                 load_afp_seed_r2(tag, split),
                                 load_afp_ci(tag, split, sig_dir)))
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
    """True only for an actual bool True (NaN/None from blank CI -> False)."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def make_panel_figure(df, fig_path):
    """Bar chart: d_r2_mean per cell, CI whiskers where a bootstrap exists."""
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
            # "AttentiveFP ahead", and vice versa
            if not np.isfinite(r.ci_lo) or not _is_true(r.significant):
                colors.append("0.75")
            elif r.ci_lo > 0:
                colors.append("tab:blue")
            else:
                colors.append("tab:red")
        bars = ax.bar(x, sub["d_r2_mean"], color=colors, width=0.68,
                      edgecolor="white", zorder=2)
        for i, r in enumerate(sub.itertuples()):
            if np.isfinite(r.ci_lo):
                _ci_whiskers(ax, i, r.d_r2_mean, r.ci_lo, r.ci_hi,
                             color="black")
            else:
                bars[i].set_hatch("///")
        ax.axhline(0, color="black", linewidth=1.0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(list(TAGS), rotation=40, ha="right", fontsize=9)
        ax.set_title(f"split = {split}")
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_ylabel("ΔR² (AttentiveFP − RF), 3-seed mean")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:red"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.75"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.75", hatch="///"),
        plt.Line2D([], [], color="black", linewidth=1.6,
                   label="95% CI (paired bootstrap, mean3)"),
    ]
    labels = ["AttentiveFP significantly ahead (CI > 0)",
              "RF significantly ahead (CI < 0)",
              "CI includes 0",
              "bootstrap not run yet",
              "95% CI (paired bootstrap, mean3)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    fig.suptitle("AttentiveFP vs RF per cell: ΔR² = mean(3-seed R²) − RF R², "
                 "whiskers = mean3 paired-bootstrap 95% CI", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def make_combined_forest(sig_dir, fig_path):
    """gin vs gine vs attentivefp mean3 CIs side by side, one row per cell.

    Reads the phase-0 gin summary (results/significance/summary.csv), the
    phase-1 gine summary (summary_gine.csv; the 6 fairness cells fall back
    to their point estimate from results/gine_panel/summary.csv) and the
    phase-3 attentivefp summary (summary_attentivefp.csv). Writes a NEW
    file -- ci_panel.png / ci_panel_gine.png are never touched.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)

    def _mean3(name):
        path = os.path.join(sig_dir, name)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        rows = df[df["pair"] == "mean3"]
        if rows.empty:
            return None
        return rows.set_index(["tag", "split"])

    gin = _mean3("summary.csv")
    gine = _mean3("summary_gine.csv")
    afp = _mean3("summary_attentivefp.csv")
    if gin is None or afp is None:
        raise FileNotFoundError(
            "need results/significance/{summary,summary_attentivefp}.csv "
            "-- run paired_bootstrap.py for gin and attentivefp first")

    # fairness cells: gine point estimates without a bootstrap row
    gine_pts_path = os.path.join(BASE, "results", "gine_panel", "summary.csv")
    gine_pts = None
    if os.path.exists(gine_pts_path):
        gine_pts = pd.read_csv(gine_pts_path).set_index(["tag", "split"])

    cells = [(t, s) for t in TAGS for s in SPLITS]
    y_pos = np.arange(len(cells))[::-1]
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    off = 0.26
    colors = {"gin": "tab:orange", "gine": "tab:green",
              "attentivefp": "tab:purple"}
    offsets = {"gin": off, "gine": 0.0, "attentivefp": -off}

    for y, cell in zip(y_pos, cells):
        g = gin.loc[cell]
        ax.hlines(y + offsets["gin"], g["ci_lo"], g["ci_hi"],
                  color=colors["gin"], linewidth=1.6)
        ax.plot([g["d_r2"]], [y + offsets["gin"]], "o",
                color=colors["gin"], markersize=5)

        if gine is not None and cell in gine.index:
            r = gine.loc[cell]
            ax.hlines(y + offsets["gine"], r["ci_lo"], r["ci_hi"],
                      color=colors["gine"], linewidth=1.6)
            ax.plot([r["d_r2"]], [y + offsets["gine"]], "o",
                    color=colors["gine"], markersize=5)
        elif gine_pts is not None and cell in gine_pts.index:
            ax.plot([gine_pts.loc[cell, "d_r2_mean"]],
                    [y + offsets["gine"]], "o", markerfacecolor="white",
                    markeredgecolor=colors["gine"], markeredgewidth=1.4,
                    markersize=6)

        a = afp.loc[cell]
        ax.hlines(y + offsets["attentivefp"], a["ci_lo"], a["ci_hi"],
                  color=colors["attentivefp"], linewidth=1.6)
        ax.plot([a["d_r2"]], [y + offsets["attentivefp"]], "o",
                color=colors["attentivefp"], markersize=5)

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
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor=colors["gine"], markeredgewidth=1.4,
                   label="GINE point estimate only (fairness)"),
        plt.Line2D([], [], marker="o", ls="", color=colors["attentivefp"],
                   label="AttentiveFP mean3 (phase 3)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Paired bootstrap ΔR² (model − RF): GIN vs GINE vs "
                 "AttentiveFP, mean3 convention", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def print_table(df):
    """The 8x2 delta table with CIs, plus the significance tally."""
    print("\n=== AttentiveFP vs RF panel (8 targets x 2 splits) ===")
    print(f"  {'tag':<10} {'split':<9} {'rf_r2':>7} {'afp_r2':>17} "
          f"{'d_r2':>8}  {'95% CI (mean3)':<26} sig")
    n_sig = 0
    n_ci = 0
    for r in df.itertuples():
        afp = f"{r.afp_r2_mean:.4f}±{r.afp_r2_std:.4f}"
        if not np.isfinite(r.ci_lo):
            ci = "(bootstrap not run)"
            sig = "-"
        else:
            ci = f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
            sig = "yes" if _is_true(r.significant) else "no"
            n_sig += int(_is_true(r.significant))
        n_ci += int(np.isfinite(r.ci_lo))
        print(f"  {r.tag:<10} {r.split:<9} {r.rf_r2:>7.4f} {afp:>17} "
              f"{r.d_r2_mean:>+8.4f}  {ci:<26} {sig}")
    m3 = df[df["ci_lo"].notna()]
    rf_wins = [f"{r.tag}/{r.split}" for r in m3.itertuples()
               if _is_true(r.significant) and r.ci_hi < 0]
    afp_wins = [f"{r.tag}/{r.split}" for r in m3.itertuples()
                if _is_true(r.significant) and r.ci_lo > 0]
    print(f"\n  {n_ci}/16 cells have a bootstrap CI; {n_sig}/{n_ci} exclude 0")
    if afp_wins:
        print(f"  AttentiveFP significantly ahead ({len(afp_wins)}): "
              f"{', '.join(afp_wins)}")
    else:
        print("  AttentiveFP significantly ahead: none")
    if rf_wins:
        print(f"  RF significantly ahead ({len(rf_wins)}): "
              f"{', '.join(rf_wins)}")
    if afp_wins:
        verdict = (f"AttentiveFP beats RF significantly in "
                   f"{len(afp_wins)} cell(s); the attention readout is "
                   "real somewhere.")
    else:
        verdict = (f"AttentiveFP never beats RF significantly: 0/{n_ci} "
                   "bootstrapped cells have a CI above 0 -- attention over "
                   "the atom graph does not flip the verdict.")
    print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sig-dir", default=os.path.join(BASE, "results",
                                                     "significance"),
                   help="directory holding {tag}_{split}_attentivefp.csv")
    p.add_argument("--out-csv", default=os.path.join(BASE, "results",
                                                     "afp_panel",
                                                     "summary.csv"))
    p.add_argument("--fig", default=os.path.join(BASE, "figures",
                                                 "afp_panel",
                                                 "panel_dR2.png"))
    p.add_argument("--combined-fig",
                   default=os.path.join(BASE, "figures", "significance",
                                        "ci_panel_with_afp.png"))
    return p.parse_args(argv)


def main(args=None):
    args = parse_args(args)
    panel = build_panel(os.path.abspath(args.sig_dir))

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    panel.to_csv(out_csv, index=False)

    make_panel_figure(panel, os.path.abspath(args.fig))
    make_combined_forest(os.path.abspath(args.sig_dir),
                         os.path.abspath(args.combined_fig))

    print_table(panel)
    print(f"\nsummary -> {os.path.relpath(out_csv, BASE)} ({len(panel)} rows)")
    print(f"figure  -> {os.path.relpath(os.path.abspath(args.fig), BASE)}")
    print(f"combined -> "
          f"{os.path.relpath(os.path.abspath(args.combined_fig), BASE)}")


if __name__ == "__main__":
    main()
