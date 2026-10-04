"""Cross-target summary: RF (Morgan FP) vs GIN across ChEMBL targets.

Question: does GIN's delta over RF depend on the protein family and on
dataset size?

Inputs (per tag):
  results/comparison_{TAG}.csv                     the GNN-04 comparison
                                                   table -- the 3-seed GIN
                                                   summary lives here
  data/processed/{TAG}_fingerprints.npz            n_molecules = X rows
                                                   (= unique compounds)
  results/learning_curve/learning_curve_{TAG}.csv  optional; its
                                                   (n_train, d_r2) points
                                                   are overlaid on the figure

d_r2 always keeps the comparison file's sign convention:
d_r2_gin_minus_rf = gin_r2 - rf_r2 (positive = GIN wins).

Outputs:
  results/multi_target/summary.csv       one row per tag
  figures/multi_target/delta_vs_n.png    dR2 vs log10(n_molecules),
                                         scaffold filled / random light
                                         hollow, coloured by protein family
Exit code 0 even when some tags are skipped (e.g. RF-only bace with no
comparison file); every skip is printed so the caller can see it.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

# tag -> protein family (mapk14 maps to CHEMBL279, which is actually VEGFR2 --
# still a kinase, so the family label is unaffected)
FAMILY = {
    "egfr": "kinase",
    "egfr_full": "kinase",
    "a2a": "gpcr",
    "bace": "protease",
    "mapk14": "kinase",
    "abl1": "kinase",
    "mpro": "viral-protease",
    "hivpr": "viral-protease",
    "herg": "ion-channel",
}
FAMILY_COLOR = {
    "kinase": "tab:blue",
    "gpcr": "tab:orange",
    "protease": "tab:green",
    "viral-protease": "tab:red",
    "ion-channel": "tab:purple",
}
DEFAULT_FAMILY_COLOR = "tab:grey"
SPLITS = ("random", "scaffold")
SUMMARY_COLS = [
    "tag", "family", "n_molecules",
    "rf_r2_random", "gin_r2_mean_random", "gin_r2_std_random", "d_r2_random",
    "rf_r2_scaffold", "gin_r2_mean_scaffold", "gin_r2_std_scaffold",
    "d_r2_scaffold",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", default=None,
                   help="comma-separated tags; default = scan results/ "
                        "for comparison_*.csv")
    p.add_argument("--out", default=os.path.join(BASE, "results", "multi_target",
                                                 "summary.csv"))
    p.add_argument("--fig", default=os.path.join(BASE, "figures", "multi_target",
                                                 "delta_vs_n.png"))
    return p.parse_args()


def discover_tags():
    pat = os.path.join(BASE, "results", "comparison_*.csv")
    return sorted(os.path.basename(f)[len("comparison_"):-len(".csv")]
                  for f in glob.glob(pat))


def load_tag(tag):
    """One summary row per tag; None when the tag must be skipped."""
    cmp_path = os.path.join(BASE, "results", f"comparison_{tag}.csv")
    if not os.path.exists(cmp_path):
        print(f"[{tag}] skipped: no {os.path.relpath(cmp_path, BASE)} "
              f"(RF-only target, no GNN comparison)")
        return None
    df = pd.read_csv(cmp_path)
    if "rf_r2" not in df.columns:
        print(f"[{tag}] skipped: {os.path.basename(cmp_path)} has no rf_r2 column")
        return None

    row = {"tag": tag, "family": FAMILY.get(tag, "unknown")}
    fp_path = os.path.join(BASE, "data", "processed", f"{tag}_fingerprints.npz")
    if os.path.exists(fp_path):
        row["n_molecules"] = int(np.load(fp_path)["X"].shape[0])
    else:
        row["n_molecules"] = None
        print(f"[{tag}] warning: no {os.path.relpath(fp_path, BASE)}, "
              f"n_molecules left empty")

    has_gin = all(c in df.columns for c in ("gin_r2_mean", "gin_r2_std",
                                            "d_r2_gin_minus_rf"))
    for split in SPLITS:
        hit = df[df["split"] == split]
        if hit.empty:
            print(f"[{tag}] warning: no '{split}' row in comparison")
            has_row = False
        else:
            has_row = True
        r = hit.iloc[0] if has_row else None
        row[f"rf_r2_{split}"] = float(r["rf_r2"]) if has_row else np.nan
        if has_row and has_gin:
            row[f"gin_r2_mean_{split}"] = float(r["gin_r2_mean"])
            row[f"gin_r2_std_{split}"] = float(r["gin_r2_std"])
            # keep the comparison file's gin-minus-rf sign as-is
            row[f"d_r2_{split}"] = float(r["d_r2_gin_minus_rf"])
        else:
            row[f"gin_r2_mean_{split}"] = np.nan
            row[f"gin_r2_std_{split}"] = np.nan
            row[f"d_r2_{split}"] = np.nan
    if not np.isfinite(row["d_r2_scaffold"]) and not np.isfinite(row["d_r2_random"]):
        print(f"[{tag}] skipped: comparison has no GIN columns "
              f"(RF-only table)")
        return None
    return row


def load_curve_points(tag):
    """[(split, n_train, d_r2)] from the learning-curve CSV, if present."""
    path = os.path.join(BASE, "results", "learning_curve",
                        f"learning_curve_{tag}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if not {"split", "n_train", "model", "r2"}.issubset(df.columns):
        print(f"[{tag}] warning: unexpected learning-curve columns, ignored")
        return []
    pts = []
    for (split, n), g in df.groupby(["split", "n_train"]):
        by_model = g.groupby("model")["r2"].mean()
        if "gin" in by_model and "rf" in by_model:
            pts.append((split, int(n), float(by_model["gin"] - by_model["rf"])))
    return pts


def print_table(rows):
    print("\n=== Cross-target summary: RF (Morgan FP) vs GIN ===")
    print(f"{'tag':<10} {'family':<15} {'n':>6} | "
          f"{'RF r2':>6} {'GIN r2 (mean+-std)':>20} {'dR2':>7} | "
          f"{'RF r2':>6} {'GIN r2 (mean+-std)':>20} {'dR2':>7}")
    print(f"{'':10} {'':15} {'':6} | "
          f"{'random':>6} {'random':>20} {'':>7} | "
          f" {'scaffold':>6} {'scaffold':>20} {'':>7}")
    print("-" * 104)
    for r in rows:
        cells = [f"{r['tag']:<10} {r['family']:<15} "
                 f"{r['n_molecules'] if r['n_molecules'] is not None else '?':>6} |"]
        for split in SPLITS:
            rf = r[f"rf_r2_{split}"]
            gm, gs, d = (r[f"gin_r2_mean_{split}"], r[f"gin_r2_std_{split}"],
                         r[f"d_r2_{split}"])
            gin_txt = (f"{gm:8.3f} +-{gs:<9.3f}" if np.isfinite(gm) else
                       f"{'-':>18}")
            d_txt = f"{d:+7.3f}" if np.isfinite(d) else f"{'-':>7}"
            rf_txt = f"{rf:6.3f}" if np.isfinite(rf) else f"{'-':>6}"
            cells.append(f"{rf_txt} {gin_txt:>20} {d_txt:>7} |")
        print(" ".join(cells))
    print(f"({len(rows)} targets)")


def print_conclusion(rows):
    scaf = [r for r in rows if np.isfinite(r.get("d_r2_scaffold", np.nan))]
    wins = [r for r in scaf if r["d_r2_scaffold"] > 0]
    names = ", ".join(r["tag"] for r in wins) if wins else "none"
    print(f"\nConclusion: on the scaffold split GIN beats RF on "
          f"{len(wins)}/{len(scaf)} targets ({names}); "
          f"GIN-RF delta range "
          f"[{min(r['d_r2_scaffold'] for r in scaf):+.3f}, "
          f"{max(r['d_r2_scaffold'] for r in scaf):+.3f}]."
          if scaf else "\nConclusion: no scaffold-split GIN results to compare.")


def make_figure(rows, curve_points, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # learning-curve dR2 points (x = log10(n_train)): small grey background dots
    if curve_points:
        xs = [np.log10(n) for _, n, _ in curve_points]
        ys = [d for _, _, d in curve_points]
        ax.scatter(xs, ys, s=10, c="0.6", alpha=0.55, linewidths=0,
                   label="learning-curve point (dR2 vs n_train)")

    plotted = []
    for r in rows:
        if r["n_molecules"] is None:
            print(f"[{r['tag']}] not on figure: n_molecules unknown")
            continue
        x = np.log10(r["n_molecules"])
        fam = r["family"]
        color = FAMILY_COLOR.get(fam, DEFAULT_FAMILY_COLOR)
        d_scaf, d_rand = r["d_r2_scaffold"], r["d_r2_random"]
        if np.isfinite(d_scaf):
            ax.scatter(x, d_scaf, s=70, c=color, marker="o", zorder=3,
                       edgecolors="white", linewidths=0.6)
            ax.annotate(r["tag"], (x, d_scaf), textcoords="offset points",
                        xytext=(6, 5), fontsize=8)
        if np.isfinite(d_rand):
            ax.scatter(x, d_rand, s=70, marker="o", facecolors="none",
                       edgecolors=color, alpha=0.45,
                       linewidths=1.4, zorder=2)
        if fam not in plotted:
            plotted.append(fam)

    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="dR2 = 0")
    ax.set_xlabel("log10(n_molecules)")
    ax.set_ylabel("dR2 (GIN - RF)")
    ax.set_title("RF vs GIN delta across targets (dR2 = gin - rf, "
                 "per comparison table)")
    ax.grid(alpha=0.3)

    handles = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLOR.get(f, DEFAULT_FAMILY_COLOR),
                          label=f) for f in plotted]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                              markeredgecolor="0.4", alpha=0.6,
                              label="random split (light)"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.6",
                              markersize=5, alpha=0.55,
                              label="learning-curve point"))
    handles.append(plt.Line2D([], [], color="red", ls="--", label="dR2 = 0"))
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = discover_tags()
        if not tags:
            print(f"no results/comparison_*.csv found under "
                  f"{os.path.relpath(os.path.join(BASE, 'results'), BASE)}")
            return
    print(f"targets: {tags}")

    rows = [r for r in (load_tag(t) for t in tags) if r is not None]
    if not rows:
        print("no target produced a summary row; nothing to write")
        return

    curve_points = []
    for r in rows:
        curve_points.extend((r["tag"], n, d) for _, n, d
                            in load_curve_points(r["tag"]))

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_COLS).to_csv(out_path, index=False)

    print_table(rows)
    print_conclusion(rows)

    fig_path = os.path.abspath(args.fig)
    make_figure(rows, curve_points, fig_path)
    print(f"\ncsv     -> {os.path.relpath(out_path, BASE)} ({len(rows)} rows)")
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")


if __name__ == "__main__":
    main()
