"""GNN step 4: RF-vs-GIN comparison table + predicted-vs-actual scatter.

Inputs:
  results/metrics_{TAG}.json            RF metrics (from the QSAR pipeline)
  results/gnn_metrics_{TAG}.json        GIN metrics (gnn_03_train_gin.py)
  models/rf_{split}_split_{TAG}.joblib  saved RF models (for RF test preds)
  results/gnn_preds_{TAG}_{split}.npz   saved GIN test preds
  data/processed/splits/{TAG}_{split}.npz  (for test_idx + scaffold labels)

Outputs:
  results/comparison_{TAG}.csv                      the comparison table; carries
                                                    BOTH effect-size conventions:
                                                    d_*_mean_* (3-seed mean GIN - RF,
                                                    primary) and d_*_seed42_* (single
                                                    seed-42 run - RF, reference only)
  results/rf_preds_{TAG}_{split}.npz                RF test preds (cached for
                                                    the error-analysis step)
  figures/compare_pred_vs_actual_{TAG}_{split}.png  scatter, both models side
                                                    by side, coloured by
                                                    Bemis-Murcko scaffold
  results/scaffold_color_map_{TAG}_{split}.csv      colour legend -> scaffold

CLI: --skip-model-check skips the RF-vs-split verification that would
otherwise run before results/rf_preds_{TAG}_{split}.npz is rewritten.
gnn_04 derives those preds from the regenerated split itself, so once it
has written them every downstream "rf_preds.test_idx == split.test_idx"
comparison is self-referential; the only independent evidence that the
split matches the RF's original test set is reproducing
metrics_{TAG}.json, exactly what gnn_01_make_splits verifies (and what
--skip-model-check there disables). Pass the same flag here during a
refresh window where models/metrics predate the split data: the check
is then skipped and announced as "[not checked]" instead of silently
passing.
"""
import argparse
import json
import os

from artifact_guards import collect_seed_metrics, check_prediction

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")
RF_METRICS = os.path.join(BASE, "results", f"metrics_{TAG}.json")
GIN_METRICS = os.path.join(BASE, "results", f"gnn_metrics_{TAG}.json")
FP_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
SPLIT_NPZ = os.path.join(BASE, "data", "processed", "splits", f"{TAG}_{{}}.npz")
FIG_DIR = os.path.join(BASE, "figures")
N_COLOR_SCAFFOLDS = 8  # distinct colours; everything else is grey


def load_split_data(split, verify=True):
    d = np.load(SPLIT_NPZ.format(split))
    fp = np.load(FP_NPZ)
    X, y, test_idx = fp["X"], fp["y"], d["test_idx"]
    scaf = d["scaffold_smiles"][test_idx]

    rf = joblib.load(os.path.join(BASE, "models", f"rf_{split}_split_{TAG}.joblib"))
    rf_pred = rf.predict(X[test_idx])
    # Verify BEFORE overwriting rf_preds: re-predicting on this split must
    # reproduce metrics_{TAG}.json, which is the non-circular RF-vs-split
    # check (the metrics were stored by 04's own run, independently of the
    # regenerated split). Without it -- and gnn_01 --skip-model-check
    # disables the same verification upstream -- the only remaining
    # "checks" would compare files this script itself derives from the
    # split, i.e. a self-loop.
    if verify:
        mj = json.load(open(RF_METRICS))
        if split not in mj:
            print(f"  [not checked] {TAG}/{split}: no '{split}' entry in "
                  f"{os.path.relpath(RF_METRICS, BASE)}")
        else:
            r2 = r2_score(y[test_idx], rf_pred)
            rmse = float(np.sqrt(mean_squared_error(y[test_idx], rf_pred)))
            exp = mj[split]
            if abs(r2 - exp["r2"]) > 1e-6 or abs(rmse - exp["rmse"]) > 1e-6:
                raise SystemExit(
                    f"{TAG}/{split}: RF re-predicts R2={r2:.7f} "
                    f"RMSE={rmse:.7f} but metrics_{TAG}.json records "
                    f"R2={exp['r2']:.7f} RMSE={exp['rmse']:.7f} - the "
                    "regenerated split does not match the RF's original "
                    "test set (or models/metrics are stale). Not "
                    "overwriting rf_preds. Re-run gnn_01_make_splits "
                    "without --skip-model-check, or rerun this script "
                    "with --skip-model-check to accept unverified preds.")
            print(f"  [ok] {TAG}/{split}: RF re-prediction reproduces "
                  f"metrics_{TAG}.json (R2={r2:.6f})")
    else:
        print(f"  [not checked] {TAG}/{split}: RF-vs-split verification "
              "skipped (--skip-model-check); rf_preds are written from "
              "the regenerated split without independent evidence")
    g = np.load(os.path.join(BASE, "results", f"gnn_preds_{TAG}_{split}.npz"))
    check_prediction(g, d, y)
    np.savez(os.path.join(BASE, "results", f"rf_preds_{TAG}_{split}.npz"),
             test_idx=test_idx, y_true=y[test_idx], y_pred=rf_pred)

    return y[test_idx], rf_pred, g["y_pred"], scaf


def collect_gin_metrics():
    """GIN metrics across seeds: base file + any *_seedN.json siblings.

    Returns ({split: {r2, rmse}}, n_files); the per-split lists hold only
    seeds that actually report that split, so a seed file missing one
    split never inflates that split's count.
    """
    return collect_seed_metrics(os.path.join(BASE, "results"), TAG)


def comparison_table():
    rf_m = json.load(open(RF_METRICS))
    gin_m = json.load(open(GIN_METRICS))
    seeds, n_files = collect_gin_metrics()
    rows = []
    for split in ("random", "scaffold"):
        r, g = rf_m[split], gin_m[split]
        n_split = len(seeds[split]["r2"])
        if n_split < n_files:
            print(f"warning: {TAG}/{split}: only {n_split}/{n_files} seed "
                  f"metric file(s) carry this split; gin_n_seeds reports "
                  f"the actual contributing seeds")
        gin_r2_mean = float(np.mean(seeds[split]["r2"]))
        gin_rmse_mean = float(np.mean(seeds[split]["rmse"]))
        rows.append({
            "split": split,
            "rf_r2": r["r2"], "rf_rmse": r["rmse"], "rf_mae": r["mae"],
            "gin_r2": g["r2"], "gin_rmse": g["rmse"], "gin_mae": g["mae"],
            "gin_r2_mean": gin_r2_mean,
            "gin_r2_std": float(np.std(seeds[split]["r2"])),
            "gin_rmse_mean": gin_rmse_mean,
            "gin_rmse_std": float(np.std(seeds[split]["rmse"])),
            # actual number of seeds contributing to THIS split, never the
            # file count (a seed missing this split must not read as 3)
            "gin_n_seeds": n_split,
            # primary effect size: 3-seed mean GIN - RF
            "d_r2_mean_gin_minus_rf": gin_r2_mean - r["r2"],
            "d_rmse_mean_gin_minus_rf": gin_rmse_mean - r["rmse"],
            # reference only: the single seed-42 run - RF
            "d_r2_seed42_gin_minus_rf": g["r2"] - r["r2"],
            "d_rmse_seed42_gin_minus_rf": g["rmse"] - r["rmse"],
        })
    return rows


def write_table_csv(rows):
    import csv
    path = os.path.join(BASE, "results", f"comparison_{TAG}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n=== RF (Morgan FP) vs GIN ({TAG}) ===")
    print(f"{'split':<9} | {'RF R2':>7} {'RF RMSE':>8} | {'GIN R2 (seed42)':>15} {'GIN RMSE':>8} | "
          f"{'GIN R2 mean+-std':>17} | {'dR2 mean':>9} {'dR2 seed42':>11}")
    print("-" * 106)
    for r in rows:
        print(f"{r['split']:<9} | {r['rf_r2']:7.3f} {r['rf_rmse']:8.3f} | "
              f"{r['gin_r2']:15.3f} {r['gin_rmse']:8.3f} | "
              f"{r['gin_r2_mean']:8.3f} +-{r['gin_r2_std']:<7.3f} | "
              f"{r['d_r2_mean_gin_minus_rf']:+9.3f} {r['d_r2_seed42_gin_minus_rf']:+11.3f}")
    print(f"saved -> {os.path.relpath(path, BASE)}")


def scaffold_colors(scaf):
    """Top-N scaffolds get distinct colours; the rest share grey."""
    uniq, counts = np.unique(scaf, return_counts=True)
    order = np.argsort(-counts)
    cmap = plt.get_cmap("tab10")
    color_of = {}
    legend_rows = []
    for rank, i in enumerate(order):
        s = uniq[i]
        if rank < N_COLOR_SCAFFOLDS:
            color_of[s] = cmap(rank % 10)
            label = f"S{rank + 1} (n={counts[i]})"
        else:
            color_of[s] = (0.7, 0.7, 0.7, 0.45)
            label = "other"
        legend_rows.append({"color_label": label, "scaffold_smiles": s,
                            "n_molecules": int(counts[i])})
    colors = np.array([color_of[s] for s in scaf])
    return colors, legend_rows


def scatter_split(split, verify=True):
    y, rf_pred, gin_pred, scaf = load_split_data(split, verify=verify)
    colors, legend_rows = scaffold_colors(scaf)
    import csv
    with open(os.path.join(BASE, "results", f"scaffold_color_map_{TAG}_{split}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["color_label", "scaffold_smiles", "n_molecules"])
        w.writeheader()
        w.writerows(legend_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    lo = min(y.min(), rf_pred.min(), gin_pred.min()) - 0.3
    hi = max(y.max(), rf_pred.max(), gin_pred.max()) + 0.3
    for ax, pred, name in ((axes[0], rf_pred, "RF (Morgan FP)"),
                           (axes[1], gin_pred, "GIN (molecular graph)")):
        ax.scatter(y, pred, s=14, c=colors, edgecolors="none", alpha=0.85)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual pIC50")
        ax.set_title(name)
    axes[0].set_ylabel("Predicted pIC50")
    cmap = plt.get_cmap("tab10")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i % 10))
               for i in range(min(N_COLOR_SCAFFOLDS, len(legend_rows)))]
    labels = [r["color_label"] for r in legend_rows[:N_COLOR_SCAFFOLDS]]
    if len(legend_rows) > N_COLOR_SCAFFOLDS:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=(0.7, 0.7, 0.7, 0.9)))
        labels.append(f"other (n={sum(r['n_molecules'] for r in legend_rows[N_COLOR_SCAFFOLDS:])})")
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               title="Bemis-Murcko scaffold", title_fontsize=8)
    fig.suptitle(f"{TAG.upper()} - {split} split (test set, same molecules for both models)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = os.path.join(FIG_DIR, f"compare_pred_vs_actual_{TAG}_{split}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {os.path.relpath(out, BASE)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="GNN step 4: RF-vs-GIN comparison table + scatter")
    p.add_argument("--skip-model-check", action="store_true",
                   help="do not verify the regenerated split by "
                        "re-predicting with the saved RF against "
                        "metrics_{TAG}.json before rewriting rf_preds "
                        "(same flag as gnn_01_make_splits; the skipped "
                        "check is announced as [not checked])")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = comparison_table()
    write_table_csv(rows)
    for split in ("random", "scaffold"):
        scatter_split(split, verify=not args.skip_model_check)


if __name__ == "__main__":
    main()
