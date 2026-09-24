"""GNN step 4: RF-vs-GIN comparison table + predicted-vs-actual scatter.

Inputs:
  results/metrics_{TAG}.json            RF metrics (from the QSAR pipeline)
  results/gnn_metrics_{TAG}.json        GIN metrics (gnn_03_train_gin.py)
  models/rf_{split}_split_{TAG}.joblib  saved RF models (for RF test preds)
  results/gnn_preds_{TAG}_{split}.npz   saved GIN test preds
  data/processed/splits/{TAG}_{split}.npz  (for test_idx + scaffold labels)

Outputs:
  results/comparison_{TAG}.csv                      the comparison table
  results/rf_preds_{TAG}_{split}.npz                RF test preds (cached for
                                                    the error-analysis step)
  figures/compare_pred_vs_actual_{TAG}_{split}.png  scatter, both models side
                                                    by side, coloured by
                                                    Bemis-Murcko scaffold
  results/scaffold_color_map_{TAG}_{split}.csv      colour legend -> scaffold
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")
RF_METRICS = os.path.join(BASE, "results", f"metrics_{TAG}.json")
GIN_METRICS = os.path.join(BASE, "results", f"gnn_metrics_{TAG}.json")
FP_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
SPLIT_NPZ = os.path.join(BASE, "data", "processed", "splits", f"{TAG}_{{}}.npz")
FIG_DIR = os.path.join(BASE, "figures")
N_COLOR_SCAFFOLDS = 8  # distinct colours; everything else is grey


def load_split_data(split):
    d = np.load(SPLIT_NPZ.format(split))
    fp = np.load(FP_NPZ)
    X, y, test_idx = fp["X"], fp["y"], d["test_idx"]
    scaf = d["scaffold_smiles"][test_idx]

    rf = joblib.load(os.path.join(BASE, "models", f"rf_{split}_split_{TAG}.joblib"))
    rf_pred = rf.predict(X[test_idx])
    np.savez(os.path.join(BASE, "results", f"rf_preds_{TAG}_{split}.npz"),
             test_idx=test_idx, y_true=y[test_idx], y_pred=rf_pred)

    g = np.load(os.path.join(BASE, "results", f"gnn_preds_{TAG}_{split}.npz"))
    assert np.array_equal(g["test_idx"], test_idx)
    assert np.allclose(g["y_true"], y[test_idx])
    return y[test_idx], rf_pred, g["y_pred"], scaf


def collect_gin_metrics():
    """GIN metrics across seeds: base file + any *_seedN.json siblings."""
    import re
    per_split = {"random": {"r2": [], "rmse": []}, "scaffold": {"r2": [], "rmse": []}}
    pat = re.compile(rf"^gnn_metrics_{TAG}(_seed\d+)?\.json$")
    files = sorted(f for f in os.listdir(os.path.join(BASE, "results")) if pat.match(f))
    for f in files:
        m = json.load(open(os.path.join(BASE, "results", f)))
        for s in per_split:
            if s in m:
                per_split[s]["r2"].append(m[s]["r2"])
                per_split[s]["rmse"].append(m[s]["rmse"])
    return per_split, len(files)


def comparison_table():
    rf_m = json.load(open(RF_METRICS))
    gin_m = json.load(open(GIN_METRICS))
    seeds, n_seeds = collect_gin_metrics()
    rows = []
    for split in ("random", "scaffold"):
        r, g = rf_m[split], gin_m[split]
        rows.append({
            "split": split,
            "rf_r2": r["r2"], "rf_rmse": r["rmse"], "rf_mae": r["mae"],
            "gin_r2": g["r2"], "gin_rmse": g["rmse"], "gin_mae": g["mae"],
            "gin_r2_mean": float(np.mean(seeds[split]["r2"])),
            "gin_r2_std": float(np.std(seeds[split]["r2"])),
            "gin_rmse_mean": float(np.mean(seeds[split]["rmse"])),
            "gin_rmse_std": float(np.std(seeds[split]["rmse"])),
            "gin_n_seeds": n_seeds,
            "d_r2_gin_minus_rf": g["r2"] - r["r2"],
            "d_rmse_gin_minus_rf": g["rmse"] - r["rmse"],
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
          f"{'GIN R2 mean+-std':>17} | {'dR2':>7}")
    print("-" * 92)
    for r in rows:
        print(f"{r['split']:<9} | {r['rf_r2']:7.3f} {r['rf_rmse']:8.3f} | "
              f"{r['gin_r2']:15.3f} {r['gin_rmse']:8.3f} | "
              f"{r['gin_r2_mean']:8.3f} +-{r['gin_r2_std']:<7.3f} | "
              f"{r['d_r2_gin_minus_rf']:+7.3f}")
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


def scatter_split(split):
    y, rf_pred, gin_pred, scaf = load_split_data(split)
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


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = comparison_table()
    write_table_csv(rows)
    for split in ("random", "scaffold"):
        scatter_split(split)


if __name__ == "__main__":
    main()
