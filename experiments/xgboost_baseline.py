"""XGBoost baseline: is the traditional-ML ceiling higher than RF suggests?

Identical evaluation contract as the learning curve: same persisted
2048-bit Morgan r=2 fingerprints (the headline featurization), same
GNN-paired train pool (train_idx), same persisted test set, fixed
hyperparameters, 3 seeds x 2 splits. The same-pool RF control values are
read from the learning-curve results when available (not re-run).

Note: XGBoost runs on CPU (tree_method="hist") - deterministic for a
fixed seed, and fast enough at this scale. libomp is required on macOS:
    brew install libomp

Outputs:
  results/xgboost/xgboost_{TAG}.csv  one row per run
  results/xgboost/summary_{TAG}.json (embeds the RF control values)
  figures/xgboost/xgboost_{TAG}.png
"""
import argparse
import csv
import json
import os
import time

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
HEADER = "split,seed,r2,rmse,mae,train_seconds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--splits", default="random,scaffold")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--out", default=os.path.join(BASE, "results", "xgboost"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "xgboost"))
    return p.parse_args()


def load_rf_control(tag):
    """Same-pool RF control from the learning curve (preferred) or headline."""
    lc = os.path.join(BASE, "results", "learning_curve",
                      f"learning_curve_{tag}.csv")
    if os.path.exists(lc):
        import pandas as pd
        df = pd.read_csv(lc)
        n_max = df["n_train"].max()
        out = {}
        for split in sorted(df["split"].unique()):
            g = df[(df["split"] == split) & (df["model"] == "rf")
                   & (df["n_train"] == n_max)]
            out[split] = {"r2_mean": float(g["r2"].mean()),
                          "r2_std": float(g["r2"].std()),
                          "source": f"learning_curve rf n={n_max} (same pool)"}
        return out
    mj = json.load(open(os.path.join(BASE, "results", f"metrics_{tag}.json")))
    return {split: {"r2_mean": mj[split]["r2"], "r2_std": None,
                    "source": "headline metrics json (full pool, not same-pool)"}
            for split in ("random", "scaffold")}


def make_figure(csv_path, fig_path, tag, control):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    splits = sorted(df["split"].unique())
    x = np.arange(len(splits))
    w = 0.35
    xgb_means = [df[df["split"] == s]["r2"].mean() for s in splits]
    xgb_stds = [df[df["split"] == s]["r2"].std() for s in splits]
    rf_means = [control[s]["r2_mean"] for s in splits]
    rf_stds = [control[s]["r2_std"] or 0.0 for s in splits]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, xgb_means, w, yerr=xgb_stds, capsize=4,
           label="XGBoost", color="tab:orange")
    ax.bar(x + w / 2, rf_means, w, yerr=rf_stds, capsize=4,
           label="RF (control)", color="tab:blue")
    ax.set_xticks(x, splits)
    ax.set_ylabel("test R2 (3 seeds, mean +/- std)")
    ax.set_title(f"XGBoost vs RF - {tag}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out, f"xgboost_{args.tag}.csv")
    fig_path = os.path.join(args.fig_dir, f"xgboost_{args.tag}.png")
    sum_path = os.path.join(args.out, f"summary_{args.tag}.json")

    try:
        import xgboost as xgb
    except Exception as e:  # ImportError, or libomp load failure on macOS
        print(f"ERROR: xgboost unavailable: {e}\n"
              "On macOS: brew install libomp && pip install xgboost")
        raise SystemExit(2)

    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{args.tag}_fingerprints.npz"))
    X, y = fp["X"], fp["y"]
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)

    control = load_rf_control(args.tag)
    print(f"[{args.tag}] RF control: "
          + json.dumps({k: v["r2_mean"] for k, v in control.items()}))

    rows = []
    csv_f = open(csv_path, "w")
    csv_f.write(HEADER + "\n")

    def emit(row):
        rows.append(row)
        csv_f.write(",".join(str(row[k]) for k in HEADER.split(",")) + "\n")
        csv_f.flush()

    print(f"[{args.tag}] seeds={seeds} splits={splits} "
          f"n_est={args.n_estimators} lr={args.lr} depth={args.max_depth} "
          f"device=cpu/hist (xgboost {xgb.__version__})")
    for split in splits:
        sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                                  f"{args.tag}_{split}.npz"))
        train_idx, test_idx = sp["train_idx"], sp["test_idx"]
        print(f"[{args.tag}/{split}] pool={len(train_idx)} test={len(test_idx)}")
        for seed in seeds:
            t0 = time.time()
            model = xgb.XGBRegressor(
                n_estimators=args.n_estimators, learning_rate=args.lr,
                max_depth=args.max_depth, random_state=seed,
                tree_method="hist", n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            m = {"r2": float(r2_score(y[test_idx], pred)),
                 "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
                 "mae": float(mean_absolute_error(y[test_idx], pred))}
            dt = time.time() - t0
            print(f"  seed={seed} R2={m['r2']:.4f} RMSE={m['rmse']:.4f} ({dt:.0f}s)")
            emit({"split": split, "seed": seed, **m,
                  "train_seconds": round(dt, 1)})
    csv_f.close()

    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r["r2"])
    summary = {f"{s}": {"r2_mean": float(np.mean(v)),
                        "r2_std": float(np.std(v)),
                        "rmse_mean": float(np.mean([r["rmse"] for r in rows
                                                    if r["split"] == s])),
                        "delta_vs_rf_control": (float(np.mean(v))
                                                - control[s]["r2_mean"]),
                        "runs": len(v)}
               for s, v in sorted(by_split.items())}
    json.dump({"tag": args.tag, "seeds": seeds,
               "xgb_params": {"n_estimators": args.n_estimators,
                              "learning_rate": args.lr,
                              "max_depth": args.max_depth,
                              "tree_method": "hist", "device": "cpu"},
               "xgboost_version": xgb.__version__,
               "rf_control": control, "by_split": summary},
              open(sum_path, "w"), indent=2)
    print(f"summary -> {os.path.relpath(sum_path, BASE)}")

    make_figure(csv_path, fig_path, args.tag, control)
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")
    print(f"csv     -> {os.path.relpath(csv_path, BASE)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
