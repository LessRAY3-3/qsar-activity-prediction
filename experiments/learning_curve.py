"""Learning curve: RF vs GIN as a function of training-set size.

Question: how do the two models scale with data, and is either of them
saturated at the current dataset size?

Protocol (everything frozen except the one variable):
  * test set and validation set: the persisted split indices, untouched
  * hyperparameters: the exact values the original pipelines selected
    (RF: the grid-search winner on the full train pool; GIN: gnn_03's
    defaults -- same architecture, optimizer, scheduler, early stopping)
  * target standardisation: full-train-pool mean/std for every n
  * the one variable: the number of training molecules n, subsampled
    from the GNN train pool (the 80% train minus the 10% validation
    carve-out persisted in data/processed/splits/)

Paired by design: for a given (split, n, seed) both models see the exact
same molecules (one RandomState(seed) subsample of the train pool), so
RF-vs-GIN differences at fixed n are not subsample noise. GIN results
must be read as 3-seed mean +- std (single-seed GNN numbers are noise,
especially across hardware).

Note on absolute numbers: the headline RF metrics (0.747 random /
0.562 scaffold) were trained on the full 4932-molecule 80% pool because
the RF pipeline needs no validation set. This curve trains RF on the
4438/4436-molecule GNN train pool so both models stay paired; RF's top
of curve therefore sits slightly below its headline. The GIN headline
(0.691 / 0.544, seed 42) used exactly this pool and is directly
comparable at n close to full.

Outputs:
  results/learning_curve/learning_curve_{TAG}.csv  one row per run
  results/learning_curve/summary_{TAG}.json        mean+-std per (split,model,n)
  figures/learning_curve/learning_curve_{TAG}.png  R2 vs n, RF vs GIN
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from gnn_03_train_gin import GINRegressor, run_epoch, predict, metrics  # noqa: E402
from gnn_graph_dataset import MoleculeGraphDataset  # noqa: E402

RF_GRID_KEYS = ("n_estimators", "max_depth", "min_samples_split")
HEADER = "split,n_train,seed,model,r2,rmse,mae,best_epoch,train_seconds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--sizes", default="500,1000,1500,2000,2500,3000,3500,4000,4400")
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--splits", default="random,scaffold")
    p.add_argument("--out", default=os.path.join(BASE, "results", "learning_curve"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "learning_curve"))
    # GIN hyperparameters -- frozen at gnn_03_train_gin's defaults
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def load_rf_params(tag):
    """Freeze RF hyperparameters to the original grid-search winner."""
    mj = os.path.join(BASE, "results", f"metrics_{tag}.json")
    bp = json.load(open(mj))["best_params_random"]
    return {k: bp[k] for k in RF_GRID_KEYS}


def gin_train_eval(graphs, subset, valid_idx, test_idx, y_mean, y_std, args, seed):
    """Identical recipe to gnn_03_train_gin.py, evaluated on the full test set."""
    import torch
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_ds = MoleculeGraphDataset(graphs, indices=subset)
    valid_ds = MoleculeGraphDataset(graphs, indices=valid_idx)
    test_ds = MoleculeGraphDataset(graphs, indices=test_idx)
    for ds in (train_ds, valid_ds, test_ds):
        ds.y = (ds.y - y_mean) / y_std

    g = torch.Generator().manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    valid_ld = DataLoader(valid_ds, batch_size=512)
    test_ld = DataLoader(test_ds, batch_size=512)

    model = GINRegressor(args.hidden, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-5)

    best = {"rmse": float("inf"), "state": None, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_ld, device, opt)
        vp, vy = predict(model, valid_ld, device)
        v_rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))
        sched.step(v_rmse)
        if v_rmse < best["rmse"] - 1e-5:
            best = {"rmse": v_rmse, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch}
        if epoch - best["epoch"] >= args.patience:
            break

    model.load_state_dict(best["state"])
    tp_raw, ty_raw = predict(model, test_ld, device)
    tp, ty = tp_raw * y_std + y_mean, ty_raw * y_std + y_mean
    return metrics(ty, tp), best["epoch"]


def make_figure(csv_path, fig_path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.5),
                             sharey=True)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        sub = df[df["split"] == split]
        for model, color, marker in (("rf", "tab:blue", "o"),
                                     ("gin", "tab:orange", "s")):
            g = sub[sub["model"] == model].groupby("n_train")["r2"].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], marker=marker, color=color,
                    label=model.upper())
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=color, alpha=0.2)
        ax.set_title(f"{split} split")
        ax.set_xlabel("training molecules")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test R2")
    axes[0].legend()
    fig.suptitle(f"Learning curve - {tag} (3-seed mean +/- std)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    sizes = sorted(int(s) for s in args.sizes.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    csv_path = os.path.join(args.out, f"learning_curve_{args.tag}.csv")
    fig_path = os.path.join(args.fig_dir, f"learning_curve_{args.tag}.png")
    sum_path = os.path.join(args.out, f"summary_{args.tag}.json")

    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{args.tag}_fingerprints.npz"))
    X, y = fp["X"], fp["y"]
    graphs = os.path.join(BASE, "data", "processed", f"{args.tag}_graphs.npz")
    rf_params = load_rf_params(args.tag)

    rows = []
    print(f"[{args.tag}] sizes={sizes} seeds={seeds} splits={splits}")
    with open(csv_path, "w") as f:
        f.write(HEADER + "\n")
    for split in splits:
        sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                                  f"{args.tag}_{split}.npz"))
        train_idx, valid_idx, test_idx = sp["train_idx"], sp["valid_idx"], sp["test_idx"]
        y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std())
        print(f"[{args.tag}/{split}] pool={len(train_idx)} "
              f"valid={len(valid_idx)} test={len(test_idx)}")
        for n in sizes:
            if n > len(train_idx):
                print(f"  n={n}: skipped (larger than train pool)")
                continue
            for seed in seeds:
                rng = np.random.RandomState(seed)
                subset = np.sort(rng.choice(train_idx, size=n, replace=False))

                from sklearn.ensemble import RandomForestRegressor
                t0 = time.time()
                rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
                rf.fit(X[subset], y[subset])
                m = metrics(y[test_idx], rf.predict(X[test_idx]))
                dt = time.time() - t0
                print(f"  n={n} seed={seed} RF   R2={m['r2']:.4f} "
                      f"RMSE={m['rmse']:.4f} ({dt:.0f}s)")
                rows.append({"split": split, "n_train": n, "seed": seed,
                             "model": "rf", **m, "best_epoch": -1,
                             "train_seconds": round(dt, 1)})

                t0 = time.time()
                m, best_epoch = gin_train_eval(graphs, subset, valid_idx,
                                               test_idx, y_mean, y_std, args, seed)
                dt = time.time() - t0
                print(f"  n={n} seed={seed} GIN  R2={m['r2']:.4f} "
                      f"RMSE={m['rmse']:.4f} (best epoch {best_epoch}, {dt:.0f}s)")
                rows.append({"split": split, "n_train": n, "seed": seed,
                             "model": "gin", **m, "best_epoch": best_epoch,
                             "train_seconds": round(dt, 1)})

                with open(csv_path, "a") as f:
                    for r in rows[-2:]:
                        f.write(",".join(str(r[k]) for k in HEADER.split(",")) + "\n")

    summary = {}
    for r in rows:
        key = f"{r['split']}/{r['model']}/n={r['n_train']}"
        summary.setdefault(key, []).append(r)
    summary = {k: {"r2_mean": float(np.mean([r["r2"] for r in v])),
                   "r2_std": float(np.std([r["r2"] for r in v])),
                   "rmse_mean": float(np.mean([r["rmse"] for r in v])),
                   "rmse_std": float(np.std([r["rmse"] for r in v])),
                   "runs": len(v)}
               for k, v in sorted(summary.items())}
    json.dump({"tag": args.tag, "sizes": sizes, "seeds": seeds,
               "rf_params": rf_params, "by_group": summary},
              open(sum_path, "w"), indent=2)
    print(f"summary -> {os.path.relpath(sum_path, BASE)}")

    make_figure(csv_path, fig_path, args.tag)
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")
    print(f"csv     -> {os.path.relpath(csv_path, BASE)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
