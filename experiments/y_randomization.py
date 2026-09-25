"""Y-randomization: is the models' signal real, or an artifact of leakage?

Training and validation labels are permuted within the train+valid pool
(fixed seeds); the test set keeps its REAL labels. A model that scores
R2 >= ~0.2 after being trained on permuted labels is learning something it
should not - the canonical leakage/sanity test. A real-label control group
(seed 42) must land on the known baselines.

Protocol matches the main experiments: GNN train pool (train_idx) for RF,
permuted train/valid for the GIN's early stopping, frozen hyperparameters
(RF grid winner, gnn_03 recipe), targets standardised with the PERMUTED
train statistics (the model only ever sees permuted targets).

Outputs:
  results/y_randomization/y_randomization_{TAG}.csv  one row per run
  results/y_randomization/summary_{TAG}.json
  figures/y_randomization/y_randomization_{TAG}.png
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

HEADER = "split,label,seed,model,r2,rmse,mae,best_epoch,train_seconds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--control-seed", type=int, default=42)
    p.add_argument("--splits", default="random,scaffold")
    p.add_argument("--out", default=os.path.join(BASE, "results", "y_randomization"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "y_randomization"))
    # GIN hyperparameters -- frozen at gnn_03_train_gin's defaults
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def gin_train_eval(graphs, train_idx, valid_idx, test_idx, y_all,
                   args, seed):
    """gnn_03 recipe; targets are taken from y_all (real or permuted)."""
    import torch
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    y_mean = float(y_all[train_idx].mean())
    y_std = float(y_all[train_idx].std())

    train_ds = MoleculeGraphDataset(graphs, indices=train_idx)
    valid_ds = MoleculeGraphDataset(graphs, indices=valid_idx)
    test_ds = MoleculeGraphDataset(graphs, indices=test_idx)
    # ds.y must stay FULL-LENGTH: __getitem__ indexes it by global graph id
    for ds in (train_ds, valid_ds, test_ds):
        ds.y = (y_all - y_mean) / y_std

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
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.5),
                             sharey=True)
    if len(splits) == 1:
        axes = [axes]
    rng_jitter = np.random.default_rng(0)
    for ax, split in zip(axes, splits):
        sub = df[df["split"] == split]
        for xpos, model in enumerate(["rf", "gin"]):
            for label, color, off in (("real", "tab:green", -0.12),
                                      ("shuffled", "tab:red", 0.12)):
                g = sub[(sub["model"] == model) & (sub["label"] == label)]
                if not len(g):
                    continue
                jx = g["r2"].to_numpy() * 0 + xpos + off + \
                    rng_jitter.uniform(-0.03, 0.03, len(g))
                ax.scatter(jx, g["r2"], c=color, s=45,
                           label=label if xpos == 0 else None, zorder=3)
        ax.axhline(0.2, color="k", ls="--", lw=1)
        ax.text(0.02, 0.21, "suspicion threshold 0.2", fontsize=8)
        ax.set_xticks([0, 1], ["RF", "GIN"])
        ax.set_title(f"{split} split")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test R2 (real test labels)")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Y-randomization - {tag} (shuffled train labels vs real control)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out, f"y_randomization_{args.tag}.csv")
    fig_path = os.path.join(args.fig_dir, f"y_randomization_{args.tag}.png")
    sum_path = os.path.join(args.out, f"summary_{args.tag}.json")

    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{args.tag}_fingerprints.npz"))
    X, y = fp["X"], fp["y"]
    graphs = os.path.join(BASE, "data", "processed", f"{args.tag}_graphs.npz")
    from sklearn.ensemble import RandomForestRegressor

    mj = json.load(open(os.path.join(BASE, "results", f"metrics_{args.tag}.json")))
    bp = mj["best_params_random"]
    rf_params = {k: bp[k] for k in ("n_estimators", "max_depth", "min_samples_split")}

    rows = []
    csv_f = open(csv_path, "w")
    csv_f.write(HEADER + "\n")

    def emit(row):
        rows.append(row)
        csv_f.write(",".join(str(row[k]) for k in HEADER.split(",")) + "\n")
        csv_f.flush()

    print(f"[{args.tag}] seeds={seeds} splits={splits} control_seed={args.control_seed}")
    for split in splits:
        sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                                  f"{args.tag}_{split}.npz"))
        train_idx, valid_idx, test_idx = sp["train_idx"], sp["valid_idx"], sp["test_idx"]
        pool = np.concatenate([train_idx, valid_idx])
        print(f"[{args.tag}/{split}] pool={len(pool)} test={len(test_idx)}")

        # ---- control: real labels ----
        t0 = time.time()
        rf = RandomForestRegressor(random_state=args.control_seed, n_jobs=-1, **rf_params)
        rf.fit(X[train_idx], y[train_idx])
        m = metrics(y[test_idx], rf.predict(X[test_idx]))
        print(f"  CONTROL real RF   R2={m['r2']:.4f} ({time.time()-t0:.0f}s)")
        emit({"split": split, "label": "real", "seed": args.control_seed,
              "model": "rf", **m, "best_epoch": -1,
              "train_seconds": round(time.time() - t0, 1)})
        t0 = time.time()
        m, be = gin_train_eval(graphs, train_idx, valid_idx, test_idx, y, args,
                               args.control_seed)
        print(f"  CONTROL real GIN  R2={m['r2']:.4f} (best epoch {be}, {time.time()-t0:.0f}s)")
        emit({"split": split, "label": "real", "seed": args.control_seed,
              "model": "gin", **m, "best_epoch": be,
              "train_seconds": round(time.time() - t0, 1)})

        # ---- shuffled labels ----
        for seed in seeds:
            rng = np.random.RandomState(seed)
            permuted = y.copy()
            permuted[pool] = y[rng.permutation(pool)]
            t0 = time.time()
            rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **rf_params)
            rf.fit(X[train_idx], permuted[train_idx])
            m = metrics(y[test_idx], rf.predict(X[test_idx]))
            print(f"  shuffled seed={seed} RF   R2={m['r2']:.4f} ({time.time()-t0:.0f}s)")
            emit({"split": split, "label": "shuffled", "seed": seed,
                  "model": "rf", **m, "best_epoch": -1,
                  "train_seconds": round(time.time() - t0, 1)})
            t0 = time.time()
            m, be = gin_train_eval(graphs, train_idx, valid_idx, test_idx,
                                   permuted, args, seed)
            print(f"  shuffled seed={seed} GIN  R2={m['r2']:.4f} (best epoch {be}, {time.time()-t0:.0f}s)")
            emit({"split": split, "label": "shuffled", "seed": seed,
                  "model": "gin", **m, "best_epoch": be,
                  "train_seconds": round(time.time() - t0, 1)})
    csv_f.close()

    summary = {}
    for r in rows:
        key = f"{r['split']}/{r['model']}/{r['label']}"
        summary.setdefault(key, []).append(r["r2"])
    summary = {k: {"r2_values": [round(v, 4) for v in vs],
                   "r2_mean": float(np.mean(vs)),
                   "r2_std": float(np.std(vs)),
                   "runs": len(vs)}
               for k, vs in sorted(summary.items())}
    flagged = [k for k, v in summary.items()
               if "shuffled" in k and v["r2_mean"] >= 0.2]
    json.dump({"tag": args.tag, "seeds": seeds, "rf_params": rf_params,
               "groups": summary,
               "verdict": "LEAKAGE SUSPECTED: " + ", ".join(flagged) if flagged
               else "ok: all shuffled-label means < 0.2"},
              open(sum_path, "w"), indent=2)
    print(f"summary -> {os.path.relpath(sum_path, BASE)}")

    make_figure(csv_path, fig_path, args.tag)
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")
    print(f"csv     -> {os.path.relpath(csv_path, BASE)} ({len(rows)} rows)")
    if flagged:
        print("WARNING: shuffled-label R2 >= 0.2 detected -> " + ", ".join(flagged))
        sys.exit(1)


if __name__ == "__main__":
    main()
