"""GNN step 7: multi-task kinase pooling (experiment C).

Pool egfr_full + abl1 + mapk14 into one dataset and train a multi-task GIN
(shared trunk + one linear head per target), testing whether the
"multi-task + large data" industrial setting lets the GNN overtake the
single-task RF baseline that the single-task experiments did not beat.

Data
    Pools data/processed/{egfr_full,abl1,mapk14}_graphs.npz.  Node ids in
    edge_index are molecule-local (verified for these caches, same format as
    gnn_03), so concatenation needs no offset shifting - plain CSR
    concatenation.  Each molecule carries a task index 0/1/2.

Splits
    Uses each tag's persisted split npz. Train = union of the three
    train_idx (with task labels); valid/test stay per-target.  y is
    standardised with each target's own train statistics; metrics are
    reported back in raw pIC50 units per target.

Model (same trunk recipe as gnn_03)
    AtomEncoder -> 4 x [GINConv -> BN -> ReLU -> dropout, residual]
    -> global mean pool (hidden 128, dropout 0.2)
    -> per-target linear head (multi-task) OR one shared linear head
    (pooled single-task negative control, ignores the task label).

Training
    Each step samples a task uniformly, then a batch from that task's
    train molecules; loss = MSE on that task (standardised units).
    Early stopping + ReduceLROnPlateau on the MEAN of the three targets'
    valid RMSE (standardised units).  epochs/patience match the main
    experiments (100/20).

Modes
    --model {mt,pooled} --split {random,scaffold} --seed S   train one run
    --aggregate     build results/multitask/summary.csv +
                    figures/multitask/{split}.png from mt_metrics*,
                    pooled_st_metrics*, gnn_metrics*, comparison_*

Outputs (per run)
    {outdir}/mt_metrics{,_seed1,_seed2}.json   (or pooled_st_metrics*)
    {outdir}/{model}_preds_{split}{suffix}.npz
    {outdir}/models/{model}_{split}{suffix}.pt

Seeds follow the gnn_03 convention: 42 -> base file, 1 -> _seed1, 2 -> _seed2.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_graph_dataset import ATOM_FEATURE_DIMS  # noqa: E402

TASK_TAGS = ["egfr_full", "abl1", "mapk14"]
METHODS = ["st_rf", "st_gin", "pooled_st", "mt_gin"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mt", "pooled"],
                   help="mt = multi-task heads, pooled = single shared head "
                        "negative control (ignored with --aggregate)")
    p.add_argument("--split", choices=["random", "scaffold"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", type=int, default=3,
                   help="use only the first N targets (smoke tests)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--outdir", default=os.path.join(BASE, "results", "multitask"))
    p.add_argument("--figdir", default=os.path.join(BASE, "figures", "multitask"))
    p.add_argument("--aggregate", action="store_true",
                   help="build summary.csv + figures from existing metric files")
    return p.parse_args()


def seed_suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


class AtomEncoder(nn.Module):
    """Sum of per-feature embeddings (PyG AtomEncoder equivalent)."""

    def __init__(self, hidden):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(d, hidden) for d in ATOM_FEATURE_DIMS])
        for e in self.embs:
            nn.init.xavier_uniform_(e.weight.data)

    def forward(self, x):
        return sum(self.embs[k](x[:, k]) for k in range(x.shape[1]))


class Trunk(nn.Module):
    """gnn_03 trunk: AtomEncoder + N x [GINConv->BN->ReLU->dropout, residual]
    + global mean pool."""

    def __init__(self, hidden=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = AtomEncoder(hidden)
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, data):
        h = self.encoder(data.x)
        for conv, bn in zip(self.convs, self.bns):
            h_new = nn.functional.relu(bn(conv(h, data.edge_index)))
            h_new = nn.functional.dropout(h_new, self.dropout, training=self.training)
            h = h + h_new
        return global_mean_pool(h, data.batch)


class PooledGIN(nn.Module):
    """Shared trunk + per-task linear heads (mt) or one shared head (pooled)."""

    def __init__(self, n_tasks, shared_head=False, hidden=128, num_layers=4,
                 dropout=0.2):
        super().__init__()
        self.trunk = Trunk(hidden, num_layers, dropout)
        n_heads = 1 if shared_head else n_tasks
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])
        self.shared_head = shared_head

    def forward(self, data, task=0):
        hg = self.trunk(data)
        head = self.heads[0] if self.shared_head else self.heads[task]
        return head(hg).squeeze(-1)


def load_pool(tags):
    """CSR-concatenate the per-tag graph caches (molecule-local edge ids)."""
    keys = ["node_feat", "edge_index", "edge_feat", "n_nodes", "n_edges"]
    cols = {k: [] for k in keys}
    ys, smis, tasks, offsets, off = [], [], [], [], 0
    for t_idx, tag in enumerate(tags):
        d = np.load(os.path.join(BASE, "data", "processed", f"{tag}_graphs.npz"))
        n = len(d["y"])
        offsets.append(off)
        off += n
        for k in keys:
            cols[k].append(d[k])
        ys.append(d["y"].astype(np.float32))
        smis.append(d["smiles"].astype(str))
        tasks.append(np.full(n, t_idx, dtype=np.int64))
    arrays = {
        "node_feat": np.concatenate(cols["node_feat"]),
        "edge_index": np.concatenate(cols["edge_index"], axis=1),
        "edge_feat": np.concatenate(cols["edge_feat"]),
        "n_nodes": np.concatenate(cols["n_nodes"]),
        "n_edges": np.concatenate(cols["n_edges"]),
        "y": np.concatenate(ys),
        "smiles": np.concatenate(smis),
        "task": np.concatenate(tasks),
        "offsets": offsets,               # pooled index of tag i's raw index 0
    }
    arrays["node_ptr"] = np.concatenate([[0], np.cumsum(arrays["n_nodes"])])
    arrays["edge_ptr"] = np.concatenate([[0], np.cumsum(arrays["n_edges"])])
    return arrays


class PooledGraphDataset(torch.utils.data.Dataset):
    """Subset view over the pooled CSR arrays.

    edge_index node ids are molecule-local, so per-molecule slicing needs no
    offset shifting.  y is standardised per task with the given train stats.
    """

    def __init__(self, arrays, indices=None, y_mean=None, y_std=None):
        self.a = arrays
        self.indices = (np.arange(len(arrays["y"])) if indices is None
                        else np.asarray(indices))
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        g = int(self.indices[i])
        a = self.a
        ns, ne = int(a["node_ptr"][g]), int(a["node_ptr"][g + 1])
        es, ee = int(a["edge_ptr"][g]), int(a["edge_ptr"][g + 1])
        y = float(a["y"][g])
        if self.y_mean is not None:
            t = int(a["task"][g])
            y = (y - float(self.y_mean[t])) / float(self.y_std[t])
        return Data(
            x=torch.from_numpy(a["node_feat"][ns:ne].astype(np.int64)),
            edge_index=torch.from_numpy(a["edge_index"][:, es:ee].astype(np.int64)),
            edge_attr=torch.from_numpy(a["edge_feat"][es:ee].astype(np.int64)),
            y=torch.tensor([y], dtype=torch.float32),
        )


@torch.no_grad()
def predict(model, loader, device, task):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch, task).cpu())
        ys.append(batch.y.cpu())
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_run(args):
    if args.model is None or args.split is None:
        raise SystemExit("--model and --split are required (unless --aggregate)")
    tags = TASK_TAGS[:args.tasks]
    n_tasks = len(tags)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ---- splits & per-target train statistics ----------------------------
    split_idx, y_mean, y_std = {}, np.zeros(n_tasks), np.zeros(n_tasks)
    for i, t in enumerate(tags):
        split_idx[t] = np.load(os.path.join(BASE, "data", "processed", "splits",
                                            f"{t}_{args.split}.npz"))
        y_all = np.load(os.path.join(BASE, "data", "processed",
                                     f"{t}_graphs.npz"))["y"]
        tr = split_idx[t]["train_idx"]
        y_mean[i], y_std[i] = float(y_all[tr].mean()), float(y_all[tr].std())

    print(f"[{args.model}/{args.split}/seed{args.seed}] tasks={tags} "
          f"device={device}", flush=True)
    for i, t in enumerate(tags):
        print(f"  {t}: y_mean={y_mean[i]:.3f} y_std={y_std[i]:.3f} "
              f"n_train={len(split_idx[t]['train_idx'])} "
              f"n_valid={len(split_idx[t]['valid_idx'])} "
              f"n_test={len(split_idx[t]['test_idx'])}", flush=True)

    arrays = load_pool(tags)
    off = arrays["offsets"]
    train_idx_t = [split_idx[t]["train_idx"] + off[i] for i, t in enumerate(tags)]
    valid_idx_t = [split_idx[t]["valid_idx"] + off[i] for i, t in enumerate(tags)]
    test_idx_t = [split_idx[t]["test_idx"] + off[i] for i, t in enumerate(tags)]

    def view(indices):
        return PooledGraphDataset(arrays, indices=indices,
                                  y_mean=y_mean, y_std=y_std)

    def make_train_ld(task_i, epoch_seed):
        g = torch.Generator().manual_seed(epoch_seed)
        return DataLoader(view(train_idx_t[task_i]), batch_size=args.batch_size,
                          shuffle=True, generator=g)

    train_iters = [iter(make_train_ld(i, args.seed + i)) for i in range(n_tasks)]
    valid_lds = [DataLoader(view(idx), batch_size=512) for idx in valid_idx_t]
    test_lds = [DataLoader(view(idx), batch_size=512) for idx in test_idx_t]

    total_train = sum(len(i) for i in train_idx_t)
    steps_per_epoch = math.ceil(total_train / args.batch_size)
    print(f"  pooled train={total_train} (per-task "
          f"{[len(i) for i in train_idx_t]}), {steps_per_epoch} steps/epoch",
          flush=True)

    model = PooledGIN(n_tasks, shared_head=(args.model == "pooled"),
                      hidden=args.hidden, num_layers=args.num_layers,
                      dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-5)

    rng = np.random.default_rng(args.seed)
    best = {"rmse": float("inf"), "state": None, "epoch": -1, "valid": None}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for _ in range(steps_per_epoch):
            task = int(rng.integers(n_tasks))          # uniform task per step
            try:
                batch = next(train_iters[task])
            except StopIteration:                      # reshuffle that task
                train_iters[task] = iter(make_train_ld(
                    task, args.seed + task + 1000 * epoch))
                batch = next(train_iters[task])
            batch = batch.to(device)
            pred = model(batch, task)
            loss = nn.functional.mse_loss(pred, batch.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(batch.y)
            n += len(batch.y)
        tr_loss = total / n

        # validation: per-target RMSE, standardised units + raw pIC50
        v_rmse_std, v_rmse_raw = [], []
        for i in range(n_tasks):
            vp, vy = predict(model, valid_lds[i], device, i)
            s = float(np.sqrt(np.mean((vp - vy) ** 2)))
            v_rmse_std.append(s)
            v_rmse_raw.append(s * y_std[i])
        v_mean = float(np.mean(v_rmse_std))
        sched.step(v_mean)
        lr_now = opt.param_groups[0]["lr"]
        marker = ""
        if v_mean < best["rmse"] - 1e-5:
            best = {"rmse": v_mean, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch, "valid": list(v_rmse_std)}
            marker = " *"
        if epoch % 10 == 0 or marker:
            per = "  ".join(f"{t}:std={s:.3f}/raw={r:.3f}"
                            for t, s, r in zip(tags, v_rmse_std, v_rmse_raw))
            print(f"  epoch {epoch:3d}  train_mse={tr_loss:.4f}  "
                  f"mean_valid_rmse_std={v_mean:.4f}  lr={lr_now:.1e}{marker}\n"
                  f"           {per}", flush=True)
        if epoch - best["epoch"] >= args.patience:
            print(f"  early stop at epoch {epoch} (best epoch {best['epoch']})",
                  flush=True)
            break

    model.load_state_dict(best["state"])

    # ---- per-target test metrics in raw pIC50 units ----------------------
    per_task, preds_out = {}, {}
    for i, t in enumerate(tags):
        tp_s, ty_s = predict(model, test_lds[i], device, i)
        tp = tp_s * y_std[i] + y_mean[i]
        ty = ty_s * y_std[i] + y_mean[i]
        m = metrics(ty, tp)
        m["best_epoch"] = best["epoch"]
        m["best_valid_rmse_std_mean"] = best["rmse"]
        m["best_valid_rmse_std_by_task"] = {
            tt: float(s) for tt, s in zip(tags, best["valid"])}
        m["best_valid_rmse_raw"] = float(best["valid"][i] * y_std[i])
        per_task[t] = m
        preds_out[f"test_idx_{t}"] = split_idx[t]["test_idx"]
        preds_out[f"y_true_{t}"] = ty.astype(np.float32)
        preds_out[f"y_pred_{t}"] = tp.astype(np.float32)
        print(f"  TEST {t:9s} R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}", flush=True)
    print(f"  ({time.time() - t0:.0f}s, params={n_params})", flush=True)

    # ---- persist ----------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)
    sfx = seed_suffix(args.seed)
    prefix = "mt_metrics" if args.model == "mt" else "pooled_st_metrics"
    params = {k: v for k, v in vars(args).items() if k != "aggregate"}
    payload = {args.split: {**per_task, "_run": {
        "model": args.model, "seed": args.seed, "tags": tags,
        "params": params, "train_seconds": round(time.time() - t0, 1)}}}
    mj = os.path.join(args.outdir, f"{prefix}{sfx}.json")
    all_m = json.load(open(mj)) if os.path.exists(mj) else {}
    all_m.update(payload)
    json.dump(all_m, open(mj, "w"), indent=2)

    np.savez(os.path.join(args.outdir,
                          f"{args.model}_preds_{args.split}{sfx}.npz"), **preds_out)
    torch.save({"state_dict": best["state"], "args": params,
                "y_mean": y_mean.tolist(), "y_std": y_std.tolist(), "tags": tags},
               os.path.join(args.outdir, "models",
                            f"{args.model}_{args.split}{sfx}.pt"))
    print(f"  saved -> {os.path.relpath(mj, BASE)}, "
          f"{args.model}_preds_{args.split}{sfx}.npz, "
          f"models/{args.model}_{args.split}{sfx}.pt", flush=True)


# --------------------------------------------------------------------------
# aggregation: summary.csv + figures
# --------------------------------------------------------------------------
def _load_seeds(path_fmt):
    """Load metric files for seeds 42/1/2 that exist -> list of dicts."""
    out = []
    for sfx in ("", "_seed1", "_seed2"):
        path = path_fmt.format(sfx=sfx)
        if os.path.exists(path):
            out.append(json.load(open(path)))
    return out


def _collect_r2(files, split, tag=None):
    vals = []
    for f in files:
        node = f.get(split)
        if node is None:
            continue
        if tag is None:
            vals.append(node["r2"])
        elif tag in node:
            vals.append(node[tag]["r2"])
    return vals


def aggregate(args):
    tags = TASK_TAGS[:args.tasks]
    rows, table = [], []
    for tag in tags:
        rf = {}
        with open(os.path.join(BASE, "results", f"comparison_{tag}.csv")) as fh:
            for row in csv.DictReader(fh):
                rf[row["split"]] = float(row["rf_r2"])
        st_files = _load_seeds(os.path.join(
            BASE, "results", f"gnn_metrics_{tag}{{sfx}}.json"))
        mt_files = _load_seeds(os.path.join(args.outdir, "mt_metrics{sfx}.json"))
        po_files = _load_seeds(os.path.join(args.outdir,
                                            "pooled_st_metrics{sfx}.json"))
        for split in ("random", "scaffold"):
            per_method = {
                "st_rf": [rf[split]],
                "st_gin": _collect_r2(st_files, split),
                "pooled_st": _collect_r2(po_files, split, tag),
                "mt_gin": _collect_r2(mt_files, split, tag),
            }
            rf_mean = per_method["st_rf"][0]
            for method in METHODS:
                vals = per_method[method]
                if vals:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals)) if len(vals) > 1 else 0.0
                else:
                    mean, std = float("nan"), float("nan")
                rows.append({"task": tag, "split": split, "method": method,
                             "r2_mean": mean, "r2_std": std,
                             "n_seeds": len(vals),
                             "delta_vs_st_rf": mean - rf_mean})
            table.append((tag, split, per_method, rf_mean))

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "split", "method",
                                           "r2_mean", "r2_std", "n_seeds",
                                           "delta_vs_st_rf"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {os.path.relpath(out_csv, BASE)}")

    print(f"\n{'task':10s} {'split':9s} " +
          " ".join(f"{m:>24s}" for m in METHODS))
    for tag, split, per_method, _ in table:
        cells = []
        for m in METHODS:
            vals = per_method[m]
            if vals:
                mu, sd = np.mean(vals), (np.std(vals) if len(vals) > 1 else 0.0)
                cells.append(f"{mu:.3f}±{sd:.3f} (Δ{mu - per_method['st_rf'][0]:+.3f})")
            else:
                cells.append("n/a")
        print(f"{tag:10s} {split:9s} " + " ".join(f"{c:>24s}" for c in cells))

    # figures: one grouped bar chart per split
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.figdir, exist_ok=True)
    colors = {"st_rf": "#4C72B0", "st_gin": "#DD8452",
              "pooled_st": "#55A868", "mt_gin": "#C44E52"}
    for split in ("random", "scaffold"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(tags))
        width = 0.2
        lo = 0.0
        for k, method in enumerate(METHODS):
            means, stds = [], []
            for tag in tags:
                r = next(r for r in rows if r["task"] == tag
                         and r["split"] == split and r["method"] == method)
                means.append(r["r2_mean"])
                stds.append(r["r2_std"] if r["n_seeds"] > 1 else 0.0)
            arr = np.array(means) - np.array(stds)
            if not np.all(np.isnan(arr)):
                lo = min(lo, float(np.nanmin(arr)) - 0.05)
            pos = x + (k - 1.5) * width
            ax.bar(pos, means, width, yerr=stds, capsize=3,
                   label=method, color=colors[method],
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(tags)
        ax.set_ylabel("test R2 (3-seed mean ± std)")
        ax.set_title(f"Multi-task kinase pooling vs baselines ({split} split)")
        ax.legend(ncols=4, fontsize=9)
        ax.set_ylim(lo, 0.9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fp = os.path.join(args.figdir, f"{split}.png")
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"saved -> {os.path.relpath(fp, BASE)}")


def main():
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        train_run(args)


if __name__ == "__main__":
    main()
