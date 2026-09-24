"""GNN step 3: train a GIN baseline on the molecular graphs.

Model (MoleculeNet-style GIN):
    AtomEncoder (sum of 9 per-feature embeddings)
    -> num_layers x [GINConv -> BatchNorm -> ReLU -> dropout, residual]
    -> global mean pool -> MLP head -> pIC50

Notes on the choices:
  * bond features are NOT used: plain GINConv aggregates node features
    only (GINEConv would take edges). Keeps the baseline canonical.
  * mean pooling instead of the paper's sum pooling: molecule sizes vary
    3-100 atoms here; sum pooling couples prediction scale to size.
  * y is standardised with TRAIN mean/std for optimisation stability and
    mapped back before metrics, so R2/RMSE stay directly comparable to RF.

Early stopping on validation RMSE (patience configurable); the best
checkpoint (in memory) is restored before the test evaluation.

Outputs:
  results/gnn_models/{TAG}_gin_{split}.pt   best model state_dict
  results/gnn_preds_{TAG}_{split}.npz       test_idx, y_true, y_pred
  results/gnn_metrics_{TAG}.json            updated {split: {r2, rmse, mae,
                                            params, best_epoch}}
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_graph_dataset import MoleculeGraphDataset, ATOM_FEATURE_DIMS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--split", required=True, choices=["random", "scaffold"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suffix", default="",
                   help="suffix for model/preds/metrics files (used to keep "
                        "hyperparameter-search runs separate from the final ones)")
    return p.parse_args()


class AtomEncoder(nn.Module):
    """Sum of per-feature embeddings (PyG AtomEncoder equivalent)."""

    def __init__(self, hidden):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(d, hidden) for d in ATOM_FEATURE_DIMS])
        for e in self.embs:
            nn.init.xavier_uniform_(e.weight.data)

    def forward(self, x):
        return sum(self.embs[k](x[:, k]) for k in range(x.shape[1]))


class GINRegressor(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        h = self.encoder(data.x)
        for conv, bn in zip(self.convs, self.bns):
            h_new = nn.functional.relu(bn(conv(h, data.edge_index)))
            h_new = nn.functional.dropout(h_new, self.dropout, training=self.training)
            h = h + h_new  # residual
        hg = global_mean_pool(h, data.batch)
        return self.head(hg).squeeze(-1)


def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = nn.functional.mse_loss(pred, batch.y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += float(loss) * len(batch.y)
            n += len(batch.y)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu())
        ys.append(batch.y.cpu())
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    split = np.load(os.path.join(BASE, "data", "processed", "splits",
                                 f"{args.tag}_{args.split}.npz"))
    graphs = os.path.join(BASE, "data", "processed", f"{args.tag}_graphs.npz")

    y_all = np.load(graphs)["y"]
    y_mean, y_std = float(y_all[split["train_idx"]].mean()), float(y_all[split["train_idx"]].std())
    print(f"[{args.tag}/{args.split}] device={device}  y_mean={y_mean:.3f} y_std={y_std:.3f}")

    train_ds = MoleculeGraphDataset(graphs, indices=split["train_idx"])
    valid_ds = MoleculeGraphDataset(graphs, indices=split["valid_idx"])
    test_ds = MoleculeGraphDataset(graphs, indices=split["test_idx"])
    # standardise targets in-place (cache arrays are copies loaded per dataset)
    for ds, mean, std in ((train_ds, y_mean, y_std), (valid_ds, y_mean, y_std), (test_ds, y_mean, y_std)):
        ds.y = (ds.y - y_mean) / y_std

    g = torch.Generator().manual_seed(args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    valid_ld = DataLoader(valid_ds, batch_size=512)
    test_ld = DataLoader(test_ds, batch_size=512)

    model = GINRegressor(args.hidden, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-5)

    best = {"rmse": float("inf"), "state": None, "epoch": -1}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_ld, device, opt)
        vp, vy = predict(model, valid_ld, device)
        v_rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))
        sched.step(v_rmse)
        lr_now = opt.param_groups[0]["lr"]
        marker = ""
        if v_rmse < best["rmse"] - 1e-5:
            best = {"rmse": v_rmse,
                    "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            marker = " *"
        if epoch % 10 == 0 or marker:
            print(f"  epoch {epoch:3d}  train_mse={tr_loss:.4f}  valid_rmse={v_rmse:.4f}"
                  f"  lr={lr_now:.1e}{marker}")
        if epoch - best["epoch"] >= args.patience:
            print(f"  early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    tp_raw, ty_raw = predict(model, test_ld, device)
    tp, ty = tp_raw * y_std + y_mean, ty_raw * y_std + y_mean  # back to pIC50 units
    m = metrics(ty, tp)
    print(f"  TEST  R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
          f"({time.time() - t0:.0f}s)")

    os.makedirs(os.path.join(BASE, "results", "gnn_models"), exist_ok=True)
    torch.save({"state_dict": best["state"], "args": vars(args),
                "y_mean": y_mean, "y_std": y_std},
               os.path.join(BASE, "results", "gnn_models",
                            f"{args.tag}_gin_{args.split}{args.suffix}.pt"))
    np.savez(os.path.join(BASE, "results", f"gnn_preds_{args.tag}_{args.split}{args.suffix}.npz"),
             test_idx=split["test_idx"], y_true=ty, y_pred=tp)

    mj = os.path.join(BASE, "results", f"gnn_metrics_{args.tag}{args.suffix}.json")
    all_m = json.load(open(mj)) if os.path.exists(mj) else {}
    all_m[args.split] = {**m, "best_epoch": best["epoch"],
                         "best_valid_rmse_std": best["rmse"],
                         "params": {k: v for k, v in vars(args).items() if k != "split"}}
    json.dump(all_m, open(mj, "w"), indent=2)
    print(f"  saved -> results/gnn_models/{args.tag}_gin_{args.split}{args.suffix}.pt, "
          f"results/gnn_preds_{args.tag}_{args.split}{args.suffix}.npz, "
          f"{os.path.relpath(mj, BASE)}")


if __name__ == "__main__":
    main()
