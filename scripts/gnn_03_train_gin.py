"""GNN step 3: train a GIN / GINE / AttentiveFP model on the molecular graphs.

Models (--model, default "gin"):
    gin (MoleculeNet-style GIN, the baseline):
        AtomEncoder (sum of 9 per-feature embeddings)
        -> num_layers x [GINConv -> BatchNorm -> ReLU -> dropout, residual]
        -> global mean pool -> MLP head -> pIC50
    gine: same shell, GINEConv consumes BondEncoder(data.edge_attr)
    attentivefp: PyG AttentiveFP on top of AtomEncoder + BondEncoder
        (no residual/BN shell - it has its own GRU readout)

Notes on the choices:
  * the default gin does NOT use bond features: plain GINConv aggregates
    node features only, keeping the baseline canonical; --model gine /
    attentivefp feed the cached 3-dim bond features through BondEncoder.
  * mean pooling instead of the paper's sum pooling: molecule sizes vary
    3-100 atoms here; sum pooling couples prediction scale to size.
  * y is standardised with TRAIN mean/std for optimisation stability and
    mapped back before metrics, so R2/RMSE stay directly comparable to RF.

Early stopping on validation RMSE (patience configurable); the best
checkpoint (in memory) is restored before the test evaluation.
--min-epochs N defers the early-stop check until epoch N (default 0 =
behaviour unchanged), so a stalled run cannot stop before epoch N.

Convergence check: after training, a run whose best epoch is below 30%
of --epochs while its train MSE is still > 0.5 (standardised y, i.e. it
explains less than half the train variance) prints a prominent WARNING
and records convergence_flag=true + best_epoch/max_epochs/train_mse in
the metrics json (see convergence_flag()).

The metrics json also records the device the run actually trained on (the
select_device() result, e.g. 'cuda' / 'mps' / 'cpu') as `device`, at the
same level as convergence_flag; metrics files written by runs before this
field existed do not have it.
--device {auto,cpu,mps,cuda} pins the backend; an unavailable explicitly
requested accelerator fails before loading data or writing artifacts.
Runtime metadata records the seed, PyTorch version and thread counts.

Outputs (model name in the filename only when it is not the default gin,
so existing gin artifacts and commands keep their exact names):
  results/gnn_models/{TAG}_{model}_{split}{suffix}.pt
  results/gnn_preds_{TAG}[_{model}]_{split}{suffix}.npz
  results/gnn_metrics_{TAG}[_{model}]{suffix}.json   updated
      {split: {r2, rmse, mae, params, best_epoch,
               convergence_flag, max_epochs, train_mse, device}}
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from time_labels import load_time_split
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP, GINEConv, GINConv, global_mean_pool

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_graph_dataset import (  # noqa: E402
    ATOM_FEATURE_DIMS,
    BOND_FEATURE_DIMS,
    MoleculeGraphDataset,
)
from qsar_common import select_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--split", required=True, choices=["random", "scaffold", "time"])
    p.add_argument("--model", default="gin", choices=["gin", "gine", "attentivefp"],
                   help="architecture (default gin keeps the baseline artifact names)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=0,
                   help="defer early stopping until this epoch (0 = off); "
                        "a stalled run cannot stop before it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                   help="execution backend; explicit accelerators never fall back to CPU")
    p.add_argument("--suffix", default="",
                   help="suffix for model/preds/metrics files (used to keep "
                        "hyperparameter-search runs separate from the final ones)")
    return p.parse_args()


def resolve_device(requested):
    """Keep legacy auto selection, but fail closed for pinned accelerators."""
    if requested == "auto":
        return select_device()
    if requested == "cpu":
        return requested
    available = (torch.cuda.is_available() if requested == "cuda"
                 else torch.backends.mps.is_available() if requested == "mps"
                 else False)
    if not available:
        raise ValueError(f"requested device {requested!r} is unavailable; no CPU fallback")
    return requested


def configure_torch_threads():
    """Apply explicit process thread limits; unset variables preserve defaults."""
    settings = []
    for name, setter in (("QSAR_TORCH_THREADS", torch.set_num_threads),
                         ("QSAR_TORCH_INTEROP_THREADS", torch.set_num_interop_threads)):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be a positive integer") from None
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        settings.append((setter, value))
    # Validate both settings before changing either process-wide thread pool.
    for setter, value in settings:
        setter(value)


def runtime_metadata(args, device):
    return {"requested_device": args.device, "device": device, "seed": args.seed,
            "torch_version": str(torch.__version__),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


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


class BondEncoder(nn.Module):
    """Sum of per-feature embeddings for the 3 bond features (OGB style).

    BOND_FEATURE_DIMS are the exact embedding-table sizes: the featurizer
    clamps every index into [0, dim-1], so max+1 <= dim by construction.
    """

    def __init__(self, hidden):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(d, hidden) for d in BOND_FEATURE_DIMS])
        for e in self.embs:
            nn.init.xavier_uniform_(e.weight.data)

    def forward(self, e):
        return sum(self.embs[k](e[:, k]) for k in range(e.shape[1]))


class GINERegressor(nn.Module):
    """GIN with bond features: GINEConv(edge_dim=hidden), same shell as GIN.

    Identical to GINRegressor (AtomEncoder -> [conv, BN, ReLU, dropout,
    residual] x num_layers -> mean pool -> MLP head) except the conv
    consumes BondEncoder(edge_attr) through GINEConv's edge linear.
    """

    def __init__(self, hidden=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = AtomEncoder(hidden)
        self.bond_encoder = BondEncoder(hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                edge_dim=hidden)
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
        e = self.bond_encoder(data.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h_new = nn.functional.relu(bn(conv(h, data.edge_index, e)))
            h_new = nn.functional.dropout(h_new, self.dropout, training=self.training)
            h = h + h_new  # residual
        hg = global_mean_pool(h, data.batch)
        return self.head(hg).squeeze(-1)


class AttentiveFPRegressor(nn.Module):
    """PyG AttentiveFP on top of AtomEncoder + BondEncoder front-ends.

    AttentiveFP has its own GRU/attention readout (no residual/BN shell),
    so the embedding front-end is applied and the raw model is used as-is.
    """

    def __init__(self, hidden=128, num_layers=4, dropout=0.2, num_timesteps=30):
        super().__init__()
        self.encoder = AtomEncoder(hidden)
        self.bond_encoder = BondEncoder(hidden)
        self.fp = AttentiveFP(in_channels=hidden, hidden_channels=hidden,
                              out_channels=1, edge_dim=hidden,
                              num_layers=num_layers, num_timesteps=num_timesteps,
                              dropout=dropout)

    def forward(self, data):
        x = self.encoder(data.x)
        e = self.bond_encoder(data.edge_attr)
        out = self.fp(x=x, edge_index=data.edge_index, edge_attr=e, batch=data.batch)
        return out.squeeze(-1)


MODELS = {"gin": GINRegressor, "gine": GINERegressor, "attentivefp": AttentiveFPRegressor}


def build_model(args):
    return MODELS[args.model](args.hidden, args.num_layers, args.dropout)


# Convergence-warning thresholds (train MSE on standardised y, so 1.0 =
# no-skill and 0.5 = half the train variance explained). Healthy runs in
# the campaigns end near 0.24-0.31; the stalled abl1/scaffold seed-42 run
# stopped at best_epoch 15/100 with train MSE 0.57.
CONVERGENCE_EPOCH_FRAC = 0.3   # best epoch below this fraction of max epochs
CONVERGENCE_TRAIN_MSE = 0.5    # train MSE still above this = fitting poorly


def convergence_flag(best_epoch, max_epochs, train_mse):
    """Flag an early best checkpoint with high training error for investigation.

    Both conditions must hold: best_epoch < 30% of max_epochs AND the
    best-checkpoint eval-mode train MSE > 0.5. A run that trains to a late best epoch
    is never flagged, no matter its train MSE.
    """
    return bool(best_epoch < CONVERGENCE_EPOCH_FRAC * max_epochs
                and train_mse > CONVERGENCE_TRAIN_MSE)


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
    configure_torch_threads()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    runtime = runtime_metadata(args, device)

    split_dir = os.path.join(BASE, "data", "processed", "splits")
    graphs = os.path.join(BASE, "data", "processed", f"{args.tag}_graphs.npz")
    with np.load(graphs) as graph_cache:
        y_all = graph_cache["y"]
        if args.split == "time":
            split = load_time_split(split_dir, args.tag, graph_cache["smiles"], y_all)
            if split is None:
                return
            y_all = split["y"]
        else:
            split = dict(np.load(os.path.join(split_dir, f"{args.tag}_{args.split}.npz")))
    label_meta = {"label_id": str(split["label_id"])} if args.split == "time" else {}
    y_mean, y_std = float(y_all[split["train_idx"]].mean()), float(y_all[split["train_idx"]].std())
    print(f"[{args.tag}/{args.split}] device={device}  y_mean={y_mean:.3f} y_std={y_std:.3f}")

    train_ds = MoleculeGraphDataset(graphs, indices=split["train_idx"])
    valid_ds = MoleculeGraphDataset(graphs, indices=split["valid_idx"])
    test_ds = MoleculeGraphDataset(graphs, indices=split["test_idx"])
    # Replace cached all-year targets before standardizing any temporal pool.
    for ds in (train_ds, valid_ds, test_ds):
        ds.y = (y_all.copy() - y_mean) / y_std

    g = torch.Generator().manual_seed(args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    valid_ld = DataLoader(valid_ds, batch_size=512)
    test_ld = DataLoader(test_ds, batch_size=512)

    model = build_model(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-5)

    best = {"rmse": float("inf"), "state": None, "epoch": -1}
    if args.min_epochs > 0:
        print(f"  early stopping deferred until epoch {args.min_epochs} "
              f"(--min-epochs {args.min_epochs})")
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
        if epoch >= args.min_epochs and epoch - best["epoch"] >= args.patience:
            print(f"  early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    last_epoch_train_mse = tr_loss
    model.load_state_dict(best["state"])
    train_eval_ld = DataLoader(train_ds, batch_size=512, shuffle=False)
    best_train_mse = run_epoch(model, train_eval_ld, device)
    flagged = convergence_flag(best["epoch"], args.epochs, best_train_mse)
    if flagged:
        print("=" * 72)
        print(f"WARNING: possible non-convergence for {args.tag}/{args.split} "
              f"({args.model}, seed {args.seed}): best epoch {best['epoch']} is "
              f"below {int(CONVERGENCE_EPOCH_FRAC * 100)}% of max "
              f"{args.epochs} epochs while best-checkpoint eval train MSE is "
              f"{best_train_mse:.4f} (> {CONVERGENCE_TRAIN_MSE}); this "
              f"is a fitting diagnostic, not proof of convergence or a reason "
              f"to select a new run from test scores")
        print("=" * 72)

    tp_raw, ty_raw = predict(model, test_ld, device)
    tp, ty = tp_raw * y_std + y_mean, ty_raw * y_std + y_mean  # back to pIC50 units
    m = metrics(ty, tp)
    print(f"  TEST  R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
          f"({time.time() - t0:.0f}s)")

    # gin (default) keeps the original artifact names: the model name only
    # appears in the filename when it is not the default ("gin").
    model_infix = "" if args.model == "gin" else f"_{args.model}"
    pt_name = f"{args.tag}_{args.model}_{args.split}{args.suffix}.pt"
    preds_name = f"gnn_preds_{args.tag}{model_infix}_{args.split}{args.suffix}.npz"

    os.makedirs(os.path.join(BASE, "results", "gnn_models"), exist_ok=True)
    torch.save({"state_dict": best["state"], "args": vars(args),
                "y_mean": y_mean, "y_std": y_std, "runtime": runtime, **label_meta},
               os.path.join(BASE, "results", "gnn_models", pt_name))
    np.savez(os.path.join(BASE, "results", preds_name),
             test_idx=split["test_idx"], y_true=ty, y_pred=tp, **label_meta)

    mj = os.path.join(BASE, "results",
                      f"gnn_metrics_{args.tag}{model_infix}{args.suffix}.json")
    all_m = json.load(open(mj)) if os.path.exists(mj) else {}
    all_m[args.split] = {**m, **label_meta, "best_epoch": best["epoch"],
                         "best_valid_rmse_std": best["rmse"],
                         "convergence_flag": flagged,
                         "device": device,
                         "runtime": runtime,
                         "max_epochs": args.epochs,
                         "train_mse": best_train_mse,
                         "train_mse_source": "best_checkpoint_eval",
                         "last_epoch_train_mse": last_epoch_train_mse,
                         "params": {k: v for k, v in vars(args).items() if k != "split"}}
    json.dump(all_m, open(mj, "w"), indent=2)
    print(f"  saved -> results/gnn_models/{pt_name}, "
          f"results/{preds_name}, "
          f"{os.path.relpath(mj, BASE)}")


if __name__ == "__main__":
    main()
