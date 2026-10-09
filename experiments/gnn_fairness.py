"""Experiment B: give the GIN a fair chance (bounded tuning + GINE edges).

Question: is "GIN loses to RF" an artefact of the frozen recipe? On the
three targets with the smallest scaffold delta(mean) (vegfr2/VEGFR2
-0.006, herg -0.013, abl1 -0.013) we run a bounded tuning pass and a
GINEConv edge-feature variant, everything else held at gnn_03's defaults.

Protocol (README 7.4 lessons baked in):
  * bounded grid: hidden {128,256} x num_layers {4,5} x dropout {0.2,0.3}
    = 8 configs, trained ONLY on the scaffold split, seeds 42/1/2.
    Config selection = lowest 3-seed MEAN valid RMSE - never a single
    seed (one seed's best-epoch valid RMSE is noise on ~950 val mols).
  * the winning config is then re-run from scratch on both splits x
    3 seeds (fresh runs, never reused tuning runs).
  * GINE variant: GINEConv consumes the cached 3-dim bond features
    (bond type / stereo / conjugated) through a BondEncoder with one
    embedding per dimension, table sizes = BOND_FEATURE_DIMS (exact
    sizes from the featurizer, values are clamped into range there).
    Single config (hidden 128, 4 layers, dropout 0.2), both splits x
    3 seeds. Mean pooling kept (sum pool couples scale to molecule
    size - documented in gnn_03's docstring).
  * frozen everything else, identical to gnn_03_train_gin.py: Adam lr
    1e-3, ReduceLROnPlateau(0.5, patience 7, min_lr 1e-5), early stop
    patience 20 on valid RMSE (checkpoint if improved by >1e-5), 100
    epochs, batch 128, train-pool y standardisation, residual blocks.
  * all reported numbers are 3-seed mean +- std (population std, same
    convention as the rest of the campaign).

Outputs (same paths locally and on m3):
  results/fairness/{TAG}_tuning.csv   one row per config x seed (valid/test)
  results/fairness/summary_{TAG}.json winner, final runs, aggregates, deltas
  results/fairness/summary.csv        tag x model x split R2 3-seed mean+-std
                                      (baseline read from results/gnn_metrics_{TAG}*.json)
  figures/fairness/{TAG}.png          RF vs baseline vs tuned vs GINE bars

Usage:
  python experiments/gnn_fairness.py --tag vegfr2
  plot only (no training): --tag vegfr2 --plot-only
    (rebuilds summary.csv + figures/fairness/{TAG}.png from
    results/fairness/summary_{TAG}.json + the baseline metric files)
  dry run:  --tag abl1 --configs 1 --seeds 42 --epochs 3 --patience 2 \
            --out results/fairness_dry --fig-dir figures/fairness_dry
"""
import argparse
import copy
import csv
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from gnn_03_train_gin import (  # noqa: E402
    AtomEncoder, GINRegressor, metrics, predict, run_epoch,
)
from gnn_graph_dataset import BOND_FEATURE_DIMS, MoleculeGraphDataset  # noqa: E402
from qsar_common import select_device  # noqa: E402

GRID_HIDDEN = (128, 256)
GRID_LAYERS = (4, 5)
GRID_DROPOUT = (0.2, 0.3)
GINE_CONFIG = {"hidden": 128, "num_layers": 4, "dropout": 0.2}
SPLITS = ("random", "scaffold")
MODELS = ("baseline_gin", "tuned_gin", "gine")

TUNING_HEADER = ("hidden,num_layers,dropout,split,seed,valid_rmse,"
                 "valid_rmse_std,test_rmse,test_r2,test_mae,best_epoch,"
                 "train_seconds")
SUMMARY_HEADER = ("tag,model,split,n_seeds,r2_mean,r2_std,rmse_mean,"
                  "rmse_std")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "vegfr2"))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--configs", type=int, default=8,
                   help="cap on grid configs evaluated (dry runs)")
    p.add_argument("--phases", default="tune,final,gine,summary",
                   help="comma list from {tune,final,gine,summary}")
    p.add_argument("--plot-only", action="store_true",
                   help="no training: rebuild summary.csv + the figure from "
                        "summary_{TAG}.json and the baseline metric files "
                        "(equivalent to --phases summary)")
    p.add_argument("--out", default=os.path.join(BASE, "results", "fairness"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "fairness"))
    # frozen gnn_03 defaults (exposed only so dry runs can shorten them)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()
    if args.plot_only:
        args.phases = "summary"
    return args


# ---------------------------------------------------------------- models

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

    Identical to gnn_03's GINRegressor (AtomEncoder -> [conv, BN, ReLU,
    dropout, residual] x num_layers -> mean pool -> MLP head) except the
    conv consumes BondEncoder(edge_attr) through GINEConv's edge linear.
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


# ---------------------------------------------------------------- training

def train_eval(model_cls, model_kwargs, graphs, train_idx, valid_idx,
               test_idx, y_mean, y_std, args, seed):
    """gnn_03's exact recipe; returns valid + test metrics for one run."""
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device()

    ds_tr = MoleculeGraphDataset(graphs, indices=train_idx)
    ds_va = MoleculeGraphDataset(graphs, indices=valid_idx)
    ds_te = MoleculeGraphDataset(graphs, indices=test_idx)
    for ds in (ds_tr, ds_va, ds_te):
        ds.y = (ds.y - y_mean) / y_std

    g = torch.Generator().manual_seed(seed)
    tr_ld = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, generator=g)
    va_ld = DataLoader(ds_va, batch_size=512)
    te_ld = DataLoader(ds_te, batch_size=512)

    model = model_cls(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-5)

    best = {"rmse": float("inf"), "state": None, "epoch": -1}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, tr_ld, device, opt)
        vp, vy = predict(model, va_ld, device)
        v_rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))  # standardised
        sched.step(v_rmse)
        if v_rmse < best["rmse"] - 1e-5:
            best = {"rmse": v_rmse, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch}
        if epoch - best["epoch"] >= args.patience:
            break

    model.load_state_dict(best["state"])
    tp_raw, ty_raw = predict(model, te_ld, device)
    tp, ty = tp_raw * y_std + y_mean, ty_raw * y_std + y_mean
    tm = metrics(ty, tp)
    return {
        "valid_rmse_std": best["rmse"],
        "valid_rmse": best["rmse"] * y_std,  # pIC50 units (y_std of this split)
        "test_rmse": tm["rmse"],
        "test_r2": tm["r2"],
        "test_mae": tm["mae"],
        "best_epoch": best["epoch"],
        "train_seconds": round(time.time() - t0, 1),
        "device": device,
    }


def split_indices(tag, split):
    sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                              f"{tag}_{split}.npz"))
    return sp["train_idx"], sp["valid_idx"], sp["test_idx"]


def run_config(model_cls, model_kwargs, graphs, tag, split, seeds, args):
    """Train one config on one split for every seed; list of run dicts."""
    train_idx, valid_idx, test_idx = split_indices(tag, split)
    y = np.load(graphs)["y"]
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std())
    runs = []
    for seed in seeds:
        r = train_eval(model_cls, model_kwargs, graphs, train_idx, valid_idx,
                       test_idx, y_mean, y_std, args, seed)
        r["seed"] = seed
        runs.append(r)
        print(f"    seed={seed} valid_rmse={r['valid_rmse']:.4f} "
              f"test_rmse={r['test_rmse']:.4f} test_r2={r['test_r2']:.4f} "
              f"(best epoch {r['best_epoch']}, {r['train_seconds']:.0f}s)",
              flush=True)
    return runs


# ---------------------------------------------------------------- phases

def phase_tune(args, state, graphs, tuning_csv):
    """Grid configs on the scaffold split; winner = mean valid RMSE."""
    grid = list(itertools.product(GRID_HIDDEN, GRID_LAYERS, GRID_DROPOUT))
    grid = grid[:args.configs]
    state["grid"] = [dict(zip(("hidden", "num_layers", "dropout"), c))
                     for c in grid]
    seeds = state["seeds"]
    with open(tuning_csv, "w") as f:
        f.write(TUNING_HEADER + "\n")
    rows = []
    for cfg in grid:
        kw = dict(zip(("hidden", "num_layers", "dropout"), cfg))
        print(f"  [tune] {kw}", flush=True)
        runs = run_config(GINRegressor, kw, graphs, args.tag, "scaffold",
                          seeds, args)
        for r in runs:
            row = {**kw, "split": "scaffold", **r}
            rows.append(row)
            with open(tuning_csv, "a") as f:
                f.write(",".join(str(row[k]) for k in TUNING_HEADER.split(","))
                        + "\n")
    # selection: 3-seed mean valid RMSE (raw units; y_std is the same for
    # every config on this split, so the ranking equals standardised units)
    per_cfg = []
    for cfg in grid:
        kw = dict(zip(("hidden", "num_layers", "dropout"), cfg))
        vr = [r["valid_rmse"] for r in rows
              if all(r[k] == kw[k] for k in kw)]
        per_cfg.append({**kw, "valid_rmse_mean": float(np.mean(vr)),
                        "valid_rmse_std": float(np.std(vr)), "runs": len(vr)})
    winner = min(per_cfg, key=lambda c: c["valid_rmse_mean"])
    state["tuning"] = {"split": "scaffold", "configs": per_cfg,
                       "selection": "min 3-seed mean valid RMSE",
                       "winner": winner}
    print(f"  [tune] winner={ {k: winner[k] for k in ('hidden','num_layers','dropout')} } "
          f"mean valid RMSE={winner['valid_rmse_mean']:.4f}", flush=True)


def winner_kwargs(state):
    w = state.get("tuning", {}).get("winner")
    if w is None:
        sys.exit("no winner in summary json - run the 'tune' phase first")
    return {k: w[k] for k in ("hidden", "num_layers", "dropout")}


def phase_final(args, state, graphs):
    """Winner config, fresh runs on both splits x seeds."""
    kw = winner_kwargs(state)
    print(f"  [final] tuned_gin {kw}", flush=True)
    runs = []
    for split in SPLITS:
        print(f"    split={split}", flush=True)
        for r in run_config(GINRegressor, kw, graphs, args.tag, split,
                            state["seeds"], args):
            runs.append({"model": "tuned_gin", "split": split, **r})
    state["final_runs"] = runs


def phase_gine(args, state, graphs):
    """GINEConv + bond features, single config, both splits x seeds."""
    print(f"  [gine] {GINE_CONFIG}", flush=True)
    runs = []
    for split in SPLITS:
        print(f"    split={split}", flush=True)
        for r in run_config(GINERegressor, GINE_CONFIG, graphs, args.tag,
                            split, state["seeds"], args):
            runs.append({"model": "gine", "split": split, **r})
    state["gine_runs"] = runs


# ---------------------------------------------------------------- summary

def _agg(runs):
    r2 = [r["test_r2"] for r in runs]
    rm = [r["test_rmse"] for r in runs]
    return {"r2_mean": float(np.mean(r2)), "r2_std": float(np.std(r2)),
            "rmse_mean": float(np.mean(rm)), "rmse_std": float(np.std(rm)),
            "n_seeds": len(r2)}


def load_baseline(tag):
    """3-seed baseline GIN from results/gnn_metrics_{tag}[_seedN].json."""
    out = {}
    for split in SPLITS:
        runs = []
        for suf in ("", "_seed1", "_seed2"):
            path = os.path.join(BASE, "results", f"gnn_metrics_{tag}{suf}.json")
            m = json.load(open(path))[split]
            runs.append({"test_r2": m["r2"], "test_rmse": m["rmse"]})
        out[split] = _agg(runs)
    return out


def load_rf(tag):
    d = json.load(open(os.path.join(BASE, "results", f"metrics_{tag}.json")))
    return {sp: {"r2": d[sp]["r2"], "rmse": d[sp]["rmse"]} for sp in SPLITS}


def aggregate(state, rf):
    res = {"baseline_gin": load_baseline(state["tag"])}
    for model, key in (("tuned_gin", "final_runs"), ("gine", "gine_runs")):
        runs = state.get(key)
        if not runs:
            continue
        res[model] = {sp: _agg([r for r in runs if r["split"] == sp])
                      for sp in SPLITS}
    state["rf"] = rf
    state["results"] = res
    state["delta_r2_gin_minus_rf"] = {
        m: {sp: res[m][sp]["r2_mean"] - rf[sp]["r2"] for sp in SPLITS}
        for m in res}
    return res


def write_summary_csv(out_dir, summary_files):
    """Rebuild the cross-tag summary.csv from every summary_{tag}.json."""
    rows = []
    for path in sorted(summary_files):
        s = json.load(open(path))
        for model in MODELS:
            for split in SPLITS:
                agg = s.get("results", {}).get(model, {}).get(split)
                if agg is None:
                    continue
                rows.append({"tag": s["tag"], "model": model, "split": split,
                             **{k: agg[k] for k in
                                ("n_seeds", "r2_mean", "r2_std",
                                 "rmse_mean", "rmse_std")}})
    rows.sort(key=lambda r: (r["tag"], MODELS.index(r["model"]),
                             SPLITS.index(r["split"])))
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER.split(","))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"summary.csv -> {os.path.relpath(path, BASE)} ({len(rows)} rows)",
          flush=True)


def make_figure(state, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, rf = state["results"], state["rf"]
    labels = ["RF", "GIN\nbaseline", "GIN\ntuned", "GINE"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
    for ax, split in zip(axes, SPLITS):
        means = [rf[split]["r2"]]
        stds = [0.0]
        for m in ("baseline_gin", "tuned_gin", "gine"):
            if m in res:
                means.append(res[m][split]["r2_mean"])
                stds.append(res[m][split]["r2_std"])
            else:
                means.append(np.nan)
                stds.append(0.0)
        bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, ecolor="black")
        for b, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"{split} split")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("test R2")
    fig.suptitle(f"GIN fairness - {state['tag']} (3-seed mean +/- std)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {os.path.relpath(fig_path, BASE)}", flush=True)


def save_state(state, out_dir):
    path = os.path.join(out_dir, f"summary_{state['tag']}.json")
    json.dump(state, open(path, "w"), indent=2, default=float)
    return path


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    graphs = os.path.join(BASE, "data", "processed", f"{args.tag}_graphs.npz")
    if not os.path.exists(graphs):
        sys.exit(f"missing graph cache: {graphs}")
    tuning_csv = os.path.join(args.out, f"{args.tag}_tuning.csv")
    summary_path = os.path.join(args.out, f"summary_{args.tag}.json")

    state = {"tag": args.tag, "seeds": seeds, "phases": []}
    if "tune" not in phases and os.path.exists(summary_path):
        state = json.load(open(summary_path))
        state["seeds"] = seeds

    print(f"[{args.tag}] seeds={seeds} phases={phases}", flush=True)
    if "tune" in phases:
        phase_tune(args, state, graphs, tuning_csv)
        state["phases"].append("tune")
        save_state(state, args.out)
    if "final" in phases:
        phase_final(args, state, graphs)
        state["phases"].append("final")
        save_state(state, args.out)
    if "gine" in phases:
        phase_gine(args, state, graphs)
        state["phases"].append("gine")
        save_state(state, args.out)
    if "summary" in phases:
        aggregate(state, load_rf(args.tag))
        state["phases"].append("summary")
        save_state(state, args.out)
        print(f"summary -> {os.path.relpath(summary_path, BASE)}", flush=True)
        write_summary_csv(args.out,
                          sorted(glob.glob(os.path.join(args.out,
                                                        "summary_*.json"))))
        if not args.no_figure:
            make_figure(state, os.path.join(args.fig_dir, f"{args.tag}.png"))

    if "final" in state.get("phases", []):
        w = winner_kwargs(state)
        print(f"[{args.tag}] done: winner={w}", flush=True)


if __name__ == "__main__":
    main()
