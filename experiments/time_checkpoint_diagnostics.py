"""Evaluate saved temporal GIN checkpoints; never train or overwrite metrics.

python experiments/time_checkpoint_diagnostics.py --threads 2
Needs frozen time splits, graph caches and the 24 published checkpoints.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gnn_graph_dataset import MoleculeGraphDataset  # noqa: E402
from time_labels import load_time_split  # noqa: E402
from time_campaign_bundle import TAGS  # noqa: E402

trainer = importlib.import_module("gnn_03_train_gin")


def diagnose(root, threads=2):
    root = Path(root)
    torch.set_num_threads(threads)
    acceptance = json.loads((root / "results/time_split/campaign_acceptance.json").read_text())
    rows = []
    for tag in TAGS:
        graph = root / f"data/processed/{tag}_graphs.npz"
        with np.load(graph, allow_pickle=False) as cache:
            split = load_time_split(root / "data/processed/splits", tag, cache["smiles"], cache["y"])
        if split is None:
            raise ValueError(f"{tag}: no active split")
        y = split["y"]
        mean, std = float(y[split["train_idx"]].mean()), float(y[split["train_idx"]].std())
        train_ds = MoleculeGraphDataset(graph, indices=split["train_idx"])
        valid_ds = MoleculeGraphDataset(graph, indices=split["valid_idx"])
        for ds in (train_ds, valid_ds):
            ds.y = (y.copy() - mean) / std
        train_ld = DataLoader(train_ds, batch_size=512, shuffle=False)
        valid_ld = DataLoader(valid_ds, batch_size=512, shuffle=False)
        for seed in (42, 1, 2):
            suffix = "" if seed == 42 else f"_seed{seed}"
            name = f"results/gnn_models/{tag}_gin_time{suffix}.pt"
            checkpoint_path = root / name
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            job = next(j for j in acceptance["jobs"] if j["role"] == "gin" and j["tag"] == tag and j["seed"] == seed)
            if checkpoint_sha != job["returned_artifact_sha256"][name]:
                raise ValueError(f"{name}: checkpoint hash differs from accepted job")
            graph_sha = hashlib.sha256(graph.read_bytes()).hexdigest()
            if graph_sha != job["provenance"]["inputs"][str(graph.relative_to(root))]:
                raise ValueError(f"{tag}: graph hash differs from accepted job")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("label_id") != str(split["label_id"]) or checkpoint["args"]["seed"] != seed:
                raise ValueError(f"{name}: checkpoint version/seed mismatch")
            if checkpoint["y_mean"] != mean or checkpoint["y_std"] != std:
                raise ValueError(f"{name}: training normalization mismatch")
            model = trainer.build_model(argparse.Namespace(**checkpoint["args"]))
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            mse = trainer.run_epoch(model, train_ld, "cpu")
            vp, vy = trainer.predict(model, valid_ld, "cpu")
            valid_rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))
            metric = json.loads((root / f"results/gnn_metrics_{tag}{suffix}.json").read_text())["time"]
            if not np.isclose(valid_rmse, metric["best_valid_rmse_std"], atol=2e-6, rtol=1e-5):
                raise ValueError(f"{name}: best validation RMSE replay mismatch")
            rows.append(dict(tag=tag, seed=seed, label_id=str(split["label_id"]),
                             checkpoint_sha256=checkpoint_sha, best_epoch=metric["best_epoch"],
                             max_epochs=metric["max_epochs"], last_epoch_train_mse=metric["train_mse"],
                             originally_reported_flag=metric["convergence_flag"],
                             best_checkpoint_train_mse=mse,
                             best_checkpoint_flag=trainer.convergence_flag(metric["best_epoch"], metric["max_epochs"], mse),
                             validation_rmse_replayed=valid_rmse, device="cpu", threads=threads))
            print(f"{tag} seed={seed}: eval train MSE={mse:.6f}, flag={rows[-1]['best_checkpoint_flag']}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    frame = diagnose(args.root, args.threads)
    out = args.out or args.root / "results/time_split/checkpoint_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{int(frame.best_checkpoint_flag.sum())}/{len(frame)} flagged; diagnostic only, no retraining")


if __name__ == "__main__":
    main()
