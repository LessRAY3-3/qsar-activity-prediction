"""Compare frozen temporal predictions with a train-mean constant.

No model training or test-set calibration. The R2 decomposition is arithmetic,
not evidence of a causal drift mechanism. Bootstrap resamples test molecules,
conditional on fitted models; it does not include training uncertainty.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from time_campaign_bundle import TAGS  # noqa: E402
from time_labels import load_time_split, require_prediction_version  # noqa: E402


def decomposition(y, rf, gin):
    """RF-to-GIN R2 change: level plus symmetric scale/correlation terms."""
    y, rf, gin = (np.asarray(x, dtype=float) for x in (y, rf, gin))
    sy = y.std()
    if sy <= 0 or rf.std() <= 0 or gin.std() <= 0:
        raise ValueError("decomposition needs nonconstant labels and predictions")
    br, bg = rf.mean() - y.mean(), gin.mean() - y.mean()
    qr, qg = rf.std() / sy, gin.std() / sy
    rr, rg = np.corrcoef(y, rf)[0, 1], np.corrcoef(y, gin)[0, 1]
    return dict(rf_bias=br, gin_bias=bg, rf_correlation=rr, gin_correlation=rg,
                rf_sd_ratio=qr, gin_sd_ratio=qg,
                level=(br**2 - bg**2) / sy**2,
                scale=(qr - qg) * (qr + qg - rr - rg),
                correlation=(rg - rr) * (qr + qg))


def constant_comparison(y, predictions, constant, n_boot=10000, seed=42):
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if (y.ndim != 1 or predictions.ndim != 2 or predictions.shape[1] != len(y)
            or len(y) < 2 or not np.isfinite(y).all()
            or not np.isfinite(predictions).all() or not np.isfinite(constant)
            or n_boot < 1 or y.var() <= 0):
        raise ValueError("invalid baseline comparison inputs")
    errors = (y - constant)**2 - (predictions - y)**2
    point = errors.mean(axis=1) / y.var()
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, n_boot, 128):
        indices = rng.integers(len(y), size=(min(128, n_boot-start), len(y)))
        variance = y[indices].var(axis=1)
        valid = variance > np.finfo(float).eps
        if valid.any():
            samples.append(errors[:, indices[:, :]][:, valid, :].mean(axis=2) / variance[valid])
    if not samples:
        raise ValueError("all bootstrap samples are degenerate")
    samples = np.concatenate(samples, axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=1)
    return point, lo, hi, samples.shape[1]


def build(root, n_boot=10000, seed=42):
    root = Path(root)
    rows = []
    for tag in TAGS:
        sp = load_time_split(root / "data/processed/splits", tag)
        if sp is None:
            raise ValueError(f"{tag}: time split unavailable")
        values = []
        for name in [f"rf_preds_{tag}_time.npz"] + [f"gnn_preds_{tag}_time{s}.npz" for s in ("", "_seed1", "_seed2")]:
            with np.load(root / "results" / name, allow_pickle=False) as p:
                require_prediction_version(p, sp)
                values.append(p["y_pred"].astype(float))
        rf, gin = values[0], np.mean(values[1:], axis=0)
        y = np.asarray(sp["y"], dtype=float)[sp["test_idx"]]
        c = np.asarray(sp["y"], dtype=float)[sp["train_idx"]].mean()
        point, lo, hi, valid = constant_comparison(y, [rf, gin], c, n_boot, seed)
        constant_r2 = 1 - np.mean((y-c)**2)/y.var()
        row = dict(tag=tag, label_id=str(sp["label_id"]), n_test=len(y),
                   test_fraction=len(y)/len(sp["y"]), train_mean=c, constant_r2=constant_r2,
                   n_boot=n_boot, valid_boot=valid, seed=seed)
        for i, model in enumerate(("rf", "gin_ensemble")):
            row.update({f"{model}_minus_constant": point[i], f"{model}_ci_lo": lo[i],
                        f"{model}_ci_hi": hi[i]})
        row.update(decomposition(y, rf, gin))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    frame = build(args.root)
    out = args.out or args.root / "results/time_split/baseline_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False))
    print(json.dumps({"output": str(out), "bootstrap_replicates": 10000, "seed": 42}))


if __name__ == "__main__":
    main()
