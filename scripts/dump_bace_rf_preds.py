"""Dump per-molecule RF predictions for the BACE-1 splits.

The two splits are reproduced verbatim from scripts/04_train_and_evaluate.py
(random: train_test_split(80/20, random_state=42); scaffold: Murcko
scaffold split, big scaffolds first). The saved models
models/rf_{random,scaffold}_split_bace.joblib are re-loaded and asked to
predict on the matching test rows.

Outputs (same keys as the EGFR files: test_idx, y_true, y_pred):
  results/rf_preds_bace_random.npz
  results/rf_preds_bace_scaffold.npz

Self-check: r2/rmse/mae recomputed from the saved npz must match
results/metrics_bace.json within 1e-4 (otherwise exit non-zero).
"""
import json
import os
import sys

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = "bace"
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
MODEL_DIR = os.path.join(BASE, "models")
METRICS_JSON = os.path.join(BASE, "results", f"metrics_{TAG}.json")
RANDOM_STATE = 42
TOL = 1e-4


def scaffold_split(smiles, test_size=0.2):
    """Assign whole Bemis-Murcko scaffolds to train/test (test ~= test_size)."""
    scaffolds = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaf = smi  # fallback: treat molecule as its own scaffold
        else:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.setdefault(scaf, []).append(i)

    # big scaffolds first -> test set fills up to roughly test_size
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    n_test = int(len(smiles) * test_size)
    test_idx, train_idx, n = [], [], 0
    for g in groups:
        if n < n_test:
            test_idx.extend(g)
            n += len(g)
        else:
            train_idx.extend(g)
    return np.array(train_idx), np.array(test_idx)


def dump_split(split, test_idx, X, y):
    """Predict with the saved model, write the npz, return recomputed metrics."""
    model = joblib.load(os.path.join(MODEL_DIR, f"rf_{split}_split_{TAG}.joblib"))
    out_path = os.path.join(BASE, "results", f"rf_preds_{TAG}_{split}.npz")
    np.savez(out_path,
             test_idx=test_idx,
             y_true=y[test_idx],
             y_pred=model.predict(X[test_idx]))

    r = np.load(out_path)
    yt, yp = r["y_true"], r["y_pred"]
    metrics = {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
    }
    print(f"[{split}] saved {os.path.relpath(out_path, BASE)}: "
          f"test_idx={r['test_idx'].shape} {r['test_idx'].dtype}, "
          f"y_true={yt.shape} {yt.dtype}, y_pred={yp.shape} {yp.dtype}")
    return metrics


def main():
    d = np.load(IN_NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    print(f"Loaded {os.path.relpath(IN_NPZ, BASE)}: X={X.shape}, y={y.shape}, "
          f"smiles={smiles.shape}")

    # ---- random split (same call as 04_train_and_evaluate.py) ----
    _, te_rand = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=RANDOM_STATE
    )
    # ---- scaffold split (verbatim copy of 04_train_and_evaluate.py) ----
    _, te_scaf = scaffold_split(smiles, test_size=0.2)

    with open(METRICS_JSON) as f:
        expected = json.load(f)

    rows, ok = [], True
    for split, test_idx in [("random", te_rand), ("scaffold", te_scaf)]:
        got = dump_split(split, np.asarray(test_idx), X, y)
        for key in ("r2", "rmse", "mae"):
            exp = float(expected[split][key])
            diff = abs(got[key] - exp)
            ok &= diff < TOL
            rows.append((split, key, exp, got[key], diff))

    print(f"\n{'split':<10} {'metric':<6} {'metrics_bace.json':>20} "
          f"{'recomputed':>20} {'abs diff':>12}  status")
    for split, key, exp, got, diff in rows:
        status = "OK" if diff < TOL else "FAIL"
        print(f"{split:<10} {key:<6} {exp:>20.10f} {got:>20.10f} "
              f"{diff:>12.3e}  {status}")

    if not ok:
        print(f"\nSELF-CHECK FAILED: some |diff| >= {TOL}")
        sys.exit(1)
    print(f"\nSelf-check passed: all |diff| < {TOL}")


if __name__ == "__main__":
    main()
