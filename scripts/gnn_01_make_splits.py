"""GNN step 1: reproduce and persist the exact splits used by the RF QSAR
pipeline (scripts/04_train_and_evaluate.py), and validate them.

Background
----------
04_train_and_evaluate.py never saved split indices. Without them the
RF-vs-GNN comparison could silently evaluate on different test molecules
and be invalid. Both splits are deterministic, so they are reproduced here:

  * random split   : train_test_split(np.arange(n), test_size=0.2,
                     random_state=42)
  * scaffold split : molecules grouped by Bemis-Murcko scaffold (RDKit
                     only, same algorithm as before); largest scaffold
                     groups go to test until ~20% of molecules is filled.

Validation
----------
The RF models trained on these splits were saved under models/. Re-
predicting on the regenerated test sets must reproduce
results/metrics_{TAG}.json (R2/RMSE) to high precision; that proves the
regenerated indices are identical to the original ones, molecule by
molecule. The script aborts if this check fails.

Extra: GNN training needs a validation set for early stopping (RF used
5-fold CV instead, so it had none). We carve 10% out of each TRAINING set
with seed 42. Test sets are not touched, so comparability with the
published RF numbers is preserved.

Output: data/processed/splits/{TAG}_{split}.npz with
  train_idx, valid_idx, test_idx : int arrays, indices into the row order
      of data/processed/{TAG}_fingerprints.npz (same order as
      {TAG}_pic50_clean.csv)
  scaffold_smiles : per-molecule Bemis-Murcko scaffold (used later to
      colour the error analysis by chemotype)
  smiles          : copy of the smiles array for a self-consistency check
"""
import json
import os

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # egfr | bace
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")
MODEL_DIR = os.path.join(BASE, "models")
METRICS_JSON = os.path.join(BASE, "results", f"metrics_{TAG}.json")
RANDOM_STATE = 42
VALID_FRAC_OF_TRAIN = 0.1


def scaffold_split(smiles, test_size=0.2):
    """Assign whole Bemis-Murcko scaffolds to train/test (test ~= test_size).

    Copied verbatim from scripts/04_train_and_evaluate.py so the output is
    bit-identical to the original run.
    """
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


def murcko_scaffold_list(smiles):
    """Per-molecule scaffold SMILES (fallback: the SMILES itself)."""
    scafs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scafs.append(smi)
        else:
            scafs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return np.array(scafs, dtype=str)


def check_disjoint(n_total, **named_idx):
    sizes = {k: len(v) for k, v in named_idx.items()}
    union = set().union(*(set(v.tolist()) for v in named_idx.values()))
    assert sum(sizes.values()) == len(union) == n_total, (
        f"splits overlap or leave molecules out: sizes={sizes}, union={len(union)}, n={n_total}"
    )


def verify_against_saved_model(split_name, test_idx, X, y, expected):
    """Re-predict with the saved RF; metrics must match the original run."""
    model = joblib.load(os.path.join(MODEL_DIR, f"rf_{split_name}_split_{TAG}.joblib"))
    pred = model.predict(X[test_idx])
    r2 = r2_score(y[test_idx], pred)
    rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))
    exp_r2, exp_rmse = expected["r2"], expected["rmse"]
    ok = abs(r2 - exp_r2) < 1e-6 and abs(rmse - exp_rmse) < 1e-6
    status = "OK " if ok else "MISMATCH"
    print(f"  [{status}] {split_name:8s}  R2={r2:.7f} (expected {exp_r2:.7f})   "
          f"RMSE={rmse:.7f} (expected {exp_rmse:.7f})")
    if not ok:
        raise SystemExit(
            f"regenerated '{split_name}' split does NOT reproduce the saved RF "
            "metrics - the RF-vs-GNN comparison would be invalid. Aborting."
        )


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    d = np.load(IN_NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    n = len(smiles)
    print(f"[{TAG}] {n} molecules | X={X.shape} | y range [{y.min():.2f}, {y.max():.2f}]")

    scaf = murcko_scaffold_list(smiles)
    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    splits = {
        "random": train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE),
        "scaffold": scaffold_split(smiles, test_size=0.2),
    }

    for name, (tr, te) in splits.items():
        # validation carved out of TRAIN only; test stays identical to the RF run
        tr_new, va = train_test_split(tr, test_size=VALID_FRAC_OF_TRAIN,
                                      random_state=RANDOM_STATE)
        check_disjoint(n, train_idx=tr_new, valid_idx=va, test_idx=te)
        out = os.path.join(SPLIT_DIR, f"{TAG}_{name}.npz")
        np.savez(out, train_idx=tr_new, valid_idx=va, test_idx=te,
                 scaffold_smiles=scaf, smiles=smiles)
        print(f"  saved {os.path.relpath(out, BASE)}  "
              f"train={len(tr_new)} valid={len(va)} test={len(te)}")
        verify_against_saved_model(name, te, X, y, metrics[name])

    print("All splits reproduced and verified against saved RF models.")


if __name__ == "__main__":
    main()
