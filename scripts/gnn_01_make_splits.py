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
results/metrics_{TAG}.json (R2/RMSE) to high precision; this is a consistency check and does not prove molecular identity. The script aborts if this check fails.

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

CLI: --skip-model-check skips the verification against the saved RF
models. Verification is impossible while the RF models are being
retrained on refreshed data (the saved models and metrics are from the
old data and would fail the check); re-run without the flag once the
new models land. The splits themselves are unaffected - they are
reproduced deterministically from the fingerprints.
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from artifact_guards import check_partition
from qsar_common import RANDOM_STATE, murcko_scaffold_list, scaffold_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # egfr | bace
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")
MODEL_DIR = os.path.join(BASE, "models")
METRICS_JSON = os.path.join(BASE, "results", f"metrics_{TAG}.json")
VALID_FRAC_OF_TRAIN = 0.1


def check_disjoint(n_total, **named_idx):
    check_partition(n_total, **named_idx)


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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-model-check", action="store_true",
                   help="do not re-predict with the saved RF models to verify the "
                        "regenerated splits; needed while the RF retrain on the new "
                        "data is still pending (default keeps the verification)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(SPLIT_DIR, exist_ok=True)
    d = np.load(IN_NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    n = len(smiles)
    print(f"[{TAG}] {n} molecules | X={X.shape} | y range [{y.min():.2f}, {y.max():.2f}]")

    scaf = murcko_scaffold_list(smiles)
    metrics = None
    if not args.skip_model_check:
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
        if args.skip_model_check:
            print(f"  [SKIP] {name} split: RF model verification disabled "
                  "(--skip-model-check)")
        else:
            verify_against_saved_model(name, te, X, y, metrics[name])

    if args.skip_model_check:
        print("All splits reproduced (model verification skipped - "
              "re-run without --skip-model-check once the new RF models land).")
    else:
        print("All splits reproduced and verified against saved RF models.")


if __name__ == "__main__":
    main()
