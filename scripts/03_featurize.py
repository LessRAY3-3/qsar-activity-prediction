"""Step 3: Featurize SMILES -> Morgan fingerprints (RDKit).

Why Morgan fingerprints:
  A molecule is a graph; ML needs numbers. Morgan fingerprint (radius=2,
  ECFP4-like) encodes every atom's circular neighbourhood up to 2 bonds as
  a hashed 2048-bit vector. Structurally similar molecules -> similar
  fingerprints, which is exactly the signal a similarity-based ML model
  (random forest / XGBoost) can learn from.
  Invalid SMILES (rare in ChEMBL, but they exist) make RDKit return None.
  03 does NOT drop them silently: every downstream artifact (splits,
  rf_preds test_idx, year joins) stores clean-CSV row positions, so
  losing a row would shift every later index while the run still exits 0.
  Any unparseable input is counted and listed, then the script aborts;
  fix the SMILES upstream (02 keeps unparseable input as a raw fallback
  parent) instead of letting the row counts diverge.
Why parent_smiles: 02 deduplicates by parent structure (salts/counter-ions
  stripped, charges neutralized), so one compound = one row. Fingerprinting
  the group's first canonical variant instead would embed salt bits and give
  the same parent different fingerprints in different rows, while scaffold
  splits (04, gnn_01) group on the exported smiles array - keeping the
  fingerprint and split unit at parent_smiles closes the salt leak M1
  opened. Old clean CSVs without the column fall back to canonical_smiles.
Output: data/processed/egfr_fingerprints.npz  (X, y, smiles)
"""
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsar_common import FP_NBITS, morgan_fp_numpy  # noqa: E402

TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_CSV = os.path.join(BASE, "data", "processed", f"{TAG}_pic50_clean.csv")
OUT_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} cleaned compounds.")

    if "parent_smiles" in df.columns:
        smiles_col = "parent_smiles"
    else:
        # backward compat: clean CSVs from before the M1 fix have no parent column
        smiles_col = "canonical_smiles"
        print("No 'parent_smiles' column in input (older clean CSV); "
              "falling back to 'canonical_smiles'.")

    X = np.zeros((len(df), FP_NBITS), dtype=np.uint8)
    y = df["pic50"].to_numpy(dtype=np.float32)
    valid_idx = []
    invalid = []

    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append((i, str(smi)))
            continue
        X[i] = morgan_fp_numpy(mol)
        valid_idx.append(i)

    if invalid:
        # Loud abort: dropping these rows would shift every later index --
        # splits / rf_preds test_idx are positions into THIS row order --
        # while still exiting 0, so downstream files would silently refer
        # to the wrong molecules.
        preview = ", ".join(f"row {i}: {s[:60]!r}" for i, s in invalid[:5])
        more = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
        print(f"ERROR: {len(invalid)} of {len(df)} SMILES failed RDKit "
              f"parsing -> {preview}{more}", flush=True)
        raise SystemExit(
            f"{TAG}: {len(invalid)} unparseable SMILES in "
            f"{os.path.relpath(IN_CSV, BASE)}; refusing to drop them. "
            "Dropping would desynchronise the fingerprint rows from the "
            "clean CSV and shift every downstream split/prediction index. "
            "Fix these SMILES upstream (scripts/02_clean_data.py keeps "
            "unparseable input as a raw fallback parent) and rerun 03.")

    X = X[valid_idx]
    y = y[valid_idx]
    smiles = np.asarray(df[smiles_col].to_numpy()[valid_idx], dtype=str)
    target = str(df["target"].iloc[0]) if "target" in df.columns else TAG
    print(f"X shape {X.shape}, y shape {y.shape}")

    np.savez_compressed(OUT_NPZ, X=X, y=y, smiles=smiles, target=np.asarray(target, dtype=str))
    print(f"Saved feature matrix -> {os.path.abspath(OUT_NPZ)}")


if __name__ == "__main__":
    main()
