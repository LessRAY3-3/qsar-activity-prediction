"""Step 3: Featurize SMILES -> Morgan fingerprints (RDKit).

Why Morgan fingerprints:
  A molecule is a graph; ML needs numbers. Morgan fingerprint (radius=2,
  ECFP4-like) encodes every atom's circular neighbourhood up to 2 bonds as
  a hashed 2048-bit vector. Structurally similar molecules -> similar
  fingerprints, which is exactly the signal a similarity-based ML model
  (random forest / XGBoost) can learn from.
  Invalid SMILES (rare in ChEMBL, but they exist) make RDKit return None;
  those compounds are dropped.
Output: data/processed/egfr_fingerprints.npz  (X, y, smiles)
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_CSV = os.path.join(BASE, "data", "processed", f"{TAG}_pic50_clean.csv")
OUT_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")

RADIUS = 2
NBITS = 2048


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} cleaned compounds.")

    X = np.zeros((len(df), NBITS), dtype=np.uint8)
    y = df["pic50"].to_numpy(dtype=np.float32)
    valid_idx = []

    for i, smi in enumerate(df["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
        X[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
        valid_idx.append(i)

    n_invalid = len(df) - len(valid_idx)
    X = X[valid_idx]
    y = y[valid_idx]
    smiles = np.asarray(df["canonical_smiles"].to_numpy()[valid_idx], dtype=str)
    target = str(df["target"].iloc[0]) if "target" in df.columns else TAG
    print(f"Dropped {n_invalid} invalid SMILES -> X shape {X.shape}, y shape {y.shape}")

    np.savez_compressed(OUT_NPZ, X=X, y=y, smiles=smiles, target=np.asarray(target, dtype=str))
    print(f"Saved feature matrix -> {os.path.abspath(OUT_NPZ)}")


if __name__ == "__main__":
    main()
