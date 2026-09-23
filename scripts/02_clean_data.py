"""Step 2: Clean the raw ChEMBL activity data.

What / why:
  1. Keep only rows whose IC50 unit is exactly 'nM'  -> comparable numbers.
  2. Drop rows with missing standard_value (IC50) or empty canonical_smiles.
  3. Drop non-positive IC50 values (a 0 breaks the log transform).
  4. Convert IC50 (nM) -> pIC50 = -log10(IC50 * 1e-9).
     Why: IC50 spans many orders of magnitude (1 nM ... 100 uM). Raw values
     are heavy-tailed and would dominate model training; pIC50 is roughly
     normal and is THE standard target transform in QSAR.
  5. One compound may be measured in many papers/assays -> duplicates.
     Group by canonical_smiles and take the MEDIAN pIC50 (robust to outliers).
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_CSV = os.path.join(BASE, "data", "raw", f"{TAG}_activities.csv")
OUT_CSV = os.path.join(BASE, "data", "processed", f"{TAG}_pic50_clean.csv")


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} raw rows.")

    # 1. unit must be nM
    df = df[df["standard_units"] == "nM"].copy()
    print(f"After unit == nM: {len(df)}")

    # 2. drop missing IC50 / SMILES
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    print(f"After dropping missing IC50/SMILES: {len(df)}")

    # 3. non-positive IC50 is physically meaningless for the log transform
    df = df[df["standard_value"] > 0].copy()
    print(f"After dropping IC50 <= 0: {len(df)}")

    # 4. IC50 (nM) -> pIC50
    df["pic50"] = -np.log10(df["standard_value"] * 1e-9)

    # sanity cross-check against ChEMBL's own pchembl_value (should match)
    chk = df.dropna(subset=["pchembl_value"])
    agree = np.isclose(chk["pic50"], chk["pchembl_value"].astype(float), atol=0.02).mean()
    print(f"Sanity check: pIC50 matches ChEMBL pchembl_value for {agree:.1%} of rows.")

    # 5. deduplicate by canonical SMILES -> median pIC50
    n_dup = df.duplicated(subset=["canonical_smiles"]).sum()
    target_name = df["target_pref_name"].dropna().iloc[0] if df["target_pref_name"].notna().any() else TAG
    clean = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(
            pic50=("pic50", "median"),
            ic50_nm_median=("standard_value", "median"),
            n_measurements=("pic50", "size"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
        )
        .reset_index(drop=True)
    )
    clean["target"] = target_name
    print(f"Target: {target_name}")
    print(f"Removed {n_dup} duplicate rows -> {len(clean)} unique compounds.")

    print("\npIC50 distribution:")
    print(clean["pic50"].describe())

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    clean.to_csv(OUT_CSV, index=False)
    print(f"\nSaved cleaned data -> {os.path.abspath(OUT_CSV)}")


if __name__ == "__main__":
    main()
