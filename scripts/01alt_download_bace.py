"""Fallback for Step 1: MoleculeNet BACE-1 dataset (used only if ChEMBL API
stays unavailable).

BACE-1 (beta-secretase 1) is an Alzheimer's-related protease; MoleculeNet's
BACE set has 1513 compounds with pIC50 labels and SMILES. We convert pIC50
back to IC50 (nM) so the rest of the pipeline (cleaning -> fingerprints ->
model) is IDENTICAL to the ChEMBL route:
    IC50_nM = 10 ** (9 - pIC50)
"""
import os
import numpy as np
import pandas as pd

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_CSV = os.path.join(OUT_DIR, "bace_activities.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(URL)
    print(f"Downloaded {len(df)} BACE-1 rows from MoleculeNet.")

    out = pd.DataFrame({
        "molecule_chembl_id": df["CID"].astype(str),
        "canonical_smiles": df["mol"],
        "standard_type": "IC50",
        "standard_value": 10 ** (9 - df["pIC50"]),  # nM
        "standard_units": "nM",
        "pchembl_value": df["pIC50"],
        "assay_type": "B",
        "assay_description": "BACE-1 FRET assay (MoleculeNet BACE)",
        "target_pref_name": "Beta-secretase 1 (BACE-1)",
        "bao_label": None,
        "document_chembl_id": None,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {os.path.abspath(OUT_CSV)}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
