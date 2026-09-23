"""Step 6 (advanced): map the most important fingerprint bits to substructures.

Random forests expose feature_importances_ per bit. A fingerprint bit alone is
opaque; here we find, for each top bit, an example molecule in the dataset
where the bit is set and reconstruct the exact substructure (atom environment)
that hashed into that bit, using RDKit's bitInfo machinery.
Output:
  results/top_bits.csv          bit, substructure SMILES, #molecules carrying it
  figures/top_bits.png          grid drawing of those substructures
"""
import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
MODEL = os.path.join(BASE, "models", f"rf_random_split_{TAG}.joblib")
OUT_CSV = os.path.join(BASE, "results", f"top_bits_{TAG}.csv")
OUT_PNG = os.path.join(BASE, "figures", f"top_bits_{TAG}.png")
TOP_N = 9
RADIUS = 2
NBITS = 2048


def substructure_for_bit(mol, atom_idx, radius):
    """Extract the atom environment that set a fingerprint bit.

    Returns (smiles, drawable_submol). Uses PathToSubmol (a real sub-molecule)
    instead of MolFragmentToSmiles, whose output often fails to re-parse
    (fragment valences don't round-trip).
    """
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    submol = Chem.PathToSubmol(mol, env)
    return Chem.MolToSmiles(submol), submol


def main():
    model = joblib.load(MODEL)
    d = np.load(IN_NPZ)
    smiles = d["smiles"]

    importances = model.feature_importances_
    top_bits = np.argsort(importances)[::-1][:TOP_N]
    print("Top bits:", top_bits.tolist())

    # single pass: count carriers per bit AND keep the largest example
    # environment found (small fragments like a bare "O" are valid bits but
    # meaningless to show)
    counts = {int(b): 0 for b in top_bits}
    best = {int(b): (0, None, None) for b in top_bits}  # bit -> (n_heavy, smi, mol)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS, bitInfo=info)
        for bit in top_bits:
            bit = int(bit)
            if bit not in info:
                continue
            counts[bit] += 1
            na_cur = best[bit][0]
            if na_cur >= 6:
                continue
            for atom_idx, r in info[bit]:
                sub_smi, submol = substructure_for_bit(mol, atom_idx, r)
                na = submol.GetNumHeavyAtoms()
                if na > best[bit][0]:
                    best[bit] = (na, sub_smi, submol)
                if best[bit][0] >= 6:
                    break

    rows, mols, legends = [], [], []
    for bit in top_bits:
        bit = int(bit)
        na, sub_smi, submol = best[bit]
        rows.append({"bit": bit,
                     "importance": float(importances[bit]),
                     "substructure_smiles": sub_smi,
                     "n_heavy_atoms_in_example": na,
                     "n_molecules_with_bit": counts[bit]})
        if submol is not None and na >= 2:
            mols.append(submol)
            legends.append(f"bit {bit} (imp={importances[bit]:.3f})")
        print(f"bit {bit:4d}  importance={importances[bit]:.4f}  "
              f"in {counts[bit]:5d} mols  e.g. {sub_smi}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(320, 320),
                                   legends=legends)
        img.save(OUT_PNG)
        print(f"Saved -> {os.path.abspath(OUT_CSV)} and {os.path.abspath(OUT_PNG)}")


if __name__ == "__main__":
    main()
