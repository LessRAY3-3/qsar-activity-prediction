"""Shared helpers: fingerprint generation and scaffold splitting.

Single source of truth for every stage of the pipeline (RF, QC, GNN,
experiments): all Morgan fingerprint generation and the deterministic
Bemis-Murcko scaffold split import from this module, so the outputs stay
bit-identical to the historical runs. Function bodies were moved verbatim
from their original scripts; only docstrings changed.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RANDOM_STATE = 42
FP_RADIUS = 2
FP_NBITS = 2048

_MORGAN_GEN = None


def get_morgan_generator():
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
    return _MORGAN_GEN


def make_morgan_generator(radius=FP_RADIUS, fp_size=FP_NBITS):
    """Factory for parametric sweeps over radius/n_bits (fingerprint ablation)."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


def morgan_fp_numpy(mol):
    """2048-bit Morgan fp as uint8 0/1 array (modern rdFingerprintGenerator API)."""
    return np.asarray(get_morgan_generator().GetFingerprintAsNumPy(mol), dtype=np.uint8)


def morgan_fp_with_info(mol):
    """-> (ExplicitBitVect, {bit: [(atomIdx, radius)]}) for substructure explanations."""
    gen = get_morgan_generator()
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = gen.GetFingerprint(mol, additionalOutput=ao)
    info = {int(bit): [(int(a), int(r)) for a, r in lst] for bit, lst in ao.GetBitInfoMap().items()}
    return fp, info


def scaffold_split(smiles, test_size=0.2):
    """Assign whole Bemis-Murcko scaffolds to train/test (test ~= test_size).

    Single source of truth; moved from gnn_01_make_splits.py (the bodies of
    the historical copies in 04_train_and_evaluate.py and
    dump_bace_rf_preds.py were identical).
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
