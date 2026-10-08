"""Fingerprint generation must stay bit-identical to the legacy RDKit API.

The pipeline's historical results were produced with
``AllChem.GetMorganFingerprintAsBitVect``; ``qsar_common`` switched to the
modern ``rdFingerprintGenerator`` API, so every test here pins the two
against each other.  If a future RDKit removes the legacy API entirely,
these tests skip instead of failing.
"""
import warnings

import numpy as np
import pytest
from rdkit import Chem

from qsar_common import (
    get_morgan_generator,
    make_morgan_generator,
    morgan_fp_numpy,
    morgan_fp_with_info,
)

# fixed panel: >=5 molecules, caffeine included (plus a chiral and a fused one)
FIXED_SMILES = [
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "CN1CCC[C@H]1c1cccnc1",           # nicotine (chiral)
    "c1ccc2c(c1)ccc1ccccc12",         # phenanthrene (fused)
]


def _require_legacy():
    from rdkit.Chem import AllChem

    if not hasattr(AllChem, "GetMorganFingerprintAsBitVect"):
        pytest.skip("RDKit build removed the legacy GetMorganFingerprintAsBitVect API")


def _legacy_bitvect(mol):
    from rdkit.Chem import AllChem

    _require_legacy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # legacy-API deprecation warning
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


@pytest.mark.parametrize("smi", FIXED_SMILES)
def test_morgan_fp_numpy_matches_legacy_bit_for_bit(smi):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    legacy = _legacy_bitvect(mol)
    mine = morgan_fp_numpy(mol)
    assert mine.shape == (2048,)
    assert mine.dtype == np.uint8
    legacy_arr = np.array([legacy.GetBit(i) for i in range(2048)], dtype=np.uint8)
    assert np.array_equal(mine, legacy_arr)


@pytest.mark.parametrize("smi", FIXED_SMILES)
def test_morgan_fp_with_info_matches_legacy_bitinfo(smi):
    from rdkit.Chem import AllChem

    _require_legacy()
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    legacy_bit_info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # legacy-API deprecation warning
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=legacy_bit_info)

    fp, info = morgan_fp_with_info(mol)
    assert set(info) == {int(b) for b in legacy_bit_info}
    for bit, hits in legacy_bit_info.items():
        assert sorted(map(tuple, info[int(bit)])) == sorted(map(tuple, hits))
    # every explained bit is actually on in the returned fingerprint
    assert set(info) <= set(fp.GetOnBits())


def test_make_morgan_generator_custom_shape():
    gen = make_morgan_generator(3, 1024)
    mol = Chem.MolFromSmiles(FIXED_SMILES[0])
    arr = gen.GetFingerprintAsNumPy(mol)
    assert arr.shape == (1024,)
    assert set(np.unique(arr)) <= {0, 1}
    # the parametric factory must not return the shared 2048-bit singleton
    assert gen is not get_morgan_generator()
