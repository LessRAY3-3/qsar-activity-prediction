"""03 must featurize and export parent_smiles, not canonical_smiles (M1, 2nd half).

02 deduplicates by parent structure (salts/counter-ions stripped), so 03 has
to fingerprint and export the same unit: otherwise the exported smiles array
- and every scaffold split computed from it (04, gnn_01) - groups on salt
variants of one parent instead of the parent itself.  Older clean CSVs
without the parent_smiles column keep working via a canonical_smiles
fallback with a printed note.
"""
import numpy as np
import pandas as pd
from rdkit import Chem

from qsar_common import morgan_fp_numpy

PARENT = "CC(=O)Oc1ccccc1C(=O)O"           # aspirin
SALT = "CC(=O)Oc1ccccc1C(=O)O.Cl"          # same parent, multi-fragment form
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


def _run_03(tmp_path, monkeypatch, load_script, df):
    in_csv = tmp_path / "clean.csv"
    out_npz = tmp_path / "fingerprints.npz"
    df.to_csv(in_csv, index=False)

    mod = load_script("03_featurize")
    monkeypatch.setattr(mod, "IN_CSV", str(in_csv))
    monkeypatch.setattr(mod, "OUT_NPZ", str(out_npz))
    mod.main()
    return np.load(out_npz)


def test_npz_smiles_and_fingerprints_use_parent(tmp_path, monkeypatch, load_script):
    # row 0: canonical variant carries a counter-ion, parent does not
    df = pd.DataFrame(
        {
            "parent_smiles": [PARENT, CAFFEINE],
            "canonical_smiles": [SALT, CAFFEINE],
            "pic50": [7.0, 6.5],
            "target": ["Test Kinase", "Test Kinase"],
        }
    )
    d = _run_03(tmp_path, monkeypatch, load_script, df)

    assert d["smiles"].tolist() == [PARENT, CAFFEINE]
    # the fingerprint itself comes from the parent, not the salt-bearing variant
    assert np.array_equal(d["X"][0], morgan_fp_numpy(Chem.MolFromSmiles(PARENT)))
    assert not np.array_equal(d["X"][0], morgan_fp_numpy(Chem.MolFromSmiles(SALT)))
    assert np.array_equal(d["X"][1], morgan_fp_numpy(Chem.MolFromSmiles(CAFFEINE)))


def test_missing_parent_column_falls_back_to_canonical(
    tmp_path, monkeypatch, load_script, capsys
):
    # pre-M1 clean CSV: no parent_smiles column at all
    df = pd.DataFrame(
        {
            "canonical_smiles": [SALT, CAFFEINE],
            "pic50": [7.0, 6.5],
            "target": ["Test Kinase", "Test Kinase"],
        }
    )
    d = _run_03(tmp_path, monkeypatch, load_script, df)

    out = capsys.readouterr().out
    assert "parent_smiles" in out and "falling back" in out  # printed notice
    assert d["smiles"].tolist() == [SALT, CAFFEINE]  # unchanged legacy behaviour
    assert d["X"].shape == (2, 2048)
