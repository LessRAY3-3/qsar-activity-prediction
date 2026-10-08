"""End-to-end smoke: 02 clean -> 03 featurize -> 04 train, all inside tmp_path.

A 24-molecule synthetic raw CSV is pushed through the real mains; only the
module-level path constants are monkeypatched.  04's grid is shrunk to a
single 3-tree candidate so the whole test stays in the seconds range.
"""
import json

import numpy as np
import pandas as pd

# columns 04's upstream (01d/02 contract) guarantees to exist in the raw CSV
RAW_COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_type",
    "assay_description",
    "target_pref_name",
    "bao_label",
    "document_chembl_id",
]


def _raw_rows(smiles):
    rows = []
    for i, smi in enumerate(smiles):
        ic50_nm = 5.0 * (i + 1)  # nM, spans ~5-120 nM -> pIC50 ~6.9-8.3
        rows.append(
            {
                "molecule_chembl_id": f"CHEMBL{i + 1:04d}",
                "canonical_smiles": smi,
                "standard_type": "IC50",
                "standard_value": ic50_nm,
                "standard_units": "nM",
                "pchembl_value": -np.log10(ic50_nm * 1e-9),
                "assay_type": "B",
                "assay_description": "synthetic smoke-test assay",
                "target_pref_name": "Test Kinase",
                "bao_label": "PROTEIN",
                "document_chembl_id": f"CHEMBL{90000 + i}",
            }
        )
    return rows


def test_pipeline_02_03_04_smoke(tmp_path, monkeypatch, load_script, valid_smiles):
    smiles = valid_smiles[:24]
    assert len(smiles) == 24 and len(set(smiles)) == 24

    # ---- 02: clean raw -> pic50 CSV ----
    raw_csv = tmp_path / "raw_activities.csv"
    pd.DataFrame(_raw_rows(smiles), columns=RAW_COLUMNS).to_csv(raw_csv, index=False)

    clean_csv = tmp_path / "clean.csv"
    mod02 = load_script("02_clean_data")
    monkeypatch.setattr(mod02, "IN_CSV", str(raw_csv))
    monkeypatch.setattr(mod02, "OUT_CSV", str(clean_csv))
    mod02.main()

    assert clean_csv.exists()
    clean = pd.read_csv(clean_csv)
    assert len(clean) == 24  # all unique -> nothing deduplicated away
    assert {"canonical_smiles", "pic50", "target"}.issubset(clean.columns)
    assert np.isfinite(clean["pic50"]).all()

    # ---- 03: clean CSV -> fingerprints npz ----
    npz_path = tmp_path / "fingerprints.npz"
    mod03 = load_script("03_featurize")
    monkeypatch.setattr(mod03, "IN_CSV", str(clean_csv))
    monkeypatch.setattr(mod03, "OUT_NPZ", str(npz_path))
    mod03.main()

    d = np.load(npz_path)
    assert set(d.files) == {"X", "y", "smiles", "target"}
    X, y, smi = d["X"], d["y"], d["smiles"]
    assert X.shape == (24, 2048) and X.dtype == np.uint8
    assert y.shape == (24,) and np.isfinite(y).all()
    assert len(smi) == 24
    assert smi.tolist() == clean["canonical_smiles"].tolist()  # row alignment
    assert set(np.unique(X)) <= {0, 1}
    assert str(d["target"]) == "Test Kinase"

    # ---- 04: npz -> metrics json + models ----
    mod04 = load_script("04_train_and_evaluate")
    metrics_json = tmp_path / "metrics.json"
    monkeypatch.setattr(mod04, "IN_NPZ", str(npz_path))
    monkeypatch.setattr(mod04, "FIG_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(mod04, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(mod04, "METRICS_JSON", str(metrics_json))
    # shrink the grid: 1 candidate x 3 trees (5-fold CV kept, still tiny)
    monkeypatch.setattr(
        mod04,
        "PARAM_GRID",
        {"n_estimators": [3], "max_depth": [None], "min_samples_split": [2]},
    )
    mod04.main()

    payload = json.loads(metrics_json.read_text())
    assert {"random", "scaffold"} <= set(payload)
    for split in ("random", "scaffold"):
        m = payload[split]
        assert isinstance(m["r2"], float) and np.isfinite(m["r2"])
        assert isinstance(m["rmse"], float) and np.isfinite(m["rmse"])
        assert isinstance(m["mae"], float) and np.isfinite(m["mae"])
    assert payload["random"]["split"] == "random"
    assert payload["scaffold"]["split"] == "scaffold"

    tag = mod04.TAG
    assert (tmp_path / "models" / f"rf_random_split_{tag}.joblib").exists()
    assert (tmp_path / "models" / f"rf_scaffold_split_{tag}.joblib").exists()
    assert (tmp_path / "figures" / f"pred_vs_actual_random_{tag}.png").exists()
    assert (tmp_path / "figures" / f"pred_vs_actual_scaffold_{tag}.png").exists()
