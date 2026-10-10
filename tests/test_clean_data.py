"""Unit tests for scripts/02_clean_data.py: censoring filter (H1) and parent
standardization / deduplication by parent_smiles (M1).

Inputs are small inline CSVs in both supported raw shapes: the ChEMBL download
format (with standard_relation, as 01d/01e now write it) and the bace download
format (no standard_relation column, where the censoring filter must be skipped
so bace behaviour is unchanged).
"""
import numpy as np
import pandas as pd
import pytest

BASE_COLUMNS = [
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


def _row(chembl_id, smiles, ic50_nm, relation="=", pchembl=None):
    row = dict.fromkeys(BASE_COLUMNS)
    row.update(
        {
            "molecule_chembl_id": chembl_id,
            "canonical_smiles": smiles,
            "standard_type": "IC50",
            "standard_value": ic50_nm,
            "standard_units": "nM",
            "pchembl_value": pchembl,
            "assay_type": "B",
            "assay_description": "synthetic unit-test assay",
            "target_pref_name": "Test Kinase",
            "bao_label": "single protein format",
            "document_chembl_id": "CHEMBL999999",
        }
    )
    if relation is not None:
        row["standard_relation"] = relation
    return row


def _run_02(tmp_path, load_script, rows, capsys):
    raw_csv = tmp_path / "raw_activities.csv"
    columns = list(rows[0])
    pd.DataFrame(rows, columns=columns).to_csv(raw_csv, index=False)
    clean_csv = tmp_path / "clean.csv"
    mod02 = load_script("02_clean_data")
    # patch via monkeypatch-free setattr: each test loads its own module instance
    mod02.IN_CSV = str(raw_csv)
    mod02.OUT_CSV = str(clean_csv)
    mod02.main()
    out = capsys.readouterr().out
    clean = pd.read_csv(clean_csv)
    # deterministic row order for downstream npz alignment
    assert clean["parent_smiles"].is_monotonic_increasing
    return clean, out


def test_censoring_filter_keeps_only_exact_rows(tmp_path, load_script, capsys):
    pic50 = lambda ic50: -np.log10(ic50 * 1e-9)  # noqa: E731
    rows = [
        _row("CHEMBL1", "c1ccncc1", 100.0, "=", pchembl=pic50(100.0)),
        _row("CHEMBL2", "c1ccoc1", 100.0, ">", pchembl=pic50(100.0)),
        _row("CHEMBL3", "c1ccsc1", 100.0, "<", pchembl=pic50(100.0)),
        _row("CHEMBL4", "c1cnc[nH]1", 100.0, ">=", pchembl=pic50(100.0)),
        _row("CHEMBL5", "c1cn[nH]c1", 100.0, "<=", pchembl=pic50(100.0)),
        _row("CHEMBL6", "C1CCCCC1", 100.0, ">>", pchembl=pic50(100.0)),
        _row("CHEMBL7", "C1CCNCC1", 100.0, "~", pchembl=pic50(100.0)),
        _row("CHEMBL8", "C1CCOCC1", 100.0, None, pchembl=pic50(100.0)),
        _row("CHEMBL9", "C1CCSCC1", 50.0, "=", pchembl=pic50(50.0)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 2  # only the two "=" rows survive
    assert set(clean["parent_smiles"]) == {"c1ccncc1", "C1CCSCC1"}
    # the printout breaks removed rows down by relation bucket
    assert "removed 7 non-exact rows" in out
    for bucket in ("'>'", "'<'", "'>='", "'<='", "'>>'", "'~'", "(empty)"):
        assert bucket in out, f"missing bucket {bucket} in:\n{out}"


def test_parent_standardization_merges_salt_forms(tmp_path, load_script, capsys):
    pic50 = lambda ic50: -np.log10(ic50 * 1e-9)  # noqa: E731
    rows = [
        # free base and sodium salt of the same parent (acetic acid)
        _row("CHEMBL1", "CC(=O)O", 100.0, "=", pchembl=pic50(100.0)),
        _row("CHEMBL2", "CC(=O)[O-].[Na+]", 400.0, "=", pchembl=pic50(400.0)),
        # a distinct control compound that must stay separate
        _row("CHEMBL3", "c1ccncc1", 10.0, "=", pchembl=pic50(10.0)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 2  # salt pair collapses to one parent compound
    parent_row = clean[clean["parent_smiles"] == "CC(=O)O"].iloc[0]
    assert parent_row["canonical_smiles"] == "CC(=O)O"  # first, for provenance
    assert parent_row["n_measurements"] == 2
    assert parent_row["n_structural_variants"] == 2
    assert parent_row["pic50"] == pytest.approx(np.median([pic50(100.0), pic50(400.0)]))
    pyridine_row = clean[clean["parent_smiles"] == "c1ccncc1"].iloc[0]
    assert pyridine_row["n_measurements"] == 1
    assert pyridine_row["n_structural_variants"] == 1
    # summary prints: multi-fragment input detected, duplicate group merged
    assert "multi-fragment inputs: 1" in out
    assert "Removed 1 duplicate rows -> 2 unique parent compounds" in out
    assert "1 held >1 distinct" in out  # merged group had two canonical SMILES


def test_uncharger_covers_charged_single_fragments(tmp_path, load_script, capsys):
    # deprotonated acetic acid (one fragment, charge only) -> neutral parent
    rows = [_row("CHEMBL1", "CC(=O)[O-]", 100.0, "=", pchembl=-np.log10(100.0 * 1e-9))]
    clean, _ = _run_02(tmp_path, load_script, rows, capsys)
    assert len(clean) == 1
    assert clean["parent_smiles"].iloc[0] == "CC(=O)O"


def test_bace_style_input_without_relation_column_is_unchanged(tmp_path, load_script, capsys):
    # no standard_relation column at all -> filter skipped, everything kept,
    # including rows whose values would look censored in the ChEMBL format
    rows = [
        _row("CHEMBL1", "c1ccncc1", 100.0, relation=None, pchembl=-np.log10(100.0 * 1e-9)),
        _row("CHEMBL2", "c1ccoc1", 200.0, relation=None, pchembl=-np.log10(200.0 * 1e-9)),
        _row("CHEMBL3", "C1CCCCC1", 50.0, relation=None, pchembl=-np.log10(50.0 * 1e-9)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 3  # nothing filtered, nothing merged
    assert "skipping censoring filter" in out
    assert "Censoring filter:" not in out


def test_invalid_smiles_falls_back_to_raw_string(tmp_path, load_script, capsys):
    rows = [
        _row("CHEMBL1", "INVALID_SMILES_XYZ", 100.0, "=", pchembl=7.0),
        _row("CHEMBL2", "c1ccncc1", 100.0, "=", pchembl=7.0),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 2
    bad = clean[clean["canonical_smiles"] == "INVALID_SMILES_XYZ"].iloc[0]
    assert bad["parent_smiles"] == "INVALID_SMILES_XYZ"  # fallback, no crash
    assert "standardization failed (kept raw SMILES): 1" in out


def test_sanity_check_reports_empty_pchembl_rows(tmp_path, load_script, capsys):
    # exact relation but no pChEMBL ("Outside typical range" boundary rows)
    rows = [
        _row("CHEMBL1", "c1ccncc1", 500000.0, "=", pchembl=None),
        _row("CHEMBL2", "c1ccoc1", 100.0, "=", pchembl=7.0),
        _row("CHEMBL3", "c1ccsc1", 50.0, "=", pchembl=7.3),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 3  # empty-pChEMBL exact rows are KEPT (policy: relation, not pchembl)
    assert "Retained rows with empty pchembl_value: 1" in out
    assert "pIC50 matches for 100.0%" in out
