"""Unit tests for scripts/02_clean_data.py: censoring filter (H1), fail-closed
behaviour when standard_relation is missing (N5), and parent standardization /
deduplication by parent_smiles (M1).

Inputs are small inline CSVs in the supported raw shapes: the ChEMBL download
format (with standard_relation, as 01d/01e now write it) and the bace download
format (no standard_relation column). The missing column is only tolerated for
genuine non-ChEMBL inputs (TAG == 'bace' or non-CHEMBL-prefixed molecule IDs);
a CHEMBL-prefixed file without it predates the 2026-10-11 refresh and must
abort instead of silently skipping the censoring filter.
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


def _prepare_02(tmp_path, load_script, rows, tag=None):
    """Write rows to a raw CSV and return (module, clean_csv_path)."""
    raw_csv = tmp_path / "raw_activities.csv"
    columns = list(rows[0])
    pd.DataFrame(rows, columns=columns).to_csv(raw_csv, index=False)
    clean_csv = tmp_path / "clean.csv"
    mod02 = load_script("02_clean_data")
    # patch via monkeypatch-free setattr: each test loads its own module instance
    mod02.IN_CSV = str(raw_csv)
    mod02.OUT_CSV = str(clean_csv)
    if tag is not None:
        mod02.TAG = tag
    return mod02, clean_csv


def _run_02(tmp_path, load_script, rows, capsys, tag=None):
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows, tag=tag)
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
        # Three structural variants of one parent (acetic acid), 4 records:
        # the free base is measured twice, plus the sodium salt and the
        # deprotonated anion.  records (4) != variants (3) and median !=
        # mean over the group, so both the "mean instead of median" and the
        # "records instead of variants" mutants of the parent groupby fail.
        _row("CHEMBL1", "CC(=O)O", 100.0, "=", pchembl=pic50(100.0)),       # 7.000
        _row("CHEMBL2", "CC(=O)[O-].[Na+]", 400.0, "=", pchembl=pic50(400.0)),  # 6.398
        _row("CHEMBL3", "CC(=O)[O-]", 10.0, "=", pchembl=pic50(10.0)),      # 8.000
        _row("CHEMBL4", "CC(=O)O", 1000.0, "=", pchembl=pic50(1000.0)),     # 6.000
        # a distinct control compound that must stay separate
        _row("CHEMBL5", "c1ccncc1", 10.0, "=", pchembl=pic50(10.0)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 2  # salt/form group collapses to one parent compound
    parent_row = clean[clean["parent_smiles"] == "CC(=O)O"].iloc[0]
    assert parent_row["canonical_smiles"] == "CC(=O)O"  # first, for provenance
    assert parent_row["n_measurements"] == 4
    assert parent_row["n_structural_variants"] == 3
    # a "records instead of variants" mutant would report 4 here
    assert parent_row["n_measurements"] != parent_row["n_structural_variants"]

    record_values = [pic50(100.0), pic50(400.0), pic50(10.0), pic50(1000.0)]
    expected_median = float(np.median(record_values))
    expected_mean = float(np.mean(record_values))
    # the fixture must discriminate median from mean at all
    assert not np.isclose(expected_median, expected_mean, rtol=0.0, atol=1e-6)
    # merged pic50 is the MEDIAN pIC50 over the parent group's records
    assert parent_row["pic50"] == pytest.approx(expected_median, rel=1e-9)
    # a "mean instead of median" mutant would land on expected_mean
    assert parent_row["pic50"] != pytest.approx(expected_mean, rel=1e-6)

    pyridine_row = clean[clean["parent_smiles"] == "c1ccncc1"].iloc[0]
    assert pyridine_row["n_measurements"] == 1
    assert pyridine_row["n_structural_variants"] == 1
    # summary prints: multi-fragment input detected, duplicate group merged
    assert "multi-fragment inputs: 1" in out
    assert "Removed 3 duplicate rows -> 2 unique parent compounds" in out
    assert "1 held >1 distinct" in out  # merged group held three canonical SMILES


def test_uncharger_covers_charged_single_fragments(tmp_path, load_script, capsys):
    # deprotonated acetic acid (one fragment, charge only) -> neutral parent
    rows = [_row("CHEMBL1", "CC(=O)[O-]", 100.0, "=", pchembl=-np.log10(100.0 * 1e-9))]
    clean, _ = _run_02(tmp_path, load_script, rows, capsys)
    assert len(clean) == 1
    assert clean["parent_smiles"].iloc[0] == "CC(=O)O"


def test_bace_tag_input_without_relation_column_is_unchanged(tmp_path, load_script, capsys):
    # QSAR_TAG=bace: no standard_relation column at all -> filter skipped,
    # everything kept (bace behaviour unchanged even with CHEMBL-style IDs)
    rows = [
        _row("CHEMBL1", "c1ccncc1", 100.0, relation=None, pchembl=-np.log10(100.0 * 1e-9)),
        _row("CHEMBL2", "c1ccoc1", 200.0, relation=None, pchembl=-np.log10(200.0 * 1e-9)),
        _row("CHEMBL3", "C1CCCCC1", 50.0, relation=None, pchembl=-np.log10(50.0 * 1e-9)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys, tag="bace")

    assert len(clean) == 3  # nothing filtered, nothing merged
    assert "skipping censoring filter" in out
    assert "Censoring filter:" not in out


def test_non_chembl_ids_without_relation_column_skip_under_default_tag(tmp_path, load_script, capsys):
    # genuine non-ChEMBL input (bace-style molecule IDs like BACE_1) under the
    # default TAG: the skip is still allowed
    rows = [
        _row("BACE_1", "c1ccncc1", 100.0, relation=None, pchembl=-np.log10(100.0 * 1e-9)),
        _row("BACE_2", "c1ccoc1", 200.0, relation=None, pchembl=-np.log10(200.0 * 1e-9)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 2
    assert "skipping censoring filter" in out


def test_chembl_input_without_relation_column_fails_closed(tmp_path, load_script, capsys):
    # ChEMBL-format file (CHEMBL IDs, standard_value present) but NO
    # standard_relation column: this is the stale artifacts-v1 release shape.
    # It must abort with a non-zero exit (N5), not silently skip censoring —
    # otherwise 'IC50 > 10000 nM' would be kept as exact pIC50 5.0 (H1 again).
    rows = [
        _row("CHEMBL1", "c1ccncc1", 100.0, relation=None, pchembl=-np.log10(100.0 * 1e-9)),
        _row("CHEMBL2", "c1ccoc1", 10000.0, relation=None, pchembl=-np.log10(10000.0 * 1e-9)),
    ]
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows)

    with pytest.raises(SystemExit) as excinfo:
        mod02.main()

    msg = str(excinfo.value)
    assert "predates the 2026-10-11 data refresh" in msg
    assert "cannot be verified" in msg
    assert "standard_relation" in msg
    assert not clean_csv.exists()  # fail closed: no output written


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


def test_all_empty_relation_column_fails_closed(tmp_path, load_script, capsys):
    """A present-but-entirely-empty standard_relation column must abort.

    Mutant killed (6c1e3bb, PR #5 Low): without the fail-closed guard every
    row fails the '=' filter, a header-only clean CSV is written and the
    run still exits 0 -- silently 'cleaning' the dataset to nothing.
    """
    rows = [
        _row("CHEMBL1", "c1ccncc1", 100.0, relation=np.nan, pchembl=7.0),
        _row("CHEMBL2", "c1ccoc1", 200.0, relation=np.nan, pchembl=6.7),
    ]
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows)

    with pytest.raises(SystemExit) as excinfo:
        mod02.main()

    msg = str(excinfo.value)
    assert "standard_relation" in msg
    assert "header-only" in msg
    assert not clean_csv.exists()  # fail closed: no output written


def test_padded_chembl_ids_without_relation_column_fail_closed(
        tmp_path, load_script, capsys
):
    """Whitespace-padded CHEMBL IDs still count as ChEMBL-style input.

    Mutant killed (6c1e3bb, PR #5 Low): ID detection without strip() --
    " CHEMBL1 " would look non-ChEMBL, the censoring filter would be
    silently skipped and censored bounds kept as exact labels.
    """
    rows = [
        # BOTH ids carry a LEADING space: without the strip() the prefix
        # check fails on each of them and the mutant slips through
        _row(" CHEMBL1 ", "c1ccncc1", 100.0, relation=None, pchembl=7.0),
        _row(" CHEMBL2 ", "c1ccoc1", 200.0, relation=None, pchembl=6.7),
    ]
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows)

    with pytest.raises(SystemExit) as excinfo:
        mod02.main()

    assert "predates the 2026-10-11 data refresh" in str(excinfo.value)
    assert not clean_csv.exists()


def test_numeric_chembl_ids_without_relation_column_fail_closed(
        tmp_path, load_script, capsys
):
    """Bare numeric ChEMBL numbers (e.g. 4069737) count as ChEMBL-style.

    Mutant killed (6c1e3bb, PR #5 Low): ID detection without the
    fullmatch(r"\\d+") arm -- a numeric-ID raw file would skip the
    censoring filter and reintroduce the H1 censored-label bug.
    """
    rows = [
        _row("4069737", "c1ccncc1", 100.0, relation=None, pchembl=7.0),
        _row("233", "c1ccoc1", 200.0, relation=None, pchembl=6.7),
    ]
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows)

    with pytest.raises(SystemExit) as excinfo:
        mod02.main()

    assert "predates the 2026-10-11 data refresh" in str(excinfo.value)
    assert not clean_csv.exists()


def test_float_coerced_numeric_ids_without_relation_fail_closed(
        tmp_path, load_script, capsys
):
    """A pandas float-coerced ID column ("4069737.0") is still ChEMBL-style.

    Mutant killed (6c1e3bb extension, PR #6 Low): without the
    fullmatch(r"\\d+\\.0+") arm the float round-trip of a numeric ID
    column would look non-ChEMBL and skip the censoring filter.
    """
    rows = [
        _row("4069737.0", "c1ccncc1", 100.0, relation=None, pchembl=7.0),
        _row("233.0", "c1ccoc1", 200.0, relation=None, pchembl=6.7),
    ]
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows)

    with pytest.raises(SystemExit) as excinfo:
        mod02.main()

    assert "predates the 2026-10-11 data refresh" in str(excinfo.value)
    assert not clean_csv.exists()


def test_blank_id_column_without_relation_fail_closed(
        tmp_path, load_script, capsys
):
    """An all-empty molecule_chembl_id column cannot certify non-ChEMBL.

    Mutant killed (6c1e3bb extension, PR #6 Low): the old check counted
    only CHEMBL-style IDs, so an all-blank ID column (n_chembl == 0)
    silently skipped the censoring filter.
    """
    rows = [
        _row("", "c1ccncc1", 100.0, relation=None, pchembl=7.0),
        _row("", "c1ccoc1", 200.0, relation=None, pchembl=6.7),
    ]
    mod02, clean_csv = _prepare_02(tmp_path, load_script, rows)

    with pytest.raises(SystemExit) as excinfo:
        mod02.main()

    assert "blank ID cell" in str(excinfo.value)
    assert not clean_csv.exists()


def test_padded_relation_values_still_count_as_exact(
        tmp_path, load_script, capsys
):
    """Relation values are compared AFTER strip(): " = " is an exact row.

    Mutant killed (6c1e3bb extension, PR #6 Low): a raw-column comparison
    would drop every padded-equal row, writing a header-only clean CSV
    with exit 0.
    """
    pic50 = lambda ic50: -np.log10(ic50 * 1e-9)  # noqa: E731
    rows = [
        _row("CHEMBL1", "c1ccncc1", 100.0, " = ", pchembl=pic50(100.0)),
        _row("CHEMBL2", "c1ccoc1", 50.0, "= ", pchembl=pic50(50.0)),
        _row("CHEMBL3", "c1ccsc1", 200.0, " > ", pchembl=pic50(200.0)),
    ]
    clean, out = _run_02(tmp_path, load_script, rows, capsys)

    assert len(clean) == 2  # the two padded-equals rows survive
    assert set(clean["parent_smiles"]) == {"c1ccncc1", "c1ccoc1"}
    assert "removed 1 non-exact rows" in out
