"""07_qc_checks ultra-potent audit counts only label-policy-kept rows.

The headline in ``check_ultra_potent`` must count rows the model actually
saw (``standard_relation == "="``, the policy scripts/02_clean_data.py
applies); raw rows above pIC50 9 with ``<``/``>``/NaN relations are
reported on a separate line and never enter the headline fraction.

Assertions anchor on the numbers (fraction pairs and the IC50 nM range),
not on console wording, so message rewrites by other owners keep the test
green while a mutant that counts every hot row fails it.
"""
import re

import pandas as pd

# standard_value is nM: pic50 = -log10(value * 1e-9), so value < 1 -> pic50 > 9.
# Rows 0-1: kept hot ('='); rows 2-3: hot but dropped by the label policy;
# rows 4-5: cold '=' rows, so the headline denominator (kept rows) is 4.
RAW_ROWS = {
    "standard_value": [0.5, 0.1, 0.4, 0.6, 5000.0, 2000.0],
    "standard_relation": ["=", "=", "<", ">", "=", "="],
    "canonical_smiles": ["CCO"] * 6,
}


def test_ultra_potent_headline_counts_only_relation_kept_rows(
        tmp_path, monkeypatch, load_script, capsys
):
    raw = tmp_path / "raw.csv"
    pd.DataFrame(RAW_ROWS).to_csv(raw, index=False)

    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(raw))
    mod.check_ultra_potent()
    out = capsys.readouterr().out

    # exactly one "hot / kept" fraction: 2 kept-hot of 4 '=' rows. The two
    # hot '<'/'>' rows must not be added (mutant: 4 / 4 or 4 / 6).
    assert re.findall(r"(\d+) / (\d+)", out) == [("2", "4")]
    # kept-hot IC50 range is 0.1..0.5 nM; 0.6 belongs to the '>' row and
    # must never show up as the range maximum
    assert "0.1000 .. 0.5000" in out


def test_ultra_potent_without_relation_column_counts_every_row(
        tmp_path, monkeypatch, load_script, capsys
):
    """Sources without standard_relation (BACE) keep the whole-file count."""
    raw = tmp_path / "raw_bace.csv"
    pd.DataFrame({
        "standard_value": [0.5, 0.6, 5000.0],
        "canonical_smiles": ["CCO", "CCC", "CCCC"],
    }).to_csv(raw, index=False)

    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(raw))
    mod.check_ultra_potent()
    out = capsys.readouterr().out

    assert re.findall(r"(\d+) / (\d+)", out) == [("2", "3")]
    assert "0.5000 .. 0.6000" in out


def test_ultra_potent_mirrors_nM_unit_filter(tmp_path, monkeypatch,
                                             load_script, capsys):
    """Rows with standard_units != 'nM' never become labels in 02, so they
    must not enter 07's labelled-rows denominator either (6c1e3bb, PR #5
    Low: "the labelled rows count skips 02's nM and empty-SMILES filters").

    Mutant killed: without the unit-filter mirror the two uM rows are
    scored as nM (pic50 computed from the raw number), turning the
    headline into 3/4 instead of 1/2.
    """
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        # 0.1 uM and 0.05 uM are hot ONLY if mis-unit-converted to nM
        "standard_value": [0.5, 5000.0, 0.1, 0.05],
        "standard_units": ["nM", "nM", "uM", "uM"],
        "standard_relation": ["=", "=", "=", "="],
        "canonical_smiles": ["CCO", "CCC", "CCO", "CCCC"],
    }).to_csv(raw, index=False)

    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(raw))
    mod.check_ultra_potent()
    out = capsys.readouterr().out

    # after the unit filter: 2 nM rows, 1 hot (mutant: ("3", "4"))
    assert re.findall(r"(\d+) / (\d+)", out) == [("1", "2")]
    assert "02 unit filter" in out
    assert "0.5000 .. 0.5000" in out


def test_ultra_potent_mirrors_empty_smiles_filter(tmp_path, monkeypatch,
                                                  load_script, capsys):
    """Rows with empty canonical_smiles never become labels in 02, so they
    must not enter 07's labelled-rows denominator either (6c1e3bb, PR #5
    Low, same mirror).

    Mutant killed: without the SMILES-filter mirror the empty-SMILES hot
    row counts, turning the headline into 3/4 instead of 2/3.
    """
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "standard_value": [0.5, 0.1, 0.05, 5000.0],
        "standard_relation": ["=", "=", "=", "="],
        # " " survives read_csv as a non-NaN whitespace SMILES -- exactly
        # what 02's str.strip() != "" filter exists to drop
        "canonical_smiles": ["CCO", " ", "CC", "CCC"],
    }).to_csv(raw, index=False)

    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(raw))
    mod.check_ultra_potent()
    out = capsys.readouterr().out

    # after the SMILES filter: 3 rows, 2 hot (mutant: ("3", "4"))
    assert re.findall(r"(\d+) / (\d+)", out) == [("2", "3")]
    assert "02 SMILES filter" in out
    assert "0.0500 .. 0.5000" in out


def test_ultra_potent_skips_non_chembl_tag(tmp_path, monkeypatch,
                                           load_script, capsys):
    """bace/esol never run 02's censoring policy, so the labelled-rows
    audit is skipped for them with an explanation (6c1e3bb extension,
    PR #6 Low) -- not silently counted as if a policy existed."""
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "standard_value": [0.5, 5000.0],
        "standard_relation": ["=", "="],
        "canonical_smiles": ["CCO", "CCC"],
    }).to_csv(raw, index=False)

    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(raw))
    monkeypatch.setattr(mod, "TAG", "bace")
    mod.check_ultra_potent()
    out = capsys.readouterr().out

    assert "skipped" in out and "bace" in out
    assert re.findall(r"(\d+) / (\d+)", out) == []   # no headline fraction


def test_ultra_potent_skips_chembl_ids_without_relation_column(
        tmp_path, monkeypatch, load_script, capsys
):
    """A ChEMBL-style file without standard_relation is refused by 02, so
    there is no label policy to audit here (6c1e3bb extension, PR #6
    Low) -- the old else-branch asserted a kept-set that does not exist."""
    raw = tmp_path / "raw.csv"
    pd.DataFrame({
        "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
        "standard_value": [0.5, 5000.0],
        "canonical_smiles": ["CCO", "CCC"],
    }).to_csv(raw, index=False)

    mod = load_script("07_qc_checks")
    monkeypatch.setattr(mod, "RAW", str(raw))
    mod.check_ultra_potent()
    out = capsys.readouterr().out

    assert "skipped" in out and "no standard_relation column" in out
    assert re.findall(r"(\d+) / (\d+)", out) == []   # no headline fraction
