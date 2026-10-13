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
