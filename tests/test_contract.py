"""Cross-script path contracts: what 01d writes is what 02 reads, and so on.

The modules are loaded by file path (names start with digits, so plain
``import`` cannot address them); importing them only evaluates module-level
constants (``os.environ.get``, ``os.path.join``) and never runs main().
"""
import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(relpath):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_01d_raw_output_is_02_raw_input(monkeypatch):
    monkeypatch.delenv("QSAR_TAG", raising=False)  # pin the default tag (egfr)
    mod01d = _load("scripts/01d_download_incremental.py")
    mod02 = _load("scripts/02_clean_data.py")

    assert os.path.basename(mod02.IN_CSV) == os.path.basename(mod01d.OUT_CSV)
    assert os.path.basename(mod01d.OUT_CSV) == "egfr_activities.csv"
    # 02's required columns must be part of 01d's download schema
    required = {
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "target_pref_name",
    }
    assert required <= set(mod01d.COLUMNS)


def test_02_clean_output_is_03_clean_input(monkeypatch):
    monkeypatch.delenv("QSAR_TAG", raising=False)
    mod02 = _load("scripts/02_clean_data.py")
    mod03 = _load("scripts/03_featurize.py")

    assert os.path.basename(mod03.IN_CSV) == os.path.basename(mod02.OUT_CSV)
    assert os.path.basename(mod02.OUT_CSV) == "egfr_pic50_clean.csv"
