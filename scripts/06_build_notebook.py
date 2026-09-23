"""Build notebooks/qsar_pipeline.ipynb from the step scripts.

Keeps the notebook and the scripts in sync automatically: each section of the
notebook is the corresponding script under scripts/, wrapped in markdown
explanations. Run after any script edit to regenerate the notebook.
"""
import os
import nbformat as nbf

BASE = os.path.join(os.path.dirname(__file__), "..")
NB_PATH = os.path.join(BASE, "notebooks", "qsar_pipeline.ipynb")

SECTIONS = [
    ("01d_download_incremental.py", "Step 1 - Download bioactivity data from ChEMBL",
     "Query the ChEMBL REST API for **EGFR (CHEMBL203)** activities: "
     "`standard_type=IC50`, `standard_units=nM`, binding/functional assays.\n\n"
     "The API was intermittently returning HTTP 500 during this project, so "
     "the downloader retries every page, saves progress after each page, and "
     "skips poisoned windows (see `scripts/01d_download_incremental.py`). "
     "A ready-made fallback source (MoleculeNet BACE-1) is provided in "
     "`scripts/01alt_download_bace.py` - set `QSAR_TAG=bace` to use it."),
    ("02_clean_data.py", "Step 2 - Data cleaning & pIC50 transform",
     "Keep nM rows, drop missing SMILES/IC50, convert to pIC50 "
     "(`-log10(IC50 * 1e-9)`) and collapse duplicate compounds to the "
     "**median** pIC50."),
    ("03_featurize.py", "Step 3 - Morgan fingerprints (RDKit)",
     "Each SMILES becomes a 2048-bit Morgan fingerprint (radius 2). "
     "Invalid SMILES that RDKit cannot parse are dropped."),
    ("04_train_and_evaluate.py", "Steps 4-6 - Split, train, evaluate",
     "Random 80/20 split vs **scaffold split** (Bemis-Murcko scaffolds kept "
     "whole - simulates predicting new chemotypes). RandomForest with "
     "GridSearchCV; metrics: R2 / RMSE / MAE + scatter plots."),
    ("05_explain_bits.py", "Step 6 (advanced) - What do the top bits mean?",
     "Map the most important fingerprint bits back to the molecular "
     "substructures that set them (RDKit bitInfo + PathToSubmol)."),
    ("07_qc_checks.py", "QC - distributions, outliers, honest importance",
     "Three sanity checks: (1) test-set pIC50 distributions of both splits "
     "(RMSE is scale-dependent), (2) audit of ultra-potent records "
     "(pIC50 > 9) by assay/source, (3) permutation importance as an "
     "unbiased cross-check of MDI, which is inflated for correlated "
     "fingerprint bits."),
]

INTRO = """# QSAR: machine-learning prediction of compound activity

End-to-end pipeline: **ChEMBL data -> cleaning -> Morgan fingerprints ->
RandomForest QSAR model -> evaluation -> substructure interpretation**.

The code cells are auto-generated from `scripts/` (see
`scripts/06_build_notebook.py`), so the notebook never drifts from the
runnable pipeline. Set the environment variable `QSAR_TAG` to `egfr`
(default) or `bace` to switch datasets.
"""


def read_script(name):
    with open(os.path.join(BASE, "scripts", name)) as f:
        return f.read()


def main():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(INTRO))
    for fname, title, md in SECTIONS:
        nb.cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{md}"))
        nb.cells.append(nbf.v4.new_code_cell(read_script(fname)))
    nbf.write(nb, NB_PATH)
    print(f"Wrote {os.path.abspath(NB_PATH)}")


if __name__ == "__main__":
    main()
