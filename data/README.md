# Data

Raw activity files (`raw/`) are not committed (see `.gitignore`) because they
can be re-downloaded from the sources below. Cleaned files (`processed/`)
are small and committed.

| File | Provenance |
|---|---|
| `raw/egfr_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL203`, `standard_type=IC50`, `standard_units=nM`, assay types B/F. Re-download: `python scripts/01d_download_incremental.py` |
| `raw/bace_activities.csv` | MoleculeNet BACE-1 (`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv`), pIC50 converted back to IC50 nM so the same cleaning pipeline applies. Re-download: `QSAR_TAG=bace python scripts/01alt_download_bace.py` |
| `raw/esol_delaney.csv` | MoleculeNet ESOL / Delaney solubility (1128 compounds). MoleculeNet's S3 mirror is gone (403); identical file vendored in the DeepChem repo. Re-download: `python scripts/gnn_06_esol_prep.py` |
| `processed/*_pic50_clean.csv` | One row per unique compound: `canonical_smiles`, median `pic50`, `ic50_nm_median`, `n_measurements`, `target` |
| `processed/*_fingerprints.npz` | `X` (n x 2048 uint8), `y` (pIC50), `smiles`, `target` |
| `processed/*_graphs.npz` | molecular graph cache (CSR-style): per-atom 9-feature array, edge index/attrs, counts, `y`, `smiles`, `scaffold_smiles`. Built by `scripts/gnn_02_build_graphs.py` |
| `processed/splits/*.npz` | frozen train/valid/test indices + scaffold labels (EGFR + ESOL). Regenerated and verified by `scripts/gnn_01_make_splits.py`; not committed (regenerable in seconds) |
