# Data

Raw activity files (`raw/`) are not committed (see `.gitignore`) because they
can be re-downloaded from the sources below. Cleaned files (`processed/`)
are small and committed.

The sha256 and line count of each raw CSV are listed in `raw/sha256sums.txt`
(the `data/` directory is gitignored; when the raw files are obtained via the
Release or re-downloaded, verify them against this manifest).

| File | Provenance |
|---|---|
| `raw/egfr_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL203`, `standard_type=IC50`, `standard_units=nM`, assay types B/F. Re-download: `python scripts/01d_download_incremental.py` |
| `raw/bace_activities.csv` | MoleculeNet BACE-1 (`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv`), pIC50 converted back to IC50 nM so the same cleaning pipeline applies. Re-download: `QSAR_TAG=bace python scripts/01alt_download_bace.py` |
| `raw/esol_delaney.csv` | MoleculeNet ESOL / Delaney solubility (1128 compounds). MoleculeNet's S3 mirror is gone (403); identical file vendored in the DeepChem repo. Re-download: `python scripts/gnn_06_esol_prep.py` |
| `raw/egfr_full_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL203`, `standard_type=IC50`, `standard_units=nM`, assay types B/F, **full pull** (`--max-rows 0`, no early stop) - 24,560 raw rows -> **13,497 unique compounds** (downloaded 2026-10-04). Relationship to `raw/egfr_activities.csv` (egfr v1): egfr_full is a strict superset - the superset check found all **6165/6165** v1 compounds, with median abs(ΔpIC50) **0.0000** and max **2.98** (a few compounds gained new measurements in the full pull, moving their median). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL203 --tag egfr_full --max-rows 0` |
| `raw/a2a_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL251` (Adenosine receptor A2a), IC50/nM, assay types B/F - 2,136 raw rows -> **1,746 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL251 --tag a2a` |
| `raw/abl1_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL1862` (Tyrosine-protein kinase ABL1), IC50/nM, assay types B/F - 5,142 raw rows -> **2,698 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL1862 --tag abl1` |
| `raw/mpro_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL4523582` (Replicase polyprotein 1ab - the ChEMBL target behind the `mpro` tag), IC50/nM, assay types B/F - 6,050 raw rows -> **4,424 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL4523582 --tag mpro` |
| `raw/hivpr_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL243` (Human immunodeficiency virus type 1 protease), IC50/nM, assay types B/F - 3,527 raw rows -> **2,799 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL243 --tag hivpr` |
| `raw/herg_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL240` (Voltage-gated inwardly rectifying potassium channel KCNH2 / hERG), IC50/nM, assay types B/F - 14,609 raw rows -> **12,019 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL240 --tag herg` |
| `raw/vegfr2_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL279` (VEGFR2 / KDR), IC50/nM, assay types B/F - 16,554 raw rows -> **11,797 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL279 --tag vegfr2`. Formerly tagged `mapk14`. |
| `processed/*_pic50_clean.csv` | One row per unique compound: `canonical_smiles`, median `pic50`, `ic50_nm_median`, `n_measurements`, `target` |
| `processed/*_fingerprints.npz` | `X` (n x 2048 uint8), `y` (pIC50), `smiles`, `target` |
| `processed/*_graphs.npz` | molecular graph cache (CSR-style): per-atom 9-feature array, edge index/attrs, counts, `y`, `smiles`, `scaffold_smiles`. Built by `scripts/gnn_02_build_graphs.py` |
| `processed/splits/*.npz` | frozen train/valid/test indices + scaffold labels for every tag, regenerated and verified bit-identically by `scripts/gnn_01_make_splits.py` (aborts on any metric mismatch). **Not committed** for the new targets: each npz stores full-width SMILES arrays and lands at 1.4-42 MB (regenerable in seconds, same rule as `processed/*_graphs.npz`). The `egfr`/`esol` copies were committed earlier and remain in the repo |
