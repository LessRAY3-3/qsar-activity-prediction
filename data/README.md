# Data

Raw activity files (`raw/`) are not committed (see `.gitignore`) because they
can be re-downloaded from the sources below. Cleaned files (`processed/`)
are small and committed.

The sha256 and line count of each raw CSV are listed in `raw/sha256sums.txt`
- the manifest itself is the only committed exception to the raw-data
gitignore (force-added; no `!` rule). When the raw files are obtained via
the Release or re-downloaded, verify them against this manifest.

Two dating notes for the table below: the ChEMBL raw files were
re-downloaded on 2026-10-10 with three extra columns
(`standard_relation`, `data_validity_comment`, `activity_id`). Seven of
the eight kept their original row counts, but `egfr` was re-pulled and
rebuilt in the same pass (commit `e37c572`, "rebuild egfr (8k
early-stop)"): **8,936 -> 8,000 data rows** - the pre-refresh count is
the `egfr_activities.csv` inside `release/raw_data.tar.gz`, the current
one is `raw/sha256sums.txt`'s 8,001 lines (header included). The
manifest (dated 2026-10-10) is therefore the checksum of what is on disk
now; and a "unique compound" count in a provenance row is from the
original cleaning - after
the 2026-10-10 label/parent refresh the clean sets are smaller (e.g. a2a
1,746 -> 1,366, egfr_full 13,497 -> 10,729), with current sizes equal to
the row counts of the `processed/` files (tabulated in
`docs/reviews/readme-numbers-2026-10-11.md`, appendix A.2).

| File | Provenance |
|---|---|
| `raw/egfr_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL203`, `standard_type=IC50`, `standard_units=nM`, assay types B/F - **8,000 data rows** after the 2026-10-10 re-pull (8,936 before, commit `e37c572`); the other seven ChEMBL pulls kept their row counts. Re-download: `python scripts/01d_download_incremental.py` |
| `raw/bace_activities.csv` | MoleculeNet BACE-1 (`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv`, HTTP 200 when checked 2026-10-12, same 1,513 data rows), pIC50 converted back to IC50 nM so the same cleaning pipeline applies. Re-download: `QSAR_TAG=bace python scripts/01alt_download_bace.py` |
| `raw/esol_delaney.csv` | MoleculeNet ESOL / Delaney solubility (1128 compounds). The MoleculeNet S3 mirror is serving again: checked 2026-10-12, `datasets/delaney-processed.csv` returns HTTP 200 and is byte-identical to this file (sha256 `8c06a76f0c6487d2…`), though the old `datasets/delaney.csv` name still returns 403; the identical file is also vendored in the DeepChem repo (what `scripts/gnn_06_esol_prep.py` fetches). Re-download: `python scripts/gnn_06_esol_prep.py` |
| `raw/egfr_full_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL203`, `standard_type=IC50`, `standard_units=nM`, assay types B/F, **full pull** (`--max-rows 0`, no early stop) - 24,560 raw rows -> **13,497 unique compounds** (downloaded 2026-10-04). Relationship to `raw/egfr_activities.csv` (egfr v1): egfr_full is a strict superset - the superset check found all **6165/6165** v1 compounds, with median abs(ΔpIC50) **0.0000** and max **2.98** (a few compounds gained new measurements in the full pull, moving their median). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL203 --tag egfr_full --max-rows 0` |
| `raw/a2a_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL251` (Adenosine receptor A2a), IC50/nM, assay types B/F - 2,136 raw rows -> **1,746 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL251 --tag a2a` |
| `raw/abl1_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL1862` (Tyrosine-protein kinase ABL1), IC50/nM, assay types B/F - 5,142 raw rows -> **2,698 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL1862 --tag abl1` |
| `raw/mpro_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL4523582` (Replicase polyprotein 1ab - the ChEMBL target behind the `mpro` tag), IC50/nM, assay types B/F - 6,050 raw rows -> **4,424 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL4523582 --tag mpro` |
| `raw/hivpr_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL243` (Human immunodeficiency virus type 1 protease), IC50/nM, assay types B/F - 3,527 raw rows -> **2,799 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL243 --tag hivpr` |
| `raw/herg_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL240` (Voltage-gated inwardly rectifying potassium channel KCNH2 / hERG), IC50/nM, assay types B/F - 14,609 raw rows -> **12,019 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL240 --tag herg` |
| `raw/vegfr2_activities.csv` | ChEMBL REST API: `target_chembl_id=CHEMBL279` (VEGFR2 / KDR), IC50/nM, assay types B/F - 16,554 raw rows -> **11,797 unique compounds** (downloaded 2026-10-03). Downloader: `scripts/01e_download_chembl.py`. Re-download: `python scripts/01e_download_chembl.py --target-chembl-id CHEMBL279 --tag vegfr2`. Formerly tagged `mapk14`. |
| `processed/*_pic50_clean.csv` | One row per unique parent compound, sorted by `parent_smiles`: `parent_smiles` (dedup key - `LargestFragmentChooser` + `Uncharger`, see `docs/methodology.md` Data QC), `canonical_smiles` (first original SMILES, kept for provenance), median `pic50`, `ic50_nm_median`, `n_measurements`, `n_structural_variants`, `molecule_chembl_id`, `target`. The 8 ChEMBL-tag files were rebuilt on 2026-10-10/11 with the `standard_relation == "="` label filter and parent dedup (`docs/reviews/h1-sensitivity-2026-10-10.md`); `bace` predates the refresh and still carries the old `canonical_smiles`-based schema (no parent columns - its raw has no relation column, so the label filter never applies there) |
| `processed/*_fingerprints.npz` | `X` (n x 2048 uint8), `y` (pIC50), `smiles` (the standardized parent SMILES, row-aligned with the clean csv; falls back to `canonical_smiles` for pre-refresh inputs), `target` |
| `processed/*_graphs.npz` | molecular graph cache (CSR-style): per-atom 9-feature array, edge index/attrs, counts, `y`, `smiles` (parent SMILES, same row order as the fingerprints npz), `scaffold_smiles`. Built by `scripts/gnn_02_build_graphs.py` |
| `processed/splits/*.npz` | frozen train/valid/test indices + scaffold labels for every tag, regenerated and verified bit-identically by `scripts/gnn_01_make_splits.py` (aborts on any metric mismatch). **Not committed** for the new targets: each npz stores full-width SMILES arrays and lands at 1.4-42 MB (regenerable in seconds, same rule as `processed/*_graphs.npz`). The `egfr`/`esol` copies were committed earlier and remain in the repo |
