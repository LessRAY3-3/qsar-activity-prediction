# QSAR Activity Prediction with Machine Learning

Predicting the bioactivity (pIC50) of compounds against a drug target from
their molecular structure, using the classic QSAR workflow:
**ChEMBL data -> cleaning -> Morgan fingerprints -> RandomForest -> evaluation -> interpretation**.

## 1. What this project does

Given a compound's structure (SMILES), predict how potently it inhibits the
target (pIC50). This is the computational first stage of **virtual
screening**: rank large virtual libraries in silico before spending wet-lab
budget on the top hits.

## 2. Results

EGFR (primary dataset, 6165 unique compounds, test = 20%):

| Split | R2 | RMSE | MAE |
|---|---|---|---|
| random | **0.747** | 0.692 | 0.492 |
| scaffold | **0.562** | 0.950 | 0.708 |

The gap between the two splits is the expected fingerprint of analogue
leakage: a random split lets near-identical compounds land on both sides and
inflates the score; the scaffold split (whole Bemis-Murcko scaffolds held
out) approximates predicting genuinely new chemotypes - the honest number.

Predicted vs actual (EGFR, random split):

![pred vs actual](figures/pred_vs_actual_random_egfr.png)

Fallback dataset (BACE-1, 1513 compounds):

| Split | R2 | RMSE | MAE |
|---|---|---|---|
| random | 0.646 | 0.792 | 0.572 |
| scaffold | 0.640 | 0.616 | 0.481 |

For BACE the two RMSEs are inverted (random > scaffold) **because the test
sets have different potency spreads** - the random test set is wider
(std 1.33, range 2.7-10.5) than the scaffold one (std 1.03, range 3.9-9.0).
R2 is scale-free, RMSE is not. See `figures/testset_hist_bace.png`.

## 3. Data

Two sources, switched via the `QSAR_TAG` environment variable:

| Dataset | Target | Compounds | Source |
|---|---|---|---|
| `egfr` (default) | EGFR (CHEMBL203) | 6165 | ChEMBL REST API |
| `bace` (fallback) | Beta-secretase 1 | 1513 | MoleculeNet BACE |

Primary data comes from **ChEMBL** (EBI): `target_chembl_id=CHEMBL203`,
`standard_type=IC50`, `standard_units=nM`, binding/functional assays. While
this project was built the ChEMBL API was intermittently returning HTTP 500
(server-side outage; a VPN did not help). The downloader
(`scripts/01d_download_incremental.py`) therefore retries every page,
flushes each page to disk, and skips poisoned result windows - 8936 rows
survived into 6165 unique compounds.

Cleaning (`scripts/02_clean_data.py`):
- keep only IC50 reported in **nM** (one unit -> comparable numbers)
- drop rows with missing SMILES/IC50 and non-positive IC50
- IC50 (nM) -> **pIC50 = -log10(IC50 x 1e-9)**; cross-checked against
  ChEMBL's own `pchembl_value` (100% agreement)
- compounds measured multiple times are collapsed to the **median** pIC50

## 4. Method

- **Featurization**: 2048-bit Morgan fingerprints (radius 2) with RDKit.
  Each bit encodes a circular atomic neighbourhood, so similar structures
  yield similar vectors.
- **Model**: `RandomForestRegressor`, tuned with `GridSearchCV` (5-fold CV
  on the training set only): n_estimators x max_depth x min_samples_split.
- **Splits**: random 80/20 (`random_state=42`) and a **scaffold split**
  implemented with RDKit only (no DeepChem dependency).

## 5. Interpretation

EGFR top fingerprint bits mapped back to the substructures that set them
(`scripts/05_explain_bits.py`):

![top bits](figures/top_bits_egfr.png)

The dominant bit (MDI 0.233, carried by 2762/6165 compounds) decodes to an
**aminopyrimidine** - the classic kinase hinge-binding motif, exactly what a
medicinal chemist would expect for EGFR.

Because RandomForest's built-in (MDI) importance is known to be inflated
for correlated features - and 2048 fingerprint bits contain many chemically
equivalent ones - every importance claim above was cross-checked with
**permutation importance on the held-out test set**
(`scripts/07_qc_checks.py`):

- EGFR: bit 1367 drops test R2 by **0.60** when permuted - the MDI ranking
  *under*casts it. Top-100 MDI-vs-permutation Spearman = 0.664.
- BACE: agreement is weaker (Spearman 0.349). MDI ranks the rare
  fluorinated-aromatic bit 964 first; permutation ranks the broadly carried
  oxygen environment first and bit 964 third - still important, but the
  substructure story must be told at the set level, not as "the one bit".

![mdi vs perm](figures/importance_mdi_vs_perm_egfr.png)

## 6. Data QC

- **Ultra-potent audit** (pIC50 > 9, i.e. IC50 < 1 nM): EGFR has 326 such
  rows (3.7%), but they spread across many independent assays and documents
  (largest single source: 14%) - consistent with genuinely potent
  inhibitors, not unit mislabels. BACE has 8 rows (0.5%) from a single FRET
  assay, also kept. No filtering rule was needed; the audit itself is the
  deliverable.
- **Inactive-end censoring**: the EGFR scatter plot shows vertical
  striations at low pIC50 (e.g. many compounds pinned at exactly 5.0 =
  IC50 10000 nM) - assay detection limits reporting "inactive above X".
  Known QSAR nuisance, visible in the data.
- **Test-set distributions** are compared before trusting any RMSE
  (see section 2).

## Repository layout

```
qsar-activity-prediction/
├── README.md
├── data/
│   ├── raw/            # downloaded activities (gitignored, see data/README.md)
│   └── processed/      # cleaned pIC50 CSVs + fingerprint npz
├── notebooks/          # qsar_pipeline.ipynb (auto-generated from scripts/)
├── scripts/            # runnable pipeline, one step per file
├── figures/            # scatter plots, substructure grids, QC plots
├── models/             # trained RandomForests (gitignored, large)
└── results/            # metrics.json, top_bits.csv, permutation_importance.csv
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export QSAR_TAG=egfr   # or: bace
python scripts/01d_download_incremental.py     # egfr only
python scripts/02_clean_data.py
python scripts/03_featurize.py
python scripts/04_train_and_evaluate.py
python scripts/05_explain_bits.py
python scripts/07_qc_checks.py
```

## Why pIC50?

Raw IC50 spans several orders of magnitude. The heavy tail dominates
least-squares training; pIC50 is roughly normal and is the standard QSAR
target transform.

## Interview talking points

1. **Why pIC50** - order-of-magnitude target standardisation.
2. **Why scaffold split** - random splits leak analogues across
   train/test and inflate scores; scaffold splits emulate new-chemotype
   prediction. This project's EGFR numbers show the effect directly
   (0.747 random vs 0.562 scaffold).
3. **Honest importance** - MDI is biased under correlated features; all
   substructure claims were validated with permutation importance on the
   held-out set before being made.
4. **Industry use** - first stage of virtual screening: in silico triage of
   large libraries before experimental validation, cutting wet-lab cost by
   orders of magnitude.
