# QSAR Activity Prediction with Machine Learning

![CI](https://github.com/LessRAY3-3/qsar-activity-prediction/actions/workflows/ci.yml/badge.svg)

Predicting how potently a compound inhibits a target (pIC50) from
structure, two ways under identical verified train/test conditions:
classic QSAR (Morgan fingerprints -> RandomForest) and a GIN graph
network, across eight ChEMBL targets plus a MoleculeNet fallback.
Headline, an honest negative result: **the GIN does not beat the tuned
fingerprint baseline on this panel**. Deep detail:
[docs/methodology.md](docs/methodology.md); interview prep:
[docs/interview.md](docs/interview.md).

## Results

### EGFR (primary dataset, 6165 unique compounds, test = 20%)

| Split | R2 | RMSE | MAE |
|---|---|---|---|
| random | **0.747** | 0.692 | 0.492 |
| scaffold | **0.562** | 0.950 | 0.708 |

The random-vs-scaffold gap is the expected fingerprint of analogue
leakage: random splits let near-identical compounds land on both sides,
inflating scores; the scaffold split (whole Bemis-Murcko scaffolds held
out) approximates genuinely new chemotypes - the honest number.

![pred vs actual](figures/pred_vs_actual_random_egfr.png)

### Fallback dataset (BACE-1, 1513 compounds)

| Split | R2 | RMSE | MAE |
|---|---|---|---|
| random | 0.646 | 0.792 | 0.572 |
| scaffold | 0.640 | 0.616 | 0.481 |

For BACE the two RMSEs are inverted (random > scaffold) **because the
test sets have different potency spreads** - random is wider (std 1.33,
range 2.7-10.5) vs scaffold (std 1.03, range 3.9-9.0). R2 is scale-free,
RMSE is not. See `figures/testset_hist_bace.png`.

### Eight-target panel

Test R2; GIN = 3-seed mean +- std; **Δ = 3-seed mean GIN − RF**, the
effect size every verdict below uses:

| Tag | Target (ChEMBL id) | Family | n | RF rand | GIN rand | Δ rand (mean) | RF scaf | GIN scaf | Δ scaf (mean) |
|---|---|---|---|---|---|---|---|---|---|
| `a2a` | A2a adenosine receptor (CHEMBL251) | GPCR | 1,746 | 0.733 | 0.687±0.003 | −0.046 | 0.691 | 0.666±0.004 | −0.026 |
| `abl1` | ABL1 tyrosine kinase (CHEMBL1862) | kinase | 2,698 | 0.792 | 0.752±0.006 | −0.039 | 0.723 | 0.710±0.037 | −0.013 |
| `egfr` | EGFR (CHEMBL203) | kinase | 6,165 | 0.747 | 0.690±0.001 | −0.056 | 0.562 | 0.526±0.016 | −0.036 |
| `egfr_full` | EGFR, full B/F pull (CHEMBL203) | kinase | 13,497 | 0.759 | 0.707±0.006 | −0.053 | 0.593 | 0.558±0.010 | −0.035 |
| `herg` | hERG / KCNH2 channel (CHEMBL240) | ion channel | 12,019 | 0.615 | 0.616±0.017 | **+0.000 (tie)** | 0.430 | 0.417±0.007 | −0.013 |
| `hivpr` | HIV-1 protease (CHEMBL243) | viral protease | 2,799 | 0.737 | 0.726±0.005 | −0.011 | 0.554 | 0.527±0.028 | −0.027 |
| `vegfr2` | **VEGFR2 / KDR** (CHEMBL279) | kinase | 11,797 | 0.736 | 0.678±0.007 | −0.058 | 0.617 | 0.611±0.024 | **−0.006** |
| `mpro` | Replicase polyprotein 1ab / Mpro (CHEMBL4523582) | viral protease | 4,424 | 0.730 | 0.707±0.003 | −0.023 | 0.509 | 0.408±0.022 | **−0.102** |

BACE is not in the table (RF baseline only). Columns from
`results/multi_target/summary.csv`; Δ computed at full precision, then
rounded.

### Reproduce: fast path

```bash
pip install -r requirements.txt
python scripts/fetch_artifacts.py --all   # 下载训练好的 models/ 和 data/raw/（GitHub Release）
```

From here Phase 2 runs directly (`gnn_01_make_splits.py` onward): saved
RF models and regenerated splits verify against the published metrics,
no retraining. Full retraining, incl. phase 3:
[methodology](docs/methodology.md).

## Data

Sources, switched via the `QSAR_TAG` environment variable. All ChEMBL
targets come from the same generic downloader
(`scripts/01e_download_chembl.py`, `standard_type=IC50`,
`standard_units=nM`, assay types B/F):

| Dataset tag | Target | ChEMBL id | Unique compounds | Source |
|---|---|---|---|---|
| `egfr` (default) | EGFR | CHEMBL203 | 6165 | ChEMBL REST API (`01d`, early-stopped pull) |
| `egfr_full` | EGFR, full B/F pull | CHEMBL203 | 13497 | ChEMBL REST API (`01e`, `--max-rows 0`) |
| `bace` (fallback) | Beta-secretase 1 | – | 1513 | MoleculeNet BACE |
| `a2a` | A2a adenosine receptor | CHEMBL251 | 1746 | ChEMBL REST API (`01e`) |
| `abl1` | ABL1 tyrosine kinase | CHEMBL1862 | 2698 | ChEMBL REST API (`01e`) |
| `mpro` | Replicase polyprotein 1ab / Mpro | CHEMBL4523582 | 4424 | ChEMBL REST API (`01e`) |
| `hivpr` | HIV-1 protease | CHEMBL243 | 2799 | ChEMBL REST API (`01e`) |
| `herg` | hERG / KCNH2 channel | CHEMBL240 | 12019 | ChEMBL REST API (`01e`) |
| `vegfr2` | **VEGFR2 / KDR** | CHEMBL279 | 11797 | ChEMBL REST API (`01e`) |

The `vegfr2` dataset was formerly tagged `mapk14` (misnomer fixed 2026-10-08).

Primary data: **ChEMBL** (EBI), IC50/nM, binding/functional assays. The
resilient downloader (page retries, disk flushes; the original `egfr`
pull survived intermittent HTTP 500s) and full per-file provenance:
[data/README.md](data/README.md). Counts: `egfr` 8936 rows -> 6165
unique; `egfr_full` (`--max-rows 0`) 24560 -> 13497 unique, a superset
check confirming all 6165 `egfr` v1 compounds (median |ΔpIC50| 0.0000).

Cleaning (`scripts/02_clean_data.py`): IC50 in **nM** only; drop
missing/non-positive rows; **pIC50 = -log10(IC50 x 1e-9)** (100%
agreement with ChEMBL's `pchembl_value`); repeats collapsed to the
**median**.

## Method

- **Featurization**: 2048-bit Morgan fingerprints (radius 2), RDKit.
- **Model**: `RandomForestRegressor` + `GridSearchCV` (5-fold CV on train
  only) over n_estimators x max_depth x min_samples_split.
- **Splits**: random 80/20 (`random_state=42`) + RDKit-only **scaffold
  split** (no DeepChem); phase 2 mirrors these exactly.
- **Why pIC50**: raw IC50's heavy tail spans orders of magnitude and
  dominates least-squares training; pIC50 is roughly normal - the
  standard QSAR target.

## Interpretation & data QC

EGFR top fingerprint bits mapped back to the substructures that set them
(`scripts/05_explain_bits.py`):

![top bits](figures/top_bits_egfr.png)

The dominant bit (MDI 0.233, carried by 2762/6165 compounds) decodes to
an **aminopyrimidine** - the classic kinase hinge-binding motif. MDI is
inflated by correlated features, so every importance claim was
cross-checked with **permutation importance on the held-out test set**
(numbers: [methodology](docs/methodology.md)).

![mdi vs perm](figures/importance_mdi_vs_perm_egfr.png)

Data QC (ultra-potent audit, censoring, test-set spreads): audited, no
filtering rule - the audit is the deliverable
([methodology](docs/methodology.md)).

## GNN vs traditional QSAR: a controlled comparison

**On the same molecules and the same train/test partitions, does a graph
neural network beat Morgan fingerprints + RandomForest?**

- Same data (cleaned EGFR, 6165, pIC50); same splits regenerated with
  identical code (random_state=42, deterministic scaffolds), **verified
  bit-identically**: saved RF models reproduce the published metrics to
  7 decimals on the regenerated indices (`gnn_01_make_splits.py` aborts
  on mismatch; indices in `data/processed/splits/`).
- The GIN gets a 10% validation carve-out from *train* (seed 42); test
  stays untouched, keeping the comparison paired.
- Molecules are graphs (MoleculeNet/OGB convention); architecture:
  [methodology](docs/methodology.md).

| Split | RF R2 / RMSE | GIN R2 / RMSE (seed 42) | GIN R2 mean +- std (3 seeds) |
|---|---|---|---|
| random | 0.747 / 0.692 | 0.691 / 0.765 | 0.690 +- 0.001 |
| scaffold | 0.562 / 0.950 | 0.544 / 0.969 | 0.526 +- 0.016 |

**On this dataset the GIN does not beat the fingerprint RF.** The random
gap (-0.06 R2) is stable across seeds; the scaffold gap sits within seed
noise (RF ~2 std above the GIN mean). Both lose ~0.2 R2 from random to
scaffold - novel chemotypes are hard for both feature types. Tuning
negatives, error analysis, ESOL: [methodology](docs/methodology.md).

Same test molecules, coloured by scaffold:

![compare scatter](figures/compare_pred_vs_actual_egfr_scaffold.png)

**Do the models fail on the same molecules?** Their worst test
predictions overlap far above chance on both splits: many failures are
**molecule-intrinsic** (noisy or assay-censored measurements), not
model-specific - a data-quality conclusion, not a model-ranking one.
Chemotypes still fail asymmetrically (chromones hurt the GIN,
thienopyrimidines hurt RF; numbers, figure:
[methodology](docs/methodology.md)).

## Cross-target generality & data scaling

**Finding 1 - on random splits the GIN never clearly leads.** Δ(mean) runs
from **+0.000 (tie)** (`herg`) down to −0.058 (`vegfr2`/VEGFR2): 7 of 8
targets are negative and the eighth is a tie. With analogues leaking across the
split, message passing buys nothing over circular substructure
counts on any target in the panel - the EGFR conclusion above is not an
EGFR artefact.

**Finding 2 - on scaffold splits no target gives the GIN a clear lead, and
the deficit's size is a property of the target, not of n.** All eight
Δ(mean) are negative: closest to parity `vegfr2`/VEGFR2 at **−0.006**, worst
`mpro` at **−0.102**, with `abl1` (2,698 compounds) and `herg` (12,019) both
at **−0.013** to three decimals - a 4.5x difference in n and the same
deficit, which is what "Δ does not follow n" looks like. The single-seed
column would have told a different story (two scaffold "wins": `abl1`
+0.030, VEGFR2 +0.028); both flip sign under the 3-seed mean - the
3-seed-mean convention (see [docs/interview.md](docs/interview.md)) - and
the scaling verdict below has no measured crossover either. What varies is
the *magnitude*: scaffold diversity and assay structure set how far behind
the GIN falls - the sign never flips.

![delta vs n](figures/multi_target/delta_vs_n.png)

Learning curve (n = 9,717, no crossover), extrapolated n\* with its
weak-identifiability caveat, training-budget caveat, cross-target failure
replication: [methodology](docs/methodology.md).

## Giving the GNN a fair chance (bounded tuning + GINE)

Sections above deliberately froze the GIN recipe - every Δ is a
**frozen-recipe** Δ. `experiments/gnn_fairness.py` (36 runs/target) gives
it a bounded chance: on the three nearest-parity targets
(`vegfr2` −0.006, `herg` −0.013, `abl1` −0.013), 8 configs (hidden
{128, 256} x layers {4, 5} x dropout {0.2, 0.3}) train **only on the
scaffold split**, selected by **3-seed mean validation RMSE**, never a
single seed. Winners: `vegfr2` 256/4L/0.2 (0.7056), `herg` 256/5L/0.2
(0.5145), `abl1` 256/4L/0.2 (0.8427) - hidden 256 on all three. A **GINE**
variant adds a `BondEncoder` (`BOND_FEATURE_DIMS = [13, 7, 2]`); all else
is `gnn_03`'s.

Test R2, 3-seed mean ± std; Δ = GIN − RF:

| Target | Model | random R2 | Δ rand | scaffold R2 | Δ scaf |
|---|---|---|---|---|---|
| `vegfr2` (VEGFR2) | RF | 0.736 | — | 0.617 | — |
| | baseline GIN | 0.678±0.007 | −0.058 | 0.611±0.024 | −0.006 |
| | tuned GIN | 0.697±0.006 | −0.039 | 0.622±0.012 | **+0.005** |
| | GINE | 0.674±0.003 | −0.062 | 0.594±0.013 | −0.023 |
| `herg` | RF | 0.615 | — | 0.430 | — |
| | baseline GIN | 0.616±0.017 | +0.000 (tie) | 0.417±0.007 | −0.013 |
| | tuned GIN | 0.612±0.016 | −0.003 | 0.455±0.009 | **+0.025** |
| | GINE | 0.593±0.025 | −0.023 | 0.454±0.007 | **+0.024** |
| `abl1` | RF | 0.792 | — | 0.723 | — |
| | baseline GIN | 0.752±0.006 | −0.039 | 0.710±0.037 | −0.013 |
| | tuned GIN | 0.756±0.014 | −0.035 | 0.693±0.063 | −0.030 |
| | GINE | 0.753±0.007 | −0.039 | 0.638±0.083 | −0.085 |

![fairness vegfr2](figures/fairness/vegfr2.png)
![fairness herg](figures/fairness/herg.png)
![fairness abl1](figures/fairness/abl1.png)

**(i) The frozen recipe really does understate the GIN - but only a
little, and only on scaffold.** 2 of the 3 nearest targets flip positive
on the mean convention (`vegfr2`, `herg`) - the recipe was
capacity-starved. Fragile: `abl1` overfits its **216** validation
molecules (test R2 and seed std both degrade), random flips on none of the three.

**(ii) GINE bond features are not a general gain.** They help only `herg`
scaffold, and hurt `vegfr2` (both splits) and `abl1` scaffold - edge
features are not the missing ingredient.

**(iii) Boundary, not a reversal.** "No clear lead on any of 16 cells" is
a *frozen-recipe* statement: tuning pushes a minority of near-parity
targets over zero on scaffold and nowhere else, 1 of 3 targets regresses,
and the wins (+0.005 / +0.025) are the size of seed noise.

## Multi-task pooling: more data does not rescue the GIN

`scripts/gnn_07_multitask.py` tests the industrial setting single-task
runs cannot: pool `egfr_full` + `abl1` + `vegfr2` into **27,992 molecule
rows** (25,143 unique SMILES; **2,525 = 10.0%** appear under more than
one target). Two arms x 3 seeds x 2 splits: **MT-GIN** (shared trunk +
one linear head per target) and **pooled**, a negative control with one
shared head that ignores the task label; per-target splits/statistics.

Test R2, 3-seed mean ± std (Δ = vs the single-task RF):

| Task | Split | ST-RF | ST-GIN | pooled | MT-GIN |
|---|---|---|---|---|---|
| `egfr_full` | random | 0.759 | 0.707±0.006 (−0.053) | 0.535±0.051 (−0.224) | 0.530±0.033 (−0.229) |
| `egfr_full` | scaffold | 0.593 | 0.558±0.010 (−0.035) | 0.451±0.027 (−0.141) | 0.356±0.078 (−0.236) |
| `abl1` | random | 0.792 | 0.752±0.006 (−0.039) | 0.638±0.060 (−0.153) | 0.625±0.035 (−0.167) |
| `abl1` | scaffold | 0.723 | 0.710±0.037 (−0.013) | 0.554±0.072 (−0.169) | 0.614±0.032 (−0.109) |
| `vegfr2` | random | 0.736 | 0.678±0.007 (−0.058) | 0.195±0.023 (−0.541) | 0.554±0.031 (−0.182) |
| `vegfr2` | scaffold | 0.617 | 0.611±0.024 (−0.006) | 0.182±0.160 (−0.435) | 0.538±0.020 (−0.079) |

![multitask random](figures/multitask/random.png)
![multitask scaffold](figures/multitask/scaffold.png)

**Pooling loses on every cell.** Neither arm beats the single-task RF in
any of the 6 cells; both are *worse* than single-task GIN everywhere:
MT-GIN trails RF by **−0.079 to −0.236**, pooled by **−0.141 to −0.541**,
vs **−0.006 to −0.058** for single-task GIN. More molecules from more
targets made the model worse, not better. Cross-split leakage can only
*inflate* these numbers; the linear-head handicap does not decide the
verdict - both in [methodology](docs/methodology.md).

## Repository layout

```
qsar-activity-prediction/
├── README.md
├── docs/            # methodology.md (deep detail), interview.md (prep notes)
├── data/            # raw/ (gitignored), processed/ (cleaned, splits/)
├── notebooks/       # qsar_pipeline.ipynb (from scripts/)
├── scripts/         # pipeline (01-07_* QSAR, gnn_* GNN), fetch_artifacts.py
├── figures/         # plots for the results quoted here
├── models/          # trained RandomForests (gitignored)
├── tests/           # pytest suite (CI)
├── results/         # metrics/predictions for both phases
└── pyproject.toml   # tool config
```

## Reproduce

Fast path: top of this README (pretrained `models/` + `data/raw/`,
then Phase 2 from `gnn_01_make_splits.py`). Full retraining - phases 1-3
incl. the 8-target panel - in [methodology](docs/methodology.md).

Docs: [methodology](docs/methodology.md) · [interview
notes](docs/interview.md) · [data provenance](data/README.md).
