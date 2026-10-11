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

> **Status - 2026-10-11 refresh.** Every number below comes from the
> rebuilt datasets (exact-measurement label filter + parent-structure
> dedup; review outcome: the 19-item
> [fix checklist](docs/reviews/fixes-2026-10-11.md), label-policy
> [sensitivity report](docs/reviews/h1-sensitivity-2026-10-10.md)).
> The **GINE / AttentiveFP architecture panels**, the
> **fingerprint-ablation** re-run (all 8 tags) and the **fairness
> campaign** (`vegfr2`/`herg`/`abl1`) are refreshed against those
> datasets (`experiments/LOG.md`).

## Results

### EGFR (primary dataset, 4,560 unique compounds, test = 20%)

| Split | R2 | RMSE | MAE |
|---|---|---|---|
| random | **0.721** | 0.733 | 0.535 |
| scaffold | **0.649** | 1.015 | 0.807 |

The random-vs-scaffold gap is the expected fingerprint of analogue
leakage: random splits let near-identical compounds land on both sides,
inflating scores; the scaffold split (whole Bemis-Murcko scaffolds held
out; **largest scaffold groups are assigned to test first** - the
reverse of DeepChem's `ScaffoldSplitter`, see [Method](#method))
approximates genuinely new chemotypes - the honest number.

![pred vs actual](figures/pred_vs_actual_random_egfr.png)

### Fallback dataset (BACE-1, 1513 compounds)

| Split | R2 | RMSE | MAE |
|---|---|---|---|
| random | 0.643 | 0.795 | 0.574 |
| scaffold | 0.643 | 0.613 | 0.478 |

For BACE the two RMSEs are inverted (random > scaffold) **because the
test sets have different potency spreads** - random is wider (std 1.33,
range 2.7-10.5) vs scaffold (std 1.03, range 3.9-9.0). R2 is scale-free,
RMSE is not. See `figures/testset_hist_bace.png`.

### Eight-target panel

Test R2; GIN = 3-seed mean +- std; **Δ = 3-seed mean GIN − RF**, the
effect size every verdict below uses. **sig** = which model's paired
bootstrap 95% CI excludes 0 (B = 10,000, mean3 *ensemble* convention;
`results/significance/summary.csv`); `ns` = CI straddles 0.

| Tag | Target (ChEMBL id) | Family | n | RF rand | GIN rand | Δ rand (mean) | sig | RF scaf | GIN scaf | Δ scaf (mean) | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `a2a` | A2a adenosine receptor (CHEMBL251) | GPCR | 1,366 | 0.710 | 0.621±0.009 | −0.089 | RF | 0.702 | 0.654±0.025 | −0.049 | ns |
| `abl1` | ABL1 tyrosine kinase (CHEMBL1862) | kinase | 2,120 | 0.822 | 0.767±0.010 | −0.055 | RF | 0.714 | 0.737±0.027 | **+0.022** | GIN |
| `egfr` | EGFR (CHEMBL203) | kinase | 4,560 | 0.721 | 0.665±0.004 | −0.057 | RF | 0.649 | 0.612±0.013 | −0.037 | ns |
| `egfr_full` | EGFR, full B/F pull (CHEMBL203) | kinase | 10,729 | 0.729 | 0.667±0.006 | −0.062 | RF | 0.608 | 0.588±0.006 | −0.020 | ns |
| `herg` | hERG / KCNH2 channel (CHEMBL240) | ion channel | 8,403 | 0.607 | 0.619±0.009 | **+0.012** | GIN | 0.384 | 0.352±0.038 | −0.032 | ns |
| `hivpr` | HIV-1 protease (CHEMBL243) | viral protease | 2,573 | 0.692 | 0.632±0.003 | −0.061 | RF | 0.538 | 0.457±0.010 | −0.081 | ns |
| `vegfr2` | **VEGFR2 / KDR** (CHEMBL279) | kinase | 9,643 | 0.670 | 0.605±0.004 | −0.064 | RF | 0.585 | 0.583±0.002 | −0.003 | ns |
| `mpro` | Replicase polyprotein 1ab / Mpro (CHEMBL4523582) | viral protease | 3,410 | 0.602 | 0.572±0.015 | −0.029 | ns | 0.433 | 0.322±0.049 | **−0.111** | RF |

BACE is not in the table (RF baseline only). Columns from
`results/multi_target/summary.csv`; Δ computed at full precision, then
rounded. Significance: **RF significantly wins 7 of 16 cells, the GIN 2
(`abl1`/scaffold and `herg`/random), 7 are undecided** - bootstrap CIs
quoted under *Significance* below. The CI is computed on the *ensemble*
Δ, so it can differ from the per-seed-mean Δ shown here (reading rule:
[Panel Δ conventions](docs/methodology.md)).

### Reproduce: fast path

```bash
pip install -r requirements.txt
python scripts/fetch_artifacts.py --all   # 下载训练好的 models/ 和 data/raw/（GitHub Release）
```

From here Phase 2 runs directly (`gnn_01_make_splits.py` onward): saved
RF models and regenerated splits verify against the published metrics,
no retraining. Full retraining, incl. phase 3:
[methodology](docs/methodology.md).

`requirements.txt` is a **snapshot of the working environment at campaign
time, not a hard boundary**: development ran the full test suite on
slightly different minor versions (e.g. scikit-learn / torch patch
levels) and the code is insensitive to them, so a fresh environment that
resolves newer compatible versions will usually work - realign to the
snapshot only if something actually breaks.

## Data

Sources, switched via the `QSAR_TAG` environment variable. All ChEMBL
targets come from the same generic downloader
(`scripts/01e_download_chembl.py`, `standard_type=IC50`,
`standard_units=nM`, assay types B/F):

| Dataset tag | Target | ChEMBL id | Unique compounds | Source |
|---|---|---|---|---|
| `egfr` (default) | EGFR | CHEMBL203 | 4560 | ChEMBL REST API (`01d`, early-stopped pull) |
| `egfr_full` | EGFR, full B/F pull | CHEMBL203 | 10729 | ChEMBL REST API (`01e`, `--max-rows 0`) |
| `bace` (fallback) | Beta-secretase 1 | – | 1513 | MoleculeNet BACE |
| `a2a` | A2a adenosine receptor | CHEMBL251 | 1366 | ChEMBL REST API (`01e`) |
| `abl1` | ABL1 tyrosine kinase | CHEMBL1862 | 2120 | ChEMBL REST API (`01e`) |
| `mpro` | Replicase polyprotein 1ab / Mpro | CHEMBL4523582 | 3410 | ChEMBL REST API (`01e`) |
| `hivpr` | HIV-1 protease | CHEMBL243 | 2573 | ChEMBL REST API (`01e`) |
| `herg` | hERG / KCNH2 channel | CHEMBL240 | 8403 | ChEMBL REST API (`01e`) |
| `vegfr2` | **VEGFR2 / KDR** | CHEMBL279 | 9643 | ChEMBL REST API (`01e`) |

Counts are the post-refresh clean sizes (2026-10-11 label/parent
rebuild; the raw pulls keep their original row counts - provenance and
the pre-refresh superset check live in
[data/README.md](data/README.md); what changed and why:
[fix checklist](docs/reviews/fixes-2026-10-11.md) +
[sensitivity report](docs/reviews/h1-sensitivity-2026-10-10.md)). The
`vegfr2` dataset was formerly tagged `mapk14` (misnomer fixed
2026-10-08).

Primary data: **ChEMBL** (EBI), IC50/nM, binding/functional assays. The
resilient downloader (page retries, disk flushes; the original `egfr`
pull survived intermittent HTTP 500s) and full per-file provenance:
[data/README.md](data/README.md).

Cleaning (`scripts/02_clean_data.py`): IC50 in **nM** only; exact
measurements only (`standard_relation == "="` - censored `>`, `<`, `~`
rows are dropped, they are bounds, not labels); salts collapsed to the
parent structure before dedup (fingerprint key = parent SMILES); drop
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

**Scaffold-split convention - largest groups fill *test* (DeepChem does
the opposite).** `scaffold_split` (`scripts/qsar_common.py`) groups
molecules by Bemis-Murcko scaffold and assigns groups to the **test**
set **largest-first** until ~20%; DeepChem/MoleculeNet's
`ScaffoldSplitter` convention is the reverse - largest groups fill
*train*, leftovers become test. Consequence: our test set contains the
dominant chemotype families (members mutually similar), so scaffold
scores are **slightly optimistic** relative to a DeepChem-style split
and absolute scaffold R2 is **not comparable across the two
implementations**; the conclusions hold either way - random and
scaffold both read GIN ≤ RF on 7 of 8 targets. Chosen deliberately: big
families in test enable
chemotype-level error analysis (`scripts/gnn_05_error_analysis.py`);
the cost: campaigns already run stay as-is, and aligning with
MoleculeNet later would require full retraining. Details:
[methodology](docs/methodology.md).

## Interpretation & data QC

EGFR top fingerprint bits mapped back to the substructures that set them
(`scripts/05_explain_bits.py`):

![top bits](figures/top_bits_egfr.png)

The dominant bit (bit 1367) decodes to an **aminopyrimidine** - the
classic kinase hinge-binding motif. MDI is inflated by correlated
features, so every importance claim was cross-checked with
**permutation importance on the held-out test set** (numbers:
[methodology](docs/methodology.md)).

![mdi vs perm](figures/importance_mdi_vs_perm_egfr.png)

Data QC (ultra-potent audit, censoring, test-set spreads): audited, no
filtering rule - the audit is the deliverable
([methodology](docs/methodology.md)).

**Cross-model attribution alignment.**
`experiments/interpretability_alignment.py` loads the committed RF and
compares its top fingerprint bits (SHAP) with the GIN's GNNExplainer top
atoms on the scaffold test split, molecule by molecule, for the
dominant held-out chemotype of each tag. On the four tags where the
family query returns molecules - `egfr`/`egfr_full` (thienopyrimidine)
and `herg`/`mpro` (chromone), 5 molecules each - the mean share of GNN
top atoms inside the RF's top-bit coverage runs **0.35-0.68**, and
**0.21-0.74** of GNN top atoms sit on the scaffold core; per-family RF
recurring bits are in `results/interpretability/{tag}/findings.md`.
`a2a`, `abl1` and `hivpr` completed but analysed **zero molecules**
(their queried family has no members in the held-out set) - no numbers
to report; `vegfr2` was not run. GNNExplainer/SHAP caveats:
[methodology](docs/methodology.md).

**Y-randomization (leakage sanity check).** Trained on permuted labels,
every shuffled group mean collapses towards zero and stays far under the
pass bar: across the seven tags run (the driver skips `vegfr2`), 2 splits
x {RF, GIN} x 3 seeds, all **28 shuffled group means lie between −0.843
and +0.010** - the largest (`egfr`/random shuffled GIN, **+0.010**) is
still more than an order of magnitude under the 0.2 verdict threshold,
every shuffled RF mean is ≤ −0.054, and each tag's
`results/y_randomization/summary_*.json` verdict reads **`ok: all
shuffled-label means < 0.2`**. Individual shuffled GIN seeds can still
land slightly positive (up to +0.026) - that is why the check is on
group means. The real-label control RF reads **0.7179** vs the headline
**0.7214** (`egfr`/random). That ~0.003 gap is the control's training
pool, not skipped tuning: `experiments/y_randomization.py` fits its RF
on the GNN train pool (`egfr`/random 3,283 rows), while the headline RF
also trained on the 10% validation carve-out (3,648 rows, 365 more) -
expected in direction and magnitude, not leakage evidence. Real and
shuffled controls always fit on the same pool, so the comparison stays
valid; the test only asks that shuffled means collapse (file verdict:
< 0.2) with the real control inside the baseline band. Details:
[methodology](docs/methodology.md).

## GNN vs traditional QSAR: a controlled comparison

**On the same molecules and the same train/test partitions, does a graph
neural network beat Morgan fingerprints + RandomForest?**

- Same data (cleaned EGFR, 4,560, pIC50); same splits regenerated with
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
| random | 0.721 / 0.733 | 0.669 / 0.799 | 0.665 ± 0.004 |
| scaffold | 0.649 / 1.015 | 0.631 / 1.042 | 0.612 ± 0.013 |

**On this dataset the GIN does not beat the fingerprint RF.** The random
gap (-0.057 R2) is an order of magnitude wider than seed noise; the
scaffold gap (-0.037) is ~3 seed std - real, but closer to the noise
floor. Both models lose ~0.05-0.07 R2 from random to scaffold - novel
chemotypes are still harder for both feature types. Tuning negatives,
error analysis, ESOL: [methodology](docs/methodology.md).

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

**Finding 1 - on random splits the GIN clearly leads exactly one
target.** Δ(mean) runs from **+0.012** (`herg`) down to −0.089 (`a2a`):
7 of 8 targets are negative, and the single positive is also the one
random cell where the bootstrap significantly favours the GIN
(`herg`/random, ensemble Δ **+0.038 [+0.006, +0.069]**, small but real).
Six random cells significantly favour RF. With analogues leaking across
the split, message passing buys nothing over circular substructure
counts on seven of eight targets - the EGFR conclusion above is not an
EGFR artefact.

**Finding 2 - on scaffold splits one target flips positive, and the
deficit's size is a property of the target, not of n.** Seven of eight
Δ(mean) are negative: closest to parity `vegfr2` at **−0.003**, worst
`mpro` at **−0.111**. The exception is `abl1`: **+0.022** per-seed mean,
and it survives every convention - seed42 **+0.060 [+0.025, +0.096]**,
3-seed ensemble **+0.037 [+0.006, +0.069]** (significantly GIN). This is
a real flip, not the seed-42 mirage the pre-refresh data showed. What
still varies is the *magnitude*: scaffold diversity and assay structure
set how far behind the GIN falls - `mpro` (3,410 compounds) at −0.111
vs `egfr_full` (10,729) at −0.020 is three times the data and
one-fifth the deficit, so Δ does not follow n.

**Significance, upgraded to bootstrap CIs (P8).** The Δ(mean) statements
above are backed by a paired bootstrap (B = 10,000; test molecules
resampled with replacement, same indices for both models; CI on the
3-seed-mean prediction). Of the 16 random/scaffold cells, **9 have a
95% CI excluding 0**:

- **7 favour RF**: `a2a`/random −0.073 [−0.138, −0.013], `abl1`/random
  −0.044 [−0.076, −0.016], `egfr`/random −0.044 [−0.079, −0.010],
  `egfr_full`/random −0.039 [−0.059, −0.021], `hivpr`/random −0.048
  [−0.083, −0.014], `mpro`/scaffold −0.074 [−0.133, −0.018],
  `vegfr2`/random −0.044 [−0.068, −0.021].
- **2 favour the GIN**: `abl1`/scaffold +0.037 [+0.006, +0.069] and
  `herg`/random +0.038 [+0.006, +0.069] - both read
  `frac_boot_gin_better` ≈ 0.99.
- **7 are undecided** (CI straddles 0); the near-miss is
  `vegfr2`/scaffold, +0.024 with `frac_boot_gin_better` = 0.97.

`frac_boot_gin_better` is the
fraction of the 10,000 bootstrap deltas above 0 - **not a p-value**; a
`p < 0.05`-style filter would drop every GIN win. The seed-42 column
still overstates leads (`abl1`/scaffold +0.060 vs +0.037 at mean3), so
the 3-seed-mean convention stays the reference. Bond features change
nothing - the refreshed GINE panel reads **0 of 10 bootstrapped cells
with CI > 0** (5 cells significantly favour RF, 5 straddle 0); the
other 6 cells (`abl1`/`herg`/`vegfr2`, whose GINE runs hold no
per-seed predictions) carry no CI and read point estimates only, 2 of
them positive (`abl1`/scaffold +0.028, `herg`/random +0.006) - see
the status note on that section. Tables
`results/significance/summary.csv`,
`results/significance/summary_partial_gine.csv`,
`results/gine_panel/summary.csv`; forest plots
`figures/significance/ci_panel.png`,
`figures/significance/ci_panel_with_gine.png`,
`figures/gine_panel/panel_dR2.png`. The ±std
columns throughout are auxiliary spread descriptions - the CIs are the
significance statement.

One reading rule for everything above: the panel tables' Δ(mean) and the
bootstrap CI are **two different conventions** - per-seed-mean Δ vs
mean3 *ensemble* Δ (R² of the per-molecule mean prediction) - and the
ensemble side always carries a small non-negative gain, which is why
`egfr`/random is −0.057 in the panel table but −0.044 in the
significance table. Read significance from the CI, point estimates from
the panel, and never compare one against the other; full formulas and
the worked example are in **Panel Δ conventions** in
[methodology](docs/methodology.md).

![delta vs n](figures/multi_target/delta_vs_n.png)

Learning curves (7 tags re-run on the refreshed data; each tag's n grid
is capped by its train pool, up to 4,400 - `herg`'s random curve has the
GIN ahead at 7 of its 9 points, ending 0.576 vs RF 0.549 at n = 4,400,
and `abl1`/scaffold leads at every point, while `a2a`'s pool only fits a
single n = 500 point), extrapolated n\* with its weak-identifiability
caveat, training-budget caveat, cross-target failure replication:
[methodology](docs/methodology.md).

## Giving the GNN a fair chance (bounded tuning + GINE + AttentiveFP)

> **Status - panels + fairness refreshed (2026-10-11).** The GINE and
> AttentiveFP panels and the fairness campaign (`vegfr2`/`herg`/`abl1`)
> are retrained against the refreshed data (`experiments/LOG.md`), so
> every number below is a single-vintage batch; the fairness table's RF
> references equal the refreshed panel RF. `gnn_fairness.py` still
> saves no per-seed predictions, hence its GINE cells are point
> estimates without bootstrap CIs.

Sections above deliberately froze the GIN recipe - every Δ is a
**frozen-recipe** Δ. `experiments/gnn_fairness.py` (36 runs/target) gives
it a bounded chance: on the three near-parity targets (frozen-recipe
scaffold Δ on the refreshed panel: `vegfr2` −0.003, `herg` −0.032,
`abl1` +0.022), 8 configs
(hidden {128, 256} x layers {4, 5} x dropout {0.2, 0.3}) train **only on
the scaffold split**, selected by **3-seed mean validation RMSE**, never a
single seed. Winners: `vegfr2` 256/5L/0.2 (0.6937), `herg` 256/5L/0.2
(0.6215), `abl1` 256/5L/0.2 (0.7833) - 256/5L/0.2 on all three. A **GINE**
variant adds a `BondEncoder` (`BOND_FEATURE_DIMS = [13, 7, 2]`); all else
is `gnn_03`'s.

Test R2, 3-seed mean ± std; Δ = GIN − RF (RF = this campaign's
reference RF, equal to the refreshed panel RF):

| Target | Model | random R2 | Δ rand | scaffold R2 | Δ scaf |
|---|---|---|---|---|---|
| `vegfr2` (VEGFR2) | RF | 0.670 | — | 0.585 | — |
| | baseline GIN | 0.605±0.004 | −0.064 | 0.583±0.002 | −0.003 |
| | tuned GIN | 0.620±0.003 | −0.050 | 0.600±0.006 | **+0.015** |
| | GINE | 0.600±0.006 | −0.070 | 0.563±0.010 | −0.022 |
| `herg` | RF | 0.607 | — | 0.384 | — |
| | baseline GIN | 0.619±0.009 | **+0.012** | 0.352±0.038 | −0.032 |
| | tuned GIN | 0.616±0.005 | **+0.009** | 0.385±0.015 | +0.000 (tie) |
| | GINE | 0.613±0.009 | **+0.006** | 0.352±0.017 | −0.033 |
| `abl1` | RF | 0.822 | — | 0.714 | — |
| | baseline GIN | 0.767±0.010 | −0.055 | 0.737±0.027 | **+0.022** |
| | tuned GIN | 0.765±0.024 | −0.058 | 0.770±0.014 | **+0.056** |
| | GINE | 0.779±0.009 | −0.043 | 0.742±0.015 | **+0.028** |

![fairness vegfr2](figures/fairness/vegfr2.png)
![fairness herg](figures/fairness/herg.png)
![fairness abl1](figures/fairness/abl1.png)

**(i) The frozen recipe really does understate the GIN - but only a
little, and only on scaffold.** Bounded tuning lifts the scaffold Δ on
all three targets: `vegfr2` crosses zero (−0.003 → **+0.015**), `herg`
climbs to a tie (−0.032 → +0.000), and `abl1`, already positive under
the frozen recipe (+0.022), widens to **+0.056** while its seed spread
shrinks (0.027 → 0.014) - the recipe was capacity-starved. Random flips
on none of the three: tuned `vegfr2` and `abl1` stay negative (−0.050 /
−0.058, `abl1`/random the one cell where tuning costs a little and the
seed spread grows 0.010 → 0.024), while `herg`/random was already
positive and eases from +0.012 to +0.009.

**(ii) GINE bond features are not a general gain.** In the fairness
table above they help only `abl1` scaffold (**+0.028**) and `herg`
random (+0.006, marginal), and hurt `vegfr2` on both splits
(−0.070 / −0.022), `herg` scaffold (−0.033) and `abl1` random
(−0.043) - edge features are not the missing ingredient. Panel-wide
against the refreshed RF this holds with bootstrap CIs: the full GINE
panel wins **0 of 10 bootstrapped cells** (5 significantly favour RF -
`a2a`/scaffold, `egfr`/random, `egfr_full`/random, `hivpr`/scaffold,
`mpro`/scaffold - and 5 straddle 0); the remaining 6 cells (`abl1`,
`herg`, `vegfr2` - no per-seed GINE predictions) carry no CI and read
point estimates only, 2 of them positive on the refreshed fairness
runs (`abl1`/scaffold **+0.028**, `herg`/random +0.006). Tables
`results/significance/summary_partial_gine.csv` +
`results/gine_panel/summary.csv` (the 10 bootstrapped cells),
`results/fairness/summary_{tag}.json` (the 3 no-CI tags); figures
`figures/gine_panel/panel_dR2.png`,
`figures/significance/ci_panel_with_gine.png`.

**(iii) Boundary, not a reversal.** The frozen recipe itself now shows
two significant leads (`abl1`/scaffold and `herg`/random - the two GIN
cells in *Significance* above); bounded tuning crosses zero on scaffold
and nowhere else - `vegfr2` −0.003 → **+0.015** is inside seed-noise
range, `herg` only ties (+0.000), and `abl1` widens its already-positive
lead to **+0.056** - while no random-split cell flips. Three widened or
new positive cells out of six tuned ones move the boundary; they do not
reverse the panel verdict.

**AttentiveFP control: attention + GRU + bond features, still no
reversal.** The strongest published small-molecule readout - PyG
AttentiveFP, gated attention with a 30-timestep GRU over the same
atom/bond encoders, frozen recipe otherwise - is the third control: 8
targets x 2 splits x 3 seeds = 48 runs (`experiments/afp_panel.py`,
`--model attentivefp`), retrained against the refreshed data. Under the
mean3 paired bootstrap it wins **2 of 16 cells - `herg`/scaffold,
ensemble Δ +0.062 [+0.030, +0.092], and `herg`/random, +0.034
[+0.002, +0.065]** (per-seed point estimates +0.020 and −0.022; the
two Δ conventions are defined in **Panel Δ conventions** in
[methodology](docs/methodology.md)) - while **8 cells significantly
favour RF** (`a2a`/random, `abl1` both splits, `egfr_full` both
splits, `hivpr`/scaffold, `mpro`/scaffold, `vegfr2`/random) and 6 are
undecided; on the random split all 8 point estimates are negative
(−0.167 `a2a` to −0.022 `herg`), and `herg`/random is the one cell
where the ensemble convention turns a negative point estimate into a
significant AttentiveFP lead - the other 4 significant random cells
favour RF. `herg`/scaffold is the friendliest cell for the enhanced
variants (tuned GIN +0.000 tie, GINE −0.033, AttentiveFP +0.020
per-seed) - only AttentiveFP reads positive there now, and it is the
one that carries a lead across significance, as does `herg`/random for
AttentiveFP alone; the frozen GIN reads −0.032 there on the refreshed
panel. Tables `results/afp_panel/summary.csv`,
`results/significance/summary_attentivefp.csv`; figures
`figures/afp_panel/panel_dR2.png` and the three-model forest plot
`figures/significance/ci_panel_with_afp.png` (gin / gine / afp side by
side). Bond features, bounded tuning and now AttentiveFP have each
failed to overturn the panel-level verdict, so "the GIN does not beat
the fingerprint RF" is no longer a complaint about the GIN
implementation - it is a data-scale / signal statement.

## XGBoost: a stronger tree baseline, still under the RF

`experiments/xgboost_baseline.py` asks the obvious follow-up: is the RF
ceiling an artefact of a weak tree ensemble? Same 2048-bit Morgan
features, same splits, XGBoost 3.4.1 (`hist`, CPU) with fixed generic
hyperparameters (500 trees, lr 0.05, depth 6); the control is the
headline RF. Δ = XGB − RF per split (`results/xgboost/summary_*.json`,
controls recomputed against the refreshed `results/metrics_*.json`):

| Tag | Δ rand | Δ scaf |
|---|---|---|
| `a2a` | −0.022 | −0.022 |
| `abl1` | −0.003 | −0.007 |
| `egfr` | **+0.010** | −0.132 |
| `egfr_full` | −0.053 | −0.038 |
| `herg` | −0.037 | **+0.007** |
| `hivpr` | −0.002 | −0.063 |
| `mpro` | −0.004 | −0.023 |
| `vegfr2` | −0.052 | **+0.004** |

**XGBoost does not beat the tuned RF**: 13 of 16 cells are ≤ RF, the
three exceptions are hair-breadth (`egfr`/random +0.010,
`herg`/scaffold +0.007, `vegfr2`/scaffold +0.004) while the worst cell
is its classic failure mode - novel chemotypes (`egfr`/scaffold 0.517
vs the RF's 0.649, Δ −0.132). The RF-vs-GIN verdict is not an artefact
of a weak tree baseline.

**"3 seeds" here is a deterministic repeat, not an uncertainty
estimate.** All three seeds are bit-identical on every cell
(`r2_std = 0.0` everywhere): with default `subsample`/`colsample` = 1.0
under `hist`/CPU there is no random path left for `random_state` to
seed - a property of the algorithm as configured, not a data or split
uncertainty (full note in [methodology](docs/methodology.md)).

Figures: `figures/xgboost/`.

## Fingerprint ablation: radius and bit width

`experiments/fingerprint_ablation.py` re-runs the RF over
r ∈ {2, 3} x bits ∈ {1024, 2048} (3 seeds) against the r=2/2048 anchor
on the refreshed data.

**Status (2026-10-11): all 8 tags landed** - 8 tags x 2 splits x 4
configs x 3 seeds = 192 RF fits, one `summary_<tag>.json` per tag in
`results/fp_ablation/`. Δ is measured against each run's own r=2/2048
anchor (3-seed mean); that anchor reproduces the published RF to within
0.006 in 13 of the 16 cells - the third outlier is `vegfr2`/scaffold
(+0.006 above) - and it sits 0.021 (`hivpr`/scaffold) and 0.012
(`mpro`/scaffold) below it, the two flagged `anchor_failures` in the JSON.

| Tag | random anchor R² | random Δ range | scaffold anchor R² | scaffold Δ range |
|---|---|---|---|---|
| `a2a` | 0.709 | −0.017 … −0.006 | 0.707 | +0.007 … +0.018 |
| `abl1` | 0.822 | −0.019 … −0.008 | 0.714 | −0.040 … −0.012 |
| `egfr` | 0.718 | −0.009 … +0.002 | 0.648 | −0.021 … −0.003 |
| `egfr_full` | 0.728 | −0.016 … −0.003 | 0.611 | −0.027 … +0.000 |
| `herg` | 0.608 | −0.016 … −0.008 | 0.388 | −0.020 … +0.004 |
| `hivpr` | 0.694 | −0.015 … +0.001 | 0.517 | −0.042 … +0.041 |
| `mpro` | 0.604 | −0.007 … +0.003 | 0.421 | −0.003 … +0.002 |
| `vegfr2` | 0.670 | −0.010 … −0.002 | 0.592 | −0.043 … −0.011 |

Each Δ range spans the other three configs (r2/1024, r3/1024, r3/2048)
of that tag. What the eight tags say:

- On **random** splits the tag-to-tag spread is tiny: all 24
  non-anchor cells land between **−0.019 and +0.003** of their
  anchor, and no tag's random range is wider than 0.016 - the
  headline r=2/2048 default is not leaving obvious performance
  behind on any target.
- On **scaffold** the spread is ~4x wider (**−0.043 … +0.041**) and
  **bit width is the lever**: at r=3, cutting to 1024 bits costs
  0.003-0.033 on 6 of 8 tags (worst `vegfr2` −0.033, `hivpr` −0.028,
  `abl1` −0.026), and the three worst cells of the whole grid are all
  r3/1024 - `vegfr2` −0.043, `hivpr` −0.042, `abl1` −0.040. The sign
  is not universal (`a2a` +0.012 and `egfr_full` +0.007 prefer 1024
  at r=3; at r=2 `hivpr` gains +0.041 from halving the bits), so
  2048 bits is a strong default rather than a law.
- **Radius 3 at 2048 bits buys nothing**: −0.011 … +0.003 on random,
  −0.027 … +0.007 on scaffold - r=2/2048 is also the radius default.

Figures: `figures/fp_ablation/`; tables: `results/fp_ablation/`.

## Multi-task pooling: more data does not rescue the GIN

`scripts/gnn_07_multitask.py` tests the industrial setting single-task
runs cannot: pool `egfr_full` + `abl1` + `vegfr2` into **22,492 molecule
rows** (21,093 unique SMILES; **1,193 = 5.66%** appear under more than
one target - down from the pre-refresh 10.0%, see the fix checklist).
Two arms x 3 seeds x 2 splits: **MT-GIN** (shared trunk +
one linear head per target) and **pooled**, a negative control with one
shared head that ignores the task label; per-target splits/statistics.

Test R2, 3-seed mean ± std (Δ = vs the single-task RF):

| Task | Split | ST-RF | ST-GIN | pooled | MT-GIN |
|---|---|---|---|---|---|
| `egfr_full` | random | 0.729 | 0.667±0.006 (−0.062) | 0.451±0.041 (−0.278) | 0.491±0.014 (−0.238) |
| `egfr_full` | scaffold | 0.608 | 0.588±0.006 (−0.020) | 0.444±0.007 (−0.164) | 0.377±0.044 (−0.231) |
| `abl1` | random | 0.822 | 0.767±0.010 (−0.055) | 0.673±0.017 (−0.149) | 0.700±0.009 (−0.122) |
| `abl1` | scaffold | 0.714 | **0.737±0.027 (+0.022)** | 0.669±0.035 (−0.045) | 0.692±0.046 (−0.022) |
| `vegfr2` | random | 0.670 | 0.605±0.004 (−0.064) | 0.309±0.057 (−0.361) | 0.510±0.010 (−0.160) |
| `vegfr2` | scaffold | 0.585 | 0.583±0.002 (−0.003) | 0.240±0.006 (−0.345) | 0.533±0.014 (−0.052) |

Columns from `results/multitask/summary.csv`; the `vegfr2` rows use the
refreshed single-task RF (2026-10-11) for the ST-RF column and the four
Δs (the summary file still carries the pre-refresh RF there).

![multitask random](figures/multitask/random.png)
![multitask scaffold](figures/multitask/scaffold.png)

**Pooling loses on every cell.** Neither pooling arm beats the
single-task RF in any of the 6 cells, and both are *worse* than the
single-task GIN in every cell: MT-GIN trails RF by **−0.022 to −0.238**,
pooled by **−0.045 to −0.361**, vs **−0.064 to +0.022** for single-task
GIN. One cell does flip - `abl1`/scaffold, where the **single-task** GIN
beats the RF by +0.022 - and pooling drags it straight back below the RF
(MT-GIN −0.022): more molecules from more targets made the model worse,
not better. Cross-split leakage can only *inflate* these numbers; the
linear-head handicap does not decide the verdict - both in
[methodology](docs/methodology.md).

## Publication-year time split: mostly the GIN degrades less, neither model is dependable

Random and scaffold splits let the model train on molecules published
*after* the molecules it is tested on. The deployment-honest cut
(`scripts/gnn_08_time_splits.py`; a molecule's year = the earliest
publication it appears in, from `data/raw/document_years.csv` - 4,147
ChEMBL documents across the 8 tags) withholds the newest years:
**test = newest publication years cumulating to ~21-28% of molecules,
valid = newest ~10% of the rest**. Because one recent year can carry a
lot of data, the test fraction runs over target on `a2a` (59%) and
`mpro` (51%) - both datasets end in 2024/2025. (`egfr`'s window reads
2013-2016: after the label refresh no `egfr` measurement is dated later
than 2016.) RF trained via
`scripts/04_train_and_evaluate.py --split time`, GIN as always
(3 seeds, same npz).

Test R2 (random for reference; GIN = 3-seed mean; GIN ens = R2 of the
3-seed ensemble prediction; **Δ = GIN ens − RF**, recomputed from the
committed per-seed prediction files, with its paired-bootstrap **95% CI**
(B = 10,000, mean3 ensemble convention,
`results/significance/summary_time.csv`); split sizes, years and the
RF/GIN columns from `results/time_split/summary.csv`; **bold Δ** = the CI
excludes 0):

| Tag | RF rand | GIN rand | RF time | GIN time | GIN ens | Δ time (ensemble) | 95% CI |
|---|---|---|---|---|---|---|---|
| `a2a` | 0.710 | 0.621 | −0.301 | −0.023±0.059 | +0.024 | **+0.326** | [+0.188, +0.467] |
| `abl1` | 0.822 | 0.767 | +0.560 | +0.271±0.182 | +0.324 | **−0.236** | [−0.275, −0.197] |
| `egfr` | 0.721 | 0.665 | +0.123 | +0.088±0.073 | +0.162 | +0.039 | [−0.012, +0.093] |
| `egfr_full` | 0.729 | 0.667 | −0.736 | −0.176±0.052 | −0.144 | **+0.592** | [+0.536, +0.656] |
| `herg` | 0.607 | 0.619 | −0.058 | −0.053±0.016 | −0.026 | +0.032 | [−0.007, +0.073] |
| `hivpr` | 0.692 | 0.632 | −0.027 | +0.029±0.076 | +0.088 | **+0.115** | [+0.036, +0.196] |
| `mpro` | 0.602 | 0.572 | +0.006 | +0.055±0.019 | +0.090 | **+0.084** | [+0.043, +0.128] |
| `vegfr2` | 0.670 | 0.605 | +0.160 | +0.068±0.033 | +0.108 | **−0.052** | [−0.077, −0.026] |

Panel means: RF **0.694 (random) → −0.034 (time)**, GIN **0.643 →
+0.032** (time, per-seed mean; ensemble **+0.078**).

**CI column restored (2026-10-11).** The paired bootstrap was re-run
against the refreshed time predictions: all 8 tags at B = 10,000 merged
into `results/significance/summary_time.csv` (schema now carries
`frac_boot_gin_better` and `n_boot`). **6 of 8 CIs exclude 0** - GIN
ahead on `a2a`, `egfr_full`, `hivpr`, `mpro`, RF ahead on `abl1` and
`vegfr2` - while `egfr` (+0.039 [−0.012, +0.093]) and `herg` (+0.032
[−0.007, +0.073]) straddle 0. `herg` is the verdict that moved: the
pre-refresh cell read +0.076 [0.034, 0.120] (significantly GIN), the
refreshed predictions read +0.032 with the CI crossing zero, so it is now
undecided (see `experiments/LOG.md`).

**Reading the Δ column.** Δ is computed on the **3-seed ensemble
prediction** (per-molecule mean of the three GIN runs; the `GIN ens`
column is that prediction's R2), not from subtracting the two displayed
columns - ensemble Δ and per-seed-mean Δ are two different conventions
(see **Panel Δ conventions** in [methodology](docs/methodology.md)) and
the ensemble is systematically ≥ the per-seed mean. `egfr` is the worked
example: GIN time mean − RF time = 0.088 − 0.123 = **−0.035** (GIN
looks worse) while the ensemble Δ is **+0.039** (GIN better) - opposite
signs from the same three runs.

**Point estimates under drift.** The ensemble Δ favours the GIN on **6
of 8** targets and RF on `abl1` (−0.236) and `vegfr2` (−0.052); with the
CIs in the table, **4 of those 6** GIN cells hold up (`egfr` and `herg`
straddle 0) and both RF cells do too - **6 of 8** CIs exclude 0. Some of
those GIN "wins" are just RF collapsing faster (`egfr_full` +0.592,
`a2a` +0.326 - both models near or below zero), so read the absolute
level next: RF time is negative on 4 of 8 targets (worst `egfr_full`
**−0.736**, then `a2a` −0.301, `herg` −0.058, `hivpr` −0.027); the GIN
ensemble is negative on `egfr_full` (−0.144) and `herg` (−0.026), and
outside `abl1` (0.560 / 0.324 ens) nothing on either model clears 0.17.
At R2 ≈ 0 or below on future chemistry neither model is usable - the
honest industrial number. And the one target where RF still works under
drift (`abl1`, +0.560) is precisely where the GIN falls furthest behind:
"the GIN degrades less" is the 6-of-8 pattern, not a law. Timelines per
target: `figures/time_split/{tag}_timeline.png`; aggregate:
`figures/time_split/panel_3splits.png`; details:
[methodology](docs/methodology.md).

![3 splits](figures/time_split/panel_3splits.png)

## Uncertainty & applicability domain: RF confidence is usable, GNN confidence is not

`experiments/uncertainty_ad.py` closes the loop from calibrated
uncertainty to an actual screening decision (8 targets x random/scaffold,
RF and GIN, plus the GINE ensemble; `results/uncertainty/`, figures
under `figures/uncertainty/`). Calibration/AD numbers below cover the
14 cells from the first 2026-10-11 refresh pass (7 tags x 2 splits);
screening numbers (P@100, precision) cover all 16 cells:

- **RF tree-std calibrates**: monotone quintile calibration (RMSE rising
  with predicted std) in **6 of 14** cells, Spearman(std, |err|)
  averaging **0.345** (0.156-0.490) - usable as a per-molecule
  confidence, imperfectly. The **GNN 3-seed ensemble std does not**
  (rho **0.147**, 0.092-0.231; monotone in only 3 of 14): the seeds are
  too close to each other to span the error. Any "GNN uncertainty" claim
  in this repo would be noise.
- **AD (max Tanimoto to train ≥ 0.4, coverage ~96.4%)**: in-AD test R2
  averages **0.638** (0.392-0.825) vs out-of-AD **−0.367** (mean of the
  12 cells with out-of-AD test mass; worst `egfr`/random **−1.451**) -
  out-of-domain predictions are indeed bad.
- **The counter-intuitive part**: AD-filtering the screen changes P@100
  in **1 of 16 cells** (15/16 unchanged; the exception `herg`/random
  0.65 → 0.66) - the RF ranking never puts an out-of-AD molecule in its
  top 100; the ranking is itself an implicit AD filter, so a second AD
  pass buys nothing on precision. What does help is slicing the
  report-positive set {pred ≥ 7} to its confident half (std ≤ median):
  mean precision **0.827 → 0.889** across the 16 cells. And confidence
  must stay a *filter*, not a sort key: re-ranking the library by low
  std first destroys P@100 (**0.602 vs 0.913** mean).
- Caveat: these conclusions are only as good as the std estimate, so
  they apply to the RF today; the GNN arm has no calibrated std to
  screen with.

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
notes](docs/interview.md) · [data provenance](data/README.md) ·
[review fix checklist](docs/reviews/fixes-2026-10-11.md) · [label-policy
sensitivity](docs/reviews/h1-sensitivity-2026-10-10.md).
