# Experiments Log

Running log of the deep-dive experiments. Each entry: what ran, headline
numbers (3-seed mean +/- std unless noted), and the commit holding the
artifacts. Newest first. Maintained by M4 Air; results produced on M3 Max.

## 2026-10-04

### Cross-target panel + egfr_full data scaling — DELIVERED — campaign complete

- Commits: `c462607` (`feat: generic ChEMBL downloader (01e) + chain
  runner docs`), `a950e1d` (`results: 6-target panel (a2a/abl1/mpro/hivpr/
  herg/mapk14) + error analyses`), `39ab879` (`results: egfr_full expansion
  (13,497 cpds) + learning curve to n=9717 + scaling verdict`); the three
  docs (this entry, README section 8, data provenance) are the follow-up
  `docs: LOG + README cross-target section + data provenance` commit.
- Question: sections 2-7 settled RF-vs-GIN on EGFR only. Is "the GIN does
  not beat fingerprint+RF" a property of EGFR, and how much data would it
  take for the GIN to catch up?

**Generic downloader (scripts/01e_download_chembl.py)**
- Generalises the hardcoded 01d EGFR downloader to any ChEMBL target:
  `--target-chembl-id`, `--tag`, `--standard-type/--units`, `--max-rows 0`
  (= pull everything, no early stop), `--page`. Same resilience contract as
  01d (retry each page x8, flush every page, skip a poisoned offset window
  after 8 failures, abandon an assay_type after >6 skips), two passes
  (assay_type B then F), and a limit=1 probe up front that prints the
  expected `total_count`. Output: `data/raw/{tag}_activities.csv`.
- One command per target; cleaning/featurizing/splitting/GIN code paths are
  untouched - only `QSAR_TAG` changes.

**6-target panel (a2a / abl1 / mpro / hivpr / herg / mapk14)**
- Same protocol as EGFR: Morgan r=2/2048 + grid-tuned RF, splits regenerated
  and verified bit-identically by `gnn_01_make_splits.py`, GIN 3 seeds
  (42/1/2). Test R2, RF / GIN mean+-std:
  - a2a     n=1,746    random 0.733 / 0.687+-0.003    scaffold 0.691 / 0.666+-0.004
  - abl1    n=2,698    random 0.792 / 0.752+-0.006    scaffold 0.723 / 0.710+-0.037
  - mpro    n=4,424    random 0.730 / 0.707+-0.003    scaffold 0.509 / 0.408+-0.022
  - hivpr   n=2,799    random 0.737 / 0.726+-0.005    scaffold 0.554 / 0.527+-0.028
  - herg    n=12,019   random 0.615 / 0.616+-0.017    scaffold 0.430 / 0.417+-0.007
  - mapk14  n=11,797   random 0.736 / 0.678+-0.007    scaffold 0.617 / 0.611+-0.024
- **Naming note: CHEMBL279 is VEGFR2 (KDR), not MAPK14.** The `mapk14` tag is
  a misnomer kept for file-name continuity - every activity row carries
  `target_pref_name = "Vascular endothelial growth factor receptor 2"`
  (verified against the ChEMBL target API). Treat the row as VEGFR2
  everywhere (README section 8, `results/multi_target/summary.csv`).
- Headline: random split GIN loses 6/6 (and 8/8 including egfr/egfr_full);
  scaffold is mixed - abl1 +0.030 and (VEGFR2) +0.028 for the GIN, hivpr a
  tie -0.001, mpro a large -0.130. Delta convention: `d_r2_gin_minus_rf`
  from `results/comparison_*.csv` = **seed-42 GIN − RF** (positive = GIN
  wins); the RF/GIN pairs above are RF vs 3-seed GIN mean. Where seed
  noise is large the two disagree in sign (abl1 scaffold +0.030 seed-42 vs
  -0.013 on the mean; herg random -0.021 vs +0.000) - the two scaffold
  wins are within seed noise.

**egfr_full expansion (CHEMBL203, full B/F pull)**
- `01e --target-chembl-id CHEMBL203 --tag egfr_full --max-rows 0` ->
  **24,560 raw rows** (assay types B and F) -> cleaning ->
  **13,497 unique compounds** (vs 6,165 in egfr v1; 10,902 duplicate rows
  collapsed to the per-compound median pIC50).
- Superset check vs egfr v1: **6165/6165** v1 compounds present in
  egfr_full, |delta pIC50| **median 0.0000, max 2.9793** - the median
  compound is bit-identical; the tail is compounds that gained new
  measurements in the full pull.
- RF (Morgan r=2/2048, grid-tuned): **0.759 random / 0.593 scaffold**
  (0.7592275 / 0.5925707). GIN 3 seeds: **0.707+-0.006 / 0.558+-0.010**
  (delta -0.059 / -0.033). Same verdict, 2.2x the data.
- Split verification evidence, `logs/egfr_full_chain.log` step5:
  `[OK ] random    R2=0.7592275 (expected 0.7592275)   RMSE=0.7092954 (expected 0.7092954)` /
  `[OK ] scaffold  R2=0.5925707 (expected 0.5925707)   RMSE=0.8900525 (expected 0.8900525)` /
  `All splits reproduced and verified against saved RF models.`

**Learning curve extended to n=9,717**
- `experiments/learning_curve.py` re-run on egfr_full with sizes
  500/1000/2000/4000/8000/**9717** x seeds 42/1/2 x {random, scaffold}.
- Random split, full-pool point n=9,717: **RF 0.7502 vs GIN 0.7009**
  (3-seed means; gap -0.0492). Scaffold curve tops out at n=8,000
  (RF 0.5737 / GIN 0.5184, gap -0.0554). **No crossover anywhere in the
  measured range.**
- Earlier points: n=500 RF 0.4311 / GIN 0.3421 -> n=8000 RF 0.7355 /
  GIN 0.6942 (random). The GIN keeps climbing, the gap shrinks from
  -0.089 to -0.049, but does not close.

**Scaling verdict (`experiments/scaling_analysis.py`)**
- Model R2(n) = R2_inf - a*n^(-b) (a,b>0, curve_fit multi-start), crossover
  = root of the gap on [n_max, 10*n_max]. Artifacts:
  `results/learning_curve/scaling_egfr_full.json`,
  `figures/learning_curve/scaling_egfr_full.png`.
- random: **n* = 28,156** (pooled fit); per-seed n* mean **33,522 +- 13,558**,
  95% CI **[19,765, 45,215]**, **3/3** seeds find a root
  (45,702 / 35,950 / 18,913).
- scaffold: **n* = 25,112** (pooled fit); per-seed only **1/3** succeeds
  (15,786; seeds 1 and 2 have no root in [8,000, 80,000]), CI
  **unavailable** (n* std/percentile undefined with a single value).
- **R2_inf > 1 warnings**: random RF 1.19, random GIN 2.37, scaffold GIN
  1.88 - a fitted asymptote above perfect R2 is impossible, so the
  asymptote (and hence n*) is only weakly identified by 6 curve points.
  Read n* as an order-of-magnitude extrapolation (~25k-35k, ~3x the
  current pool), not a forecast. `no_crossover_projected` is false for
  both splits - i.e. the fit does project an eventual crossover.

**gnn_05 shared-failure reproduced on 4 targets**
- Worst-50 overlap (random / scaffold) and Spearman rho of per-molecule
  |error|, from `results/error_analysis_{tag}_{split}.csv`:
  - egfr    **25 / 21**, rho **0.52 / 0.52**  (random expectation ~2)
  - mapk14  **25 / 30**, rho **0.49 / 0.55**  (expectation ~1)
  - herg    **32 / 42**, rho **0.52 / 0.60**  (expectation ~1)
  - a2a     **32 / 32**, rho **0.63 / 0.65**  (expectation ~7)
- Overlap spans 21-42 against an expected 1-7 (10-40x chance), rho
  0.49-0.65: "both models fail on the same molecules" is **molecule-
  intrinsic and holds across targets**, not an EGFR quirk. Corroborates
  README 7.5 and conclusion 8.

**Execution environment**
- The egfr_full chain (download -> clean -> superset check -> featurize ->
  RF -> splits+verify -> graphs -> GIN x6 -> compare) and the extended
  learning curve + scaling analysis ran on the **M3 Max over the
  Thunderbolt bridge**, inside the miniforge `qsar` env
  (`/Users/leisirui/qsar-activity-prediction`, python 3.12); the chain and
  curve logs are kept as `logs/egfr_full_chain.log` and
  `logs/egfr_full_curve.log`. Docs written on the M4 Air.
- herg / mapk14 `gnn_01` verification lines were not in the archived logs;
  re-run on the M4 Air for this sign-off (deterministic, seconds):
  `[OK ] random R2=0.6152846 (expected 0.6152846)` /
  `[OK ] scaffold R2=0.4296432 (expected 0.4296432)` (herg) and
  `[OK ] random R2=0.7361120 (expected 0.7361120)` /
  `[OK ] scaffold R2=0.6169340 (expected 0.6169340)` (mapk14); regenerated
  split arrays are content-identical to the archived ones.

## 2026-10-03

### README Conclusions (m4-010) — DELIVERED — campaign complete
- Commit: 767f75b (`docs: README conclusions from deep-dive experiments`,
  README.md only, +35 lines)
- Six numbered conclusions in README: (1) GIN does not beat fingerprint+RF;
  (2) RF saturates early, GIN still climbing at n=4400; (3) signal is real
  (Y-randomization); (4) XGBoost does not beat the tuned RF either;
  (5) 2048 bits not dispensable on scaffold splits; (6) the two
  architectures attend to different molecular parts, in proportion to how
  differently they fail.
- All three batches verified end-to-end. deep-dive is now a complete
  story arc; merging into main is the user's decision.

## 2026-10-03

### Interpretability alignment (m4-008) — DELIVERED
- Commit: f9d62de (`results: interpretability alignment (GNNExplainer + SHAP)`)
- Question: on the chemotypes where RF and GIN fail asymmetrically
  (chromone n=37 hurts the GIN; thienopyrimidine n=47 hurts the RF),
  do the two models look at the same parts of the molecule?
- Method: 5 disagreement-maximal molecules per family; GNNExplainer
  (edge-mask aggregated to atoms; node masks cannot backprop through
  integer categorical embeddings) vs SHAP interventional bit values
  (path-dependent mode returns garbage on current sklearn) decoded to
  atom sets via Morgan bitInfo spheres.
- Results: cross-model atom overlap mean 0.47 (chromone 0.40 /
  thienopyrimidine 0.54); GNN attention 66% on the scaffold core;
  bit 1367 (aminopyrimidine hinge-binding motif) in the RF top-5 for
  10/10 molecules across BOTH families; recurring bits 1645/1452/329.
- Conclusion: the two architectures attend differently in exact
  proportion to how differently they fail - fingerprints lean on the
  recurring hinge-binding substructure, the GIN spreads attention over
  the scaffold core. A mechanism-level corroboration of README 7.5.
  Caveat: GNNExplainer is post-hoc/approximate; low overlap means
  "attends differently", not "one is wrong".

## 2026-09-25

### Fingerprint ablation (m4-006) — DELIVERED
- Commit: 862b3f4 (`results: fingerprint ablation r={2,3} bits={1024,2048}`)
- Question: how sensitive is the RF conclusion to the Morgan fingerprint
  parameters? Anchor r=2/2048 (the headline featurization) reproduced
  headline to 0.001 (random) / 0.004 (scaffold), well within 0.01.
- Results (test R2, 3-seed mean +/- std, delta vs anchor):
  - random:   r2/1024 0.7448 (-0.0008) | r2/2048 0.7456 (anchor)
              | r3/1024 0.7454 (-0.0002) | r3/2048 0.7489 (+0.0033)
  - scaffold: r2/1024 0.5468 (-0.0114) | r2/2048 0.5582 (anchor)
              | r3/1024 0.5264 (-0.0318) | r3/2048 0.5567 (-0.0015)
- Conclusions: random split is insensitive to fingerprint config (spread
  0.004). Scaffold exposes 1024-bit as the weak point (r3/1024 -0.032);
  with 2048 bits radius 2 and 3 are equivalent. The headline r=2/2048
  default is sound - and 2048 bits cannot be trimmed away.

### XGBoost baseline (m4-005) — DELIVERED
- Commit: a044dea (`results: xgboost baseline vs RF, 3 seeds`)
- Question: does a stronger gradient-boosted tree beat the RF ceiling on
  the same features (2048-bit Morgan r=2) and same 4438 paired pool?
- Results (test R2, 3 seeds): random XGB 0.7316 vs RF 0.7411 (delta -0.010);
  scaffold XGB 0.5172 vs RF 0.5616 (delta -0.044). XGBoost 3.4.1 hist/CPU
  was bit-identical across seeds (std=0).
- Conclusion: XGBoost with fixed generic hypers does NOT beat the tuned RF;
  it lags clearly on scaffold (novel chemotypes). The RF-vs-GIN story is
  not an artifact of a weak tree baseline.

### Y-randomization (m4-004) — DELIVERED
- Commit: be6d532 (`results: y-randomization, RF + GIN, 3 seeds`)
- Sanity test: train/valid labels permuted (3 seeds), test labels real.
  Controls (real labels, seed 42) must land on known baselines.
- Results (test R2): controls RF 0.7414/0.5598 (random/scaffold),
  GIN 0.7034/0.5244 - all in band. Shuffled: RF -0.177+/-0.028 / -0.152+/-0.012,
  GIN -0.006+/-0.017 / -0.023+/-0.007 - every shuffled mean is NEGATIVE
  (shuffled GIN early-stops by epoch 9).
- Conclusion: the reported signal is real; no leakage. Both models agree.

### Learning curve (m4-002) — DELIVERED
- Commit: 6ea16a6 (`results: learning curve n=500-4400, RF vs GIN, 3 seeds`)
- Question: how do RF / GIN scale with training-set size; is either saturated?
- Protocol: test/val/hyperparameters frozen; only n varies (500..4400,
  9 points); RF and GIN trained on identical molecules per (n, seed).
- Results (test R2, 3-seed mean +/- std):
  - random:   RF 0.541+/-0.007 (n=500) -> 0.741+/-0.000 (n=4400);
              GIN 0.400+/-0.035 -> 0.701+/-0.011
  - scaffold: RF 0.438+/-0.018 -> 0.562+/-0.002;
              GIN 0.389+/-0.014 -> 0.526+/-0.020
- Conclusions: RF saturates early (flat from ~n=2000); GIN still climbing
  at n=4400; scaffold gap narrows to 0.036, random gap 0.040. More data
  likely helps the GIN more than the RF.
- Note: curve RF tops slightly below headline (0.741 vs 0.747) by design -
  the paired pool is the 4438-molecule GNN train pool; headline RF trained
  on the full 4932.

### Batch-1 scripts ready (task books m4-004/005/006 posted)
- Commit: 82bebd0 - y_randomization.py, xgboost_baseline.py,
  fingerprint_ablation.py (+ xgboost==3.4.1 pin, libomp)
- Smoke-tested on M4 Air (incl. one dataset-y alignment bug caught and
  fixed before dispatch). M3 Max executing; results pending.

## 2026-09-24

### Reference baseline (main @ 8807296)
- EGFR, 6165 unique compounds; test = 20%; splits persisted under
  data/processed/splits/.
- RF (Morgan r=2/2048, grid-tuned): random R2=0.747, scaffold R2=0.562.
- GIN (MoleculeNet-style, seed 42): random 0.691 / scaffold 0.544;
  3-seed means 0.690+/-0.001 / 0.526+/-0.016.
- Headline finding: on this dataset the GIN does not beat fingerprint+RF;
  both lose ~0.2 R2 from random to scaffold split.
