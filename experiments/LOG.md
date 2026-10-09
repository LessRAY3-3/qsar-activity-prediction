# Experiments Log

Running log of the deep-dive experiments. Each entry: what ran, headline
numbers (3-seed mean +/- std unless noted), and the commit holding the
artifacts. Newest first. Maintained by M4 Air; results produced on M3 Max.

## 2026-10-09

### P8 campaign: paired-bootstrap significance + GINE panel + publication-year time split + uncertainty/AD — DELIVERED

- Commit: pending commit (working tree). This entry and the docs it
  covers (README section 8/9 CI upgrade + new README sections 11-12;
  methodology "Statistical significance", "Publication-year (time)
  split", "Uncertainty & applicability domain" + the GNN model update)
  all land together in that pending commit.
- **Question**: stress-test the frozen-recipe verdict ("no clear GIN
  lead") from four directions: (1) is any per-cell Δ separable from
  resampling noise? (2) do bond features (GINE) flip "GNN does not beat
  fingerprint RF"? (3) does the verdict - and the deployment story -
  survive a publication-year (time) split? (4) can uncertainty estimates
  + an applicability domain support a virtual-screening loop?
- **Method** (one bullet per phase):
  - **Significance** (`experiments/paired_bootstrap.py`): paired
    bootstrap, B=10,000, test MOLECULES resampled with replacement (one
    index matrix applied to both models); every cell carries both
    conventions - `seed42` (reference) and `mean3` (primary: per-molecule
    mean of the 3 GIN seed predictions vs RF); 95% percentile CI +
    one-sided empirical p = #(ΔR2_boot > 0)/B; significant = CI excludes
    0.
  - **GINE panel** (`experiments/gine_panel.py`): 8 targets x 2 splits; 5
    new tags (a2a/egfr/egfr_full/hivpr/mpro) trained x3 seeds with
    `gnn_03 --model gine`; vegfr2/herg/abl1 reuse the fairness run's
    `gine_runs` - point estimates only, gnn_fairness saved no per-seed
    predictions, so those 6 cells get no bootstrap.
  - **Time split**: `scripts/fetch_document_years.py` ->
    `data/raw/document_years.csv` (4,147 ChEMBL documents across the 8
    tags, deduped; 3 documents have no year in ChEMBL);
    `scripts/gnn_08_time_splits.py` cuts on publication year (molecule
    year = MIN document year = first report; test = newest years
    cumulating to >= ~20% of molecules; valid = newest ~10% of the
    remaining pool; NA guards on year span / year coverage / busiest-year
    density); RF via `04_train_and_evaluate.py --split time`, GIN
    unchanged (gnn_03 reads the same npz).
  - **Uncertainty + AD** (`experiments/uncertainty_ad.py`, run for gin and
    gine): RF per-tree std vs GNN 3-seed ensemble std, quintile
    calibration (5 bins), AD = max Morgan Tanimoto to the train set
    >= 0.4, screening strategies ranked by prediction (rf_top /
    AD-filtered / confidence) with P@100/P@200 plus the report-positive
    set {pred >= 7}.
- **Results / headline numbers**:
  - **Significance, random/scaffold (gin, 16 cells)**: mean3 -> only 4
    cells have a 95% CI excluding 0, **all four RF wins**:
    egfr/random **−0.031** [−0.054, −0.008], egfr_full/random **−0.033**
    [−0.048, −0.018], mpro/scaffold **−0.086** [−0.132, −0.041],
    vegfr2/random **−0.033** [−0.053, −0.015]; **GIN 0 cells**. The seed42
    convention instead manufactures 2 "GIN significant wins" -
    abl1/scaffold +0.030 [0.002, 0.060] and vegfr2/scaffold +0.028
    [0.008, 0.047] - both vanish under mean3: the bootstrap version of
    the README's "single seed is noise" rule.
  - **GINE (16 cells, 10 with CI)**: **0 cells with CI > 0**; 5 cells
    significantly favour RF (egfr/random, egfr_full both splits,
    hivpr/scaffold **−0.118**, mpro/scaffold **−0.098**); the only
    positive point estimate is herg/scaffold **+0.024** (no CI -
    fairness vintage). Bond features do not flip the verdict.
  - **Time split (8 targets)**: panel-mean RF R2 0.731 (random) ->
    **−0.084** (time); GIN 0.695 -> **+0.014**. Worst cells: egfr_full RF
    **−0.891** (GIN −0.240), herg RF −0.034. Significance pattern
    **flips**: on time, 5/8 cells are significant and **all 5 favour
    GIN** (mean3: egfr_full **+0.676** [0.617, 0.741], a2a **+0.238**
    [0.129, 0.347], mpro **+0.143** [0.112, 0.175], herg **+0.076**
    [0.034, 0.120], egfr **+0.058** [0.005, 0.112]), **0 cells favour
    RF** - the mirror image of random/scaffold (4/16, all RF). test_frac
    runs over target on a2a (62%) and mpro (51%) because one recent year
    carries most of the data. Honest reading: under temporal drift the
    GIN degrades significantly less (relative robustness), but absolute
    R2 is ~0 or negative on 6/8 targets for both models (only abl1
    0.280/0.229 and hivpr 0.173/0.179 look usable) - neither model is
    deployable at these numbers, so the flip is a robustness ranking,
    not a win.
  - **Uncertainty + AD**: RF tree-std is usable - quintile calibration
    monotone in 12/16 cells, slope ~0.74, Spearman(std, |err|) ~0.45.
    The GNN 3-seed ensemble std is not - the seeds are too close (rho
    ~0.17, slope > 1). AD >= 0.4 covers ~97% of test molecules: in-AD R2
    ~0.67 vs out-of-AD ~−0.19 (worst hivpr/random **−1.36**).
    Counter-intuitive screening result: AD filtering changes P@100 by
    **0.000 in 16/16 cells** - the RF ranking is itself an implicit AD
    filter (no out-of-AD molecule reaches its top 100). What does help
    is slicing the report-positive set {pred >= 7} to its confident half
    (std <= median): precision **0.859 -> 0.910** (15/16 cells); while
    re-ranking the library by confidence alone destroys precision
    (P@100 0.563 vs 0.922).
- **Verdict**: the frozen-recipe verdict survives every stress test and
  gets *harder*, not softer: GINE never wins significantly (0/10 cells
  CI > 0), the 4 significant random/scaffold cells all belong to RF, and
  the only place the GIN wins significantly - the time split - is a
  place where both models score near zero (relative robustness,
  absolute failure). Operationally: ship RF with tree-std uncertainty
  and a confidence filter on the hit set; treat GNN ensemble std and any
  "GINE advantage" as absent.
- **Pitfalls**:
  - **herg RF time hung on the M4**: during the evaluation stage one
    loky worker disappeared and the main process never exited; killed
    and re-run - the retry completed normally (`logs/p8_rf_time_herg.log`
    vs `logs/p8_rf_time_herg_retry.log`). The final herg and egfr_full RF
    time runs were done on the M3 Max.
  - **Nested n_jobs=-1 thread storm on the M3**: GridSearchCV workers x
    RF workers oversubscribed the machine (load 492). Fixed by
    rate-limiting via `RF_JOBS` / `GS_JOBS` env vars - the M3 copy of
    `scripts/04_train_and_evaluate.py` carries that env patch, while the
    M4 copy in this working tree stays stock (the env patch was never
    ported over here).
- **Timing / environment**: ~12h wall clock on the M4 Air 16GB (primary)
  + ~2.5h on the M3 Max (herg/egfr_full RF time); the analysis scripts
  are single-process and fast. Docs written on the M4 Air.
- **Tests**: 6 new files (`tests/test_p8_analysis.py`,
  `test_gnn_models.py`, `test_fetch_years.py`, `test_time_splits.py`,
  `test_rf_time_split.py`, `test_gine_panel.py`) = 43 cases; suite grows
  23 -> 66, all green.
- **Artifacts**:
  - scripts: new `scripts/fetch_document_years.py`,
    `scripts/gnn_08_time_splits.py`; modified
    `scripts/gnn_03_train_gin.py` (`--model {gin,gine,attentivefp}`) and
    `scripts/04_train_and_evaluate.py` (`--split time`); data
    `data/raw/document_years.csv`, `data/processed/splits/{tag}_time.npz`
    (+ `{tag}_time_NA.json` guard markers).
  - analysis: `experiments/{paired_bootstrap,gine_panel,
    time_split_summary,uncertainty_ad}.py`.
  - results: `results/significance/{summary.csv, summary_gine.csv,
    summary_time.csv, {tag}_{split}{,_gine}.csv}`,
    `results/gine_panel/summary.csv`,
    `results/time_split/summary.csv`, `results/uncertainty/` (16 summary
    json + 32 cell csv), `results/rf_preds_{tag}_time.npz`,
    `results/gnn_preds_{tag}_time{_seedN}.npz`,
    `results/gnn_preds_{tag}_gine_{split}{_seedN}.npz`,
    `results/metrics_{tag}.json` / `results/gnn_metrics_{tag}[_gine].json`
    (time-split entries added).
  - figures: `figures/significance/{ci_panel, ci_panel_gine,
    ci_panel_time, ci_panel_with_gine}.png`,
    `figures/gine_panel/panel_dR2.png`,
    `figures/time_split/panel_3splits.png` + `{tag}_timeline.png` x8,
    `figures/pred_vs_actual_time_{tag}.png` x8,
    `figures/uncertainty/` (68 files: 32 calib, 32 ad_error, 4
    vs-screen panels).
  - tests: `tests/{test_p8_analysis, test_gnn_models, test_fetch_years,
    test_time_splits, test_rf_time_split, test_gine_panel}.py`.
  - logs: `logs/p8_fetch_years.log`, `logs/p8_gin_time.{sh,log}`,
    `logs/p8_gine_all.{sh,log}`, `logs/p8_rf_time.{sh,log}`,
    `logs/p8_rf_time_{egfr_full,herg,herg_retry}.log`,
    `logs/p8_uncertainty_gine.log`.

### P8 phase 3: AttentiveFP control (48 runs) — DELIVERED

- **Method**: `scripts/gnn_03_train_gin.py --model attentivefp` - PyG
  AttentiveFP (gated attention + a GRU readout over 30 timesteps) on the
  same `AtomEncoder` + `BondEncoder` front-end, every other element of
  the frozen `gnn_03` recipe untouched - run as **8 targets x 2 splits x
  3 seeds = 48 runs, all rc=0** (`logs/p8_afp_all.sh` ->
  `logs/p8_afp_all.log`), then `experiments/afp_panel.py` +
  `experiments/paired_bootstrap.py --model attentivefp` (mean3, B=10,000).
  Every run kept per-seed predictions, so - unlike the GINE panel's
  fairness-vintage rows - all 16 cells are bootstrappable.
- **Results** (16 cells, mean3): **9 CIs exclude 0 - 8 favour RF**
  (a2a/random, abl1/random, abl1/scaffold, egfr/random, egfr_full/random,
  hivpr/scaffold **−0.239**, mpro/scaffold, vegfr2/scaffold), **exactly 1
  favours AttentiveFP: `herg`/scaffold +0.013 [+0.035, +0.092]**, and 7
  CIs contain 0. **Random: 8/8 Δ negative** (−0.026 to −0.103).
  Reference frame at `herg`/scaffold: GIN **−0.013** (negative), GINE
  **+0.024** (point estimate only, fairness vintage) - the one cell the
  enhanced variants keep landing positive on (tuned GIN +0.025, GINE
  +0.024, AFP +0.013), and the first time it crosses significance (its
  `seed42` row is −0.006, not significant). Under the `seed42`
  convention the panel reads **12/16 significant, every cell to RF** -
  the third panel in a row where single-seed and mean3 pick different
  significance sets (gin manufactured 2 GIN wins; here the only AFP win
  exists only under mean3).
- **Verdict**: the verdict hardens once more. AttentiveFP - attention,
  GRU readout and bond features together, the strongest published
  small-molecule readout - still loses 8 cells significantly to the
  fingerprint RF, wins exactly one, and is negative on all 8 random
  cells. With bond features (GINE), bounded tuning and AttentiveFP now
  each tried and none flipping a verdict, "the GIN does not beat the
  fingerprint RF" is no longer a complaint about our GIN implementation -
  it is a data-scale / signal statement.
- **Caveat** (convention, same as `experiments/gine_panel.py`): the
  panel's `d_r2_mean` is the per-seed-mean Δ (mean of the 3 seed R2s
  minus RF R2) while `ci_lo/ci_hi` is the mean3 *ensemble* Δ (R2 of the
  per-molecule mean prediction minus RF R2); ensembling lifts R2, so the
  point estimate and the interval need not coincide - `herg`/scaffold
  reads **+0.013** as a point Δ against a **[+0.035, +0.092]** CI,
  `hivpr`/scaffold **−0.239** against an ensemble centre of −0.193. Both
  numbers are reported per their own convention.
- **Timing / environment**: **~4.3h wall clock on the M3 Max**
  (caffeinate-wrapped, unattended: 48 runs 01:34 -> 05:17 local, plus
  wrap-up). The driver's auto push was rejected as non-fast-forward, so
  the M4 fetched `20980df` over the LAN and merged it as `854a82c`
  (05:55). Docs written on the M4 Air.
- **Tests**: 66 -> **73**, all green (`tests/test_afp_panel.py` new;
  `tests/test_p8_analysis.py` extended for `--model attentivefp`).
- **Artifacts**:
  - analysis: `experiments/afp_panel.py`,
    `experiments/paired_bootstrap.py --model attentivefp`.
  - results: `results/afp_panel/summary.csv` (16 rows),
    `results/significance/summary_attentivefp.csv` + 16 per-cell
    `results/significance/{tag}_{split}_attentivefp.csv`;
    `results/gnn_{metrics,preds}_{tag}_attentivefp*` from the 48 runs.
  - figures: `figures/afp_panel/panel_dR2.png`,
    `figures/significance/ci_panel_attentivefp.png`,
    `figures/significance/ci_panel_with_afp.png` (gin / gine / afp mean3
    CIs side by side).
  - tests: `tests/test_afp_panel.py`.
  - logs: `logs/p8_afp_all.{sh,log}`.

### Review fixes: delta conventions spelled out, scaffold-split convention note, cuda device branch, best_params backfill, raw-data checksums

Presentation-layer fixes from the user's review - **none of them changes
any published result number, verdict or figure**. Docs items below land
with this entry (the same docs pass also added the y-randomization
real-control 0.7414 vs 0.7466 footnote and the requirements-snapshot
note in README's Reproduce); the scripts/results items come from the
parallel pass. Commit: pending.

- **Δ conventions spelled out**: `docs/methodology.md`'s "Panel Δ
  conventions" now defines both formulas (per-seed-mean vs mean3
  ensemble), why the ensemble side is always at least as large
  (ensemble gain never negative), which artifact reports which, and the
  read rule (significance from CIs, point estimates from panels, never
  mixed); README §8 points there and the long AttentiveFP explanation
  shrank to one line. Why: the conventions had only been explained
  inside `afp_panel.py`, so `egfr`/random −0.056 (panel) vs −0.031
  (significance) and `herg` +0.013 vs [+0.035, +0.092] read as
  inconsistencies.
- **Scaffold-split convention note**: README (first mention + Method)
  and a new methodology section record that `scaffold_split` fills the
  **test** set with the largest Bemis-Murcko groups first - the reverse
  of DeepChem/MoleculeNet's `ScaffoldSplitter` - plus the direction of
  the difference (scaffold scores slightly optimistic vs DeepChem-style
  splits; absolute numbers not comparable across implementations,
  conclusions unchanged), the reason for the choice (chemotype-level
  error analysis needs the big families in test) and the cost (no
  re-runs; MoleculeNet alignment later would mean full retraining).
- **cuda device branch** (scripts/experiments): the shared
  `select_device()` in `scripts/qsar_common.py` (cuda > mps > cpu)
  replaces the seven hardcoded `mps else cpu` picks in the GNN and
  experiment scripts, so CUDA hosts train on cuda; on Apple machines
  the choice - and every logged device string - is unchanged.
- **best_params backfill** (results): run metadata that was missing its
  `best_params_*` block is backfilled with the parameters the run
  actually used, so stored records read complete; read-side only, the
  trained models are untouched.
- **raw-data checksums** (data): the raw-data provenance record now
  carries per-file checksums, so `data/raw/` downloads can be verified
  against what was published.

## 2026-10-08

### Dataset tag rename: `mapk14` → `vegfr2`

- The tag `mapk14` was renamed to `vegfr2` globally: CHEMBL279 is
  VEGFR2/KDR, so the old tag was a misnomer - all files, results and
  figures for this dataset were renamed accordingly (data, models,
  results, figures, logs, code defaults, README, this log).

## 2026-10-05

### Experiment B: GIN fairness - bounded tuning + GINE (vegfr2 / herg / abl1) — DELIVERED

- Commits: `8ba7329` (`feat: fairness experiment (bounded GIN tuning +
  GINE) + multitask kinase pooling scripts`), `e7cefd9` (`results:
  fairness (2/3 targets flip on scaffold after tuning) + multitask
  pooling (worse)`); this entry and README sections 9-10 land in the
  follow-up `docs:` commit.
- **Question**: is section 8's "no clear GIN lead" an artefact of the
  frozen `gnn_03` recipe? Run on the 3 targets nearest parity in section
  8 (Δmean scaffold: `vegfr2`/VEGFR2 −0.006, `herg` −0.013,
  `abl1` −0.013).
- **Protocol** (`experiments/gnn_fairness.py`): bounded grid
  hidden {128,256} x layers {4,5} x dropout {0.2,0.3} = 8 configs,
  trained **only on the scaffold split**, seeds 42/1/2; config chosen by
  **3-seed mean valid RMSE** (README 7.4 lesson - never a single seed).
  The winner is re-run from scratch on both splits x 3 seeds. GINE
  variant (GINEConv + BondEncoder over the cached 3-dim bond features,
  `BOND_FEATURE_DIMS=[13,7,2]`), one fixed config, both splits x 3 seeds.
  Everything else identical to `gnn_03`. **36 runs per target
  (24 tune + 6 final + 6 GINE) = 108 total, all exit 0.**
- **Winners**: `vegfr2` 256/4L/0.2 (mean valid RMSE **0.7056**),
  `herg` 256/5L/0.2 (**0.5145**), `abl1` 256/4L/0.2 (0.8427) -
  **hidden 256 wins on all three**, i.e. the frozen recipe's hidden 128
  was capacity-starved.
- **Results** (test R2, 3-seed mean+-std; Δ = GIN − RF, mean convention):
  - vegfr2/VEGFR2: baseline 0.678+-0.007 / 0.611+-0.024 (Δ −0.058 / −0.006);
    tuned 0.697+-0.006 / **0.622+-0.012** (Δ −0.039 / **+0.005**);
    GINE 0.674+-0.003 / 0.594+-0.013 (Δ −0.062 / −0.023)
  - herg: baseline 0.616+-0.017 / 0.417+-0.007 (Δ +0.000 / −0.013);
    tuned 0.612+-0.016 / **0.455+-0.009** (Δ −0.003 / **+0.025**);
    GINE 0.593+-0.025 / **0.454+-0.007** (Δ −0.023 / **+0.024**)
  - abl1: baseline 0.752+-0.006 / 0.710+-0.037 (Δ −0.039 / −0.013);
    tuned 0.756+-0.014 / 0.693+-**0.063** (Δ −0.035 / −0.030);
    GINE 0.753+-0.007 / 0.638+-**0.083** (Δ −0.039 / −0.085)
- **Verdict**: scaffold flips positive on **2/3** under the mean
  convention (VEGFR2 +0.005, herg +0.025) - the frozen recipe really did
  understate the GIN - but the effect is small and validation-noise
  sensitive: **abl1 is the counter-example** (a config picked on 216
  validation molecules overfits; test R2 *drops* and the seed std explodes
  0.037 -> 0.063), and **the random split flips on none**
  (−0.039 / −0.003 / −0.035). GINE bond features are **not a general
  gain** (help only herg scaffold +0.024; hurt vegfr2 and abl1 scaffold
  −0.085). Section 8's verdict therefore stands as a *frozen-recipe*
  statement; this entry is its boundary, not a reversal.
- **Timing**: **17h41m actual** (vegfr2 4h16m, herg 11h30m, abl1 1h55m,
  serial - one MPS user at a time) against a **4-5h estimate, i.e.
  3.5-4.4x over**. The 8-config grid on a 11.8k-molecule target dominates
  the wall clock.
- **Watcher trap**: `results/fairness/summary_{TAG}.json` is written by
  `save_state()` after **every** phase, so it already exists right after
  `tune` with `"phases": ["tune"]`. A watcher keyed on file *existence*
  fires about a third of the way into a target and reports a
  half-finished state as done. Correct signals: `phases` contains
  `"summary"`, or the `[TAG] done: winner=...` line (the finished files
  list all four phases: tune/final/gine/summary).
- Artifacts: `results/fairness/{summary.csv, summary_{tag}.json,
  {tag}_tuning.csv}`, `figures/fairness/{vegfr2,herg,abl1}.png`,
  `logs/fairness_{vegfr2,herg,abl1}.log`, `logs/fairness_all.log`,
  `logs/fairness_all.sh`, `logs/fairness_run_tag.sh`.

## 2026-10-04

### Experiment C: multi-task kinase pooling (egfr_full + abl1 + vegfr2) — DELIVERED — negative result

- Commits: same pair as experiment B (`8ba7329` scripts, `e7cefd9`
  results/figures/logs); this entry and README section 10 land in the
  follow-up `docs:` commit.
- **Question**: does the industrial "multi-task + more data" setting let
  the GNN overtake the single-task RF it never beat in sections 7-8?
- **Method** (`scripts/gnn_07_multitask.py`): pool the three datasets -
  **27,992 molecule rows**, 25,143 unique SMILES, **2,525 (10.0%) shared
  by two or more targets**. Two arms x 3 seeds x 2 splits (12 runs, all
  rc=0): **MT-GIN** = shared 4-layer trunk + one linear head per target,
  every batch drawn from a single task; **pooled** = one shared head that
  ignores the task label (negative control). Each target keeps its own
  persisted split and its own train statistics; early stopping on the
  mean of the three valid RMSEs.
- **Results** (test R2 3-seed mean, Δ vs ST-RF): **0 of 6 cells won** by
  either arm. MT-GIN Δ −0.079..−0.236, pooled Δ −0.141..−0.541, against
  ST-GIN Δ −0.006..−0.058 - pooling widens the gap to RF everywhere.
  - egfr_full r/s: ST-GIN 0.707 / 0.558 | pooled 0.535 / 0.451 | MT 0.530 / 0.356
  - abl1      r/s: ST-GIN 0.752 / 0.710 | pooled 0.638 / 0.554 | MT 0.625 / 0.614
  - vegfr2    r/s: ST-GIN 0.678 / 0.611 | pooled **0.195 / 0.182** | MT 0.554 / 0.538
- **Contamination / partial repair**: the shared head destroys `vegfr2`
  (0.195/0.182 vs 0.678/0.611 single-task); the per-task heads pull it
  back by **+0.36** on both splits (0.554/0.538) but still trail ST-GIN by
  −0.124/−0.073. MT averages **+0.11** above pooled across the 6 cells,
  yet that average is carried entirely by `vegfr2` - on `egfr_full` (both
  splits) and `abl1`/random, MT is *below* pooled.
- **Leakage**: shared molecules mean **775 unique SMILES** sit in one
  target's train set while appearing in another target's test set on the
  random split (169 on scaffold). The direction is to *inflate* pooled/MT
  scores, so the reported deficit is a lower bound.
- **Head diagnostic**: the spec fixed a **linear** head where `gnn_03`
  uses an MLP head - a single-task re-run through the linear head scores
  **0.661** against `gnn_03`'s **0.707** (egfr_full/random, ~0.04-0.05
  structural handicap). Crediting that back still leaves MT at 0.40-0.67,
  below ST-GIN in all six cells - the verdict does not rest on the head.
- **Verdict**: negative. Pooling / multi-task at this scale makes the GIN
  worse; neither the extra molecules nor the per-task heads rescue it.
- Runtime 15:28-15:56 (M3, four parallel chains + `--aggregate`).
- Artifacts: `results/multitask/{summary.csv, mt_metrics*, 
  pooled_st_metrics*, *_preds_*.npz, models/}`,
  `figures/multitask/{random,scaffold}.png`, `logs/multitask.log`,
  `logs/run_multitask.sh`.

### ep300 training-budget confirmation (egfr_full) — DELIVERED — verdict unchanged

- Commits: `ad70ce4` (`results: epoch-300 training-budget confirmation
  (mean shift << RF-GIN gap)`), `774c550` (`chore: ignore regenerable
  split npz`); this entry and the README 8.2 caveat land in the follow-up
  `docs: LOG/README training-budget note` commit. Authoritative record:
  `results/learning_curve/ep300_confirmation.md`.
- **Why**: the extended learning curve showed large-n runs pinned against
  the 100-epoch cap - all 6 `egfr_full` full-pool baseline runs (2 splits x
  3 seeds) landed at `best_epoch` **93-100** - which made the n=9717 GIN
  numbers look like they might be budget-truncated.
- **Experiment**: same data, same splits, same seeds, same hyperparameters
  (patience 20); only `--epochs 300`. `scripts/gnn_03_train_gin.py` was not
  touched. 6 runs = 2 splits x 3 seeds on the M3:
  `logs/ep300.sh` -> `logs/ep300_confirm.log` (round 1) and
  `logs/ep300_repair.sh` -> `logs/ep300_repair.log` (repair round).
- **Truncation confirmed**: every run early-stopped between epochs
  **100 and 217** (repair log's last epoch is 217), none reached 300;
  `best_epoch` moved to **80-197 with 5/6 now past the old 100 cap**. The
  100-epoch budget really was binding on large-n runs.
- **But a single pass cannot separate the budget effect from run noise**:
  - per-seed ΔR2 straddles zero: **−0.0503 … +0.0308**, **4/6** past the
    0.01 materiality threshold, no consistent direction (random/42 up,
    scaffold/42 and scaffold/1 down, scaffold/2 up; random/1 and random/2
    essentially flat);
  - **two same-seed, same-config ep300 runs differ by up to 0.0277**
    (PyG scatter/atomics non-determinism amplified by early-stop epoch
    selection) - the epoch-budget effect is the same order as run noise,
    so one paired run cannot resolve it; quantifying it needs repeated
    runs per seed.
- **3-seed mean shift (round 1 from the log / round 2 from the files)**:
  random **+0.0080 / +0.0035**, scaffold **−0.0015 / −0.0173**. The larger
  magnitude, 0.0173, is ~31% of the same-split RF−GIN gap (−0.049 random /
  −0.055 scaffold) and the random shift is ~7% of its gap - both far below
  the gap. **The "no measured crossover" verdict is unchanged**, and it is
  now known to be robust to the training budget.
- **Artifact-naming lesson**: `gnn_03_train_gin.py` writes metrics to
  `results/gnn_metrics_{TAG}{suffix}.json` and preds to
  `gnn_preds_{TAG}_{split}{suffix}.npz` - **neither filename carries the
  seed**. Round 1 ran all 6 jobs with the shared `--suffix _ep300`, so the
  three seeds overwrote each other: only seed 2's outputs survived on disk
  (preserved under `_seed2_ep300` names - that metrics file still carries
  `params.suffix = "_ep300"` inside), while seed 42 and seed 1 survived
  only in the log. The repair round re-ran those two under per-seed
  suffixes (`_ep300` = seed 42, `_seed1_ep300`) and the artifact set is
  whole again (15 files). **Every multi-seed run must now get a per-seed
  suffix.**

### Cross-target panel + egfr_full data scaling — DELIVERED — campaign complete

- Commits: `c462607` (`feat: generic ChEMBL downloader (01e) + chain
  runner docs`), `a950e1d` (`results: 6-target panel (a2a/abl1/mpro/hivpr/
  herg/vegfr2) + error analyses`), `39ab879` (`results: egfr_full expansion
  (13,497 cpds) + learning curve to n=9717 + scaling verdict`), `3aca834`
  (`docs: LOG + README cross-target section + data provenance`); the Δ
  convention was then corrected to the 3-seed mean in the follow-up
  `docs: delta convention - 3-seed mean as primary effect size` commit
  (see the delta-convention bullet below).
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

**6-target panel (a2a / abl1 / mpro / hivpr / herg / vegfr2)**
- Same protocol as EGFR: Morgan r=2/2048 + grid-tuned RF, splits regenerated
  and verified bit-identically by `gnn_01_make_splits.py`, GIN 3 seeds
  (42/1/2). Test R2, RF / GIN mean+-std:
  - a2a     n=1,746    random 0.733 / 0.687+-0.003    scaffold 0.691 / 0.666+-0.004
  - abl1    n=2,698    random 0.792 / 0.752+-0.006    scaffold 0.723 / 0.710+-0.037
  - mpro    n=4,424    random 0.730 / 0.707+-0.003    scaffold 0.509 / 0.408+-0.022
  - hivpr   n=2,799    random 0.737 / 0.726+-0.005    scaffold 0.554 / 0.527+-0.028
  - herg    n=12,019   random 0.615 / 0.616+-0.017    scaffold 0.430 / 0.417+-0.007
  - vegfr2  n=11,797   random 0.736 / 0.678+-0.007    scaffold 0.617 / 0.611+-0.024
- **Naming note: CHEMBL279 is VEGFR2 (KDR), not a MAPK target.** This
  dataset was released under an erroneous tag and renamed to `vegfr2` on
  2026-10-08 (see the rename record at the top of this log) - every
  activity row carries
  `target_pref_name = "Vascular endothelial growth factor receptor 2"`
  (verified against the ChEMBL target API). Read every panel row as
  VEGFR2 (README section 8, `results/multi_target/summary.csv`).
- Headline (**3-seed mean convention**): across the 8 panel targets x 2
  splits the GIN **never clearly leads** - 15 of 16 cells are negative and
  the 16th (herg random, +0.0003) is a tie. Random: 7 negative + herg tie,
  range +0.000 to -0.058 (worst vegfr2/VEGFR2). Scaffold: all 8 negative,
  closest vegfr2/VEGFR2 **-0.006**, worst mpro **-0.102**. Magnitude is
  target-dependent but sign is not: abl1 (2,698 cpds) and herg (12,019)
  both land at -0.013, so Δ does not follow n.
- Delta convention (**corrected 2026-10-04, docs follow-up commit**): every
  verdict above is **Δ = 3-seed mean GIN − RF**, taken from the mean
  columns of `results/multi_target/summary.csv` at full precision before
  rounding. The `d_r2_gin_minus_rf` column inside `results/comparison_*.csv`
  is **not** that number - it is the **seed-42** GIN run minus RF, and it is
  reference-only from here on: it reports scaffold "wins" for abl1 (+0.030)
  and vegfr2/VEGFR2 (+0.028) that both **flip sign** under the mean
  (-0.013 / -0.006), and a herg random -0.021 where the mean is +0.000.
  Quoting single-seed deltas contradicts README interview point 6 (GIN
  scaffold R2 varies +-0.02 across seeds), so the campaign headline is
  "no clear GIN lead on any target" - which the scaling verdict below
  (no measured crossover to n=9,717) independently corroborates.

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
  - vegfr2  **25 / 30**, rho **0.49 / 0.55**  (expectation ~1)
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
- herg / vegfr2 `gnn_01` verification lines were not in the archived logs;
  re-run on the M4 Air for this sign-off (deterministic, seconds):
  `[OK ] random R2=0.6152846 (expected 0.6152846)` /
  `[OK ] scaffold R2=0.4296432 (expected 0.4296432)` (herg) and
  `[OK ] random R2=0.7361120 (expected 0.7361120)` /
  `[OK ] scaffold R2=0.6169340 (expected 0.6169340)` (vegfr2); regenerated
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
