# Experiments Log

Running log of the deep-dive experiments. Each entry: what ran, headline
numbers (3-seed mean +/- std unless noted), and the commit holding the
artifacts. Newest first. Maintained by M4 Air (this machine, Apple M4
MacBook Air); the 2026-10-10/11 refresh-campaign panel trainings ran on
the win4070 CUDA host (logs committed under `logs/win4070_/`) and the
auxiliary batches (learning curve, y-randomization, multitask) on this
M4 Air under MPS; older 2026-09/10 entries also record M3 Max runs.

**Status pointer (2026-10-14): the repair below defines the current temporal
results. The initial 2026-10-13 entry records the preceding campaign; its hERG
scores and aggregate means are superseded. Earlier all-year-label time scores
are historical only.**

## 2026-10-14 — post-pilot expansion to eight seeds per environment/budget cell

- **Decision chronology:** the M3 three-seed pilot was completed, accepted and
  committed as `54f4c7d` before the user explicitly requested eight seeds in
  every cell. The new matched CUDA experiment had not started. This is a
  post-pilot design amendment, not an eight-seed preregistration; preserve the
  earlier design, pilot reports, parameters and result archives unchanged.
- **Current scope:** cross-computing-environment reproducibility and robustness
  testing, with fixed data/splits and the same GIN architecture. CPU/CUDA ×
  original (`100/0/20`) / extended (`200/60/40`) epochs/minimum/patience;
  eight targets × seeds **42/1/2/3/4/5/6/7** gives **64 jobs per cell,
  128 per environment, 256 total**. This is not data k-fold cross-validation
  or an architecture comparison.
- **Execution status:** M3's accepted 48 pilot jobs are retained. The additional
  80 M3 jobs (seeds 3/4/5/6/7 in both budgets) are running; completion and
  independent artifact/model-replay acceptance are pending;
  the accepted combined CPU cohort will contain 128 jobs. The complete new
  CUDA cohort contains 128 jobs and will start only after the user's
  connection-ready message. No eight-seed result is complete or claimed here.
- **Frozen computation:** retain trainer, original worker, label loader,
  graph dataset, common helpers and all data hashes from
  `90a470050367cebc4cd83dfe8d3f7fb37d53e6de`. Training mathematics are unchanged.
  New coordinator/seed-wrapper entry points have separate recorded provenance,
  with their supplement bundle frozen at `8521063`; they do not replace the
  frozen training code.
- **Reporting:** preserve the pilot's actual result (23/24 identical saved
  prediction pairs; only HIVPR seed 42 improves validation; diagnostic flags
  9/24 → 9/24). Report the complete eight-seed cohort separately, retain every
  declared seed and do not select from test performance. The original offline
  replay commands and paths remain valid in the report's default three-seed
  mode. Same-numbered CPU/CUDA seeds do not guarantee the same random sequence
  or operator results; the contrast includes backend RNG and operator
  implementations, not only floating-point differences. It is not a causally
  isolated hardware test. Main temporal/RF results remain untouched.
- **Prepared analysis path:** `time_budget_report.py` accepts a separate pilot
  and supplement tree with eight declared seeds; the future combined CPU
  report requires all 128 jobs. A fresh CUDA report uses `--training-device cuda`
  and likewise requires 128 jobs. `experiments/time_environment_comparison.py`
  requires both independently accepted reports, then emits 64 paired rows and
  five environment/budget contrasts. Missing CUDA evidence is an error, not
  a partial comparison. The additional public M3 archive is planned at
  `m3_extension/study.tar.gz` under the study results; it is not yet published.
- **Amendment record:**
  `results/time_budget_sensitivity/replication_extension.json`. It supersedes
  the earlier plan to finish all four three-seed cells before deciding on
  extra seeds; the historical entries below record what was planned and
  observed at those earlier stages.

## 2026-10-14 — M3 two-budget study accepted; new CUDA cohort pending

**Historical three-seed pilot entry.** Its results and original 96-job plan
are retained below; the current eight-seed amendment above supersedes its
next-step plan and pending CUDA job count.

- **Scope/status:** the M3 half of the cross-computing-environment
  reproducibility and robustness study is complete: eight targets × three seeds
  × two budgets = 48 GIN jobs. Fixed labels/splits make this distinct from data
  k-fold cross-validation. The two fresh 4070 CUDA arms remain pending the user's
  connection message; the full four-cell, 96-job design is not yet complete.
- **Execution:** source `90a470050367cebc4cd83dfe8d3f7fb37d53e6de`, CPU with four
  concurrent workers and 4 intra-op / 1 inter-op threads per worker. Original
  `100/0/20` and extended `200/60/40` epochs/minimum/patience share the fixed
  model, scheduler and input versions. Training wall time was 683.063 seconds
  (11 min 23 s), excluding setup, transfer and verification. This is not an
  isolated hardware-speed benchmark.
- **Acceptance:** all 48 statuses succeeded and canonical/snapshot artifacts,
  provenance, labels and parameters passed checks. Independent CPU replay
  matched train MSE, validation RMSE and test predictions within the 1e-5
  tolerance (maximum test-prediction error 9.54e-7). Logs verify the actual
  stopping epochs, including the extended minimum. All 24 paired common log
  prefixes match at their printed precision, without a claim of bitwise
  optimizer-state identity.
- **Paired result:** 23/24 saved test-prediction arrays are unchanged. Only
  `hivpr` seed 42 selects a later best epoch (44 → 84; stopping 64 → 124), with
  standardized train MSE 0.273681 → 0.166170 and validation RMSE
  0.741784 → 0.735256. Its secondary test R² is 0.012266 → 0.058906.
  Across all pairs mean Δvalidation RMSE is −0.000272; common flags remain
  9/24 → 9/24. Mean single-seed test R² changes −0.009381 → −0.007437, driven
  by that one pair and not strong evidence of a general gain. Under the current
  scheduler and these budgets, extending training did not broadly improve
  selected models; this does not prove that every model has converged.
- **Cohorts and next step:** the earlier accepted CUDA 13/24 flags are a
  separate historical reference, not the M3 original-budget count and not a
  matched new CUDA cell. Complete the new CUDA original/extended arms with the
  same frozen training source and inputs, retaining every target/seed. Only
  after the initial three seeds in all four cells, plan added seeds from
  validation variability and precision needs, never from test performance.
  CPU/CUDA differences combine systems, architecture, libraries and kernels;
  report environment-associated contrasts and interactions, not pure hardware
  causality. No main campaign weights, predictions or metric history are replaced.
- **Offline evidence:** `results/time_budget_sensitivity/m3/analysis/` contains
  48 diagnostics rows, 24 paired rows, eight target summaries, 24 log-prefix
  comparisons and the acceptance report. The frozen `inputs.tar.gz` is about
  4.6 MiB; `m3/study.tar.gz` is 29,868,373 bytes (28.5 MiB), with canonical and
  snapshot duplicates stored as tar hardlinks. All 442 restored result files
  matched the archive manifest and frozen input verification passed. The
  methodology provides hash-first extraction and full offline replay commands.

## 2026-10-14 — PR #8 review corrections and an environment × budget study

This entry records review corrections to the current repaired campaign. The
published model weights, predictions and historical metric JSON entries are
preserved. New diagnostics are separate artifacts. The authorized follow-up forms
a 2 × 2 environment/budget study. Its M3 48-job queue was running when this
review entry was first written and has since completed acceptance, as recorded
in the entry above; the 4070 CUDA campaign's 48 jobs still await the user's
connection message.

- **T1 — resampling-unit sensitivity:** the existing 24-cell scaffold comparison
  covers eight targets and GIN/GINE/AttentiveFP. Seven of ten molecule-bootstrap
  significant cells have scaffold-group intervals containing zero. README and
  methodology now expose the full sensitivity result and distinguish the two
  resampling units; this is a reinterpretation of the existing sensitivity CSV,
  not new training or 24 independent replications.
- **T2 — constant baseline and error decomposition:**
  `experiments/time_baseline_diagnostics.py` writes a separate
  `results/time_split/baseline_diagnostics.csv`, comparing RF and GIN ensemble
  with the train-label-mean constant using 10,000 paired molecule resamples.
  It also records test fractions and the arithmetic level/scale/correlation
  decomposition of the RF-to-GIN R² difference. The interpretation distinguishes
  outperforming RF from outperforming a constant; the decomposition does not
  identify a causal distribution-shift mechanism. Published predictions are
  unchanged, and the figure/summary wording now reflects these distinctions.
- **T3 — diagnose the saved best checkpoint:** the old convergence heuristic
  used the final training epoch's stochastic loss, although the model saved and
  evaluated on test was the best validation checkpoint. CPU eval-mode replay of
  all 24 accepted checkpoints verifies their training normalization and best
  validation RMSE, then measures the saved checkpoint's training MSE. Thirteen
  of 24 exceed standardized training MSE 0.5 and meet the early-best-epoch rule;
  see `results/time_split/checkpoint_diagnostics.csv`. This is a fitting
  diagnostic, **not proof of non-convergence or evidence that longer training
  will improve generalization**. Historical metric JSON is not rewritten. The
  acceptance record now distinguishes original last-epoch flags from this new
  diagnostic. Future trainer output records best-checkpoint eval MSE, its source,
  and the separate last-epoch loss before applying the heuristic.
- **T4 — direct temporal-label tests:** explicit expected medians cover train,
  validation and test year boundaries, future-measurement exclusion, exact
  relation and nM eligibility; prediction-version checks cover `label_id`, row
  order and true labels. The coordinator's targeted mutation check caught all
  six injected faults. These tests check the label rules directly rather than
  accepting agreement between two consumers of the same wrong labels.
- **T5 — fail on absent structural keys:** the splitter now checks cached parent
  keys against all standardized raw structures before document/year or label
  eligibility filtering. An absent key raises an error with cache row and key;
  an existing raw structure with missing dating/eligible-label metadata can still
  be excluded normally. Cached parent keys are never normalized a second time.
- **T6 — traceable coordination and runnable independent audit:** 16 surviving
  initial/repair coordinator scripts are archived unchanged as `.py.txt` under
  `scripts/campaign_history/`, with SHA256, original scratch paths and
  source-derived worker command templates in its manifest. These post-campaign
  snapshots preserve one-off machine paths and mutation logic and must not be
  executed; they are not a complete historical shell transcript. The separately
  added `scripts/audit_time_campaign.py` is a new parameterized audit, not a
  claim that this exact CLI was used originally. Its `data` and `full` scopes
  both passed against all 42,636 retained labels, 168 exclusions, 32 predictions
  and 16 × 10,000-replicate bootstrap rows. Eight dedicated tests and Ruff passed.
  Commands, dependencies and explicit skipped-hash semantics are documented in
  methodology under "Independently audit raw labels and current results".
- **T7 — protect accepted RF models from the old release:**
  `scripts/fetch_artifacts.py` installs only the 18 named random/scaffold RF
  models from `artifacts-v2`. It validates archive members but does not install
  its eight obsolete temporal models, including when other local models are
  missing. Existing accepted time models are preserved; new time RF weights
  remain available on M3/the coordinator rather than in that old release.
- **Additional guard:** learning-curve, y-randomization and XGBoost auxiliary
  commands now reject unsupported `time` splits explicitly, instead of silently
  applying their random/scaffold label path to temporal data.
- **Authorized 2 × 2 follow-up:** both M3 CPU and 4070 CUDA run the original
  budget (`epochs=100, min_epochs=0, patience=20`) and the extended budget
  (`epochs=200, min_epochs=60, patience=40`). Each cell contains eight targets ×
  seeds 42/1/2: **24 jobs per cell, 48 per environment, 96 total**. M3 has now
  completed acceptance; the fresh 4070 pair is waiting for the user's
  connection message. All four cells must use the same frozen current labels,
  split indices, graph inputs, training-source revision, model, optimizer,
  scheduler and validation-selection rule. Only the declared budget and
  environment factors differ; environment versions and execution settings are
  recorded rather than assumed equivalent.
- **Analysis and interpretation:** retain every target/seed result and its
  paired contrasts, not only four aggregate averages. Compare extended minus
  original within each environment, CUDA minus CPU at each budget, and the
  difference between the two within-environment budget contrasts. Primary
  diagnostics are best-checkpoint eval training MSE and validation RMSE; report
  the full cohort's test scores secondarily without selecting budgets, seeds or
  replacement models by those scores. The common early-fit diagnostic uses
  `best_epoch < 30` and standardized train MSE > 0.5 in both budget arms, avoiding
  a changed threshold merely because the maximum epoch count doubled. Operating
  system, architecture, numerical kernels and underlying libraries remain
  bundled with device, so the comparison estimates an **environment-associated
  difference, not a pure causal hardware effect**. Longer training is not
  presumed better. The old accepted CUDA fits remain historical references and
  are not substituted for the newly matched original-budget CUDA cell; all
  existing accepted results remain intact while the study is performed.

## 2026-10-14 — hERG parent-key repair and reproducible temporal inputs

- **Data correction:** time splitting was normalizing already-normalized cache
  SMILES again. For `CHEMBL2035183`, that changed the key and excluded a valid
  2012 exact 7,200 nM measurement. The raw-record key is now normalized once
  and joined to the unchanged cache key. hERG train increases 4,943 → 4,944;
  exclusions decrease 20 → 19 (168 total). Validation/test rows and labels are
  unchanged. All other seven split arrays and their retained file bytes match.
- **Training:** source `d13f9125af56028f60e099ec2e914dd2509c239e`; replace only
  hERG RF + GIN seeds 42/1/2. RF ran on M3 with 16 CV workers in 1,625 seconds;
  GIN ran on the 4070 in 47 seconds. Original model recipes are unchanged.
  The other 28 jobs retain source `7ce15c1` and their original provenance.
- **Worker correction:** each attempt starts with isolated source/input files
  and empty outputs. NA/input/source/manifest changes are rechecked before
  execution, resume and publication. No-op or partial runs cannot reuse old
  artifacts; publication failures roll back replaced files.
- **Acceptance:** four replacement jobs passed frozen source/input/label hashes,
  metric recomputation and checkpoint loading. RF replay is exact; maximum
  CUDA GIN replay error is 9.54e-7. Independent CPU replay of all three GIN
  validation scores matched exactly, with train-only normalization verified.
  All 32 current prediction files align with the current labels; non-time
  metric entries remain unchanged. Acceptance maps jobs to the initial or
  repair manifests instead of assigning one source revision to every job.
- **Updated results:** hERG RF R2 −0.056; GIN per-seed −0.065±0.033; GIN
  ensemble −0.034. Ensemble ΔR2 +0.022, 95% CI [−0.016, +0.061]. B=10,000
  bootstrap retains four GIN wins, two RF wins and two intervals containing
  zero. Eight-row means are RF −0.034, GIN per-seed +0.022, ensemble +0.062.
  The original uncertainty and overlapping-target caveats still apply.
- **Reproduction:** publish the eight frozen splits as a 2.1 MiB package in
  `results/time_split/inputs/`. `scripts/restore_time_inputs.py` verifies hashes,
  split signatures and all prediction/metric label versions before restoration.
  Conflicting local inputs are refused. The as-of constant baseline now reads
  split labels directly, so raw tables and feature caches are unnecessary for
  bootstrap/summary reproduction. CI restores and validates the package before
  running tests. Analysis code is recorded at `0b4fb29`.
- **Final verification:** 327 tests passed in both the coordinator worktree and
  a clean exported checkout; Ruff and diff checks passed. Without raw tables or
  feature caches, restore → B=10,000 bootstrap → summary reproduced all 12 CSV
  and PNG outputs byte-for-byte. An independent raw-record audit recomputed
  42,636 labels, checked all 168 exclusions and 32 predictions, and reproduced
  all 16 bootstrap intervals. Logs and reproduction evidence are committed.
- **Records:** `results/time_split/campaign_acceptance.json`,
  `repair_split_comparison.json`, `repair_independent_review.json`, repair input
  manifests and `logs/time_asof_repair_20261014/`. The original acceptance is
  preserved as `initial_campaign_acceptance.json`. RF release assets remain
  historical; accepted new joblib files are retained on M3 and the coordinator.

## 2026-10-13 — completed as-of temporal retraining

- **Frozen protocol:** source `7ce15c1f0849ae69e0ca208f781a9824517eb328`,
  `exact-dated-asof-v1`, eight targets, 169 label-eligibility exclusions.
  RF fits train only with its 12-setting, five-fold search; GIN fits the
  same pool and selects checkpoints on validation, seeds 42/1/2. No test
  scores were used to alter the recipe or discard seeds.
- **Execution:** M3 Max completed eight RF jobs in 117.9 minutes (four
  CV workers for the first five targets; sixteen for hivpr/mpro/vegfr2).
  RTX 4070 Laptop GPU, 8 GiB, torch 2.14.0+cu126 / CUDA 12.6 completed
  24 GIN fits in 225.3 seconds, excluding setup and transfer. All stopped
  after 20 epochs without validation improvement; best epochs span 2–33.
  No script convergence flags were triggered; this is not evidence of
  optimal fitting. M4 Air prepared inputs, accepted results and ran analysis.
- **Acceptance:** 32/32 jobs passed source/input/label hash checks, test-row
  alignment, checkpoint loading and metric recomputation. RF checkpoint
  replay matched exactly; maximum CUDA GIN prediction difference was
  9.54e-7. Records: `results/time_split/initial_campaign_acceptance.json`, both
  `input_manifest_*.json` files and `logs/time_asof_20261013/`.
- **Results:** B=10,000 paired molecule bootstrap of the three-seed
  ensemble favours GIN on a2a/egfr_full/hivpr/mpro and RF on abl1/vegfr2;
  egfr/herg intervals contain zero. Unweighted mean time R² is RF −0.033,
  GIN per-seed +0.020, GIN ensemble +0.061. Current figures and README
  report negative scores, train-only constant baselines and uncertainty.
  These intervals condition on fitted seeds; targets include overlapping
  EGFR datasets and are not independent replications.
- **Integration:** archived previous temporal artifacts before replacing
  predictions, GIN checkpoints and figures; merged only `time` and RF
  `best_params_time` into metric JSON files. All non-time metric entries
  are unchanged. The old MSE decomposition remains a historical record.
- **Distribution:** RF models are retained on M3 and the coordinator,
  outside Git. The `artifacts-v2` release has not been updated for this
  campaign and still contains historical RF time models.
- **Validation:** 292 tests passed; Ruff and `git diff --check` passed.
  The stored-constant regression test now independently selects train-only
  as-of labels, while preserving its historical non-test-pool branch.

## 2026-10-11

### Full refresh: data rebuild + whole-campaign retrain + README rewrite — DELIVERED (batch uncommitted at write time; commits handled centrally)

- **Data refresh (review H1 / M1)**: `01d`/`01e` downloaders now pull
  `standard_relation`/`activity_id`, and `02_clean_data.py` keeps only
  `standard_relation == "="` - censored `>`/`<`/`~` rows are bounds, not
  labels; salts collapse to the **parent structure before dedup**
  (`parent_smiles`), and `03_featurize` + the `document_years` join
  follow the parent too. 7 ChEMBL targets re-pulled raw, `egfr`
  rebuilt; panel size 55,145 -> **42,804** unique compounds (`herg`
  12,019 -> 8,403, `egfr` 6,165 -> 4,560, `egfr_full` 13,497 -> 10,729,
  `vegfr2` 11,797 -> 9,643). Records: `docs/reviews/fixes-2026-10-11.md`,
  `docs/reviews/h1-sensitivity-2026-10-10.md`, and the fill-in number
  draft `docs/reviews/readme-numbers-2026-10-11.md`.
- **Full retrain** (everything below is post-refresh unless flagged):
  - **RF**: `scripts/04_train_and_evaluate.py`, 8 tags x 3 splits
    (random / scaffold / time) -> `results/metrics_{tag}.json` +
    `rf_preds_{tag}_{split}.npz`; `vegfr2` was the last to land
    (10-11 15:31).
  - **GIN**: 8 tags x 3 splits x 3 seeds = **72 runs** ->
    `results/gnn_metrics_{tag}{,_seed1,_seed2}.json` (each file holds
    random/scaffold/time).
  - **XGBoost**: `experiments/xgboost_baseline.py`, 8 tags x 2 splits x
    3 seeds, `rf_control` re-read from the refreshed metrics (15:31).
  - **Multi-task**: `scripts/gnn_07_multitask.py`, 2 arms x 3 seeds x 2
    splits -> `results/multitask/` (10:47-11:32).
  - **Uncertainty + AD**: `experiments/uncertainty_ad.py`, 16 cells x
    {gin, gine} -> `results/uncertainty/` (15:31).
  - **Significance**: `experiments/paired_bootstrap.py`, B = 10,000,
    **both conventions** (`seed42` + `mean3`) for every cell ->
    `results/significance/summary.csv` (15:28), the per-cell
    `*_{random,scaffold}[_gine|_attentivefp].csv`,
    `summary_attentivefp.csv` + `summary_partial_gine.csv` (17:16) and
    the time CIs merged into `results/time_split/summary.csv` (16:59).
  - **Architecture panels**: `experiments/gine_panel.py` ->
    `results/gine_panel/summary.csv` + `figures/gine_panel/`;
    `experiments/afp_panel.py` (48 runs) -> `results/afp_panel/` +
    `figures/afp_panel/`; forest plots
    `figures/significance/ci_panel_with_{gine,afp}.png`.
  - **Fingerprint ablation**: 8 tags x 2 splits x {r2, r3} x {1024,
    2048} x 3 seeds = **192 RF fits** -> 8 x
    `results/fp_ablation/summary_<tag>.json`.
  - **Auxiliary family**: y-randomization (7 tags), learning curves
    (7 tags), interpretability alignment (7 tags), error analysis
    (7 tags), `multi_target` + `time_split` summaries - all re-run on
    the new data (15:45 / 12:56 / 16:59-17:00); all 7 y-randomization
    verdicts read `ok: all shuffled-label means < 0.2`. The
    `run_aux_exp.sh` tag list still excludes `vegfr2`.
- **Key new numbers**:
  - **`vegfr2`/time RF = +0.160** (pre-refresh +0.093); `vegfr2` random
    0.670 / scaffold 0.585 (was 0.736 / 0.617). With that last stale RF
    gone, every `vegfr2`-derived Δ (panel, time, multitask, xgboost,
    uncertainty, gine/afp) now sits on the refreshed baseline.
  - **Significance pattern: RF 7 / GIN 2 / ns 7** across the 16
    random+scaffold cells (pre-refresh: 4 cells, all RF). The two GIN
    wins are `abl1`/scaffold **+0.037 [+0.006, +0.069]** and
    `herg`/random **+0.038 [+0.006, +0.069]**, both
    `frac_boot_gin_better` ~0.99.
  - **`abl1`/scaffold flips**: ST-GIN 0.737±0.027 vs RF 0.714, per-seed
    Δ **+0.022** (seed 42 alone: +0.060) - the one scaffold cell where
    the GIN beats the RF, and it survives the ensemble bootstrap; the
    pre-refresh panel had all 8 scaffold cells negative.
  - **Time split**: 6/8 CIs exclude 0 - GIN ahead on `a2a` **+0.326**,
    `egfr_full` **+0.592**, `hivpr` **+0.115**, `mpro` **+0.084**; RF
    ahead on `abl1` **−0.236** and `vegfr2` **−0.052**; `egfr` (+0.039)
    and `herg` (+0.032) straddle 0 (pre-refresh: 5/8, all GIN).
  - **GINE panel**: **0 of 10** bootstrapped cells favour GINE, 5
    significantly favour RF, 5 straddle 0; the other 6 cells
    (`abl1`/`herg`/`vegfr2`) have no per-seed GINE predictions yet and
    are point estimates only (2 of them positive: `abl1`/scaffold
    +0.028, `herg`/random +0.006).
    *(superseded 2026-10-12 — see bottom entry: full-panel bootstrap, all
    16 cells carry CIs; mean3 now 1 GINE win / 7 RF wins / 8 undecided)*
  - **AttentiveFP panel**: wins **2 of 16** (`herg`/scaffold +0.062
    [+0.030, +0.092], `herg`/random +0.034 [+0.002, +0.065]), 8 cells
    favour RF, 6 undecided; all 8 random point estimates negative.
    *(superseded 2026-10-12 — see bottom entry: after the `abl1`/scaffold
    seed-42 rerun the mean3 tally is 7 RF / 2 AFP / 7 undecided)*
  - **Fingerprint ablation**: random grid stays within **−0.019 …
    +0.003** of the r=2/2048 anchor on all 8 tags; scaffold spreads
    **−0.043 … +0.041**, worst cells all r3/1024 (`vegfr2` −0.043,
    `hivpr` −0.042, `abl1` −0.040).
  - **XGBoost**: 13/16 cells ≤ RF (worst `egfr`/scaffold **−0.132**).
    *(superseded 2026-10-12 — see bottom entry: same-pool rerun, 11/16
    cells ≤ RF, worst `hivpr`/scaffold **−0.080**)*
  - **Multi-task pool**: 22,492 rows / 21,093 unique SMILES / 1,193 =
    **5.66%** appear under >1 target (pre-refresh 10.0%).
  - **Fairness campaign completed (3 tags)**: `experiments/gnn_fairness.py`
    finished on win4070 against the refreshed splits/graphs (`vegfr2`
    10-11 17:44, `herg` 18:08, `abl1` 18:12 - all four phases in every
    summary), pulled to `results/fairness/` + `figures/fairness/`; the
    README fairness table + status notes are re-stamped with the
    refreshed numbers (RF refs = refreshed panel RF: `vegfr2`
    0.670/0.585, `herg` 0.607/0.384, `abl1` 0.822/0.714; baseline GIN
    read from the refreshed `gnn_metrics_*`). `gnn_fairness` still
    saves no per-seed predictions, so those GINE cells remain point
    estimates without bootstrap CIs; `results/gine_panel/summary.csv`
    was rebuilt from these refreshed `gine_runs` (10-11 19:35), so the
    three tags now read the fairness numbers.
    *(superseded 2026-10-12 — see bottom entry: the full-panel bootstrap
    gave every GINE cell a CI, so the point-estimate-only caveat above no
    longer holds)*
- **In progress / not re-run**:
  - **`scripts/07_qc_checks.py` not re-run**: `figures/testset_hist_*.png`
    still predate the refresh (flagged in `docs/methodology.md`).
    *(superseded 2026-10-13 — 07 has since run for every tag:
    testset-hist + `permutation_importance_*` + importance figures for
    7 tags 2026-10-11 evening, bace 2026-10-12, vegfr2 2026-10-13
    (`logs/2026-10-13_art_refresh.log` §2); the methodology flag is
    gone)*
  - `results/significance/summary_time.csv` and the per-cell `*_time.csv`
    are still the 10-10 pre-refresh merge (old `p_one_sided` schema);
    the refreshed time CIs live in `results/time_split/summary.csv`.
  - `vegfr2` is outside the aux driver, so its y-randomization /
    learning curve / interpretability / error analysis remain pre-refresh;
    `permutation_importance_*` and `top_bits_*` were not re-run either.
    *(superseded 2026-10-13 — `top_bits_{bace,egfr,vegfr2}.csv`
    regenerated in the N13 sweep and `permutation_importance_*` re-run
    for all 8 tags (7 tags 10-11, bace 10-12, vegfr2 10-13); the
    vegfr2 y-randomization / learning curve / interpretability claim
    still holds)*
  - README: top Status note, the architecture-comparison section
    (GINE/AFP counts) and the fingerprint-ablation section were rewritten
    against these artifacts; QC stays flagged inline.
  - *(2026-10-13 status note — superseded in part by the bottom
    2026-10-12 entry: `results/significance/summary_time.csv` and the
    per-cell `*_time.csv` were re-merged onto the new schema with the
    refreshed values (`p_one_sided` now 0 hits anywhere under
    `results/`); `top_bits_{bace,egfr,vegfr2}.csv` and
    `permutation_importance_bace.csv` were regenerated and the vegfr2
    error analysis re-run (N13); README was re-stamped with the 10-12
    counts (GINE "1 of 16", XGBoost "11 of 16"). Still open as stated
    above: the QC testset-hist figures, vegfr2 y-randomization /
    learning curve / interpretability, and
    `permutation_importance_vegfr2.csv` (still built from the old
    vegfr2 RF).)*
  - *(2026-10-13 PR #6 status update — that "Still open" list is itself
    now stale in part: QC `figures/testset_hist_*.png` and
    `results/permutation_importance_*.csv` are re-run for every tag
    (7 tags 2026-10-11, bace 2026-10-12, vegfr2 2026-10-13 —
    `permutation_importance_vegfr2.csv` now comes from the refreshed RF,
    R² 0.6695, `logs/2026-10-13_art_refresh.log` §2); the seven
    `figures/learning_curve/learning_curve_*.png` were redrawn
    2026-10-13 with ddof=0 bands; pre-refresh
    `scaling_{egfr,egfr_full}.png` now live under
    `results/_archive-2026-10-09/scaling_ep300/` and the six
    zero-molecule interpretability findings under
    `results/_archive-2026-10-09/interpretability_flat/`
    (`by_tag/` untouched). Still open: vegfr2 y-randomization /
    learning curve / interpretability.)*

## 2026-10-10

### Review 2026-10-10 fix pass (findings PR #1 + independent verification) — DELIVERED (A-layer; no results retrained here)

- Reviews: PR #1 findings (`.claude/skills/qsar-review/references/findings-2026-10-10.md`)
  and the independent verification (`docs/reviews/codex-verification-2026-10-10.md`,
  its corrections preferred). Branch `fix/review-2026-10-10`; fixes uncommitted at write time.
- **Code (A-layer, no retraining)**:
  - `paired_bootstrap.py`: `p_one_sided` renamed to `frac_boot_gin_better` - the fraction of
    bootstrap deltas > 0, not a p-value (a strong GIN win reads ~1.0, so a `p < 0.05` filter
    would drop every GIN win). Summary readers accept both names, so committed csvs still
    saying `p_one_sided` keep working.
  - `04_train_and_evaluate.py`: grid CV is now `KFold(5, shuffle=True, random_state=42)` -
    scaffold/time train pools are ordered by scaffold group / SMILES, so unshuffled folds
    would be single-chemotype blocks.
  - `fingerprint_ablation.py` / `learning_curve.py` / `y_randomization.py` /
    `interpretability_alignment.py`: scaffold cells read `best_params_scaffold`
    (fallback to `best_params_random` with a printed note).
  - `time_split_summary.py`: `DELTA_CONVENTION` re-labelled to the ensemble convention
    (`point=ensemble_mean-pred_R2;ci=ensemble_mean-pred_R2`), matching what `d_r2_mean3`
    actually stores.
  - `fingerprint_ablation.py`: summary std now population std (ddof=0), the repo-wide JSON convention.
- **Docs**: README time-split table gains a `GIN ens` column (ensemble R2 of the 3-seed mean
  prediction) + a note that the Δ column and CI come from the ensemble bootstrap, not from
  subtracting the displayed columns (`herg`: columns give −0.042, ensemble Δ +0.076);
  y-randomization control gap re-attributed to the control's training pool (GNN train pool
  4438 rows vs headline 4932 on `egfr`/random, 494 fewer) with real and shuffled controls
  on the same pool; XGBoost seeds documented as deterministic repeats (default
  subsample/colsample = 1.0 -> std=0 is not a data/split uncertainty estimate);
  `mpro`/scaffold mean3 Δ corrected to −0.085 (full precision −0.0854787);
  y-randomization "collapse" wording scoped to group means (shuffled GIN seeds can be
  slightly positive: −0.029/+0.002/+0.011 on `egfr`/random).
- **Tests**: `test_time_split_summary.py` pins `DELTA_CONVENTION` against an independent
  recomputation from the committed `herg`/time per-seed prediction files (ensemble R2
  +0.041903, RF pinned at −0.033775, ensemble Δ +0.075677; per-seed-mean Δ −0.042404 kept
  as the contrast).
- Not in this pass: the RF re-run under shuffled-fold grid selection (recomputes
  `best_params_*`, rewrites `results/metrics_*.json` + `models/`; summary tables re-derived
  from it) and the H1/M1 label-policy sensitivity runs - logged separately when they land.

## 2026-10-09

### P8 campaign: paired-bootstrap significance + GINE panel + publication-year time split + uncertainty/AD — DELIVERED

- Commits: `a9d4547` (analysis pipeline), `af276d2` (results+figures),
  `45e5ce7`/`e25012d` (docs + M3 RF-time logs). This entry and the docs
  it covers (README section 8/9 CI upgrade + new README sections 11-12;
  methodology "Statistical significance", "Publication-year (time)
  split", "Uncertainty & applicability domain" + the GNN model update)
  landed across those four commits.
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
parallel pass. Commits: `0b9f48d` (scripts/experiments/tests +
backfilled metrics + panel summaries), `7457cda` (raw manifest),
`1373596` (this docs pass).

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

## 2026-10-12

### 2026-10-12 review-remediation campaign (PR #3 follow-up) — DELIVERED

Batch answering the 2026-10-11 re-review's N1-N14 findings (PR #3
follow-up) on branch `fix/review-2026-10-11`. Commits landed so far:
`7fb8478` (N1-N14 medium findings) and `920763d` (all-8-tag XGBoost
rerun + refreshed AttentiveFP forest plot); this entry's documentation
edits were still uncommitted at write time.
Appended at the end of this file rather than above the 2026-10-11
header, so the "newest first" order holds only for the older entries.

- **GINE full-panel bootstrap (N1/N2)**: `experiments/paired_bootstrap.py
  --model gine --n-boot 10000 --force` over all 8 tags x
  random/scaffold rewrote `results/significance/summary_gine.csv` to 32
  rows (seed42 + mean3 for 16 cells) - **every one of the 16 cells now
  carries a CI**, replacing the 5-tag partial. Mean3 tally **1 GINE win
  / 7 RF wins / 8 undecided**; the single win is `abl1`/scaffold
  **+0.058 [+0.028, +0.092]**. `figures/significance/ci_panel_with_{gine,afp}.png`
  regenerated from it, the pre-refresh `figures/significance/ci_panel_gine.png`
  deleted, `gine_panel.py`'s point-only path removed, and
  `tests/test_gine_panel.py` now asserts 16 CI-bearing rows.
- **AttentiveFP `abl1`/scaffold convergence fix (N3)**:
  `scripts/gnn_03_train_gin.py` gained `--min-epochs N` (defers the
  early-stop check) and writes `convergence_flag` + `best_epoch` /
  `max_epochs` / `train_mse` into the metrics json (flag = best epoch
  < 30% of max epochs while train MSE > 0.5). Seed 42 rerun with
  `--min-epochs 60` (log `logs/review11_n3_afp_abl1_scaffold_s42_min60.log`):
  test R2 0.260 (best epoch 15) -> **0.675** (best epoch 82), and the
  cell in `results/significance/summary_attentivefp.csv` moves from
  RF-significant −0.048 to **undecided +0.020 [−0.018, +0.057]**. The
  mean3 panel now reads **7 RF / 2 AFP / 7 undecided** (seed42: 12/16
  significant).
  *(2026-10-13 correction, PR #6/P1: attributing this flip to the
  `--min-epochs 60` floor is wrong — the flip came from the **independent
  MPS rerun itself** (fresh randomness, `device=mps`, printed at
  `logs/review11_n3_afp_abl1_scaffold_s42_min60.log:6`; the two runs
  already diverge at epoch 2). The floor never bound: the rerun's
  longest non-improving stretch is 46 -> 62, below patience 20, so it
  reaches 100 epochs with or without the floor. The original CUDA run
  **early-stopped at epoch 35 with best epoch 15 and train MSE 0.57**
  (not "stopped at epoch 15"). `README.md` and `docs/methodology.md`
  now say this; the N3 rerun json carries no `device` field.)*
- **Time-split no-skill baseline (N4)**: `experiments/time_split_summary.py`
  output gains `constant_r2` (R2 of always predicting the mean pIC50 of
  all non-test rows) and `gin_best_epochs`; `results/time_split/summary.csv`
  regenerated (8 rows, 25 columns, `d_r2_mean3` semantics unchanged).
  RF is below the constant predictor on **4 of 8** tags (`a2a` −0.301 vs
  −0.004, `egfr_full` −0.736 vs −0.455, `herg` −0.058 vs −0.003, `hivpr`
  −0.027 vs −0.006) and **20 of 24** GIN time runs stop before epoch 24
  (median best epoch 10; random/scaffold median is 80) - the "GIN
  degrades less" story is largely shrinkage toward the mean.
  *(2026-10-13 correction, PR #6/R2: the "largely shrinkage" mechanism
  label is withdrawn — time splits are now read through the exact **MSE
  error decomposition** (level / spread / correlation, Shapley split of
  the cross-term, residual ≤ 9e-16; `logs/2026-10-13_time_split_mse_decomposition.log`,
  mirrored in `README.md` / `docs/methodology.md`): a2a Δ +0.326 =
  level **+0.580** + spread +0.008 + corr −0.263 (RF predicts 0.83 log
  units low — a level effect, not shrinkage), only egfr is spread-led
  (+0.079 of +0.039), and on herg the GIN has *more* spread than RF
  (0.44 vs 0.41). The 20-of-24 early-stop count above stays as a fact
  (`gin_best_epochs`) but is no longer used as a mechanism claim.)*
- **XGBoost all-8-tag same-pool rerun (N7)**:
  `experiments/xgboost_baseline.py` now fits on train+valid (the full
  non-test pool, matching the headline RF), only marks `same_pool: true`
  when the sizes actually match, and can no longer pick a learning-curve
  RF as control. All 8 tags x 2 splits rerun -> `results/xgboost/`
  (16/16 `same_pool: true`, 3 bit-identical seeds as always). **11/16
  cells <= RF**; worst cell **`hivpr`/scaffold −0.080** (0.458 vs the
  RF's 0.538); the five cells above RF are all <= +0.032
  (`herg`/scaffold +0.032, `hivpr`/random +0.010, `abl1`/random +0.006,
  `mpro`/random +0.005, `vegfr2`/scaffold +0.003). `egfr` is now
  −0.011 / −0.045 (was +0.010 / −0.132 against a stale-pool control).
  *(2026-10-13 update: `same_pool` is no longer a size check — since
  6c1e3bb `experiments/xgboost_baseline.py:107` compares
  `pool_fingerprint` hashes of the two row-index sets and only falls
  back to a row-count comparison (with a printed WARNING) when the RF
  fingerprint is unavailable. PR #6 Low notes the remaining circularity:
  the RF pool is rebuilt from the same split npz as the XGB pool, so for
  random/scaffold the two can never disagree; a strict verdict needs the
  RF pool derived from `results/rf_preds_{tag}_{split}.npz` instead.)*
- **GINE uncertainty / AD, all 8 tags (N8)**:
  `experiments/uncertainty_ad.py --model gine` regenerated **58 files**
  (16 `results/uncertainty/*_gine.csv`, 8 `summary_*_gine.json`, 34
  `figures/uncertainty/*_gine.png`), ending the pre-refresh vintage
  (test sizes now match, e.g. `egfr` 912, `herg` 1,681). Across the 16
  cells the RF calibration is monotone in **8/16** (mean slope 0.736,
  mean Spearman(std, |err|) 0.356).
- **Notebook rebuild + drift test (N11)**: `scripts/06_build_notebook.py`
  Step 2/3/4 markdown and snippets updated to the current policy,
  `notebooks/qsar_pipeline.ipynb` rebuilt (all 6 code cells byte-identical
  to `scripts/01d/02/03/04/05/07`), and `tests/test_notebook_sync.py`
  (4 AST-based invariant tests) added; it is collected by the existing
  CI `pytest -q`.
- **Fail-closed artifacts path (N5)**: `scripts/02_clean_data.py` no
  longer skips the censoring filter silently - a ChEMBL-format raw file
  without `standard_relation` exits non-zero saying it predates the
  2026-10-11 refresh (only `TAG=bace` / non-CHEMBL ids may skip).
  `scripts/fetch_artifacts.py` verifies every extracted raw file against
  the repo's `data/raw/sha256sums.txt` (`verify_extracted_raw`), and
  `.artifacts_cache/` joined `.gitignore`. Consequence: `--raw/--all`
  now aborts on the stale `artifacts-v1` release until a refreshed one
  is published.
- **Stale-artifact regeneration (N13)**: multitask `vegfr2` `st_rf`
  row refreshed to **0.670 random / 0.585 scaffold** (was 0.736 / 0.617,
  so every vegfr2 `delta_vs_st_rf` is re-based); `results/rf_preds_bace_{random,scaffold}.npz`,
  `results/permutation_importance_bace.csv` (top-100 MDI-vs-permutation
  Spearman 0.349 -> **0.543**; bit 964 now ranks *second* by permutation
  behind bit 650) and `results/top_bits_{bace,egfr,vegfr2}.csv` +
  figures regenerated; `results/error_analysis_vegfr2_{random,scaffold}.csv`
  rerun (n 1,929 / 1,945, worst-50 overlap 23 / 20). Pre-refresh
  diagnostics moved to `results/_archive-2026-10-09/`:
  `gnn_metrics_egfr_tune{A,B}.json`, `gnn_preds_egfr_scaffold_tune{A,B}.npz`,
  `egfr_gin_scaffold_tune{A,B}.pt` (from `results/gnn_models/`), and
  `_diag_tasks1/` (now `results/_archive-2026-10-09/_diag_tasks1/`).
- **M4 regression test rewritten (N12)**:
  `tests/test_time_split_summary.py` no longer pins stale herg
  constants - it recomputes `d_r2_mean3` from the committed per-seed
  prediction npz for all 8 tags and matches
  `results/time_split/summary.csv` to 1e-9 (stored `herg` value
  **+0.032304**), plus a synthetic `build_summary` behavioural test that
  kills the "store the column difference instead of the ensemble delta"
  mutant. `pytest tests/test_time_split_summary.py` -> **3 passed**
  (this file's previous state was 1 failed).
- **qsar-review skill folder (N14)**: `.claude/skills/qsar-review/`
  repaired - `chembl_censoring_counts.py --overlap` rewritten as an
  offline raw-CSV parent-level audit, `qc_salts_censoring.py` audits
  `parent_smiles` with `Uncharger` (matching 02), `SKILL.md` re-stamped
  (test count, downloaders do request `standard_relation`, dedup is on
  `parent_smiles`, `artifacts-v1` staleness warning), the four author
  audit scripts given `--out/--dry-run` and pinned baselines, and
  `references/findings-2026-10-10.md` given a dated amendment note.
  `verify_readme_tables.py` was extended **332 -> 431 cells** (adds the
  BACE detail, GIN-vs-RF, fairness incl. GINE and dataset-size tables)
  and now **exits 1 on any mismatch** (final check here: 431 cells, 0
  mismatches, exit 0; mid-batch it correctly exited 1 on the README's
  then-stale XGBoost delta column, which the README owner has since
  updated).
- **Documentation corrections in this batch**: panel-size totals fixed
  from 45,145 -> 42,796 to **55,145 -> 42,804** (the eight per-tag rows
  sum to those; in this file's `2026-10-11` section and in
  `docs/reviews/{fixes,readme-numbers}-2026-10-11.md`), `data/README.md`
  now records the `egfr` re-pull (8,936 -> 8,000 raw rows, `e37c572`) and
  the S3 mirror's return to HTTP 200, `docs/interview.md` was re-stamped
  against the refreshed numbers, and the fixes doc's open-issue list and
  M4 test records were brought up to date (2026-10-12 addendum).
  2026-10-12 (later): `artifacts-v2` release published (refreshed raw CSVs +
  26 RF models + `document_years.csv`); `scripts/fetch_artifacts.py` now
  points at it, and the README fast-path staleness warning was removed —
  the release manifest + repo `data/raw/sha256sums.txt` double verification
  stays in place.

## 2026-10-13

### PR #6 review-remediation (findings-2026-10-13, R1-R5 + Low) — batch record (nine parallel fix batches)

Answers the third-round review (PR #6; `findings-2026-10-13.md`,
baseline `6c1e3bb`: 165 tests passed, ruff clean, CI green, README 431
cells / 0 mismatches at review time). The reviewer retracted two of
their own earlier items — P5 (under the pinned xgboost 3.4.1 the
`egfr_full`/random 0.6814376116 reproduces) and P2's per-tag "mechanism"
labels (partly wrong). Main conclusion unchanged and strengthened:
**tuned RF >= GIN**, now shown robust to dropping any single training
seed in 20 of its 21 win cells. Items R1-R5 were split across the
fetch / README / methodology / docs-records / interpretability / code /
tests / skill batches working concurrently; per-item state below is as
observed at this entry's write time.

- **R1 single-seed dependence** (analysis: README owner,
  `logs/2026-10-13_single_seed_dependence.log`): the repo's own paired
  bootstrap (B=10,000, seed=42, ensemble = mean of the kept seeds) over
  all 48 GIN/GINE/AttentiveFP random+scaffold cells reproduces every
  committed mean3 row (max |dev| 9.7e-17) and finds **14 cells whose
  verdict changes when any one training seed is dropped**: 4 of the 5
  GNN wins (all but GINE `abl1`/scaffold), 1 of the 21 RF wins (AFP
  `hivpr`/scaffold — per-seed 0.201/0.508/0.377 vs RF 0.538; drop
  seed 42 and the ensemble Δ is −0.013 [−0.076, +0.046], a tie), and
  9 of the 22 ties (all nine become RF-significant). Standard set for
  the docs: verdicts come from the 3-seed ensemble CI; any-drop
  changes ⇒ marked *conditional* — wording landed in `README.md` as
  the *Conditional verdicts* section (2026-10-13); the methodology side
  is owned by the METH batch.
- **R2 time-split error decomposition replaces mechanism labels**
  (analysis: README owner,
  `logs/2026-10-13_time_split_mse_decomposition.log`): GIN-ens − RF R²
  splits exactly (residual ≤ 8.9e-16) into level / spread / correlation
  (Shapley split of the cross-term) from the committed time
  predictions: `a2a` +0.326 = **+0.580** +0.008 −0.263 (RF predicts
  0.83 log units low — a level effect, not shrinkage), `egfr_full`
  +0.592 = +0.392 +0.032 +0.168, `abl1` −0.236 = −0.002 −0.115 −0.119;
  only `egfr` is spread-led (+0.079 of +0.039), and on `herg` the GIN
  spreads *more* than RF (0.44 vs 0.41), so "compression" never fitted.
  The P2-era labels are withdrawn — the N4 annotation at the 2026-10-12
  entry above marks the superseded text; README/methodology carry the
  table (owner batches).
- **R3 trust-anchor hardening** (FETCH): `scripts/fetch_artifacts.py`
  now refuses symlink/hardlink/special-file archive members outright,
  matches the trust anchor case-insensitively, confines every member to
  a per-asset prefix allowlist, and installs only verified regular-file
  copies with backup/rollback (verified present in the tree at write
  time; `tests/test_fetch_artifacts.py` owned by the FETCH/TESTS
  batches).
- **R4 stale-local lockout** (FETCH): a failed skip-path check no
  longer aborts the run — stale files are named, left untouched, and
  the asset falls through to download+verify (`(R4)`-tagged in the
  `fetch_artifacts.py` docstring and skip path at write time).
- **R5 egfr GIN provenance** (methodology batch): the nine committed
  `egfr` GIN results have no training log and the "panel trained on
  CUDA" sentence does not hold for them; fix = commit/rescope (egfr GIN
  logs absent from `logs/` at write time).
- **Artifacts** (ART — full record in `logs/2026-10-13_art_refresh.log`):
  the seven `figures/learning_curve/learning_curve_*.png` redrawn with
  ddof=0 bands; `results/permutation_importance_vegfr2.csv` +
  `figures/importance_mdi_vs_perm_vegfr2.png` recomputed against the
  refreshed RF (R² 0.6695, was pre-refresh 0.736); 15 ep300 files and
  `scaling_{egfr,egfr_full}.{png,json}` moved to
  `results/_archive-2026-10-09/scaling_ep300/` (the scaling pngs left
  `figures/learning_curve/`); the six zero-molecule interpretability
  findings moved to
  `results/_archive-2026-10-09/interpretability_flat/` (`by_tag/`
  untouched).
- **Release**: the `artifacts-v2` assets were re-uploaded **in place on
  2026-10-13** to add `processed_cache.tar.gz` (P4 fast path;
  `fetch_artifacts.py --cache` / `--all`). This entry is the repo-side
  record of that re-upload — nothing in the tree logged it before.
- **Docs/records (this batch, MISC)**: in-place annotations above (N3
  min-epochs attribution, N4 shrinkage reading, N7 `same_pool`
  description, stale-open claims for 07 outputs / learning-curve /
  permutation); `data/README.md` gains the `raw/document_years.csv` row
  (4,147 data rows, `scripts/fetch_document_years.py`, in the manifest
  and the release); `docs/reviews/fixes-2026-10-11.md` test counts and
  status labels synced; `docs/interview.md` tuning pass re-dated to
  2026-09-24 (the `results/_archive-2026-10-09/` folder name is the
  archive date); `docs/reviews/readme-numbers-2026-10-11.md` status
  block gains this round's line; `logs/2026-10-12_summary_attentivefp_abl1_update.log`
  annotated (original run early-stopped at epoch 35 / best epoch 15;
  the flip came from the independent MPS rerun, not the `--min-epochs`
  floor).
- **Verification**: no Python changed by this batch (docs/records only).
  Full `pytest -q` at final re-run (2026-10-13 14:04): **198 passed,
  1 failed** — the failure is another batch's in-flight file
  (`tests/test_time_split_summary.py::test_print_conclusion_labels_drift_conditionally`,
  a new drift-wording test against `experiments/time_split_summary.py`
  owned by the CODE batch; `test_fetch_artifacts.py` failures seen at
  13:56 were green again by 14:04). The suite grew from the review's
  165 to 199 tests while this batch ran; ruff untouched (no `.py`
  edits).
