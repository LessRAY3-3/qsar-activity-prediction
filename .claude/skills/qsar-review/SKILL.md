---
name: qsar-review
description: Audit this QSAR repo (ChEMBL -> pIC50 -> Morgan-fingerprint RF vs GIN/GINE/AttentiveFP) for data-quality, split-leakage, experiment-hygiene, statistics and README-consistency problems, and write the result as a ranked, evidence-backed report. Use when asked to review, audit or sanity-check the pipeline, before a new data pull or campaign, after changing scripts/ or experiments/, or when README numbers are regenerated.
---

# QSAR pipeline review

This skill re-runs the reviews done on 2026-10-10, 2026-10-11 and
2026-10-12 and keeps their lessons. The findings, with file:line anchors,
evidence and suggested fixes, are in `references/findings-2026-10-10.md`,
`references/findings-2026-10-11.md` (N1-N14, the data-refresh re-review)
and `references/findings-2026-10-12.md` (P1-P8 plus Low, reviewing the
N1-N14 remediation). Start every review by reading the newest file, then
the earlier ones - on the 2026-10-10 file, first its dated amendment note,
which points to the four claims the 2026-10-11 re-review corrected - and
checking which items are still open.

## 0. Baseline (5 min)

```bash
conda env create -f environment.yml && conda activate qsar   # Python 3.12
ruff check scripts experiments tests
pytest -q
```

`requirements.txt` pins need Python >= 3.12 (numpy 2.5.x). On 3.11, install
unpinned: the suite (97 tests at the 2026-10-11 refresh; 135 at the 2026-10-12
PR #4 head reviewed in findings-2026-10-12) and ruff were verified to pass
that way.
Many checks below need only `data/processed/` and `results/`, which are
tracked; `models/` and `data/raw/` come from `python scripts/fetch_artifacts.py --all`.

> **Release status.** `artifacts-v2` (2026-10-12) publishes the refreshed
> raw CSVs and RF models; `fetch_artifacts.py` points at it. The superseded
> `artifacts-v1` (2026-10-08, commit 5b5d244) predates the 2026-10-11 data
> refresh — its raw files lack `standard_relation` and its models were
> trained on pre-refresh data. `scripts/fetch_artifacts.py` now verifies
> extracted raw files against the repo's `data/raw/sha256sums.txt` and
> fails closed on any stale release, and `scripts/02_clean_data.py`
> refuses ChEMBL-format raw files without `standard_relation` (H1 cannot
> silently return).

## 1. Review checklist

Work layer by layer. Every claim in the report must be verified by
re-reading the code or by a number computed from the repo, never inferred.

### A. Data layer (highest leverage: every model inherits it)

- **Censored activities.** ChEMBL stores "IC50 > 10000 nM" with
  `standard_relation` `>`/`<`. Since the 2026-10-11 refresh the downloaders
  (01d/01e) request that column and `02_clean_data.py` keeps only
  `standard_relation == "="` rows before aggregating labels (H1 fixed).
  02 now fails closed on ChEMBL-format input that lacks the column
  (N5 fixed, findings-2026-10-11; re-verified in findings-2026-10-12) -
  the skip is limited to the known bace-style non-ChEMBL input; residual
  fail-open edge cases are tracked in findings-2026-10-12 (Low, code).
  Quantify upstream censoring with `scripts/chembl_censoring_counts.py`
  (live API) and audit the committed raw pulls with its `--overlap` mode
  (section 2).
- **Salt forms and parents.** `02_clean_data.py` standardizes every record
  to `parent_smiles` (largest fragment + Uncharger) and deduplicates on it
  (M1 fixed); `canonical_smiles` in the clean CSV is provenance only (the
  first variant). Count parent-level duplicates, multi-fragment parents and
  split overlap with `scripts/qc_salts_censoring.py`.
- Unit filter (`nM` only), non-positive values, median aggregation and
  invalid-SMILES handling in `02`/`03`: confirm row counts against
  `data/README.md` and `data/raw/sha256sums.txt`.

### B. Splits

- `valid_idx` must be carved from train only; `test_idx` must be asserted
  equal across RF and every GNN seed before any paired statistic.
- Parent-level leakage (salt vs free base on both sides): rows are now
  unique `parent_smiles`, so cross-split parent overlap must be exactly 0;
  a non-zero count means split/CSV misalignment, not a leaky split.
- CV folds are shuffled (L3 fixed, findings-2026-10-10; re-verified
  findings-2026-10-11): `04_train_and_evaluate.py` uses
  `KFold(n_splits=5, shuffle=True, random_state=42)`. Metrics committed
  before that fix were tuned under unshuffled folds - scaffold-split train
  indices ordered by scaffold group size, time-split ones following the
  SMILES-sorted clean CSV - so note it whenever comparing historical
  hyperparameter choices.

### C. Experiment hygiene

- **Per-split hyperparameters.** `metrics_{tag}.json` carries
  `best_params_random`, `best_params_scaffold` and `best_params_time`.
  Any experiment that refits an RF on a non-random split must read the
  matching key, not `best_params_random`.
  `grep -n best_params_random experiments/ scripts/`
- **Seeds must actually vary something.** For every multi-seed table,
  check that per-seed rows differ. Identical rows mean the "std" is not a
  variance estimate (XGBoost without `subsample`/`colsample_*` is
  deterministic).
- **Partial runs must not clobber campaign files.** Run any summary script
  with a subset of tags into a temp dir and diff against the committed
  summary; whole-file rewrites lose cells.
- **Seed counts.** Any `n_seeds` column must count runs per split, not files.

### D. Statistics conventions

The repo uses two Δ conventions (see `docs/methodology.md`, "Panel Δ
conventions"): per-seed-mean R² and R² of the 3-seed ensemble prediction.
Check that every column, label and table header names the convention it
actually holds, that no table invites subtracting columns of one
convention to get a Δ of the other, that the win-fraction column is named
`frac_boot_gin_better` (legacy files still say `p_one_sided` and are
renamed on read; the old `2*min(p, 1-p)` statistic was wrong and was
retired - L1 fixed, amendment in findings-2026-10-10) and is read as the
bootstrap fraction of replicates with ΔR² > 0, not as a p-value for a GIN
win, and that std ddof is consistent between JSON summaries and figures.

### E. README vs results

Recompute every README number from `results/` programmatically (pandas /
json), at the README's rounding. Panel, significance, time split,
multitask, fairness and uncertainty tables were all consistent on
2026-10-10 except the items listed in the findings file. As a first pass
run `scripts/verify_readme_tables.py` (section 2): it covers the markdown
tables and exits non-zero on any mismatch; the prose sections
(uncertainty-screening bullets, GINE panel verdicts, AttentiveFP notes)
are not machine-checked and need manual recomputation.

## 2. Scripts in this skill

| Script | What it does | Network |
|---|---|---|
| `scripts/qc_salts_censoring.py --tag TAG [--out f.json]` | audits the committed `parent_smiles` (multi-fragment / duplicated parents must be 0, split overlap must be 0, recomputation from provenance with 02's own standardization) and **exits 1 when any must-be-0 invariant is violated**, provenance salt context, raw-CSV censoring counts with 02's own inclusion rule (`standard_relation != "="`); skips the parent-level checks with a note on clean CSVs that predate them (bace) | no |
| `scripts/chembl_censoring_counts.py [--targets tag=CHEMBLID,...] [--overlap TAG] [--skip-counts]` | censored-record counts per target and assay type (live ChEMBL); with `--overlap`, an offline raw-CSV audit: every modelled parent keeps ≥ 1 `=` record and `n_measurements` matches, mixed exact+censored parents are counted, and raw parents dropped by the H1 filter are reported | counts: ChEMBL API; `--overlap --skip-counts`: no |
| `scripts/verify_readme_tables.py [--out report.json] [--dry-run]` | independently recomputes README table cells (panel, time split, XGBoost, multitask, fp ablation, EGFR/BACE detail, GIN-vs-RF, fairness incl. GINE rows, dataset sizes) from `results/`; **exits non-zero on any mismatch** (`--dry-run` reports without failing) | no |
| `scripts/rf_h1_sensitivity.py [--out D] [--dry-run] [--baseline-ref REF]` | the historical (pre-refresh) H1 sensitivity RF re-run; inputs pinned in `.scratch/h1_sensitivity/`, primary baseline = published pre-refresh metrics as constants, refreshed-state baseline pinned to tag `data-refresh-2026-10-11` (never HEAD); refuses `results/`/`figures/`/`models/`/`data/` outputs | no (local `git show` for the ref) |
| `scripts/reconcile_h1.py [--old-raw F] [--clean F] [--out D] [--dry-run]` | record-level historical-vs-fresh pull reconciliation for egfr_full; the pre-refresh inputs were clobbered by the refresh, so both pull defaults are the refreshed pull and a full run **refuses an old==new degenerate comparison** - run `--dry-run` first, it says what to pass; handles `standard_relation` suffixes that used to KeyError | no |
| `scripts/regen_fp_ablation_figs.py [--out D] [--dry-run] [--force]` | regenerates missing (or, with `--force`, stale) `figures/fp_ablation/*.png` through `experiments/fingerprint_ablation.py:make_figure`; import-safe (writes only under `__main__`) | no |

All scripts find the repo root on their own. Every script that writes takes
its destination from `--out` and refuses `results/`, `data/`, `models/`
(and `figures/`, except `regen_fp_ablation_figs.py` whose output *is*
`figures/fp_ablation/`) via a realpath-normalized, case-insensitive path
check - `--out Results/x` is refused too on macOS's case-insensitive
filesystem; `chembl_censoring_counts.py` writes nothing (console only);
`--dry-run` prints the plan without writing.

## 3. Report format

Rank findings by impact on the headline conclusions. For each one give:
severity, `path:line`, a one-sentence claim, a concrete failure scenario
(input or state, then the wrong output), the evidence (a command and its
number), and the smallest fix. Mark latent issues as latent. Close with
what is notably done well, so the author knows what not to change. Save a
new review as `references/findings-YYYY-MM-DD.md` and mark items of
earlier reviews as fixed or still open.
