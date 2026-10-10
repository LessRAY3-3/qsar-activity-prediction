---
name: qsar-review
description: Audit this QSAR repo (ChEMBL -> pIC50 -> Morgan-fingerprint RF vs GIN/GINE/AttentiveFP) for data-quality, split-leakage, experiment-hygiene, statistics and README-consistency problems, and write the result as a ranked, evidence-backed report. Use when asked to review, audit or sanity-check the pipeline, before a new data pull or campaign, after changing scripts/ or experiments/, or when README numbers are regenerated.
---

# QSAR pipeline review

This skill re-runs the review that was done on 2026-10-10 and keeps its
lessons. The findings of that review, with file:line anchors, evidence and
suggested fixes, are in `references/findings-2026-10-10.md`. Start every
review by reading that file and checking which items are still open.

## 0. Baseline (5 min)

```bash
conda env create -f environment.yml && conda activate qsar   # Python 3.12
ruff check scripts experiments tests
pytest -q
```

`requirements.txt` pins need Python >= 3.12 (numpy 2.5.x). On 3.11, install
unpinned: the suite (81 tests) and ruff were verified to pass that way.
Many checks below need only `data/processed/` and `results/`, which are
tracked; `models/` and `data/raw/` come from `python scripts/fetch_artifacts.py --all`.

## 1. Review checklist

Work layer by layer. Every claim in the report must be verified by
re-reading the code or by a number computed from the repo, never inferred.

### A. Data layer (highest leverage: every model inherits it)

- **Censored activities.** ChEMBL stores "IC50 > 10000 nM" with
  `standard_relation` `>`/`<`. The downloaders do not request that column,
  and `pchembl_value` is empty for every censored record, so the pChEMBL
  cross-check in `02_clean_data.py` silently covers uncensored rows only.
  Quantify with `scripts/chembl_censoring_counts.py` (section 2).
- **Salt forms and parents.** `canonical_smiles` keeps counter-ions; dedup
  is on the raw string. Count parent-level duplicates, label conflicts and
  train/test parent leakage with `scripts/qc_salts_censoring.py`.
- Unit filter (`nM` only), non-positive values, median aggregation and
  invalid-SMILES handling in `02`/`03`: confirm row counts against
  `data/README.md` and `data/raw/sha256sums.txt`.

### B. Splits

- `valid_idx` must be carved from train only; `test_idx` must be asserted
  equal across RF and every GNN seed before any paired statistic.
- Parent-level leakage (salt vs free base on both sides): random split is
  expected to leak a little, scaffold should not.
- `GridSearchCV(cv=5)` uses unshuffled `KFold`. Random-split train indices
  are shuffled, but scaffold-split ones are ordered by scaffold group size
  and time-split ones follow the SMILES-sorted clean CSV. Note it whenever
  hyperparameter choice matters.

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
convention to get a Δ of the other, that `p_one_sided` is read as
P(ΔR² > 0) and not as a p-value for a GIN win, and that std ddof is
consistent between JSON summaries and figures.

### E. README vs results

Recompute every README number from `results/` programmatically (pandas /
json), at the README's rounding. Panel, significance, time split,
multitask, fairness and uncertainty tables were all consistent on
2026-10-10 except the items listed in the findings file.

## 2. Scripts in this skill

| Script | What it does | Network |
|---|---|---|
| `scripts/qc_salts_censoring.py --tag TAG [--out f.json]` | multi-fragment SMILES, parent duplicates and their pIC50 spread, parent-level train/test leakage for every persisted split, censoring proxy from raw CSV | no |
| `scripts/chembl_censoring_counts.py [--targets tag=CHEMBLID,...] [--overlap TAG]` | censored-record counts per target and assay type; with `--overlap`, how many cleaned compounds carry only censored labels | ChEMBL API |

Both find the repo root on their own. Neither writes into `results/` or
`data/` unless `--out` points there.

## 3. Report format

Rank findings by impact on the headline conclusions. For each one give:
severity, `path:line`, a one-sentence claim, a concrete failure scenario
(input or state, then the wrong output), the evidence (a command and its
number), and the smallest fix. Mark latent issues as latent. Close with
what is notably done well, so the author knows what not to change. Save a
new review as `references/findings-YYYY-MM-DD.md` and mark items of
earlier reviews as fixed or still open.
