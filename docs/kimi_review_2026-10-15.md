# Kimi follow-up review: 2026-10-15

Baseline: main `75953818f863e4b2f4e7d2fd609755234ce2e2ef` (merged PR #7).
This patch changes validation, safe output handling and documentation. It does
not retrain published experiments or replace accepted data, models, predictions,
figures, or scientific result tables.

## Finding disposition

| Finding | Change and verification |
| --- | --- |
| M1 | Default RF training now merges random/scaffold metrics, preserving other keys as JSON values. Before training, an existing time block must match the current validated as-of split and data identity. JSON writes use a temporary file and atomic replacement. Regression cases cover preservation and rejection before fitting. |
| M2 | Error analysis validates RF/GIN test indices against the split in order, finite vector shapes, labels and fingerprint molecule order. Interpretability also validates graph order and the error CSV's molecule/scaffold/label/error values before selecting molecules. Two mutually aligned but stale prediction files are rejected. |
| M3 | Methodology distinguishes the historical 447-cell check from the final 467-cell check (20 environment-table entries added). Historical audit JSON remains unchanged. The same approximately 0.00344 RF gap is consistently shown as approximately 0.003. |
| L1 | Seed pooling requires integer `params.seed`, checks it against the conventional filename, rejects duplicate seeds and nonfinite values, and counts contributors per split. Missing identity is an error; no filename-only inference. |
| L2 | All three auxiliary RF analyses require the requested split's hyperparameters before opening result files. Previous behavior printed a fallback notice but still accepted the substituted configuration; that fallback is removed. Existing saved experiment summaries already use the correct split parameters. |
| L3 | Both ChEMBL entry points stage fresh downloads and replace the prior CSV only on success. Failed requests, premature empty pages, changed totals or inconsistent pagination abort. The explicit row cap remains intentional sampling. The legacy EGFR entry delegates to the generic downloader; neither claims resume support. |
| L4 | Newly trained RF estimators store a digest binding features, labels, molecule order and train/test indices. Plot-only validates it and replays R2/RMSE/MAE against saved metrics. Legacy models have no retrospectively recoverable digest: their metric replay is checked and the missing original identity is explicitly reported. Metric agreement alone is not proof of identity. |
| L5 | PR #9 made GNN/ESOL partition and prediction guards explicit under `python -O`, but missed the graph-builder SMILES-order assertion. It pinned ESOL to a constant matching the manifest, rather than reading the manifest. The 2026-10-16 closeout fixes the remaining graph guard and reads one valid ESOL digest directly from `data/raw/sha256sums.txt` for both caches and downloads. Atomic installation and the corrected 72/8/20 split docstring remain in place. |
| L6 | Temporal summaries reject random/scaffold comparison values that disagree with current RF metrics and independently pooled, identity-checked GIN metrics. This is a value-consistency gate for the consumed fields, not publisher authentication or proof of original training identity. |
| L7 | A fresh data-scope audit is committed in `results/time_split/data_audit_20261015.json`: 76 artifact hashes verified, 48 distinct model/cache paths explicitly skipped; labels, predictions and bootstrap checks also pass. This replaces an unsupported present-tense claim with a reproducible, committed record, without inventing the missing historical audit file. |
| L8 | README command comment is English; methodology and Release no longer describe PR #7 as pending. GIN rejects empty, nonfinite or zero-variance training labels before normalization. |

New helper dependencies are included and hashed in isolated campaign jobs.
Bundle validation requires the helper for trainers that import it; historical
frozen trainers without that dependency remain supported. The generated notebook
is synchronized with the updated scripts.

## Verification

- `pytest -q`: **754 passed**, 3 existing warnings; 30 additional regression cases.
- `ruff check scripts experiments tests .claude/skills/qsar-review/scripts`: passed.
- README verifier: **467 cells, 0 mismatches**.
- `scripts/audit_analysis_inputs.py`: all **16** tag/split cells pass prediction,
  label, index, CSV error-table, seed identity and comparison-value checks.
- `scripts/audit_time_campaign.py --scope data`: passed; 42,636 retained labels,
  168 exclusions, 32 prediction files and 16 bootstrap rows.
- No baseline tracked file under `data/`, `models/`, `results/` or `figures/`
  was changed. New audit reports are additional evidence.

Commands and readable outputs are in `logs/kimi_review_20261015/`; the read-only
analysis-input audit is saved as `results/review_kimi_20261015/analysis_inputs.json`.
No live ChEMBL crawl or scientific campaign was launched for this patch. Network
failure cases use controlled responses. The public ESOL mirror bytes were
independently downloaded and matched the already tracked SHA256.

## Limits of this acceptance

Passing tests and cache audits do not establish absence of all leakage or model
errors. Historical missing training identity is not reconstructed by a checksum
of today's files. Existing conditional-bootstrap, assay heterogeneity,
cross-environment confounding, and missing historical source/log limitations
remain as documented in the research report. No current evidence requires
rerunning the accepted scientific experiments for these guard changes.
