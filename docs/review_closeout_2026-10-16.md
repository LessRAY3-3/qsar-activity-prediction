# PR #9 / #10 review closeout — 2026-10-16

Baseline: main `09dbff10a3b87d304e897d122988999934464885` (merged PR #10).
This patch closes latent validation gaps and corrects the scope of replay claims.
No training was launched. Existing data, models, predictions and scientific
results are unchanged; new verification reports are additional evidence.

## Finding disposition

| Finding | Failure scenario / disposition | Evidence |
| --- | --- | --- |
| L5 graph guard (Low) | `scripts/gnn_02_build_graphs.py:35` used an assertion: optimized Python could accept a reordered split and write misaligned scaffold labels. It now raises `ValueError` before featurization or output. | Real subprocess tests in normal and `-O` modes reject reversed SMILES and leave no graph output. |
| ESOL manifest (Low) | A copied digest would not follow a reviewed manifest update. `verify_bytes` now reads exactly one valid `esol_delaney.csv` entry from the tracked raw-data manifest on each verification. Missing, duplicate or malformed entries fail. | Tests exercise manifest updates and malformed/duplicate/missing entries; pre-existing corrupt-cache/download tests still pass. |
| GIN replay claim (Medium wording) | The auditor calls the producer's construction, graph loader, prediction and metric implementations. README and method notes now explicitly call this model-to-prediction consistency replay, not independent implementation validation. New audit reports record `implementation_independent: false`. | The import and calls remain visible in `experiments/audit_core_gin_artifacts.py`; the historical 72-checkpoint acceptance is unchanged. |
| Historical RF comparison (Low) | A new fit could replay its own saved predictions but differ from the historical baseline and still receive acceptance. The verifier now rejects nonfinite/malformed predictions and historical maximum absolute deviation above `1e-10` (relative tolerance zero). Existing reports cannot be overwritten. | Negative drift/shape/nonfinite tests; fresh replay of all 16 archived models passes, maximum historical difference **0.0**. |
| GIN tolerance (clarified, retained) | The observed `1.9073486328125e-6` is not the acceptance threshold. The documented gate is `abs(error) <= 1e-4 + 1e-5 * abs(stored)`, with separate metric/label checks. It is not tightened post hoc to this observed cohort. | README's quantitative error remains an observed result; method notes now state the full combined tolerance. No model/inference change or new GIN training. |
| Tiny assay strata (Low) | `n=6` can produce mathematically defined but unstable R² values. Assay strata now retain counts and suppress all performance metrics for `n<20`, marking `too_few_molecules`. `n>=20` is still descriptive, not an inferential validity guarantee. | Boundary tests at n=0/6/19/20; new 50-row table suppresses 15 tiny strata. Every n>=20 value matches the previous table. |
| Temporal wording (Low) | Four GIN-favoring molecule-level temporal intervals do not all disappear: EGFR-full/time retains a positive document-cluster interval. README states this explicitly, along with both models' negative full-test R². | The existing 24-cell report and temporal table support this qualification. Different full vs exclusive-source cohorts remain explicitly distinguished. |
| RF launcher UX (Low) | A clean checkout has no root `input_manifest.json`; the trainer previously created an output directory before a raw file error. It now explains the frozen-package requirement and validates inputs before creating output. | Regression test checks exit code 2, actionable error, and no output directory. Frozen original trainer sources are preserved in the release archive. |

The original PR #9 L5 disposition is corrected in place to distinguish what
PR #9 implemented from what this closeout adds. Passing checks do not prove
absence of all possible scientific errors or reconstruct old run identities.

## Verification

- Full suite: **788 passed**, three existing warnings (14 new regression cases).
- Ruff: passed across scripts, experiments, tests and the review scripts.
- README tables: **467 checked cells, zero mismatches**.
- New RF replay: **16/16 accepted**, maximum historical prediction difference
  **0.0**; the new `1e-10` historical gate is recorded in the acceptance JSON.
- Full source analysis rerun: **24 cell summaries exactly equal**, including
  paired molecule/document bootstrap intervals; the other **26 CSV outputs are
  byte-identical**. No extra fit or changed prediction was used.
- The baseline's **1,355 tracked files** under data/models/results/figures remain
  unchanged. New reports/table are stored under `results/review_closeout_20261016/`.

Evidence: [RF acceptance](../results/review_closeout_20261016/rf_acceptance.json),
[source comparison](../results/review_closeout_20261016/source_acceptance.json),
[guarded assay table](../results/review_closeout_20261016/assay_type_metrics.csv),
[verification logs](../logs/review_closeout_20261016/).
The original source table remains a frozen historical artifact; use the linked
guarded table for presentational or downstream summaries of assay strata.

## Reproduction

Use the pinned Python environment and restore the project artifacts/time inputs
as described in the main methodology before running the full test suite.

```bash
pytest -q
ruff check scripts experiments tests .claude/skills/qsar-review/scripts
python .claude/skills/qsar-review/scripts/verify_readme_tables.py --out .scratch/readme-closeout-new.json
python experiments/source_heterogeneity.py --out-dir .scratch/source-closeout-new
python experiments/verify_rf_core_confirmation.py \
  --package .scratch/rf-confirmation-replay \
  --report .scratch/rf-closeout-new.json
```

The RF `--package` directory must first be extracted and verified using the
[RF archive instructions](../results/provenance_reconstruction/core_rf_20261015/README.md).
The trainer `experiments/rf_core_confirmation.py` itself is a frozen-package
entry point, not a clean-checkout training command. Do not invent a root manifest
or silently reuse one for different source bytes. For the already published
recipe, run the trainer contained in the verified archive with a fresh output
directory; changing the recipe requires a newly frozen manifest and new run
identity. Existing release sources/manifests remain immutable.

## What remains sound

The original label reconstruction, paired document bootstrap, fixed-parameter
RF confirmation and 72-checkpoint consistency findings remain supported. The
small-stratum guard changes reporting, not model selection or the main effect
estimates. Source heterogeneity, shared-implementation replay limitations,
missing historical identities and lack of prospective wet-lab validation still
apply.
