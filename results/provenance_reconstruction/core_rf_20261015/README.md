# Core RF fixed-parameter confirmation

Sixteen new fits: 8 datasets × random/scaffold, seed 42, published selected
parameters. Trained on M3 Max with eight RF threads; independently replayed on
the M4 coordinator with one inference thread. No hyperparameter search was
rerun and no published prediction/model was replaced. These are new runs, not
recovered historical identities. See `training_summary.json`, `receipts/`,
`input_manifest.json`, `environment.json`, `predictions/` and `acceptance.json`.

The full archive includes frozen inputs and trainer/helper sources, all 16
models, receipts, environment, predictions and training/launch logs:

- [Download complete RF archive](https://github.com/LessRAY3-3/qsar-activity-prediction/releases/download/source-provenance-20261015/rf-core-confirmation-20261015.tar.gz)
- Size: **554,406,533 bytes**, 104 regular files.
- SHA256: `3a1cd3161ff988d3d91b74ad50d47076751a05035dfcf1b5c3b1821cc40ab5a9`.
- The committed `archive.manifest.json` binds every file's size and SHA256.

After downloading into a local `downloads/` directory:

```bash
python scripts/unpack_verified_return.py \
  --archive downloads/rf-core-confirmation-20261015.tar.gz \
  --manifest results/provenance_reconstruction/core_rf_20261015/archive.manifest.json \
  --out-dir .scratch/rf-confirmation-replay
python experiments/verify_rf_core_confirmation.py \
  --package .scratch/rf-confirmation-replay \
  --report .scratch/rf-confirmation-replay-report.json
```

The verifier checks the entire input cohort, training membership/order,
receipt/environment/parameter identities, model hashes, saved model identity,
prediction alignment and metrics. The archive contains joblib serialization;
only load the trusted project artifact after verification. Hashes establish
integrity relative to the manifest, not third-party authentication of the run.

For another independent fit using the frozen recipe and inputs:

```bash
python .scratch/rf-confirmation-replay/experiments/rf_core_confirmation.py \
  --out-dir .scratch/rf-confirmation-new --jobs 8
```

Output directories must be new. This intentionally preserves every previous
attempt. Dependencies are recorded in `environment.json` (Python 3.12.14,
scikit-learn 1.9.1, NumPy 2.5.3; RDKit is required to reconstruct scaffold
membership). RF parameter selection is inherited from the published metrics;
the new experiment does not measure uncertainty from the original grid search.
An explicitly post-run [M3 package inventory](../../../logs/source_provenance_20261015/m3_postrun_environment.json)
records RDKit 2026.3.6 and the 12-performance/4-efficiency-core topology; it is
not mislabeled as a contemporaneous training snapshot.
