#!/bin/bash
# HOST-SPECIFIC EXAMPLE - not part of the reproducible pipeline.
# Kept as a record of how this campaign was launched on the original
# author's machine (pre-2026-10-11 refresh; the "model exists" skip below
# reflects the artifact state of that run). Do not run as-is to reproduce
# the published numbers; use the commands documented in README/docs instead.
# P8 Phase 2c (CPU arm): RF time-split retraining, 8 targets, serial (RAM-safe).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for tag in a2a mpro hivpr abl1 egfr vegfr2 herg egfr_full; do
  if [ -f "models/rf_time_split_${tag}.joblib" ]; then
    echo "=== RF-TIME $tag SKIP (model exists) ==="
    continue
  fi
  echo "=== RF-TIME $tag start $(date '+%F %T') ==="
  start=$(date +%s)
  QSAR_TAG=$tag $PY scripts/04_train_and_evaluate.py --split time
  rc=$?
  echo "=== END RF-TIME $tag rc=$rc duration=$(( $(date +%s) - start ))s $(date '+%F %T') ==="
done
echo "=== P8 RF-TIME ALL DONE $(date '+%F %T') ==="
