#!/bin/bash
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
