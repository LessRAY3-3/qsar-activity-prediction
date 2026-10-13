#!/bin/bash
# HOST-SPECIFIC EXAMPLE - not part of the reproducible pipeline.
# Kept as a record of how this campaign was launched on the original
# author's machine (pre-2026-10-11 refresh; PY/paths were host state).
# Do not run as-is to reproduce the published numbers; use the commands
# documented in README/docs instead.
# P8 Phase 2c (MPS arm): GIN time-split, 8 targets x 3 seeds. Runs AFTER the
# GINE driver to keep MPS serial. Per-seed suffixes (artifact-naming lesson).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  tag=$1; seed=$2; sfx=$3
  echo "=== GIN-TIME $tag seed$seed start $(date '+%F %T') ==="
  start=$(date +%s)
  QSAR_TAG=$tag $PY scripts/gnn_03_train_gin.py --split time --seed $seed --suffix "$sfx"
  rc=$?
  echo "=== END GIN-TIME $tag seed$seed rc=$rc duration=$(( $(date +%s) - start ))s $(date '+%F %T') ==="
}

for tag in a2a mpro hivpr abl1 egfr vegfr2 herg egfr_full; do
  run $tag 42 ""
  run $tag 1 "_seed1"
  run $tag 2 "_seed2"
done
echo "=== P8 GIN-TIME ALL DONE $(date '+%F %T') ==="
