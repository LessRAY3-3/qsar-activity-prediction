#!/bin/bash
# P8 Phase 1: GINE frozen-recipe panel - 5 remaining targets x 2 splits x 3 seeds.
# vegfr2/herg/abl1 GINE results already exist from experiments/gnn_fairness.py.
# MPS is serial: one run at a time. Per-seed suffixes everywhere (LOG artifact-naming lesson).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  tag=$1; split=$2; seed=$3; sfx=$4
  echo "=== GINE $tag $split seed$seed start $(date '+%F %T') ==="
  start=$(date +%s)
  QSAR_TAG=$tag $PY scripts/gnn_03_train_gin.py --split $split --model gine --seed $seed --suffix "$sfx"
  rc=$?
  echo "=== END GINE $tag $split seed$seed rc=$rc duration=$(( $(date +%s) - start ))s $(date '+%F %T') ==="
}

for tag in a2a hivpr mpro egfr egfr_full; do
  for split in random scaffold; do
    run $tag $split 42 ""
    run $tag $split 1 "_seed1"
    run $tag $split 2 "_seed2"
  done
done
echo "=== P8 GINE ALL DONE $(date '+%F %T') ==="
