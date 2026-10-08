#!/bin/bash
# P8 Phase 3: AttentiveFP panel - 8 targets x 2 splits x 3 seeds (MPS serial).
# PyG AttentiveFP (gated attention + GRU over 30 timesteps) ~3x GIN per epoch.
# Small targets first so early failures surface fast. Per-seed suffixes.
set -u
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/qsar/bin/python

run() {
  tag=$1; split=$2; seed=$3; sfx=$4
  echo "=== AFP $tag $split seed$seed start $(date '+%F %T') ==="
  start=$(date +%s)
  QSAR_TAG=$tag $PY scripts/gnn_03_train_gin.py --split $split --model attentivefp --seed $seed --suffix "$sfx"
  rc=$?
  echo "=== END AFP $tag $split seed$seed rc=$rc duration=$(( $(date +%s) - start ))s $(date '+%F %T') ==="
}

for tag in a2a hivpr mpro abl1 egfr vegfr2 herg egfr_full; do
  for split in random scaffold; do
    run $tag $split 42 ""
    run $tag $split 1 "_seed1"
    run $tag $split 2 "_seed2"
  done
done
echo "=== P8 AFP ALL DONE $(date '+%F %T') ==="

echo "=== commit + push AFP results ==="
git add results/gnn_metrics_*_attentivefp*.json \
        results/gnn_preds_*_attentivefp*.npz \
        results/gnn_models/*attentivefp*.pt \
        logs/p8_afp_all.log 2>/dev/null
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git -c user.name="$(git config user.name || echo leisirui)" \
      -c user.email="$(git config user.email || echo leisirui@users.noreply.github.com)" \
      commit -m "results: P8 AttentiveFP panel (8 targets x 2 splits x 3 seeds)" && \
  git push origin main 2>&1 | tail -2
fi
echo "=== P8 AFP WRAP-UP DONE $(date '+%F %T') ==="
