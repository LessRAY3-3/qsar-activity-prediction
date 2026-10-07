#!/bin/bash
cd /Users/leisirui/qsar-activity-prediction || exit 1
PY=/Users/leisirui/miniforge3/envs/qsar/bin/python
LOG=/Users/leisirui/qsar-activity-prediction/logs/multitask.log

chain() {
  local model=$1 split=$2
  for seed in 42 1 2; do
    echo "=== START $(date "+%F %T") model=$model split=$split seed=$seed ===" >> "$LOG"
    local t0=$(date +%s)
    $PY scripts/gnn_07_multitask.py --model "$model" --split "$split" --seed $seed >> "$LOG" 2>&1
    local rc=$?
    echo "=== END $(date "+%F %T") model=$model split=$split seed=$seed rc=$rc duration=$(( $(date +%s) - t0 ))s ===" >> "$LOG"
  done
}

echo "=== MULTITASK JOB LAUNCH $(date "+%F %T") pid=$$ ===" >> "$LOG"
chain mt random &
P1=$!
chain mt scaffold &
P2=$!
chain pooled random &
P3=$!
chain pooled scaffold &
P4=$!
wait $P1 $P2 $P3 $P4
echo "=== TRAINING DONE $(date "+%F %T"), aggregating ===" >> "$LOG"
$PY scripts/gnn_07_multitask.py --aggregate >> "$LOG" 2>&1
echo "=== MULTITASK JOB ALL DONE $(date "+%F %T") ===" >> "$LOG"
