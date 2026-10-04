#!/bin/bash
cd /Users/leisirui/qsar-activity-prediction || exit 1
PY=/Users/leisirui/miniforge3/envs/qsar/bin/python
LOG=/Users/leisirui/qsar-activity-prediction/logs/ep300_repair.log

run_one() {
  local split=$1 seedflag=$2 suffix=$3 label=$4
  echo "=== START $(date '+%F %T') split=$split $label suffix=$suffix ===" >> "$LOG"
  local t0=$(date +%s)
  QSAR_TAG=egfr_full "$PY" scripts/gnn_03_train_gin.py --split "$split" $seedflag --epochs 300 --suffix "$suffix" >> "$LOG" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "=== END $(date '+%F %T') split=$split $label rc=$rc duration=$((t1-t0))s ===" >> "$LOG"
}

chain_random() {
  run_one random "" "_ep300" "seed42"
  run_one random "--seed 1" "_seed1_ep300" "seed1"
}

chain_scaffold() {
  run_one scaffold "" "_ep300" "seed42"
  run_one scaffold "--seed 1" "_seed1_ep300" "seed1"
}

echo "=== EP300 REPAIR JOB LAUNCH $(date '+%F %T') pid=$$ ===" >> "$LOG"
chain_random &
PID_R=$!
chain_scaffold &
PID_S=$!
wait $PID_R
wait $PID_S
echo "=== EP300 REPAIR JOB ALL DONE $(date '+%F %T') ===" >> "$LOG"
