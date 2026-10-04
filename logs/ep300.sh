#!/bin/bash
cd /Users/leisirui/qsar-activity-prediction || exit 1
PY=/Users/leisirui/miniforge3/envs/qsar/bin/python
LOG=/Users/leisirui/qsar-activity-prediction/logs/ep300_confirm.log

run_one() {
  local split=$1 seedflag=$2 label=$3
  echo "=== START $(date '+%F %T') split=$split $label ===" >> "$LOG"
  local t0=$(date +%s)
  if [ -n "$seedflag" ]; then
    QSAR_TAG=egfr_full "$PY" scripts/gnn_03_train_gin.py --split "$split" $seedflag --epochs 300 --suffix _ep300 >> "$LOG" 2>&1
  else
    QSAR_TAG=egfr_full "$PY" scripts/gnn_03_train_gin.py --split "$split" --epochs 300 --suffix _ep300 >> "$LOG" 2>&1
  fi
  local rc=$?
  local t1=$(date +%s)
  echo "=== END $(date '+%F %T') split=$split $label rc=$rc duration=$((t1-t0))s ===" >> "$LOG"
}

chain_random() {
  run_one random "" "seed42"
  run_one random "--seed 1" "seed1"
  run_one random "--seed 2" "seed2"
}

chain_scaffold() {
  run_one scaffold "" "seed42"
  run_one scaffold "--seed 1" "seed1"
  run_one scaffold "--seed 2" "seed2"
}

echo "=== EP300 CONFIRM JOB LAUNCH $(date '+%F %T') pid=$$ ===" >> "$LOG"
chain_random &
PID_R=$!
chain_scaffold &
PID_S=$!
wait $PID_R
wait $PID_S
echo "=== EP300 CONFIRM JOB ALL DONE $(date '+%F %T') ===" >> "$LOG"
