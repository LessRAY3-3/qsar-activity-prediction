#!/bin/bash
# Fairness experiment runner: one target, full protocol (tune+final+gine+summary).
# Serialised with the other targets by logs/fairness_all.sh to avoid MPS contention.
cd ~/qsar-activity-prediction || exit 1
PY=/Users/leisirui/miniforge3/envs/qsar/bin/python
TAG=$1
mkdir -p logs
echo "=== fairness_${TAG} start $(date) ===" >> "logs/fairness_${TAG}.log"
$PY experiments/gnn_fairness.py --tag "$TAG" >> "logs/fairness_${TAG}.log" 2>&1
rc=$?
echo "=== fairness_${TAG} exit=$rc $(date) ===" >> "logs/fairness_${TAG}.log"
exit $rc
