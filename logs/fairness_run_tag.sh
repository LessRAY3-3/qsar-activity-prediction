#!/bin/bash
# HOST-SPECIFIC EXAMPLE - not part of the reproducible pipeline.
# Kept as a record of how this campaign was launched on the original
# author's machine (hard-coded interpreter paths, written before the
# 2026-10-11 data refresh). Do not run as-is to reproduce the published
# numbers; use the commands documented in README/docs instead.
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
