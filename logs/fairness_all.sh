#!/bin/bash
# HOST-SPECIFIC EXAMPLE - not part of the reproducible pipeline.
# Kept as a record of how this campaign was launched on the original
# author's machine (hard-coded interpreter paths, written before the
# 2026-10-11 data refresh). Do not run as-is to reproduce the published
# numbers; use the commands documented in README/docs instead.
# GIN fairness experiment B: 3 targets serially (nohup; one MPS user at a time).
#   nohup bash logs/fairness_all.sh > logs/fairness_all.log 2>&1 &
cd ~/qsar-activity-prediction || exit 1
for tag in vegfr2 herg abl1; do
  echo "[fairness_all] starting ${tag} $(date)"
  bash logs/fairness_run_tag.sh "$tag"
  rc=$?
  echo "[fairness_all] ${tag} exit=${rc} $(date)"
  if [ $rc -ne 0 ]; then
    echo "[fairness_all] aborting remaining targets"
    exit $rc
  fi
done
echo "[fairness_all] all targets done $(date)"
