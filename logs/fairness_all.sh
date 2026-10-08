#!/bin/bash
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
