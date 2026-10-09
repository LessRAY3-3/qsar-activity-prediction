"""One-off backfill: add best_params_{split} to results/metrics_{tag}.json.

For every models/rf_{split}_split_{tag}.joblib the fitted estimator's
get_params() is written under best_params_{split} in that tag's metrics
file. A key that already exists (every historical best_params_random) is
skipped, never overwritten; random/scaffold/time r2/rmse/mae must come
through untouched.

Safety: each file is snapshotted first, re-serialized with the same
json.dump(indent=2) shape (no trailing newline), checked against the
snapshot BEFORE the write, written, re-read from disk and compared again;
any r2 drift restores the backup and aborts the run.

Usage:  .venv/bin/python scripts/backfill_best_params.py
"""
import glob
import json
import os
import shutil
import tempfile

import joblib

BASE = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE, "results")
MODEL_DIR = os.path.join(BASE, "models")
SPLITS = ("random", "scaffold", "time")


def model_paths(tag):
    """{split: path} for the tag's fitted RF models (missing ones omitted)."""
    out = {}
    for split in SPLITS:
        path = os.path.join(MODEL_DIR, f"rf_{split}_split_{tag}.joblib")
        if os.path.exists(path):
            out[split] = path
    return out


def r2_snapshot(payload):
    """{split: r2} for every split block present (compared exactly)."""
    return {s: payload[s]["r2"] for s in SPLITS
            if isinstance(payload.get(s), dict) and "r2" in payload[s]}


def backfill_file(path, models):
    """Add the missing best_params_{split} keys of one metrics file.

    -> list of added keys; raises SystemExit (restoring the backup) if any
    recorded r2 differs from the pre-write snapshot.
    """
    tag = os.path.basename(path)[len("metrics_"):-len(".json")]
    with open(path, "rb") as fh:
        original = fh.read()
    payload = json.loads(original)
    before = r2_snapshot(payload)

    added, skipped = [], []
    for split in SPLITS:
        key = f"best_params_{split}"
        if key in payload:
            skipped.append(key)          # historical value wins: do not touch
            continue
        if split not in models:
            if split in payload:
                raise SystemExit(
                    f"{path}: has a '{split}' block but no "
                    f"models/rf_{split}_split_{tag}.joblib to backfill from")
            continue
        payload[key] = joblib.load(models[split]).get_params()
        added.append((key, os.path.basename(models[split])))

    if not added:
        note = f" (already present: {', '.join(skipped)})" if skipped else ""
        print(f"metrics_{tag}.json: nothing to add{note}")
        return []

    new_bytes = json.dumps(payload, indent=2).encode()   # json.dump(indent=2) shape
    if r2_snapshot(json.loads(new_bytes)) != before:
        raise SystemExit(f"ABORT {path}: r2 drifted before write; file untouched")

    backup_dir = tempfile.mkdtemp(prefix="metrics_backfill_")
    backup_path = os.path.join(backup_dir, os.path.basename(path))
    with open(backup_path, "wb") as fh:
        fh.write(original)
    with open(path, "wb") as fh:
        fh.write(new_bytes)

    if r2_snapshot(json.load(open(path))) != before:
        shutil.copyfile(backup_path, path)
        raise SystemExit(
            f"ABORT {path}: r2 drifted after write; restored from {backup_path}")

    keys = [k for k, _ in added]
    print(f"metrics_{tag}.json: + {', '.join(keys)}"
          + (f"  (skipped existing: {', '.join(skipped)})" if skipped else ""))
    for key, model in added:
        print(f"    {key} <- {model}")
    return keys


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "metrics_*.json")))
    if not paths:
        raise SystemExit(f"no metrics_*.json under {RESULTS_DIR}")
    total = {}
    for path in paths:
        tag = os.path.basename(path)[len("metrics_"):-len(".json")]
        for key in backfill_file(path, model_paths(tag)):
            total[f"{tag}/{key}"] = True
    print(f"\ndone: added {len(total)} key(s): {', '.join(sorted(total))}")


if __name__ == "__main__":
    main()
