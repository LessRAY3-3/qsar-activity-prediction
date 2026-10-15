"""Fail-closed identity checks for analysis and newly fitted RF artifacts.

Legacy model metric replay is a consistency check, not proof of identity.
No accepted prediction or model is rewritten by these helpers.
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def rf_identity(X, y, smiles, train_idx, test_idx):
    h = hashlib.sha256()
    for a in (
        np.asarray(X),
        np.asarray(y),
        np.asarray(smiles, str),
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    ):
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def check_partition(n_total, **named):
    arrays = [np.asarray(v) for v in named.values()]
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("split indices must be one-dimensional integers")
    flat = np.concatenate(arrays)
    if len(flat) != n_total or not np.array_equal(np.sort(flat), np.arange(n_total)):
        raise ValueError("splits overlap, contain invalid indices, or leave molecules out")


def check_prediction(pred, split, y=None):
    idx = np.asarray(split["test_idx"])
    n = len(split["smiles"])
    if (
        idx.ndim != 1
        or idx.dtype.kind not in "iu"
        or len(np.unique(idx)) != len(idx)
        or np.any(idx < 0)
        or np.any(idx >= n)
    ):
        raise ValueError("invalid test indices")
    if not np.array_equal(pred["test_idx"], idx):
        raise ValueError("prediction test_idx does not match split (including order)")
    for key in ("y_true", "y_pred"):
        a = np.asarray(pred[key])
        if a.shape != idx.shape or not np.isfinite(a).all():
            raise ValueError(f"prediction {key} shape or finite-value mismatch")
    if y is not None and not np.allclose(pred["y_true"], np.asarray(y)[idx], rtol=0, atol=1e-5):
        raise ValueError("prediction labels do not match current data")


def check_error_table(df, split, rf, gin, y):
    check_prediction(rf, split, y)
    check_prediction(gin, split, y)
    idx = split["test_idx"]
    for column, key in (("smiles", "smiles"), ("scaffold", "scaffold_smiles")):
        # Empty Murcko scaffolds are valid for acyclic molecules.
        if df[column].fillna("").astype(str).tolist() != np.asarray(split[key])[idx].tolist():
            raise ValueError(f"error analysis {column} does not match split order")
    for key, expected in (
        ("y_true", rf["y_true"]),
        ("err_rf", np.abs(rf["y_pred"] - rf["y_true"])),
        ("err_gin", np.abs(gin["y_pred"] - rf["y_true"])),
    ):
        values = df[key].to_numpy(float)
        if values.shape != np.shape(expected) or not np.allclose(
            values, expected, rtol=0, atol=1e-5
        ):
            raise ValueError(f"error analysis {key} is stale")


def collect_seed_metrics(results, tag):
    per_split = {s: {"r2": [], "rmse": []} for s in ("random", "scaffold")}
    seen = {s: set() for s in per_split}
    pat = re.compile(rf"^gnn_metrics_{re.escape(tag)}(_seed(\d+))?\.json$")
    files = sorted(p for p in Path(results).iterdir() if pat.fullmatch(p.name))
    for path in files:
        match = pat.fullmatch(path.name)
        expected = int(match[2]) if match[2] else 42
        data = json.loads(path.read_text())
        for s in per_split:
            if s not in data:
                continue
            seed = data[s].get("params", {}).get("seed")
            if type(seed) is not int or seed != expected:
                raise ValueError(f"{path.name}/{s}: missing or conflicting seed identity")
            if seed in seen[s]:
                raise ValueError(f"{tag}/{s}: duplicate seed {seed}")
            seen[s].add(seed)
            for metric in ("r2", "rmse"):
                value = float(data[s][metric])
                if not np.isfinite(value):
                    raise ValueError(f"{path.name}/{s}: nonfinite {metric}")
                per_split[s][metric].append(value)
    if any(not v["r2"] for v in per_split.values()):
        raise ValueError(f"{tag}: missing random/scaffold seed metrics")
    return per_split, len(files)


def require_rf_params(metrics, split):
    key = f"best_params_{split}"
    if key not in metrics:
        raise ValueError(f"{key} missing; run the corresponding RF search first")
    return {k: metrics[key][k] for k in ("n_estimators", "max_depth", "min_samples_split")}


def training_scale(y, train_idx):
    values = np.asarray(y)[train_idx]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("training labels must be nonempty and finite")
    mean, std = float(values.mean()), float(values.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("training label standard deviation must be positive")
    return mean, std
