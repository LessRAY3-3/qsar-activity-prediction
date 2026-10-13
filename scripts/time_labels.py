"""Versioned as-of labels shared by the temporal RF and GNN consumers."""
import hashlib
import json
from pathlib import Path

import numpy as np

LABEL_POLICY = "exact-dated-asof-v1"


def dataset_signature(smiles, y):
    h = hashlib.sha256(json.dumps(np.asarray(smiles, str).tolist()).encode())
    h.update(np.asarray(y, dtype="<f4").tobytes())
    return h.hexdigest()


def label_signature(split):
    h = hashlib.sha256()
    for key in ("label_policy", "dataset_signature", "source_sha256",
                "cutoff_valid", "cutoff_test"):
        h.update(str(np.asarray(split[key]).item()).encode())
        h.update(b"\0")
    for key in ("train_idx", "valid_idx", "test_idx", "excluded_idx"):
        h.update(np.asarray(split[key], dtype="<i8").tobytes())
        h.update(b"\0")
    h.update(np.asarray(split["y"], dtype="<f4").tobytes())
    h.update(json.dumps(np.asarray(split["smiles"], str).tolist()).encode())
    h.update(np.asarray(split["year"], dtype="<f4").tobytes())
    return h.hexdigest()


def load_time_split(split_dir, tag, smiles=None, original_y=None):
    """NA takes precedence; legacy or mismatched labels fail before training."""
    root = Path(split_dir)
    marker = root / f"{tag}_time_NA.json"
    if marker.exists():
        print(f"[{tag}/time] skipped: {json.loads(marker.read_text()).get('reason')}")
        return None
    path = root / f"{tag}_time.npz"
    if not path.exists():
        raise SystemExit(f"missing time split: {path}")
    with np.load(path) as archive:
        sp = {key: archive[key] for key in archive.files}
    if str(sp.get("label_policy", "")) != LABEL_POLICY:
        raise ValueError("legacy time split has all-year labels; run "
                         f"scripts/gnn_08_time_splits.py --tag {tag} and retrain")
    required = {"smiles", "year", "y", "original_y", "train_idx", "valid_idx",
                "test_idx", "excluded_idx", "dataset_signature", "source_sha256",
                "cutoff_valid", "cutoff_test", "label_id"}
    if not required.issubset(sp):
        raise ValueError(f"incomplete time split: missing {sorted(required - set(sp))}")
    for key in ("y", "smiles", "year", "original_y"):
        if sp[key].ndim != 1 or len(sp[key]) != len(sp["y"]):
            raise ValueError(f"time {key} must be a one-dimensional array matching label count")
    if not np.isfinite(sp["original_y"]).all():
        raise ValueError("time original labels must be finite")
    if dataset_signature(sp["smiles"], sp["original_y"]) != str(sp["dataset_signature"]):
        raise ValueError("time split source dataset signature mismatch")
    if str(sp.get("label_id", "")) != label_signature(sp):
        raise ValueError("time label signature mismatch; regenerate the split")
    y = sp["y"]
    parts = [sp[k] for k in ("train_idx", "valid_idx", "test_idx", "excluded_idx")]
    if any(idx.ndim != 1 for idx in parts):
        raise ValueError("time indices must be one-dimensional")
    indices = np.concatenate(parts)
    if (indices.dtype.kind not in "iu" or
            not np.array_equal(np.sort(indices), np.arange(len(y)))):
        raise ValueError("time indices must partition the data, including exclusions")
    for idx in parts[:3]:
        if len(idx) == 0 or not np.isfinite(y[idx]).all():
            raise ValueError("empty time pool or missing dated labels")
        if not np.isfinite(sp["year"][idx]).all():
            raise ValueError("time pool contains an undated molecule")
    if np.isfinite(y[parts[3]]).any():
        raise ValueError("excluded time molecules must have missing labels")
    if smiles is not None:
        if not np.array_equal(np.asarray(smiles, str), sp["smiles"]):
            raise ValueError("time split SMILES do not match the input row order")
        if original_y is not None and dataset_signature(smiles, original_y) != str(sp["dataset_signature"]):
            raise ValueError("time split data version differs from the input cache")
    return sp


def require_prediction_version(predictions, split):
    """Prevent a partly rerun campaign from mixing legacy and as-of results."""
    expected = str(split.get("label_id", ""))
    if str(predictions.get("label_id", "")) != expected:
        raise ValueError("time prediction label version differs from split; retrain all time models")
    if not expected:
        raise ValueError("time prediction validation requires a versioned split")
    if not np.array_equal(predictions.get("test_idx"), split["test_idx"]):
        raise ValueError("time prediction test indices differ from split")
    actual = np.asarray(predictions.get("y_true"))
    target = np.asarray(split["y"])[split["test_idx"]]
    if actual.shape != target.shape or not np.allclose(actual, target, rtol=0, atol=1e-6):
        raise ValueError("time prediction labels differ from split")
