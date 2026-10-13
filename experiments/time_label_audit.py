"""Audit as-of labels without replacing any split, model, or prediction.

Run from the repository root: python experiments/time_label_audit.py.
Counts compare each retained pool's historical label to its all-year label;
they quantify changed inputs, not changes in predictive performance.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from time_labels import LABEL_POLICY, dataset_signature  # noqa: E402

time_splits = importlib.import_module("gnn_08_time_splits")
TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")


def audit_tag(tag):
    cache = BASE / "data" / "processed" / f"{tag}_fingerprints.npz"
    smiles, acts, docs = time_splits.read_inputs(
        cache, BASE / "data" / "raw" / f"{tag}_activities.csv",
        BASE / "data" / "raw" / "document_years.csv")
    with np.load(cache) as fp:
        original = fp["y"]
    year = time_splits.assign_years(smiles, acts, docs)
    n_dated = time_splits.ensure_coverage(year, smiles)
    train, valid, test, cutoff_v, cutoff_t = time_splits.time_split(year)
    initial = (train, valid, test)
    y, retained, excluded = time_splits.asof_labels(
        smiles, acts, docs, initial, cutoff_v, cutoff_t)
    row = dict(tag=tag, label_policy=LABEL_POLICY, n_total=len(smiles),
               n_first_seen_dated=n_dated, cutoff_valid=cutoff_v,
               cutoff_test=cutoff_t, n_excluded=len(excluded),
               dataset_signature=dataset_signature(smiles, original))
    for name, before, kept in zip(("train", "valid", "test"), initial, retained):
        row[f"n_{name}_before"] = len(before)
        row[f"n_{name}"] = len(kept)
        row[f"n_{name}_excluded"] = len(before) - len(kept)
        row[f"n_{name}_label_changed"] = int(np.count_nonzero(
            ~np.isclose(y[kept], original[kept], rtol=0, atol=1e-6)))
    return row


def main():
    rows = []
    for tag in TAGS:
        row = audit_tag(tag)
        rows.append(row)
        print(f"[{tag}] excluded={row['n_excluded']} "
              f"changed train/valid/test="
              f"{row['n_train_label_changed']}/{row['n_valid_label_changed']}/"
              f"{row['n_test_label_changed']}", flush=True)
    output = BASE / "results" / "time_split" / "asof_label_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
