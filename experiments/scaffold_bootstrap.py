"""Sensitivity of ensemble comparisons to resampling entire Murcko groups.

Run after paired_bootstrap.py. This complements molecule-level intervals;
few held-out groups still limit inference about unseen chemical series.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import BASE, TAGS, load_cell, r2  # noqa: E402


def cluster_bootstrap(y, pred_gnn, pred_rf, groups, n_boot=10000, seed=42):
    y, pg, pr = [np.asarray(a, dtype=float) for a in (y, pred_gnn, pred_rf)]
    groups = np.asarray(groups)
    if any(a.ndim != 1 for a in (y, pg, pr, groups)) or not len(y):
        raise ValueError("inputs must be nonempty one-dimensional arrays")
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not (len(y) == len(pg) == len(pr) == len(groups)):
        raise ValueError("prediction/group lengths differ")
    if not all(np.isfinite(a).all() for a in (y, pg, pr)):
        raise ValueError("nonfinite input")
    _, inverse = np.unique(groups, return_inverse=True)
    k = int(inverse.max()) + 1
    if k < 2 or n_boot < 1:
        raise ValueError("need at least two scaffold groups and one resample")
    # Centering avoids cancellation when labels have a large common offset.
    yc = y - y.mean()
    stats = np.array([np.bincount(inverse, weights=w, minlength=k)
                      for w in (np.ones(len(y)), yc, yc**2,
                                (y - pr)**2 - (y - pg)**2)]).T
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, n_boot, 256):
        draw = rng.integers(k, size=(min(256, n_boot - start), k))
        n, sy, sy2, dsse = stats[draw].sum(axis=1).T
        sst = sy2 - sy**2 / n
        valid = sst > np.finfo(float).eps
        values.extend((dsse[valid] / sst[valid]).tolist())
    if not values:
        raise ValueError("all cluster resamples are degenerate")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"d_r2": r2(y, pg) - r2(y, pr), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_groups": k, "n_test": len(y), "n_boot": n_boot,
            "n_valid_boot": len(values), "boot_seed": seed,
            "resampling_unit": "Murcko scaffold", "pair": "mean3",
            "winner": "gnn" if lo > 0 else "rf" if hi < 0 else "tie"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(BASE, "results", "significance", "scaffold_cluster_sensitivity.csv"))
    args = parser.parse_args()
    rows = []
    for model in ("gin", "gine", "attentivefp"):
        for tag in TAGS:
            cell = load_cell(tag, "scaffold", model)
            with np.load(os.path.join(BASE, "data", "processed", "splits", f"{tag}_scaffold.npz")) as sp:
                groups = sp["scaffold_smiles"][sp["test_idx"]]
            rows.append({"tag": tag, "model": model, "split": "scaffold",
                         **cluster_bootstrap(cell["y"], cell["gin_mean3"], cell["rf"], groups)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
