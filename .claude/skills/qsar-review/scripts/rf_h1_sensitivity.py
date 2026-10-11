# 2026-10-11 数据刷新战役用：H1 敏感性 RF 重跑驱动（剔除 censored-only 化合物重跑 egfr_full RF）；原 .scratch/h1_sensitivity/rf_h1_sensitivity.py。
"""H1 sensitivity: re-run the egfr_full RF with censored-only compounds dropped.

Replicates the ORIGINAL scripts/04_train_and_evaluate.py flow at git HEAD
(PARAM_GRID, RandomForestRegressor(random_state=42, n_jobs=-1), GridSearchCV
scoring="r2", cv=KFold(5) UNSHUFFLED) while removing every compound whose
cleaned records contain no standard_relation == "=" row (censored-only) from
both the training pool and the test set. Training pool = npz train_idx+valid_idx
(verified equal to 04's own split at HEAD).

Outputs to .scratch/h1_sensitivity/: rf_h1_sensitivity.json, rf_h1_preds_*.npz,
scatter PNGs. Does NOT touch results/ or models/.
"""
import json
import os
import subprocess
import sys

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.join(BASE, ".scratch", "h1_sensitivity")
# Old-data snapshots: the workspace copies were overwritten by the C refresh,
# so the sensitivity inputs live here (verified: 13,497 rows, smiles order
# matches the OLD clean CSV extracted from git HEAD).
FP = os.path.join(OUT, "old_egfr_full_fingerprints.npz")
SPLIT_DIR = OUT  # old_egfr_full_{random,scaffold}.npz

sys.path.insert(0, os.path.join(BASE, "scripts"))
from qsar_common import RANDOM_STATE  # noqa: E402

PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}


def git_head_metrics():
    """'before' numbers from git HEAD (workspace results/ may be overwritten)."""
    raw = subprocess.run(
        ["git", "show", "HEAD:results/metrics_egfr_full.json"],
        cwd=BASE, capture_output=True, text=True, check=True).stdout
    mj = json.loads(raw)
    return {k: mj[k] for k in ("random", "scaffold")}


def main():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cens = set(pd_read_censored())
    d = np.load(FP)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    is_cens = np.array([s in cens for s in smiles])
    print(f"compounds total={len(smiles)}, censored-only={int(is_cens.sum())}",
          flush=True)

    before = git_head_metrics()
    out = {"random_state": RANDOM_STATE, "cv": "KFold(5) unshuffled",
           "censored_only_total": int(is_cens.sum()), "before_head": before,
           "splits": {}}

    for split in ["random", "scaffold"]:
        s = np.load(os.path.join(SPLIT_DIR, f"egfr_full_{split}.npz"))
        pool = np.concatenate([s["train_idx"], s["valid_idx"]])  # == 04 train pool
        test = s["test_idx"]
        tr = pool[~is_cens[pool]]
        te = test[~is_cens[test]]
        print(f"\n=== {split}: pool {len(pool)} -> {len(tr)} "
              f"(dropped {int(is_cens[pool].sum())}), "
              f"test {len(test)} -> {len(te)} (dropped {int(is_cens[test].sum())}) ===",
              flush=True)

        # RF n_jobs=1 + GridSearchCV n_jobs=-1: result-identical scheduling
        # (RF is deterministic given random_state) without nested oversubscription.
        gs = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            PARAM_GRID, cv=KFold(n_splits=5), scoring="r2", n_jobs=-1, verbose=1,
        )
        gs.fit(X[tr], y[tr])
        pred = gs.predict(X[te])
        r2 = float(r2_score(y[te], pred))
        rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
        mae = float(mean_absolute_error(y[te], pred))
        print(f"[{split}] best={gs.best_params_} (CV R2 {gs.best_score_:.4f}) "
              f"-> R2 {r2:.4f} RMSE {rmse:.4f} MAE {mae:.4f}", flush=True)

        out["splits"][split] = {
            "dropped_train_pool": int(is_cens[pool].sum()),
            "dropped_test": int(is_cens[test].sum()),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "best_params": gs.best_params_,
            "best_cv_r2": float(gs.best_score_),
            "after": {"r2": r2, "rmse": rmse, "mae": mae},
            "delta_vs_head": {
                "r2": r2 - before[split]["r2"],
                "rmse": rmse - before[split]["rmse"],
            },
        }
        np.savez(os.path.join(OUT, f"rf_h1_preds_{split}.npz"),
                 test_idx=te, y_true=y[te], y_pred=pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y[te], pred, s=10, alpha=0.4, edgecolors="none")
        lo, hi = min(y[te].min(), pred.min()), max(y[te].max(), pred.max())
        plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.title(f"egfr_full H1 sensitivity - RF ({split}, no censored-only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"rf_h1_scatter_{split}.png"), dpi=150)
        plt.close()

        with open(os.path.join(OUT, "rf_h1_sensitivity.json"), "w") as f:
            json.dump(out, f, indent=2)

    print(json.dumps(out["splits"], indent=2), flush=True)


def pd_read_censored():
    import pandas as pd
    p = os.path.join(OUT, "censored_only_smiles.csv")
    return pd.read_csv(p)["canonical_smiles"].tolist()


if __name__ == "__main__":
    main()
