"""Steps 4-6: Split data, train RandomForest (with GridSearchCV), evaluate.

Step 4 - two split strategies:
  * random split: train_test_split(80/20, random_state=42).
    EASY, but near-identical analogues can land on both sides -> score
    is optimistic.
  * scaffold split: group molecules by Bemis-Murcko scaffold and put
    whole scaffolds into train or test. Simulates predicting truly NEW
    chemotypes -> the honest number. Implemented with RDKit only (no
    DeepChem dependency).

Step 5 - model:
  RandomForestRegressor with GridSearchCV over
  n_estimators / max_depth / min_samples_split (5-fold CV on training set).

Step 6 - evaluation:
  R2, RMSE, MAE on the test set + predicted-vs-actual scatter plot.

CLI:
  --split {random,scaffold,time} runs only that split; metrics_{TAG}.json
  is then loaded and only that split's key is (over)written - every other
  key is preserved bit-for-bit. Default (no --split) keeps the original
  behaviour: train random + scaffold, rewrite the whole metrics file.
  The time split reads its indices from
  data/processed/splits/{TAG}_time.npz (train_idx/test_idx; valid_idx is
  ignored - RF uses CV, not early stopping). If that npz is missing but a
  {TAG}_time_NA.json marker exists beside it, its "reason" is printed and
  the run skips with exit 0; missing both is an error.
"""
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from qsar_common import RANDOM_STATE, scaffold_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
FIG_DIR = os.path.join(BASE, "figures")
MODEL_DIR = os.path.join(BASE, "models")
METRICS_JSON = os.path.join(BASE, "results", f"metrics_{TAG}.json")
SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")
RESULTS_DIR = os.path.join(BASE, "results")

PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}
CV_FOLDS = 5


def evaluate(model, X_test, y_test, tag):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = mean_absolute_error(y_test, pred)
    print(f"[{tag}]  R2 = {r2:.3f}   RMSE = {rmse:.3f}   MAE = {mae:.3f}")
    return {"split": tag, "r2": float(r2), "rmse": rmse, "mae": float(mae)}, pred


def scatter(y_test, pred, tag, path, target):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, s=10, alpha=0.4, edgecolors="none")
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
    plt.xlabel("Actual pIC50")
    plt.ylabel("Predicted pIC50")
    # short TAG-based title: full target names overflow the 6in canvas
    plt.title(f"{TAG.upper()} - RandomForest ({tag} split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_split(X, y, smiles, train_idx, test_idx, tag, fig_path, target):
    print(f"\n=== {tag} split: train={len(train_idx)}, test={len(test_idx)} ===")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X[train_idx], y[train_idx])
    print(f"Best params: {gs.best_params_}  (CV R2 = {gs.best_score_:.3f})")
    metrics, pred = evaluate(gs.best_estimator_, X[test_idx], y[test_idx], tag)
    scatter(y[test_idx], pred, tag, fig_path, target)
    return metrics, gs.best_estimator_, pred


def run_single(split, train_idx, test_idx, X, y, smiles, target):
    """Train/evaluate ONE split, save model + preds npz, merge its metrics key.

    Used by --split {random,scaffold,time}; the default (no --split) path
    below keeps the original train-both-then-rewrite-whole-file behaviour.
    """
    m, best, pred = run_split(
        X, y, smiles, train_idx, test_idx, split,
        os.path.join(FIG_DIR, f"pred_vs_actual_{split}_{TAG}.png"), target
    )
    joblib.dump(best, os.path.join(MODEL_DIR, f"rf_{split}_split_{TAG}.joblib"))
    # same keys/dtypes as the existing rf_preds_{TAG}_{split}.npz files:
    # test_idx int64, y_true (dataset dtype), y_pred float64
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f"rf_preds_{TAG}_{split}.npz"),
             test_idx=np.asarray(test_idx), y_true=y[test_idx], y_pred=pred)
    # merge: load whatever is there, overwrite only this split's key, rewrite
    payload = json.load(open(METRICS_JSON)) if os.path.exists(METRICS_JSON) else {}
    payload[split] = m
    with open(METRICS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nMetrics merged -> {os.path.abspath(METRICS_JSON)}")


def resolve_time_split():
    """-> (train_idx, test_idx) | None (NA marker: reason printed, skip, exit 0).

    The time split lives in data/processed/splits/{TAG}_time.npz; when a
    dataset has no usable document years, a {TAG}_time_NA.json marker sits
    beside it instead and carries the reason to print. Neither -> error.
    valid_idx is deliberately not read: RF cross-validates, no early stop.
    """
    npz_path = os.path.join(SPLIT_DIR, f"{TAG}_time.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        return d["train_idx"], d["test_idx"]
    na_path = os.path.join(SPLIT_DIR, f"{TAG}_time_NA.json")
    if os.path.exists(na_path):
        na = json.load(open(na_path))
        print(f"[{TAG}/time] skipped: {na.get('reason', na)}")
        return None
    raise SystemExit(
        f"missing time split: neither {npz_path} nor {na_path} exists")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true",
                   help="skip GridSearchCV; load the saved models, re-plot the "
                        "predicted-vs-actual figures (metrics are NOT rewritten)")
    p.add_argument("--split", choices=["random", "scaffold", "time"], default=None,
                   help="train/evaluate only this split and merge just its key "
                        "into metrics_{TAG}.json (default: train random + scaffold "
                        "and rewrite the whole metrics file, as before)")
    args = p.parse_args()

    # resolve the time split first so an NA marker skips before anything is written
    time_idx = None
    if args.split == "time":
        time_idx = resolve_time_split()
        if time_idx is None:  # NA marker printed its reason; exit 0
            return

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)

    d = np.load(IN_NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    target = str(d["target"]) if "target" in d.files else TAG
    print(f"Target: {target} | Feature matrix: X={X.shape}, y range [{y.min():.2f}, {y.max():.2f}]")

    if args.split == "time":  # indices come from the npz; no random/scaffold work
        run_single("time", time_idx[0], time_idx[1], X, y, smiles, target)
        return

    # ---- random split ----
    tr, te = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=RANDOM_STATE
    )
    # ---- scaffold split ----
    tr_s, te_s = scaffold_split(smiles, test_size=0.2)

    if args.plot_only:
        for split, train_idx, test_idx in (("random", tr, te), ("scaffold", tr_s, te_s)):
            model = joblib.load(os.path.join(MODEL_DIR, f"rf_{split}_split_{TAG}.joblib"))
            pred = model.predict(X[test_idx])
            print(f"[plot-only:{split}] test R2 = {r2_score(y[test_idx], pred):.6f}")
            scatter(y[test_idx], pred, split,
                    os.path.join(FIG_DIR, f"pred_vs_actual_{split}_{TAG}.png"), target)
        return

    if args.split is not None:  # --split random | --split scaffold
        if args.split == "random":
            run_single("random", tr, te, X, y, smiles, target)
        else:
            run_single("scaffold", tr_s, te_s, X, y, smiles, target)
        return

    m_rand, best_rand, _ = run_split(
        X, y, smiles, tr, te, "random",
        os.path.join(FIG_DIR, f"pred_vs_actual_random_{TAG}.png"), target
    )
    joblib.dump(best_rand, os.path.join(MODEL_DIR, f"rf_random_split_{TAG}.joblib"))

    m_scaf, best_scaf, _ = run_split(
        X, y, smiles, tr_s, te_s, "scaffold",
        os.path.join(FIG_DIR, f"pred_vs_actual_scaffold_{TAG}.png"), target
    )
    joblib.dump(best_scaf, os.path.join(MODEL_DIR, f"rf_scaffold_split_{TAG}.joblib"))

    with open(METRICS_JSON, "w") as f:
        json.dump({"target": target, "tag": TAG, "random": m_rand, "scaffold": m_scaf,
                   "best_params_random": best_rand.get_params()}, f, indent=2)
    print(f"\nMetrics saved -> {os.path.abspath(METRICS_JSON)}")


if __name__ == "__main__":
    main()
