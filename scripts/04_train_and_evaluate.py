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
  is then loaded and only that split's key plus best_params_{split} (the
  fitted estimator's get_params()) are (over)written - every other key is
  preserved bit-for-bit. Default (no --split) keeps the original
  behaviour: train random + scaffold, rewrite the whole metrics file
  (now carrying best_params_random and best_params_scaffold).
  The time split reads its indices from
  data/processed/splits/{TAG}_time.npz (train_idx/test_idx; valid_idx is
  ignored - RF uses CV, not early stopping). The npz must carry versioned
  as-of labels; old all-year-label splits are rejected for training.
  Any {TAG}_time_NA.json marker takes precedence, prints its reason and
  skips with exit 0; missing both is an error. --plot-only renders saved
  predictions, respects --split, and never fits or writes models/metrics.
"""
import json
import os
from contextlib import nullcontext

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from qsar_common import RANDOM_STATE, scaffold_split
from time_labels import load_time_split, require_prediction_version

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
    # Shuffled folds: scaffold/time train pools are ordered by scaffold group or
    # SMILES, so unshuffled KFold would put whole chemotype blocks in one fold.
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # QSAR_CV_BACKEND=threading dodges loky worker deaths seen on some macOS boxes; numerically identical (fixed seeds/folds).
    gs = GridSearchCV(
        # inner n_jobs=1 — the outer GridSearchCV pool already saturates the
        # cores; nesting thrashes (load ~180)
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        PARAM_GRID,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    cv_backend = os.environ.get("QSAR_CV_BACKEND", "").strip()
    if cv_backend == "loky_nommap":
        # no memmapping: the resource tracker was deleting memmap folders under
        # long grids, killing loky workers mid-run
        ctx = parallel_backend("loky", max_nbytes=None)
    elif cv_backend:
        ctx = parallel_backend(cv_backend)
    else:
        ctx = nullcontext()
    with ctx:
        gs.fit(X[train_idx], y[train_idx])
    print(f"Best params: {gs.best_params_}  (CV R2 = {gs.best_score_:.3f})")
    metrics, pred = evaluate(gs.best_estimator_, X[test_idx], y[test_idx], tag)
    scatter(y[test_idx], pred, tag, fig_path, target)
    return metrics, gs.best_estimator_, pred


def run_single(split, train_idx, test_idx, X, y, smiles, target, label_id=None):
    """Train/evaluate ONE split, save model + preds npz, merge its metrics.

    Merges two keys: {split} (r2/rmse/mae) and best_params_{split}
    (best_estimator_.get_params()). Used by --split {random,scaffold,time};
    the default (no --split) path below keeps the original
    train-both-then-rewrite-whole-file behaviour.
    """
    m, best, pred = run_split(
        X, y, smiles, train_idx, test_idx, split,
        os.path.join(FIG_DIR, f"pred_vs_actual_{split}_{TAG}.png"), target
    )
    if label_id is not None:
        best.time_label_id_ = label_id
        m["label_id"] = label_id
    joblib.dump(best, os.path.join(MODEL_DIR, f"rf_{split}_split_{TAG}.joblib"))
    # same keys/dtypes as the existing rf_preds_{TAG}_{split}.npz files:
    # test_idx int64, y_true (dataset dtype), y_pred float64
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f"rf_preds_{TAG}_{split}.npz"),
             test_idx=np.asarray(test_idx), y_true=y[test_idx], y_pred=pred,
             **({"label_id": label_id} if label_id is not None else {}))
    # merge: load whatever is there, overwrite only this split's keys, rewrite
    payload = json.load(open(METRICS_JSON)) if os.path.exists(METRICS_JSON) else {}
    payload[split] = m
    payload[f"best_params_{split}"] = best.get_params()
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
    sp = load_time_split(SPLIT_DIR, TAG)
    return None if sp is None else (sp["train_idx"], sp["test_idx"])


def plot_saved_predictions(selected):
    """Plot stored predictions without fitting or writing model/metric files."""
    for split in selected:
        if split == "time" and os.path.exists(os.path.join(SPLIT_DIR, f"{TAG}_time_NA.json")):
            print(f"[{TAG}/time] skipped: NA marker present")
            continue
        pred_path = os.path.join(RESULTS_DIR, f"rf_preds_{TAG}_{split}.npz")
        if split == "time":
            with np.load(pred_path) as pred:
                split_path = os.path.join(SPLIT_DIR, f"{TAG}_time.npz")
                if os.path.exists(split_path):
                    with np.load(split_path) as archive:
                        sp = dict(archive)
                    if "label_policy" in sp or "label_id" in sp:
                        sp = load_time_split(SPLIT_DIR, TAG)
                        if sp is None:
                            continue
                    if "label_id" in sp or "label_id" in pred:
                        require_prediction_version(pred, sp)
                elif "label_id" in pred:
                    raise ValueError("versioned time predictions require their matching split")
                if "label_id" not in pred:
                    print("[time/plot-only] historical all-year labels; not prospective validation")
                y, p = pred["y_true"], pred["y_pred"]
        else:
            # Default RF training saves models only; always predict from that
            # current model, never prefer a possibly stale random/scaffold NPZ.
            with np.load(IN_NPZ) as data:
                if split == "random":
                    _, test = train_test_split(np.arange(len(data["y"])), test_size=0.2,
                                               random_state=RANDOM_STATE)
                else:
                    _, test = scaffold_split(data["smiles"], test_size=0.2)
                model = joblib.load(os.path.join(MODEL_DIR, f"rf_{split}_split_{TAG}.joblib"))
                y, p = data["y"][test], model.predict(data["X"][test])
        os.makedirs(FIG_DIR, exist_ok=True)
        scatter(y, p, split, os.path.join(FIG_DIR, f"pred_vs_actual_{split}_{TAG}.png"), TAG)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true",
                   help="skip GridSearchCV; load saved models/predictions, re-plot the "
                        "predicted-vs-actual figures (metrics are NOT rewritten)")
    p.add_argument("--split", choices=["random", "scaffold", "time"], default=None,
                   help="train/evaluate only this split and merge just its key "
                        "into metrics_{TAG}.json (default: train random + scaffold "
                        "and rewrite the whole metrics file, as before)")
    args = p.parse_args()

    if args.plot_only:
        plot_saved_predictions([args.split] if args.split else ["random", "scaffold"])
        return

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
        sp = load_time_split(SPLIT_DIR, TAG, smiles, y)
        if sp is None:
            return
        run_single("time", time_idx[0], time_idx[1], X, sp["y"], smiles, target,
                   label_id=str(sp["label_id"]))
        return

    # ---- random split ----
    tr, te = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=RANDOM_STATE
    )
    # ---- scaffold split ----
    tr_s, te_s = scaffold_split(smiles, test_size=0.2)

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
                   "best_params_random": best_rand.get_params(),
                   "best_params_scaffold": best_scaf.get_params()}, f, indent=2)
    print(f"\nMetrics saved -> {os.path.abspath(METRICS_JSON)}")


if __name__ == "__main__":
    main()
