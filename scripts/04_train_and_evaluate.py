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
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
FIG_DIR = os.path.join(BASE, "figures")
MODEL_DIR = os.path.join(BASE, "models")
METRICS_JSON = os.path.join(BASE, "results", f"metrics_{TAG}.json")
RANDOM_STATE = 42

PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}


def scaffold_split(smiles, test_size=0.2):
    """Assign whole Bemis-Murcko scaffolds to train/test (test ~= test_size)."""
    scaffolds = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaf = smi  # fallback: treat molecule as its own scaffold
        else:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.setdefault(scaf, []).append(i)

    # big scaffolds first -> test set fills up to roughly test_size
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    n_test = int(len(smiles) * test_size)
    test_idx, train_idx, n = [], [], 0
    for g in groups:
        if n < n_test:
            test_idx.extend(g)
            n += len(g)
        else:
            train_idx.extend(g)
    return np.array(train_idx), np.array(test_idx)


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
    plt.title(f"{target} QSAR - RandomForest ({tag} split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_split(X, y, smiles, train_idx, test_idx, tag, fig_path, target):
    print(f"\n=== {tag} split: train={len(train_idx)}, test={len(test_idx)} ===")
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X[train_idx], y[train_idx])
    print(f"Best params: {gs.best_params_}  (CV R2 = {gs.best_score_:.3f})")
    metrics, pred = evaluate(gs.best_estimator_, X[test_idx], y[test_idx], tag)
    scatter(y[test_idx], pred, tag, fig_path, target)
    return metrics, gs.best_estimator_, pred


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)

    d = np.load(IN_NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    target = str(d["target"]) if "target" in d.files else TAG
    print(f"Target: {target} | Feature matrix: X={X.shape}, y range [{y.min():.2f}, {y.max():.2f}]")

    # ---- random split ----
    tr, te = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=RANDOM_STATE
    )
    m_rand, best_rand, _ = run_split(
        X, y, smiles, tr, te, "random",
        os.path.join(FIG_DIR, f"pred_vs_actual_random_{TAG}.png"), target
    )
    joblib.dump(best_rand, os.path.join(MODEL_DIR, f"rf_random_split_{TAG}.joblib"))

    # ---- scaffold split ----
    tr_s, te_s = scaffold_split(smiles, test_size=0.2)
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
