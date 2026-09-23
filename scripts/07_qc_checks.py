"""QC checks beyond the headline metrics (run after 04_train_and_evaluate.py).

Three checks:
  1. Test-set pIC50 distributions of the random vs scaffold split. RMSE is
     scale-dependent: if the random test set spans a wider potency range, its
     RMSE can be larger even when R2 is identical - worth showing explicitly.
  2. Ultra-potent outliers (pIC50 > 9, i.e. IC50 < 1 nM). Values like this are
     often assay detection limits, unit mislabels or cross-assay artefacts
     rather than true affinity. Group them by assay / document to see whether
     a few suspicious sources dominate.
  3. Permutation importance vs MDI. RandomForest's built-in (MDI) importance
     is inflated for correlated features - and 2048 fingerprint bits contain
     many chemically equivalent ones. Permutation importance on the held-out
     test set is the honest cross-check before telling the substructure story.

Splits are reconstructed exactly (same random_state / deterministic scaffold
function as scripts/04_train_and_evaluate.py).
"""
import importlib.util
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")
NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
RAW = os.path.join(BASE, "data", "raw", f"{TAG}_activities.csv")
MODEL = os.path.join(BASE, "models", f"rf_random_split_{TAG}.joblib")
FIG_HIST = os.path.join(BASE, "figures", f"testset_hist_{TAG}.png")
FIG_IMP = os.path.join(BASE, "figures", f"importance_mdi_vs_perm_{TAG}.png")
OUT_PERM = os.path.join(BASE, "results", f"permutation_importance_{TAG}.csv")
RANDOM_STATE = 42


def load_scaffold_split_func():
    spec = importlib.util.spec_from_file_location(
        "mod04", os.path.join(BASE, "scripts", "04_train_and_evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.scaffold_split


def check_testset_distributions(X, y, smiles):
    tr_r, te_r = train_test_split(np.arange(len(X)), test_size=0.2,
                                  random_state=RANDOM_STATE)
    tr_s, te_s = load_scaffold_split_func()(smiles, test_size=0.2)
    yr, ys = y[te_r], y[te_s]
    print("== 1. test-set pIC50 distributions ==")
    for name, v in [("random", yr), ("scaffold", ys)]:
        print(f"  {name:9s} n={len(v):4d}  mean={v.mean():.2f}  std={v.std():.2f}  "
              f"min={v.min():.2f}  max={v.max():.2f}")
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(min(yr.min(), ys.min()), max(yr.max(), ys.max()), 31)
    plt.hist(yr, bins=bins, alpha=0.55, label=f"random (std={yr.std():.2f})", density=True)
    plt.hist(ys, bins=bins, alpha=0.55, label=f"scaffold (std={ys.std():.2f})", density=True)
    plt.xlabel("pIC50"); plt.ylabel("density"); plt.legend()
    plt.title(f"Test-set pIC50 distribution - {TAG}")
    plt.tight_layout(); plt.savefig(FIG_HIST, dpi=150); plt.close()
    print(f"  saved -> {FIG_HIST}")


def check_ultra_potent():
    print("\n== 2. ultra-potent records (pIC50 > 9, IC50 < 1 nM) ==")
    df = pd.read_csv(RAW)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["standard_value"] > 0]
    df["pic50"] = -np.log10(df["standard_value"] * 1e-9)
    hot = df[df["pic50"] > 9]
    print(f"  {len(hot)} / {len(df)} raw rows ({len(hot)/len(df):.1%}) have pIC50 > 9")
    if len(hot) == 0:
        return
    for col in ["assay_description", "document_chembl_id"]:
        if col in hot.columns and hot[col].notna().any():
            print(f"  top sources by {col}:")
            print(hot.groupby(col).size().sort_values(ascending=False).head(5).to_string())
    print(f"  IC50 nM range of these rows: {hot['standard_value'].min():.4f} .. {hot['standard_value'].max():.4f}")


def check_importance(X, y, smiles):
    print("\n== 3. MDI vs permutation importance (random split) ==", flush=True)
    model = joblib.load(MODEL)
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2,
                              random_state=RANDOM_STATE)
    X_te, y_te = X[te], y[te]
    mdi = model.feature_importances_

    # Manual permutation importance over the top-K MDI bits only (sklearn's
    # permutation_importance would permute all 2048 columns; we only care
    # about the ranking near the top, and the model still sees full-width X).
    from sklearn.metrics import r2_score
    K, N_REP = 100, 3
    baseline = r2_score(y_te, model.predict(X_te))
    rng = np.random.RandomState(RANDOM_STATE)
    top_bits = np.argsort(mdi)[::-1][:K]
    rows = []
    for b in top_bits:
        drops = []
        for _ in range(N_REP):
            Xp = X_te.copy()
            Xp[:, b] = rng.permutation(Xp[:, b])
            drops.append(baseline - r2_score(y_te, model.predict(Xp)))
        rows.append({"bit": int(b), "mdi": float(mdi[b]),
                     "perm_mean": float(np.mean(drops)),
                     "perm_std": float(np.std(drops))})
    df = pd.DataFrame(rows).sort_values("mdi", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_PERM, index=False)
    print(df.head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    from scipy.stats import spearmanr
    rho = spearmanr(df["mdi"].rank(), df["perm_mean"].rank()).statistic
    print(f"  Spearman rank corr (MDI vs perm, top-{K} bits): {rho:.3f}", flush=True)

    plot = df.head(15).iloc[::-1]
    ypos = np.arange(len(plot))
    plt.figure(figsize=(8, 5))
    plt.barh(ypos - 0.2, plot["mdi"], height=0.38, label="MDI (built-in)")
    plt.barh(ypos + 0.2, plot["perm_mean"], height=0.38,
             xerr=plot["perm_std"], label="permutation (test set)")
    plt.yticks(ypos, [f"bit {b}" for b in plot["bit"]])
    plt.xlabel("importance"); plt.legend()
    plt.title(f"MDI vs permutation importance - {TAG} (top-{K} MDI bits)")
    plt.tight_layout(); plt.savefig(FIG_IMP, dpi=150); plt.close()
    print(f"  saved -> {OUT_PERM} and {FIG_IMP}", flush=True)


def main():
    d = np.load(NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    check_testset_distributions(X, y, smiles)
    check_ultra_potent()
    check_importance(X, y, smiles)


if __name__ == "__main__":
    main()
