"""QC checks beyond the headline metrics (run after 04_train_and_evaluate.py).

Three checks:
  1. Test-set pIC50 distributions of the random vs scaffold split. RMSE is
     scale-dependent: if the random test set spans a wider potency range, its
     RMSE can be larger even when R2 is identical - worth showing explicitly.
  2. Ultra-potent outliers (pIC50 > 9, i.e. IC50 < 1 nM). Values like this are
     often assay detection limits, unit mislabels or cross-assay artefacts
     rather than true affinity. Group them by assay / document to see whether
     a few suspicious sources dominate. Only rows the model actually saw are
     counted: 02's nM-unit and non-empty-SMILES filters apply first, then
     only rows the label policy keeps (standard_relation == "=") - rows the
     H1 filter drops are reported on a separate line, never mixed into the
     headline. Inputs 02 refuses or runs without that policy (non-ChEMBL
     TAGs bace/esol; ChEMBL-style files missing the relation column) are
     skipped with an explanation instead of being audited.
  3. Permutation importance vs MDI. RandomForest's built-in (MDI) importance
     is inflated for correlated features - and 2048 fingerprint bits contain
     many chemically equivalent ones. Permutation importance on the held-out
     test set is the honest cross-check before telling the substructure story.

Splits are reconstructed exactly (same random_state and the shared
deterministic scaffold function from scripts/qsar_common.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from qsar_common import scaffold_split

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")
NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
RAW = os.path.join(BASE, "data", "raw", f"{TAG}_activities.csv")
MODEL = os.path.join(BASE, "models", f"rf_random_split_{TAG}.joblib")
FIG_HIST = os.path.join(BASE, "figures", f"testset_hist_{TAG}.png")
FIG_IMP = os.path.join(BASE, "figures", f"importance_mdi_vs_perm_{TAG}.png")
OUT_PERM = os.path.join(BASE, "results", f"permutation_importance_{TAG}.csv")
RANDOM_STATE = 42


def check_testset_distributions(X, y, smiles):
    tr_r, te_r = train_test_split(np.arange(len(X)), test_size=0.2,
                                  random_state=RANDOM_STATE)
    tr_s, te_s = scaffold_split(smiles, test_size=0.2)
    yr, ys = y[te_r], y[te_s]
    print("== 1. test-set pIC50 distributions ==")
    for name, v in [("random", yr), ("scaffold", ys)]:
        print(f"  {name:9s} n={len(v):4d}  mean={v.mean():.2f}  std={v.std():.2f}  "
              f"min={v.min():.2f}  max={v.max():.2f}")
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(min(yr.min(), ys.min()), max(yr.max(), ys.max()), 31)
    plt.hist(yr, bins=bins, alpha=0.55, label=f"random (std={yr.std():.2f})", density=True)
    plt.hist(ys, bins=bins, alpha=0.55, label=f"scaffold (std={ys.std():.2f})", density=True)
    plt.xlabel("pIC50")
    plt.ylabel("density")
    plt.legend()
    plt.title(f"Test-set pIC50 distribution - {TAG}")
    plt.tight_layout()
    plt.savefig(FIG_HIST, dpi=150)
    plt.close()
    print(f"  saved -> {FIG_HIST}")


def check_ultra_potent():
    print("\n== 2. ultra-potent records (pIC50 > 9, IC50 < 1 nM) ==")
    # Skip the audit for non-ChEMBL campaigns: 02_clean_data runs bace
    # without the standard_relation censoring policy (and esol never goes
    # through 02 at all), so a relation-based "labelled rows" headline
    # would misstate what the model saw (Low, PR #6).
    if TAG in ("bace", "esol"):
        print(f"  skipped: QSAR_TAG={TAG!r} is not a ChEMBL campaign "
              "dataset - 02_clean_data applies the standard_relation "
              "censoring policy only to ChEMBL inputs, so the labelled-"
              "rows audit below does not apply here")
        return
    df = pd.read_csv(RAW)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    # Mirror 02's filters BEFORE counting: a row with a non-nM unit or an
    # empty SMILES never becomes a label, so it must not enter the
    # "labelled rows" denominator either (Low, PR #5).
    if "standard_units" in df.columns:
        n_units = len(df)
        df = df[df["standard_units"] == "nM"]
        if len(df) != n_units:
            print(f"  (02 unit filter) dropped {n_units - len(df)} row(s) "
                  "with standard_units != 'nM'")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    n_smiles = len(df)
    df = df[df["canonical_smiles"].str.strip() != ""]
    if len(df) != n_smiles:
        print(f"  (02 SMILES filter) dropped {n_smiles - len(df)} row(s) "
              "with empty canonical_smiles")
    df = df[df["standard_value"] > 0]
    df["pic50"] = -np.log10(df["standard_value"] * 1e-9)
    hot_all = df[df["pic50"] > 9]
    if "standard_relation" in df.columns:
        # The label policy (scripts/02_clean_data.py) keeps only
        # standard_relation == "=" rows; anything else ('<', '>', '>=',
        # NaN) never reaches pic50, so counting it here would audit rows
        # the model never saw. Headline = kept rows only; the dropped
        # ones are reported on a separate line.
        kept = df["standard_relation"] == "="
        hot = hot_all[kept.loc[hot_all.index]]
        dropped = hot_all[~kept.loc[hot_all.index]]
        n_kept = int(kept.sum())
        pct = f" ({len(hot) / n_kept:.1%})" if n_kept else ""
        print(f"  {len(hot)} / {n_kept} labelled rows (standard_relation "
              f"'='){pct} have pIC50 > 9")
        if len(dropped):
            rels = (dropped["standard_relation"].fillna("NaN")
                    .value_counts().to_dict())
            print(f"  {len(dropped)} further raw row(s) exceed pIC50 > 9 "
                  f"but were dropped by the label policy "
                  f"(relations: {rels}) - not counted above")
    else:
        # Skip what 02 refuses: a ChEMBL-style file without the relation
        # column is rejected outright by 02 (N5: censoring cannot be
        # verified), so no label policy was ever applied to build the
        # model from it -- auditing it here would assert a kept-set that
        # does not exist (Low, PR #6). Non-ChEMBL sources without the
        # column (bace-style IDs): 02 keeps every row that survived the
        # unit/SMILES filters above, so that filtered set IS the kept set.
        if "molecule_chembl_id" in df.columns:
            ids = (df["molecule_chembl_id"].fillna("").astype(str)
                   .str.strip().str.upper())
            n_chembl_style = int(
                (ids.str.startswith("CHEMBL") | ids.str.fullmatch(r"\d+")
                 | ids.str.fullmatch(r"\d+\.0+")).sum())
        else:
            n_chembl_style = 0
        if n_chembl_style:
            print(f"  skipped: {n_chembl_style} ChEMBL-style molecule ID(s) "
                  "but no standard_relation column - 02_clean_data refuses "
                  "such inputs (the censoring filter cannot be verified), "
                  "so there is no label policy to audit")
            return
        hot = hot_all
        print(f"  {len(hot)} / {len(df)} rows after 02's unit/SMILES filters "
              f"({len(hot)/len(df):.1%}) have pIC50 > 9")
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
    plt.xlabel("importance")
    plt.legend()
    plt.title(f"MDI vs permutation importance - {TAG} (top-{K} MDI bits)")
    plt.tight_layout()
    plt.savefig(FIG_IMP, dpi=150)
    plt.close()
    print(f"  saved -> {OUT_PERM} and {FIG_IMP}", flush=True)


def main():
    d = np.load(NPZ)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    check_testset_distributions(X, y, smiles)
    check_ultra_potent()
    check_importance(X, y, smiles)


if __name__ == "__main__":
    main()
