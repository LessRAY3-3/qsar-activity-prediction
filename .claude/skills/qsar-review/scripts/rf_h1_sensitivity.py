# 2026-10-11 数据刷新战役用：H1 敏感性 RF 重跑驱动（剔除 censored-only 化合物重跑 egfr_full RF）；原 .scratch/h1_sensitivity/rf_h1_sensitivity.py。
"""H1 sensitivity: re-run the egfr_full RF with censored-only compounds dropped.

Historical experiment (2026-10-10/11 campaign): replicates the ORIGINAL
scripts/04_train_and_evaluate.py flow as it stood pre-refresh (PARAM_GRID,
RandomForestRegressor(random_state=42, n_jobs=-1), GridSearchCV scoring="r2",
cv=KFold(5) UNSHUFFLED) while removing every compound whose cleaned records
contain no standard_relation == "=" row (censored-only) from both the
training pool and the test set. Training pool = npz train_idx+valid_idx.
Inputs are the pinned pre-refresh snapshots in .scratch/h1_sensitivity/.

Baselines (why no more `git show HEAD:`):
  The published deltas (−0.0250 random / +0.0088 scaffold,
  docs/reviews/h1-sensitivity-2026-10-10.md) compare the sensitivity run
  against the PRE-refresh metrics_egfr_full.json (random R2 0.7592,
  scaffold 0.5926) that lived at HEAD when the experiment ran. The
  2026-10-11 data refresh replaced that file with refreshed values
  (0.7289 / 0.6081; tag `data-refresh-2026-10-11`), so reading HEAD now
  silently reports +0.0053 / −0.0067 against a different test set. This
  script therefore pins:
    * the PRIMARY baseline to the published pre-refresh metrics, as
      constants below (identical to "before_head" in the original
      .scratch/h1_sensitivity/rf_h1_sensitivity.json), and
    * the optional refreshed-state baseline to the git tag
      `data-refresh-2026-10-11` (via --baseline-ref, never HEAD),
      reported separately as "delta_vs_pinned_ref".

Outputs go to --out (default .scratch/h1_sensitivity/): rf_h1_sensitivity.json,
rf_h1_preds_*.npz, scatter PNGs. Does NOT touch results/, figures/, models/
or data/ (refused explicitly). --dry-run validates inputs and prints the
resolved baselines without training or writing.

Usage:
    python .claude/skills/qsar-review/scripts/rf_h1_sensitivity.py --dry-run
    python .claude/skills/qsar-review/scripts/rf_h1_sensitivity.py --out /tmp/h1
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# Inputs are immutable pre-refresh snapshots; outputs default alongside them.
IN_DIR = os.path.join(BASE, ".scratch", "h1_sensitivity")
FP = os.path.join(IN_DIR, "old_egfr_full_fingerprints.npz")
SPLIT_DIR = IN_DIR  # old_egfr_full_{random,scaffold}.npz
CENSORED_CSV = os.path.join(IN_DIR, "censored_only_smiles.csv")
PINNED_REF = "data-refresh-2026-10-11"

# Published pre-refresh baseline (docs/reviews/h1-sensitivity-2026-10-10.md;
# == "before_head" of the original rf_h1_sensitivity.json).
PUBLISHED_BASELINE = {
    "random": {"r2": 0.7592274824973815, "rmse": 0.7092953672809168,
               "mae": 0.4950194546970497},
    "scaffold": {"r2": 0.5925707030729936, "rmse": 0.8900524594034126,
                 "mae": 0.6629450572128702},
}

sys.path.insert(0, os.path.join(BASE, "scripts"))
from qsar_common import RANDOM_STATE  # noqa: E402

PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}

PROTECTED_DIRS = tuple(
    os.path.join(BASE, d) for d in ("results", "figures", "models", "data")
)


def is_protected(path):
    """True when path is inside results/ figures/ models/ or data/."""
    return any(path == d or path.startswith(d + os.sep) for d in PROTECTED_DIRS)


def ref_metrics(ref):
    """metrics_egfr_full.json pinned at a git ref (never HEAD)."""
    try:
        raw = subprocess.run(
            ["git", "show", f"{ref}:results/metrics_egfr_full.json"],
            cwd=BASE, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[warn] cannot read baseline at ref {ref!r}: {exc}", flush=True)
        return None
    mj = json.loads(raw)
    return {k: mj[k] for k in ("random", "scaffold")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=IN_DIR,
                   help=f"output directory (default: {os.path.relpath(IN_DIR, BASE)}); "
                        "results/, figures/, models/ and data/ are refused")
    p.add_argument("--dry-run", action="store_true",
                   help="validate inputs and print resolved baselines; "
                        "no training, no writes")
    p.add_argument("--baseline-ref", default=PINNED_REF,
                   help=f"git ref for the refreshed-state comparison baseline "
                        f"(default: {PINNED_REF}; never HEAD)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = os.path.abspath(args.out)
    if is_protected(out):
        raise SystemExit(f"--out {out} points into a protected directory; refusing")

    censored = pd_read_censored()
    before = dict(PUBLISHED_BASELINE)
    pinned = ref_metrics(args.baseline_ref)

    inputs = {
        "fingerprints": FP, "censored_csv": CENSORED_CSV,
        "split_random": os.path.join(SPLIT_DIR, "egfr_full_random.npz"),
        "split_scaffold": os.path.join(SPLIT_DIR, "egfr_full_scaffold.npz"),
    }
    missing = [k for k, p in inputs.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing pinned inputs in {IN_DIR}: {missing}")

    print(f"baseline (published pre-refresh): "
          f"random R2 {before['random']['r2']:.4f}, "
          f"scaffold R2 {before['scaffold']['r2']:.4f}")
    if pinned:
        print(f"baseline (pinned ref {args.baseline_ref}): "
              f"random R2 {pinned['random']['r2']:.4f}, "
              f"scaffold R2 {pinned['scaffold']['r2']:.4f} "
              "(refreshed data; deltas vs this are NOT the published numbers)")
    else:
        print(f"[warn] ref {args.baseline_ref!r} unavailable; "
              "delta_vs_pinned_ref will be null")

    if args.dry_run:
        print(f"dry-run: would train 2 splits (random, scaffold) with "
              f"{len(censored)} censored-only SMILES to drop, "
              f"then write outputs to {out}")
        return 0

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)

    d = np.load(FP)
    X, y, smiles = d["X"], d["y"], d["smiles"]
    cens = set(censored)
    is_cens = np.array([s in cens for s in smiles])
    print(f"compounds total={len(smiles)}, censored-only={int(is_cens.sum())}",
          flush=True)

    result = {
        "random_state": RANDOM_STATE,
        "cv": "KFold(5) unshuffled",
        "censored_only_total": int(is_cens.sum()),
        "baseline_mode": "published pre-refresh constants",
        "before_published": before,
        "baseline_ref": args.baseline_ref,
        "before_pinned_ref": pinned,
        "splits": {},
    }

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

        entry = {
            "dropped_train_pool": int(is_cens[pool].sum()),
            "dropped_test": int(is_cens[test].sum()),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "best_params": gs.best_params_,
            "best_cv_r2": float(gs.best_score_),
            "after": {"r2": r2, "rmse": rmse, "mae": mae},
            "delta_vs_published_baseline": {
                "r2": r2 - before[split]["r2"],
                "rmse": rmse - before[split]["rmse"],
            },
            "delta_vs_pinned_ref": ({
                "r2": r2 - pinned[split]["r2"],
                "rmse": rmse - pinned[split]["rmse"],
            } if pinned else None),
        }
        result["splits"][split] = entry
        np.savez(os.path.join(out, f"rf_h1_preds_{split}.npz"),
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
        plt.savefig(os.path.join(out, f"rf_h1_scatter_{split}.png"), dpi=150)
        plt.close()

    with open(os.path.join(out, "rf_h1_sensitivity.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        s: {k: result["splits"][s][k] for k in ("after", "delta_vs_published_baseline",
                                                "delta_vs_pinned_ref")}
        for s in result["splits"]}, indent=2), flush=True)
    return 0


def pd_read_censored():
    import pandas as pd
    return pd.read_csv(CENSORED_CSV)["canonical_smiles"].tolist()


if __name__ == "__main__":
    sys.exit(main())
