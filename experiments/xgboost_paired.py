"""Recover deterministic XGB test predictions and paired, per-cell uncertainty.

One seed-42 CPU refit per historical cell; every score must reproduce the
committed 3.4.1 result to 1e-6 before it enters the summary. Training uses the
committed parameters, fingerprints and split. Outputs are supplemental and
never replace the campaign's original metrics or RF/GIN models.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from paired_bootstrap import TAGS, paired_bootstrap  # noqa: E402
from xgboost_baseline import fit_pool, pool_fingerprint  # noqa: E402

BASE = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(fp, sp, rf):
    """Require a complete, disjoint partition and exactly aligned RF labels."""
    train, test = fit_pool(sp), np.asarray(sp["test_idx"])
    all_idx = np.concatenate([train, test])
    if not np.array_equal(np.sort(all_idx), np.arange(len(fp["y"]))):
        raise ValueError("split must partition fingerprint rows exactly once")
    if not np.array_equal(test, rf["test_idx"]):
        raise ValueError("RF test_idx differs from split")
    if not np.allclose(fp["y"][test], rf["y_true"], rtol=0, atol=1e-6):
        raise ValueError("RF labels differ from fingerprint labels")
    return train, test


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=BASE / "results/xgboost_paired")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], {}
    for tag in TAGS:
        fp_path = BASE / f"data/processed/{tag}_fingerprints.npz"
        summary_path = BASE / f"results/xgboost/summary_{tag}.json"
        historical = json.loads(summary_path.read_text())
        if xgb.__version__ != historical["xgboost_version"]:
            raise ValueError("refit requires historical XGBoost version")
        fp = np.load(fp_path)
        for split in ("random", "scaffold"):
            split_path = BASE / f"data/processed/splits/{tag}_{split}.npz"
            rf_path = BASE / f"results/rf_preds_{tag}_{split}.npz"
            sp, rf = np.load(split_path), np.load(rf_path)
            train, test = verify_inputs(fp, sp, rf)
            params = {**historical["xgb_params"], "random_state": 42,
                      "n_jobs": args.n_jobs}
            model = xgb.XGBRegressor(**params)
            model.fit(fp["X"][train], fp["y"][train])
            pred = model.predict(fp["X"][test])
            score = float(r2_score(fp["y"][test], pred))
            expected = historical["by_split"][split]["r2_mean"]
            if abs(score - expected) > 1e-6:
                raise ValueError(f"{tag}/{split}: refit {score} != historical {expected}")
            res = paired_bootstrap(rf["y_true"], pred, rf["y_pred"],
                                   n_boot=10_000, seed=42)
            res["frac_boot_xgb_better"] = res.pop("frac_boot_gin_better")
            winner = "xgb" if res["ci_lo"] > 0 else ("rf" if res["ci_hi"] < 0 else "tie")
            pred_path = args.out_dir / f"preds_{tag}_{split}_seed42.npz"
            np.savez_compressed(pred_path, test_idx=test, y_true=rf["y_true"],
                                y_pred=pred, seed=42, xgboost_version=xgb.__version__)
            row = {"tag": tag, "split": split, "n_test": len(test),
                   "refit_r2": score, "historical_r2": expected,
                   "score_abs_difference": abs(score - expected),
                   "winner": winner, "boot_seed": 42, **res}
            rows.append(row)
            inputs[f"{tag}/{split}"] = {
                "files": {str(path.relative_to(BASE)): sha256(path)
                          for path in (fp_path, summary_path, split_path, rf_path)},
                "fit_pool_sha256": pool_fingerprint(train),
                "test_pool_sha256": pool_fingerprint(test),
                "prediction_sha256": sha256(pred_path), "params": params,
            }
            print(f"{tag}/{split}: R2={score:.9f}, delta={res['delta_r2_point']:+.6f} "
                  f"CI=[{res['ci_lo']:+.6f}, {res['ci_hi']:+.6f}] {winner}", flush=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "summary.csv", index=False)
    report = {"schema": "qsar-xgb-paired-v1", "training_seed": 42,
              "xgboost_version": xgb.__version__, "python": platform.python_version(),
              "platform": platform.platform(), "inputs": inputs,
              "code_sha256": {str(path.relative_to(BASE)): sha256(path) for path in
                              (Path(__file__), BASE / "experiments/paired_bootstrap.py",
                               BASE / "experiments/xgboost_baseline.py")},
              "counts": pd.Series([r["winner"] for r in rows]).value_counts().to_dict(),
              "limitations": ["Uncorrected per-cell percentile 95% intervals over test molecules.",
                              "Conditions on one RF fit and one deterministic XGB fit; excludes training uncertainty.",
                              "Scaffold test molecules need not be independent; no scaffold-cluster interval here.",
                              "Bootstrap tail share is descriptive, not a calibrated hypothesis-test p-value."]}
    (args.out_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
