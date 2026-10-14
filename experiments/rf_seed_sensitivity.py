"""Independent RF seed sensitivity on current a2a/hERG as-of time labels.

Fixed published selected parameters, train_idx only, no tuning. This is a new
five-fit sensitivity study, not a reconstruction of the review's undocumented
SD~0.006 experiment. Original metrics/models/predictions are read-only inputs.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from time_labels import load_time_split, require_prediction_version  # noqa: E402

TAGS = ("a2a", "herg")
SEEDS = (42, 1, 2, 3, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_aligned(pred, split):
    require_prediction_version(pred, split)
    if not np.array_equal(pred["test_idx"], split["test_idx"]):
        raise ValueError("prediction test_idx mismatch")
    y = split["y"][split["test_idx"]]
    if not np.allclose(pred["y_true"], y, rtol=0, atol=1e-6):
        raise ValueError("prediction labels mismatch")
    if np.asarray(pred["y_pred"]).shape != y.shape or not np.isfinite(pred["y_pred"]).all():
        raise ValueError("invalid prediction shape or values")


def summarize(frame):
    rows = []
    for tag, part in frame.groupby("tag", sort=True):
        row = {"tag": tag, "n_seeds": len(part)}
        for name in ("r2", "rmse", "mae", "delta_fixed_gin_minus_rf"):
            values = part[name].to_numpy()
            row.update({f"{name}_mean": float(np.mean(values)),
                        f"{name}_std_ddof0": float(np.std(values, ddof=0)),
                        f"{name}_std_ddof1": float(np.std(values, ddof=1)),
                        f"{name}_min": float(np.min(values)),
                        f"{name}_max": float(np.max(values))})
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/rf_seed_sensitivity")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args(argv)
    if not 1 <= args.n_jobs <= 4:
        p.error("--n-jobs must be between 1 and 4")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, evidence = [], {}
    for tag in TAGS:
        fp_path = ROOT / f"data/processed/{tag}_fingerprints.npz"
        split_path = ROOT / f"data/processed/splits/{tag}_time.npz"
        metrics_path = ROOT / f"results/metrics_{tag}.json"
        rf_path = ROOT / f"results/rf_preds_{tag}_time.npz"
        fp = np.load(fp_path)
        split = load_time_split(split_path.parent, tag, fp["smiles"], fp["y"])
        if split is None:
            raise ValueError(f"{tag}: unavailable time split")
        metrics = json.loads(metrics_path.read_text())
        published_params = metrics["best_params_time"]
        if published_params["random_state"] != 42:
            raise ValueError("published RF is not seed42")
        rf = np.load(rf_path)
        assert_aligned(rf, split)
        sources = [fp_path, split_path, metrics_path, rf_path]
        gin_predictions = []
        for suffix in ("", "_seed1", "_seed2"):
            path = ROOT / f"results/gnn_preds_{tag}_time{suffix}.npz"
            pred = np.load(path)
            assert_aligned(pred, split)
            gin_predictions.append(np.asarray(pred["y_pred"], dtype=np.float64))
            sources.append(path)
        y, train, test = split["y"], split["train_idx"], split["test_idx"]
        gin_r2 = float(r2_score(y[test], np.mean(gin_predictions, axis=0)))
        source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
        predictions = {}
        print(f"{tag}: train={len(train)}, test={len(test)}, "
              f"trees={published_params['n_estimators']}, threads={args.n_jobs}", flush=True)
        for seed in SEEDS:
            params = {**published_params, "random_state": seed, "n_jobs": args.n_jobs}
            start = time.monotonic()
            model = RandomForestRegressor(**params)
            model.fit(fp["X"][train], y[train])
            prediction = model.predict(fp["X"][test])
            difference = float(np.max(np.abs(prediction - rf["y_pred"])))
            if seed == 42 and not np.allclose(prediction, rf["y_pred"], rtol=0, atol=1e-10):
                raise ValueError(f"{tag}: seed42 differs from published RF by {difference}")
            row = {"tag": tag, "seed": seed, "n_train": len(train), "n_test": len(test),
                   "r2": float(r2_score(y[test], prediction)),
                   "rmse": float(np.sqrt(mean_squared_error(y[test], prediction))),
                   "mae": float(mean_absolute_error(y[test], prediction)),
                   "fixed_gin_ensemble_r2": gin_r2,
                   "seconds": time.monotonic() - start,
                   "max_abs_prediction_diff_vs_published_seed42": difference}
            row["delta_fixed_gin_minus_rf"] = gin_r2 - row["r2"]
            name = f"preds_{tag}_time_seed{seed}.npz"
            path = args.out_dir / name
            np.savez_compressed(path, test_idx=test, y_true=y[test], y_pred=prediction,
                                seed=seed, label_id=split["label_id"], sklearn_version=sklearn.__version__)
            predictions[name] = sha(path)
            all_rows.append(row)
            print(f"{tag}/seed{seed}: R2={row['r2']:.9f} "
                  f"delta={row['delta_fixed_gin_minus_rf']:+.9f} {row['seconds']:.1f}s", flush=True)
        if any(sha(ROOT / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("source artifact changed during study")
        evidence[tag] = {"input_sha256": source_hashes, "prediction_sha256": predictions,
                         "published_parameters": published_params,
                         "label_id": str(split["label_id"]),
                         "changes_from_published_parameters": {"random_state": list(SEEDS),
                                                               "n_jobs": args.n_jobs}}
    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.out_dir / "runs.csv", index=False)
    summary = summarize(frame)
    pd.DataFrame(summary).to_csv(args.out_dir / "summary.csv", index=False)
    report = {"schema": "qsar-rf-seed-sensitivity-v1", "seeds": list(SEEDS),
              "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
              "numpy_version": np.__version__, "platform": platform.platform(),
              "seed42_prediction_tolerance_atol": 1e-10, "seed42_prediction_tolerance_rtol": 0,
              "evidence": evidence, "summary": summary,
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                (Path(__file__), ROOT / "scripts/time_labels.py")},
              "limitations": ["New current-as-of five-seed study; original review's target, seeds and configuration are undocumented.",
                              "Not a reproduction or verification of the review-reported SD approximately 0.006.",
                              "Fixed selected hyperparameters; no search repeated, so variability excludes selection uncertainty.",
                              "Held-out cohort and GIN seed42/1/2 ensemble are fixed; no test-resampling uncertainty measured.",
                              "Five seed draws give a descriptive sensitivity estimate, not a confidence interval or equivalence test.",
                              "Published environment version is not inferred; seed42 prediction agreement is measured directly."]}
    (args.out_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
