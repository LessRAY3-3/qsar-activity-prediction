"""New fixed-parameter RF confirmations; never recovery of historical searches.

Run only against a frozen input_manifest.json. Each model is saved and replayed,
with receipts binding source, input, environment, parameters and outputs. The
original predictions/metrics/models are read-only. No test-driven selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from artifact_guards import atomic_json, check_partition, check_prediction, rf_identity  # noqa: E402
from qsar_common import scaffold_split  # noqa: E402

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_inputs(root, manifest):
    expected = {"experiments/rf_core_confirmation.py", "scripts/artifact_guards.py",
                "scripts/qsar_common.py"}
    for tag in TAGS:
        expected.update({f"data/processed/{tag}_fingerprints.npz", f"results/metrics_{tag}.json"})
        for split in ("random", "scaffold"):
            expected.update({f"data/processed/splits/{tag}_{split}.npz",
                             f"results/rf_preds_{tag}_{split}.npz"})
    if set(manifest["files"]) != expected:
        raise ValueError("incomplete or unexpected frozen input cohort")
    for rel, digest in manifest["files"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen input changed: {rel}")


def training_indices(fp, split, kind):
    check_partition(len(fp["y"]), **{k: split[k] for k in ("train_idx", "valid_idx", "test_idx")})
    if not np.array_equal(fp["smiles"], split["smiles"]):
        raise ValueError("split molecule order differs from fingerprint cache")
    # Reproduce RF's full 80% training pool AND order; concatenating GNN
    # train/valid indices changes bootstrap draws even at the same RF seed.
    if kind == "random":
        train, test = train_test_split(np.arange(len(fp["y"])), test_size=.2, random_state=42)
    else:
        train, test = scaffold_split(fp["smiles"], test_size=.2)
    if not np.array_equal(test, split["test_idx"]) or not np.array_equal(
        np.sort(train), np.sort(np.concatenate([split["train_idx"], split["valid_idx"]]))
    ):
        raise ValueError("reconstructed RF pool differs from frozen split")
    return train, test


def scores(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 16:
        parser.error("jobs must be 1..16")
    manifest_path = ROOT / "input_manifest.json"
    if not manifest_path.is_file():
        parser.error(
            "missing input_manifest.json: this trainer runs inside a frozen input package, "
            "not a clean checkout. See docs/source_provenance_followup_2026-10-15.md; "
            "to verify existing fits, use verify_rf_core_confirmation.py --package."
        )
    manifest = json.loads(manifest_path.read_text())
    verify_inputs(ROOT, manifest)
    out = args.out_dir.resolve()
    # An exclusive new directory prevents mixing attempts or resuming without
    # proving the identity of a previously completed receipt.
    out.mkdir(parents=True, exist_ok=False)
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "machine": platform.machine(),
                   "sklearn": sklearn.__version__, "numpy": np.__version__,
                   "joblib": joblib.__version__, "started_utc": utc(),
                   "jobs": args.jobs, "input_manifest_sha256": sha(ROOT / "input_manifest.json")}
    atomic_json(out / "environment.json", environment)
    receipts = []
    for tag in TAGS:
        with np.load(ROOT / f"data/processed/{tag}_fingerprints.npz") as z:
            fp = dict(z)
        metrics = json.loads((ROOT / f"results/metrics_{tag}.json").read_text())
        for kind in ("random", "scaffold"):
            job_id = f"rf_{tag}_{kind}_seed42"
            folder = out / job_id
            folder.mkdir()
            with np.load(ROOT / f"data/processed/splits/{tag}_{kind}.npz") as z:
                split = dict(z)
            with np.load(ROOT / f"results/rf_preds_{tag}_{kind}.npz") as z:
                original = dict(z)
            train, test = training_indices(fp, split, kind)
            check_prediction(original, split, fp["y"])
            published = metrics[f"best_params_{kind}"]
            if published["random_state"] != 42:
                raise ValueError("unexpected published seed")
            params = {**published, "n_jobs": args.jobs}
            receipt = {"job_id": job_id, "started_utc": utc(), "status": "running",
                       "parameters": params, "published_parameters": published,
                       "input_manifest_sha256": environment["input_manifest_sha256"],
                       "environment_sha256": sha(out / "environment.json"),
                       "data_identity": rf_identity(fp["X"], fp["y"], fp["smiles"], train, test),
                       "n_train": len(train), "n_test": len(test)}
            atomic_json(folder / "receipt.json", receipt)
            start = time.monotonic()
            print(f"{utc()} START {job_id} n_train={len(train)} params={params}", flush=True)
            model = RandomForestRegressor(**params).fit(fp["X"][train], fp["y"][train])
            model.qsar_identity_ = receipt["data_identity"]
            pred = model.predict(fp["X"][test])
            joblib.dump(model, folder / "model.joblib", compress=3)
            loaded = joblib.load(folder / "model.joblib")
            replay = loaded.predict(fp["X"][test])
            replay_error = float(np.max(np.abs(replay - pred)))
            if replay_error > 1e-10:
                raise ValueError("saved model replay differs")
            np.savez_compressed(folder / "predictions.npz", test_idx=test, train_idx=train,
                                y_true=fp["y"][test], y_pred=pred)
            result = scores(fp["y"][test], pred)
            receipt.update(status="complete", ended_utc=utc(), elapsed_seconds=time.monotonic()-start,
                           metrics=result, historical_metrics=scores(fp["y"][test], original["y_pred"]),
                           saved_model_replay_max_abs_error=replay_error,
                           historical_prediction_max_abs_difference=float(np.max(np.abs(pred-original["y_pred"]))),
                           output_sha256={name: sha(folder / name) for name in ("model.joblib", "predictions.npz")})
            atomic_json(folder / "receipt.json", receipt)
            receipts.append(receipt)
            print(f"{utc()} COMPLETE {job_id} r2={result['r2']:.9f} historical_prediction_max_abs_difference={receipt['historical_prediction_max_abs_difference']:.3g}", flush=True)
    verify_inputs(ROOT, manifest)
    atomic_json(out / "summary.json", {
        "schema": "qsar-core-rf-confirmation-v1", "status": "complete", "jobs": receipts,
        "completed_utc": utc(), "environment": environment,
        "interpretation": "New fixed-published-parameter fits. No hyperparameter search rerun, no historical identity recovered, no main results replaced.",
        "historical_identity_recovered": False,
    })


if __name__ == "__main__":
    main()
