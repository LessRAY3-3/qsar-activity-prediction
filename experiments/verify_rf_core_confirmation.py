"""Independently replay all 16 new RF models from a verified return package.

First use scripts/unpack_verified_return.py against the committed manifest.
Joblib files are executable serialization: use only the trusted, hash-verified
project archive. This verifies new fits, not original search identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_core_confirmation import TAGS, scores, sha, training_indices, verify_inputs  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from artifact_guards import atomic_json, check_prediction, rf_identity  # noqa: E402

HISTORICAL_ATOL = 1e-10


def check_historical_prediction(replay, historical):
    replay, historical = np.asarray(replay), np.asarray(historical)
    if (replay.ndim != 1 or replay.size == 0 or replay.shape != historical.shape
            or not np.isfinite(replay).all() or not np.isfinite(historical).all()):
        raise ValueError("invalid historical RF prediction comparison")
    maximum = float(np.max(np.abs(replay - historical)))
    if maximum > HISTORICAL_ATOL:
        raise ValueError(f"historical RF prediction mismatch: {maximum:.6g} > {HISTORICAL_ATOL}")
    return maximum


def validate_receipt(receipt, manifest_digest, env_digest, identity, expected_params):
    if receipt["status"] != "complete":
        raise ValueError("incomplete training receipt")
    if receipt["input_manifest_sha256"] != manifest_digest or receipt["environment_sha256"] != env_digest:
        raise ValueError("receipt provenance mismatch")
    if receipt["data_identity"] != identity or receipt["parameters"] != expected_params:
        raise ValueError("receipt data/parameter identity mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("report must be new")
    root = args.package
    run = root / "output"
    manifest = json.loads((root / "input_manifest.json").read_text())
    verify_inputs(root, manifest)
    manifest_digest = sha(root / "input_manifest.json")
    environment = json.loads((run / "environment.json").read_text())
    if environment["input_manifest_sha256"] != manifest_digest:
        raise ValueError("environment input manifest mismatch")
    summary = json.loads((run / "summary.json").read_text())
    expected = {f"rf_{tag}_{kind}_seed42" for tag in TAGS for kind in ("random", "scaffold")}
    if (summary["status"] != "complete" or len(summary["jobs"]) != 16
            or {r["job_id"] for r in summary["jobs"]} != expected):
        raise ValueError("incomplete or duplicate 16-fit cohort")
    if {p.name for p in run.iterdir() if p.is_dir()} != expected:
        raise ValueError("unexpected job directories")
    rows = []
    for tag in TAGS:
        with np.load(root / f"data/processed/{tag}_fingerprints.npz") as z:
            fp = dict(z)
        old_metrics = json.loads((root / f"results/metrics_{tag}.json").read_text())
        for kind in ("random", "scaffold"):
            job_id = f"rf_{tag}_{kind}_seed42"
            folder = run / job_id
            receipt = json.loads((folder / "receipt.json").read_text())
            if receipt != next(r for r in summary["jobs"] if r["job_id"] == job_id):
                raise ValueError("summary differs from receipt")
            with np.load(root / f"data/processed/splits/{tag}_{kind}.npz") as z:
                split = dict(z)
            train, test = training_indices(fp, split, kind)
            identity = rf_identity(fp["X"], fp["y"], fp["smiles"], train, test)
            params = {**old_metrics[f"best_params_{kind}"], "n_jobs": environment["jobs"]}
            validate_receipt(receipt, manifest_digest, sha(run / "environment.json"), identity, params)
            if set(receipt["output_sha256"]) != {"model.joblib", "predictions.npz"}:
                raise ValueError("incomplete output hashes")
            for name, digest in receipt["output_sha256"].items():
                if sha(folder / name) != digest:
                    raise ValueError("model/prediction digest mismatch")
            with np.load(folder / "predictions.npz") as z:
                prediction = dict(z)
            check_prediction(prediction, split, fp["y"])
            if not np.array_equal(train, prediction["train_idx"]):
                raise ValueError("saved RF training order differs")
            model = joblib.load(folder / "model.joblib")
            if model.qsar_identity_ != identity or model.get_params() != params:
                raise ValueError("saved model provenance differs")
            # Bounded single-thread coordinator inference also changes thread
            # reduction order compared with the eight-thread training host.
            model.n_jobs = 1
            replay = model.predict(fp["X"][test])
            maximum = float(np.max(np.abs(replay-prediction["y_pred"])))
            if maximum > 1e-10:
                raise ValueError("independent model replay mismatch")
            computed = scores(fp["y"][test], replay)
            for key, value in computed.items():
                if abs(value-receipt["metrics"][key]) > 1e-10:
                    raise ValueError("independent metric replay mismatch")
            with np.load(root / f"results/rf_preds_{tag}_{kind}.npz") as z:
                old = dict(z)
            check_prediction(old, split, fp["y"])
            old_scores = scores(fp["y"][test], old["y_pred"])
            historical_diff = check_historical_prediction(replay, old["y_pred"])
            rows.append({"job_id": job_id, "tag": tag, "split": kind, "passed": True,
                         "replay_max_abs_error": maximum,
                         "historical_prediction_max_abs_difference": historical_diff,
                         "metrics": computed, "historical_metrics": old_scores,
                         "r2_difference": computed["r2"]-old_scores["r2"],
                         "receipt_sha256": sha(folder / "receipt.json")})
            print(f"Verified {job_id} replay={maximum:.3g} historical={historical_diff:.3g}", flush=True)
    verify_inputs(root, manifest)
    atomic_json(args.report, {"schema": "qsar-core-rf-independent-replay-v1",
                             "status": "accepted", "jobs": rows, "n_jobs": len(rows),
                             "input_manifest_sha256": manifest_digest,
                             "training_summary_sha256": sha(run / "summary.json"),
                             "historical_prediction_tolerance": {"atol": HISTORICAL_ATOL, "rtol": 0},
                             "verifier_sha256": sha(__file__),
                             "verifier_dependencies_sha256": {
                                 "experiments/rf_core_confirmation.py": sha(Path(__file__).with_name("rf_core_confirmation.py")),
                                 "scripts/artifact_guards.py": sha(Path(__file__).resolve().parents[1] / "scripts/artifact_guards.py"),
                                 "scripts/qsar_common.py": sha(Path(__file__).resolve().parents[1] / "scripts/qsar_common.py")},
                             "environment": {"platform": platform.platform(), "python": platform.python_version(),
                                             "sklearn": sklearn.__version__, "numpy": np.__version__,
                                             "inference_threads": 1},
                             "historical_identity_recovered": False,
                             "scope": "New fixed-parameter fits, independently replayed; original searches not rerun, original fits not identified."})


if __name__ == "__main__":
    main()
