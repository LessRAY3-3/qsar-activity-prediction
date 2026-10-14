"""Independently replay the complete 18-fit CUDA reconstruction on CPU.

Inputs are a verified returned reconstruction directory, the ORIGINAL plan and
the EXACT outer runner shipped with it. This is inference only: no fitting,
CUDA, historical overwrite, or claim of recovered historical run identity.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader


EVIDENCE_FILES = {"job.json", "execution_files.json", "environment.json", "stdout.log",
                  "stderr.log", "status.json", "acceptance.json"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def frozen_imports(workspace, source_files):
    """Avoid a previously imported checkout shadowing the frozen sources."""
    names = {Path(name).stem for name in source_files if name.startswith("scripts/")}
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = list(sys.path)
    sys.path.insert(0, str(workspace / "scripts"))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def expected_job_ids():
    return {f"{model}_{split}_seed{seed}" for model in ("egfr_gin", "mt", "pooled")
            for split in ("random", "scaffold") for seed in (42, 1, 2)}


def verify_receipts(run_dir, plan_path, runner_path, receipt_threads=4):
    plan = json.loads(plan_path.read_text())
    ids = [job["id"] for job in plan["jobs"]]
    if len(ids) != 18 or set(ids) != expected_job_ids() or plan.get("new_backend") != "cuda":
        raise ValueError("Expected the complete independent 18-job CUDA reconstruction cohort")
    runner = load_module(runner_path, "returned_provenance_runner")
    identity = {"plan_sha256": sha256(plan_path), "runner_sha256": sha256(runner_path),
                "threads": receipt_threads, "backend": "cuda"}
    actual = {p.parent.name for p in run_dir.glob("*/complete.json")}
    if actual != set(ids):
        raise ValueError(f"Completed receipt cohort differs: missing={set(ids)-actual}, extra={actual-set(ids)}")
    verified = []
    for job in plan["jobs"]:
        job_dir = runner.safe_path(run_dir, job["id"])
        if not runner.resume_valid(job_dir, identity):
            raise ValueError(f"Incomplete job: {job['id']}")
        receipt = json.loads((job_dir / "complete.json").read_text())
        attempt = runner.safe_path(job_dir, receipt["attempt"])
        workspace = runner.safe_path(attempt, "workspace")
        files = {**plan["source_sha256"],
                 **{name: plan["input_sha256"][name] for name in job["inputs"]}}
        if receipt["execution_files_sha256"] != files:
            raise ValueError(f"Execution manifest differs from frozen plan: {job['id']}")
        if set(receipt["evidence_sha256"]) != EVIDENCE_FILES:
            raise ValueError(f"Incomplete evidence manifest: {job['id']}")
        if set(receipt["outputs_sha256"]) != set(runner.artifact_names(job)):
            raise ValueError(f"Incomplete output manifest: {job['id']}")
        if json.loads((attempt / "job.json").read_text()) != job:
            raise ValueError(f"Job payload differs from plan: {job['id']}")
        if json.loads((attempt / "execution_files.json").read_text()) != files:
            raise ValueError(f"Execution evidence differs from plan: {job['id']}")
        status = json.loads((attempt / "status.json").read_text())
        if (status.get("status") != "complete" or status.get("exit_code") != 0
                or status.get("identity") != identity or status.get("argv") != job["argv"]
                or status.get("job_id") != job["id"] or not status.get("start_utc")
                or not status.get("end_utc")):
            raise ValueError(f"Run status identity/completion mismatch: {job['id']}")
        env = json.loads((attempt / "environment.json").read_text())
        if (env.get("device") != "cuda" or not env.get("device_name")
                or env.get("torch_threads") != receipt_threads or not env.get("cuda_runtime")):
            raise ValueError(f"Recorded CUDA environment mismatch: {job['id']}")
        accepted = json.loads((attempt / "acceptance.json").read_text())
        for name in ("outputs_sha256", "comparisons", "checkpoint_validation"):
            if accepted[name] != receipt[name]:
                raise ValueError(f"Receipt/acceptance mismatch: {job['id']}/{name}")
        verified.append({"job": job, "workspace": workspace, "attempt": attempt,
                         "receipt_sha256": sha256(job_dir / "complete.json"), "environment": env})
    return plan, runner, identity, verified


def verify_normalization(workspace, job, checkpoint):
    means, stds = [], []
    for tag in job["tags"]:
        with np.load(workspace / f"data/processed/{tag}_graphs.npz", allow_pickle=False) as graphs:
            y = graphs["y"]
        with np.load(workspace / f"data/processed/splits/{tag}_{job['split']}.npz", allow_pickle=False) as split:
            training = y[split["train_idx"]]
        means.append(float(training.mean()))
        stds.append(float(training.std()))
    for key, expected in (("y_mean", means), ("y_std", stds)):
        observed = np.asarray(checkpoint[key]).reshape(-1)
        if observed.shape != (len(expected),) or not np.allclose(observed, expected, rtol=1e-7, atol=1e-7):
            raise ValueError(f"Checkpoint {key} differs from frozen training labels")
    return means, stds


def metrics(truth, pred):
    return {"r2": float(r2_score(truth, pred)),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "mae": float(mean_absolute_error(truth, pred))}


def replay_job(workspace, job, runner, source_files, pred_atol=1e-4, pred_rtol=1e-5,
               metric_atol=1e-5, metric_rtol=1e-5):
    names = runner.artifact_names(job)
    with frozen_imports(workspace, source_files):
        runner.validate_checkpoint(workspace, job, workspace / names[2])
        saved = torch.load(workspace / names[2], map_location="cpu", weights_only=False)
        means, stds = verify_normalization(workspace, job, saved)
        module = load_module(workspace / job["argv"][0], "frozen_replay_trainer")
        single = job["argv"][0].endswith("gnn_03_train_gin.py")
        params = saved["args"]
        if single:
            model = module.build_model(argparse.Namespace(**params))
            pooled = None
        else:
            model = module.PooledGIN(len(job["tags"]), shared_head=params["model"] == "pooled",
                                     hidden=params["hidden"], num_layers=params["num_layers"], dropout=params["dropout"])
            pooled = module.load_pool(job["tags"])
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to("cpu").eval()
        stored_metrics = json.loads((workspace / names[0]).read_text())[job["split"]]
        rows, arrays = [], {}
        with np.load(workspace / names[1], allow_pickle=False) as predictions:
            for task, tag in enumerate(job["tags"]):
                with np.load(workspace / f"data/processed/splits/{tag}_{job['split']}.npz", allow_pickle=False) as split:
                    indices = split["test_idx"]
                with np.load(workspace / f"data/processed/{tag}_graphs.npz", allow_pickle=False) as graphs:
                    y_all = graphs["y"]
                if single:
                    dataset = module.MoleculeGraphDataset(workspace / f"data/processed/{tag}_graphs.npz", indices=indices)
                    dataset.y = (y_all.copy() - means[task]) / stds[task]
                else:
                    dataset = module.PooledGraphDataset(pooled, indices=indices + pooled["offsets"][task],
                                                        y_mean=np.asarray(means), y_std=np.asarray(stds))
                loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
                output, labels = [], []
                with torch.no_grad():
                    for batch in loader:
                        output.append((model(batch) if single else model(batch, task)).detach().cpu().numpy())
                        labels.append(batch.y.detach().cpu().numpy())
                if not output:
                    raise ValueError(f"Empty test pool: {job['id']}/{tag}")
                replay_pred = np.concatenate(output) * stds[task] + means[task]
                replay_truth = np.concatenate(labels) * stds[task] + means[task]
                key_suffix = "" if single else f"_{tag}"
                stored_idx, truth, pred = [predictions[key + key_suffix] for key in ("test_idx", "y_true", "y_pred")]
                if not np.array_equal(stored_idx, indices):
                    raise ValueError(f"Test membership mismatch: {job['id']}/{tag}")
                for name, observed in (("predictions", replay_pred), ("labels", replay_truth)):
                    if observed.shape != truth.shape or not np.isfinite(observed).all():
                        raise ValueError(f"Invalid CPU {name}: {job['id']}/{tag}")
                if (not np.isfinite(pred).all() or not np.isfinite(truth).all()
                        or not np.allclose(replay_truth, truth, rtol=1e-5, atol=1e-5)
                        or not np.allclose(truth, y_all[indices], rtol=1e-5, atol=1e-5)):
                    raise ValueError(f"Label integrity mismatch: {job['id']}/{tag}")
                cpu = metrics(replay_truth, replay_pred)
                original = metrics(truth, pred)
                stored = stored_metrics if single else stored_metrics[tag]
                prediction_ok = bool(np.allclose(replay_pred, pred, rtol=pred_rtol, atol=pred_atol))
                saved_metric_ok = all(np.isclose(original[key], stored[key], rtol=1e-5, atol=1e-5) for key in cpu)
                metric_ok = all(np.isclose(cpu[key], stored[key], rtol=metric_rtol, atol=metric_atol) for key in cpu)
                historical = job["historical_r2"] if single else job["historical_r2"][tag]
                row = {"job_id": job["id"], "tag": tag, "split": job["split"], "seed": job["seed"],
                       "n_test": len(indices), "passed": prediction_ok and saved_metric_ok and metric_ok,
                       "predictions_within_tolerance": prediction_ok, "stored_metrics_match_predictions": bool(saved_metric_ok),
                       "cpu_metrics_within_tolerance": bool(metric_ok),
                       "max_prediction_abs_error": float(np.max(np.abs(replay_pred - pred))),
                       "max_label_abs_error": float(np.max(np.abs(replay_truth - truth))),
                       "historical_r2": historical, "saved_r2_minus_historical": original["r2"] - historical,
                       **{"cpu_" + key: value for key, value in cpu.items()},
                       **{"saved_" + key: value for key, value in original.items()}}
                rows.append(row)
                arrays.update({f"test_idx_{tag}": indices, f"y_true_{tag}": replay_truth, f"y_pred_{tag}": replay_pred})
        return rows, arrays


def build_report(run_dir, plan_path, runner_path, out_dir, cpu_threads=4, receipt_threads=4,
                 pred_atol=1e-4, pred_rtol=1e-5, metric_atol=1e-5, metric_rtol=1e-5):
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError("Report output must be a new directory")
    if out_dir.resolve().is_relative_to(run_dir.resolve()) or run_dir.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError("Report output must be separate from returned evidence")
    if cpu_threads < 1 or receipt_threads < 1 or any(not np.isfinite(v) or v < 0 for v in (pred_atol, pred_rtol, metric_atol, metric_rtol)):
        raise ValueError("Invalid threads or tolerances")
    # Verify all 18 receipts before importing any checkpoint or writing output.
    plan, runner, identity, verified = verify_receipts(run_dir, plan_path, runner_path, receipt_threads)
    torch.set_num_threads(cpu_threads)
    out_dir.mkdir(parents=True)
    (out_dir / "predictions").mkdir()
    report = {"schema": "qsar-provenance-cpu-replay-v1", "status": "running",
              "start_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False,
              "interpretation": "Independent reconstruction and CPU inference verification; historical executable, environment and logs remain unknown. Matching scores do not establish historical run identity.",
              "identity": identity, "report_source_sha256": sha256(__file__),
              "tolerance": {"prediction_atol": pred_atol, "prediction_rtol": pred_rtol,
                            "metric_atol": metric_atol, "metric_rtol": metric_rtol},
              "environment": {"device": "cpu", "threads": cpu_threads, "python": sys.version,
                              "platform": platform.platform(),
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ("torch", "torch-geometric", "numpy", "scikit-learn", "rdkit")}},
              "jobs_expected": 18, "jobs": [], "errors": []}
    rows = []
    for entry in verified:
        job = entry["job"]
        print(f"CPU replay: {job['id']}", flush=True)
        try:
            job_rows, arrays = replay_job(entry["workspace"], job, runner, plan["source_sha256"],
                                          pred_atol, pred_rtol, metric_atol, metric_rtol)
            rows.extend(job_rows)
            prediction_path = out_dir / "predictions" / (job["id"] + ".npz")
            np.savez_compressed(prediction_path, **arrays)
            report["jobs"].append({"id": job["id"], "passed": all(row["passed"] for row in job_rows),
                                   "receipt_sha256": entry["receipt_sha256"],
                                   "prediction_sha256": sha256(prediction_path),
                                   "training_environment": entry["environment"]})
        except Exception as error:
            report["errors"].append({"job_id": job["id"], "error": f"{type(error).__name__}: {error}"})
    if rows:
        with (out_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    try:
        _, _, _, after = verify_receipts(run_dir, plan_path, runner_path, receipt_threads)
        if [entry["receipt_sha256"] for entry in after] != [entry["receipt_sha256"] for entry in verified]:
            raise ValueError("Receipts changed during CPU replay")
    except Exception as error:
        report["errors"].append({"stage": "post_replay_integrity", "error": f"{type(error).__name__}: {error}"})
    passed = (len(report["jobs"]) == 18 and len(rows) == 42 and not report["errors"]
              and all(j["passed"] for j in report["jobs"]))
    report.update(status="accepted_independent_cpu_replay" if passed else "rejected",
                  rows_checked=len(rows), rows_passed=sum(row["passed"] for row in rows),
                  end_utc=datetime.now(timezone.utc).isoformat())
    report["output_sha256"] = {str(p.relative_to(out_dir)): sha256(p) for p in sorted(out_dir.rglob("*")) if p.is_file()}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "plan", "runner", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--receipt-threads", type=int, default=4)
    parser.add_argument("--prediction-atol", type=float, default=1e-4)
    parser.add_argument("--prediction-rtol", type=float, default=1e-5)
    parser.add_argument("--metric-atol", type=float, default=1e-5)
    parser.add_argument("--metric-rtol", type=float, default=1e-5)
    args = parser.parse_args()
    report = build_report(args.run_dir.resolve(), args.plan.resolve(), args.runner.resolve(), args.out_dir.absolute(),
                          args.cpu_threads, args.receipt_threads, args.prediction_atol, args.prediction_rtol,
                          args.metric_atol, args.metric_rtol)
    print(json.dumps({"status": report["status"], "jobs": len(report["jobs"]),
                      "rows": report["rows_checked"], "report": str(args.out_dir / "report.json")}))
    if report["status"] != "accepted_independent_cpu_replay":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
