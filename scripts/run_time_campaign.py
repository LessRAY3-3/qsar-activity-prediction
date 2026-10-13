"""Serial, resumable temporal workers. Dry-run is the default; no SSH or split generation.

Use one RF worker on the CPU host and one GIN worker on the CUDA host, each
with an identical frozen input bundle. Successful jobs are skipped only when
provenance and both canonical and archived artifact hashes still match.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

from time_labels import load_time_split, require_prediction_version

ROOT = Path(__file__).resolve().parent.parent
TAGS = ("a2a", "egfr_full", "abl1", "egfr", "herg", "hivpr", "mpro", "vegfr2")
SEEDS = (42, 1, 2)
ENV_KEYS = ("QSAR_RF_JOBS", "QSAR_CV_BACKEND", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def worker_lock(root, role):
    """Never steal or automatically clear another worker's lock."""
    path = Path(root) / f".time_campaign_{role}.lock"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Worker lock exists: {path}; inspect its owner before manual removal") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "role": role}, stream)
        yield
    finally:
        path.unlink()


def git_info(root):
    def read(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return "unavailable"
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"revision": read("rev-parse", "HEAD"),
            "dirty_status": read("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(read("diff", "HEAD", "--").encode()).hexdigest()}


def environment():
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "torch-geometric", "rdkit", "joblib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "variables": {key: os.environ.get(key) for key in ENV_KEYS}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("rf", "gin"))
    parser.add_argument("--tags", nargs="+", choices=TAGS, default=list(TAGS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "results" / "time_campaign")
    parser.add_argument("--execute", action="store_true", help="actually train; otherwise read-only plan")
    parser.add_argument("--rerun", action="store_true", help="allow failed or incompatible jobs to run again")
    return parser.parse_args(argv)


def build_jobs(args, root=ROOT):
    root = Path(root)
    env = environment()
    sources = ["scripts/run_time_campaign.py", "scripts/time_labels.py", "scripts/qsar_common.py"]
    sources += (["scripts/04_train_and_evaluate.py"] if args.role == "rf" else
                ["scripts/gnn_03_train_gin.py", "scripts/gnn_graph_dataset.py"])
    code = {name: digest(root / name) for name in sources}
    bundle = None
    if (root / "manifest.json").is_file():
        manifest = json.loads((root / "manifest.json").read_text())
        bundle = {"sha256": digest(root / "manifest.json"),
                  "source_git_head": manifest.get("source", {}).get("git_head")}
    jobs = []
    for tag in dict.fromkeys(args.tags):
        split_dir = root / "data" / "processed" / "splits"
        split = load_time_split(split_dir, tag)
        if split is None:
            raise ValueError(f"{tag}: NA marker present; campaign refuses stale outputs")
        cache_name = f"{tag}_{'fingerprints' if args.role == 'rf' else 'graphs'}.npz"
        cache_path = root / "data" / "processed" / cache_name
        with np.load(cache_path, allow_pickle=False) as cache:
            load_time_split(split_dir, tag, cache["smiles"], cache["y"])
        inputs = {str(p.relative_to(root)): digest(p) for p in
                  (split_dir / f"{tag}_time.npz", cache_path)}
        for seed in (SEEDS[:1] if args.role == "rf" else dict.fromkeys(args.seeds)):
            suffix = "" if seed == 42 else f"_seed{seed}"
            job_id = f"{args.role}_{tag}_seed{seed}"
            if args.role == "rf":
                command = [sys.executable, "scripts/04_train_and_evaluate.py", "--split", "time"]
                files = [f"models/rf_time_split_{tag}.joblib", f"results/rf_preds_{tag}_time.npz",
                         f"results/metrics_{tag}.json", f"figures/pred_vs_actual_time_{tag}.png"]
                params = {"seed": 42, "cv_folds": 5, "grid": "PARAM_GRID in hashed RF source",
                          "device": "cpu"}
            else:
                command = [sys.executable, "scripts/gnn_03_train_gin.py", "--tag", tag,
                           "--split", "time", "--model", "gin", "--seed", str(seed),
                           "--hidden", "128", "--num-layers", "4", "--dropout", "0.2",
                           "--lr", "0.001", "--batch-size", "128", "--epochs", "100",
                           "--patience", "20", "--min-epochs", "0", "--device", args.device]
                if suffix:
                    command += ["--suffix", suffix]
                files = [f"results/gnn_models/{tag}_gin_time{suffix}.pt",
                         f"results/gnn_preds_{tag}_time{suffix}.npz",
                         f"results/gnn_metrics_{tag}{suffix}.json"]
                params = {"seed": seed, "hidden": 128, "num_layers": 4, "dropout": 0.2,
                          "lr": 0.001, "batch_size": 128, "epochs": 100, "patience": 20,
                          "min_epochs": 0, "device": args.device}
            provenance = {"schema": 1, "role": args.role, "tag": tag, "label_id": str(split["label_id"]),
                          "inputs": inputs, "code": code, "hyperparameters": params, "environment": env,
                          "bundle_manifest": bundle,
                          # Python location is host-specific; package/Python versions above are what matters.
                          "command": command[1:]}
            jobs.append({"id": job_id, "command": command, "files": files, "provenance": provenance,
                         "signature": json_digest(provenance), "split": split})
    return jobs


def can_resume(status, job, root, directory):
    if status.get("state") != "success" or status.get("signature") != job["signature"]:
        return False
    hashes = status.get("artifacts", {})
    if set(hashes) != set(job["files"]):
        return False
    for name, expected in hashes.items():
        for path in (Path(root) / name, Path(directory) / status["snapshot"] / name):
            if not path.is_file() or digest(path) != expected:
                return False
    return True


def build_checkpoint_model(root, parameters):
    """Build the hashed GIN architecture on CPU to validate checkpoint compatibility."""
    spec = importlib.util.spec_from_file_location("campaign_gin_trainer", Path(root) / "scripts/gnn_03_train_gin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model(argparse.Namespace(**parameters))


def validate_outputs(job, root):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    for name in job["files"]:
        if not (Path(root) / name).is_file() or (Path(root) / name).stat().st_size == 0:
            raise ValueError(f"Training exited without required nonempty output: {name}")
    prediction = next(name for name in job["files"] if "_preds_" in name)
    with np.load(Path(root) / prediction) as archive:
        pred = dict(archive)
    require_prediction_version(pred, job["split"])
    if np.asarray(pred["y_pred"]).shape != np.asarray(pred["y_true"]).shape or not np.isfinite(pred["y_pred"]).all():
        raise ValueError("Invalid prediction shape or non-finite predictions")
    metrics = next(name for name in job["files"] if "metrics_" in name)
    payload = json.loads((Path(root) / metrics).read_text())
    data = payload["time"]
    label_id = job["provenance"]["label_id"]
    if data.get("label_id") != label_id:
        raise ValueError("Metric label_id differs from frozen split")
    expected_metrics = {"r2": r2_score(pred["y_true"], pred["y_pred"]),
                        "rmse": np.sqrt(mean_squared_error(pred["y_true"], pred["y_pred"])),
                        "mae": mean_absolute_error(pred["y_true"], pred["y_pred"])}
    for key, expected in expected_metrics.items():
        if not np.isfinite(expected) or not np.isclose(data.get(key, np.nan), expected, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Stored {key} does not match predictions")
    parameters = job["provenance"]["hyperparameters"]
    if data.get("runtime", {}).get("seed") != parameters["seed"]:
        raise ValueError("Metric runtime seed differs from requested seed")
    if job["provenance"]["role"] == "gin":
        import torch

        if data.get("device") != parameters["device"]:
            raise ValueError("Actual GIN device differs from requested device")
        checkpoint_path = next(Path(root) / name for name in job["files"] if name.endswith(".pt"))
        # Only load the checkpoint produced by this local job, never untrusted downloads.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("label_id") != label_id:
            raise ValueError("GIN checkpoint label_id differs from frozen split")
        expected_args = {**parameters, "tag": job["provenance"]["tag"], "model": "gin", "split": "time",
                         "suffix": "" if parameters["seed"] == 42 else f"_seed{parameters['seed']}"}
        for key, expected in expected_args.items():
            if checkpoint.get("args", {}).get(key) != expected:
                raise ValueError(f"GIN checkpoint argument mismatch: {key}")
            if key != "split" and data.get("params", {}).get(key) != expected:
                raise ValueError(f"GIN metric parameter mismatch: {key}")
        runtime = checkpoint.get("runtime", {})
        if any(runtime.get(key) != data.get("runtime", {}).get(key) for key in ("seed", "device", "requested_device")):
            raise ValueError("GIN checkpoint runtime differs from metrics")
        if runtime.get("seed") != parameters["seed"] or runtime.get("device") != parameters["device"]:
            raise ValueError("GIN checkpoint runtime differs from requested configuration")
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict) or not state or any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("GIN checkpoint has empty or invalid weights")
        for key in ("y_mean", "y_std"):
            if not np.isfinite(checkpoint.get(key, np.nan)):
                raise ValueError(f"GIN checkpoint has invalid {key}")
        if checkpoint["y_std"] <= 0:
            raise ValueError("GIN checkpoint label scaling must be positive")
        model = build_checkpoint_model(root, checkpoint["args"])
        model.load_state_dict(state, strict=True)
    else:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.utils.validation import check_is_fitted

        model_path = next(Path(root) / name for name in job["files"] if name.endswith(".joblib"))
        model = joblib.load(model_path)
        if not isinstance(model, RandomForestRegressor):
            raise ValueError("RF checkpoint is not a RandomForestRegressor")
        check_is_fitted(model)
        if not getattr(model, "estimators_", None):
            raise ValueError("RF checkpoint has no fitted trees")
        if getattr(model, "time_label_id_", None) != label_id or model.random_state != parameters["seed"]:
            raise ValueError("RF checkpoint label_id or seed differs from requested configuration")
        if model.get_params() != payload.get("best_params_time"):
            raise ValueError("RF checkpoint parameters differ from metrics")


def run_job(job, args, root=ROOT):
    root = Path(root)
    directory = args.campaign_dir.resolve() / job["id"]
    status_path = directory / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if can_resume(previous, job, root, directory):
            print(f"SKIP verified {job['id']}", flush=True)
            return
        if not args.rerun:
            raise RuntimeError(f"{job['id']}: prior status/artifacts incompatible; inspect and use --rerun")
    directory.mkdir(parents=True, exist_ok=True)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    attempt_dir = directory / attempt
    attempt_dir.mkdir()
    if status_path.exists():
        shutil.copy2(status_path, attempt_dir / "previous_status.json")
    # Preserve all old files before training may replace any of them.
    for name in job["files"]:
        path = root / name
        if path.exists():
            destination = attempt_dir / "before" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    state = {"state": "running", "signature": job["signature"], "provenance": job["provenance"],
             "command": job["command"], "git": git_info(root), "host": socket.gethostname(),
             "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "log": f"{attempt}/train.log", "snapshot": f"{attempt}/outputs"}
    atomic_json(status_path, state)
    start = time.monotonic()
    try:
        env = os.environ.copy()
        env.update({"QSAR_TAG": job["provenance"]["tag"], "PYTHONUNBUFFERED": "1"})
        with (attempt_dir / "train.log").open("w") as stream:
            subprocess.run(job["command"], cwd=root, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        # Inputs/source must not have changed while a long job was running.
        for name, expected in {**job["provenance"]["inputs"], **job["provenance"]["code"]}.items():
            if digest(root / name) != expected:
                raise ValueError(f"Input/source changed during training: {name}")
        bundle = job["provenance"].get("bundle_manifest")
        if bundle is not None and digest(root / "manifest.json") != bundle["sha256"]:
            raise ValueError("Bundle manifest changed during training")
        validate_outputs(job, root)
        hashes = {}
        for name in job["files"]:
            destination = directory / state["snapshot"] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
            hashes[name] = digest(destination)
        state.update(state="success", artifacts=hashes)
    except BaseException as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic() - start,
                     finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, state)
    print(f"DONE {job['id']} ({state['elapsed_seconds']:.1f}s)", flush=True)


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", "jobs": [
        {key: value for key, value in job.items() if key != "split"} for job in jobs]}, indent=2))
    if not args.execute:
        return
    with worker_lock(ROOT, args.role):
        for job in jobs:
            run_job(job, args)


if __name__ == "__main__":
    main()
