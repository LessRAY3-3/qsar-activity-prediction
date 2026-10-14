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
from pathlib import Path, PurePosixPath
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
ENV_KEYS = ("QSAR_RF_JOBS", "QSAR_CV_BACKEND", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
            "QSAR_TORCH_THREADS", "QSAR_TORCH_INTEROP_THREADS")


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


def reclaim_dead_lock(path):
    """Explicit local recovery only; never reclaim a live/foreign/unknown owner."""
    import fcntl

    old = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = path.stat()
        original = os.fstat(old)
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise RuntimeError("Worker lock changed while recovery was requested")
        owner = json.loads(os.read(old, 65536))
        if owner.get("host") != socket.gethostname() or type(owner.get("pid")) is not int or owner["pid"] < 1:
            raise RuntimeError("Cannot recover a foreign or invalid worker lock")
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Cannot recover a worker lock whose owner is still alive")
        path.unlink()
        # O_EXCL still protects a normal worker that wins the unlink/create gap.
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    finally:
        os.close(old)


@contextmanager
def worker_lock(root, role, recover=False):
    """Reclaim a confirmed dead local owner only with explicit recovery."""
    path = Path(root) / f".time_campaign_{role}.lock"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        if not recover:
            raise RuntimeError(f"Worker lock exists: {path}; inspect its owner before manual removal") from exc
        fd = reclaim_dead_lock(path)
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
    top = read("rev-parse", "--show-toplevel")
    if top == "unavailable" or Path(top).resolve() != Path(root).resolve():
        return {"revision": "unavailable", "dirty_status": "unavailable", "diff_sha256": None,
                "reason": "root is not a Git worktree root"}
    return {"revision": read("rev-parse", "HEAD"),
            "dirty_status": read("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(read("diff", "HEAD", "--").encode()).hexdigest()}


def verified_bundle(root, role):
    from time_campaign_bundle import verify_manifest

    path = Path(root) / "manifest.json"
    if not path.exists():
        return None
    manifest = verify_manifest(json.loads(path.read_text()), root)
    if manifest["role"] != role:
        raise ValueError("Bundle role differs from the requested worker role")
    return manifest


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
    parser.add_argument("--epochs", type=int, default=100, help="GIN maximum training epochs")
    parser.add_argument("--patience", type=int, default=20, help="GIN early-stopping patience")
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="GIN minimum epochs before early stopping is allowed")
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "results" / "time_campaign")
    parser.add_argument("--execute", action="store_true", help="actually train; otherwise read-only plan")
    parser.add_argument("--rerun", action="store_true", help="allow failed or incompatible jobs to run again")
    parser.add_argument("--recover-interrupted", action="store_true",
                        help="with --execute, recover publication and a dead local lock only; never train")
    args = parser.parse_args(argv)
    if args.recover_interrupted and args.rerun:
        parser.error("recovery is separate from training; use --execute --rerun in a later invocation")
    if args.epochs < 1 or args.patience < 1 or not 0 <= args.min_epochs <= args.epochs:
        parser.error("training budget requires --epochs >= 1, --patience >= 1, "
                     "and 0 <= --min-epochs <= --epochs")
    return args


def build_jobs(args, root=ROOT):
    root = Path(root)
    env = environment()
    sources = ["scripts/run_time_campaign.py", "scripts/time_labels.py", "scripts/qsar_common.py",
               "scripts/time_campaign_bundle.py"]
    sources += (["scripts/04_train_and_evaluate.py"] if args.role == "rf" else
                ["scripts/gnn_03_train_gin.py", "scripts/gnn_graph_dataset.py"])
    code = {name: digest(root / name) for name in sources}
    bundle = None
    if (root / "manifest.json").is_file():
        manifest = verified_bundle(root, args.role)
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
                           "--lr", "0.001", "--batch-size", "128", "--epochs", str(args.epochs),
                           "--patience", str(args.patience), "--min-epochs", str(args.min_epochs),
                           "--device", args.device]
                if suffix:
                    command += ["--suffix", suffix]
                files = [f"results/gnn_models/{tag}_gin_time{suffix}.pt",
                         f"results/gnn_preds_{tag}_time{suffix}.npz",
                         f"results/gnn_metrics_{tag}{suffix}.json"]
                params = {"seed": seed, "hidden": 128, "num_layers": 4, "dropout": 0.2,
                          "lr": 0.001, "batch_size": 128, "epochs": args.epochs, "patience": args.patience,
                          "min_epochs": args.min_epochs, "device": args.device}
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
        train_y = np.asarray(job["split"]["y"])[job["split"]["train_idx"]]
        if checkpoint["y_mean"] != float(train_y.mean()) or checkpoint["y_std"] != float(train_y.std()):
            raise ValueError("GIN checkpoint normalization differs from frozen training labels")
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


def validate_job_inputs(job, root, full_bundle=True):
    """Recheck a built plan, including a later NA marker, before trusting it."""
    root = Path(root)
    marker = root / "data/processed/splits" / f"{job['provenance']['tag']}_time_NA.json"
    if marker.exists():
        raise ValueError(f"{job['id']}: NA marker present; campaign refuses stale outputs")
    for name, expected in {**job["provenance"]["inputs"], **job["provenance"]["code"]}.items():
        if not (root / name).is_file() or digest(root / name) != expected:
            raise ValueError(f"Input/source changed since job was planned: {name}")
    bundle = job["provenance"].get("bundle_manifest")
    if bundle is not None and (not (root / "manifest.json").is_file()
                               or digest(root / "manifest.json") != bundle["sha256"]):
        raise ValueError("Bundle manifest changed since job was planned")
    if bundle is not None and full_bundle:
        verified_bundle(root, job["provenance"]["role"])


def transaction_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid publication path: {name}")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Publication path escapes root: {name}")
    return path


def restore_publication(journal_path, root, allow_commit=True):
    """Idempotently recover a process killed between per-file replacements.

    The successful status write is the commit point. SIGTERM/SIGKILL cannot
    guarantee immediate rollback; this journal remains for a later invocation.
    """
    journal = json.loads(journal_path.read_text())
    if journal["state"] in ("committed", "rolled_back"):
        return
    if journal["schema"] != 1 or journal["root"] != str(Path(root).resolve()):
        raise ValueError("Publication journal belongs to a different root")
    attempt = journal_path.parent
    status_path = attempt.parent / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("snapshot") != str(attempt.relative_to(attempt.parent) / "outputs"):
        raise ValueError("Publication journal does not belong to current job status")
    entries = journal["entries"]
    # Verify every recoverable image before changing any canonical file.
    for name, entry in entries.items():
        canonical = transaction_path(root, name)
        snapshot = transaction_path(attempt / "outputs", name)
        before = transaction_path(attempt / "before", name)
        if digest(snapshot) != entry["new_sha256"]:
            raise ValueError(f"Publication snapshot changed: {name}")
        if entry["before_sha256"] is not None and digest(before) != entry["before_sha256"]:
            raise ValueError(f"Publication backup changed: {name}")
        current = digest(canonical) if canonical.exists() else None
        if current not in (None, entry["before_sha256"], entry["new_sha256"]):
            raise ValueError(f"Canonical file changed outside interrupted publication: {name}")
    committed = (allow_commit and status.get("state") == "success"
                 and status.get("artifacts") == {name: e["new_sha256"] for name, e in entries.items()}
                 and all((Path(root) / name).is_file() and digest(Path(root) / name) == e["new_sha256"]
                         for name, e in entries.items()))
    if not committed:
        for name, entry in entries.items():
            destination = transaction_path(root, name)
            if entry["before_sha256"] is None:
                destination.unlink(missing_ok=True)
            elif not destination.is_file() or digest(destination) != entry["before_sha256"]:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + "." + attempt.name + ".restore.tmp")
                shutil.copy2(transaction_path(attempt / "before", name), temporary)
                os.replace(temporary, destination)
        status.update(state="failed", error="Recovered interrupted publication; old canonical artifacts restored")
        atomic_json(status_path, status)
    for name in entries:
        destination = transaction_path(root, name)
        for suffix in (".publish.tmp", ".restore.tmp"):
            destination.with_name(destination.name + "." + attempt.name + suffix).unlink(missing_ok=True)
    journal["state"] = "committed" if committed else "rolled_back"
    atomic_json(journal_path, journal)


def recover_publications(campaign_dir, root, role):
    for path in sorted(Path(campaign_dir).glob(f"{role}_*/*/publication.json")):
        restore_publication(path, root)


def run_job(job, args, root=ROOT):
    root = Path(root)
    directory = args.campaign_dir.resolve() / job["id"]
    for journal_path in sorted(directory.glob("*/publication.json")):
        restore_publication(journal_path, root)
    # A queued job may have become invalid since build_jobs(), even on resume.
    validate_job_inputs(job, root)
    status_path = directory / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if can_resume(previous, job, root, directory):
            validate_job_inputs(job, root)
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
    # Training never reads historical outputs or writes into the canonical tree.
    # This also leaves the old artifacts intact on a no-op, OOM or interruption.
    for name in job["files"]:
        path = root / name
        if path.exists():
            destination = attempt_dir / "before" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    work = attempt_dir / "work"
    work.mkdir()
    state = {"state": "running", "signature": job["signature"], "provenance": job["provenance"],
             "command": job["command"], "git": git_info(root), "host": socket.gethostname(),
             "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
             "log": f"{attempt}/train.log", "snapshot": f"{attempt}/outputs"}
    atomic_json(status_path, state)
    start = time.monotonic()
    journal_path = attempt_dir / "publication.json"
    try:
        names = list(job["provenance"]["inputs"]) + list(job["provenance"]["code"])
        if job["provenance"].get("bundle_manifest") is not None:
            names.append("manifest.json")
        for name in names:
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        validate_job_inputs(job, root)
        validate_job_inputs(job, work, full_bundle=False)
        env = os.environ.copy()
        env.update({"QSAR_TAG": job["provenance"]["tag"], "PYTHONUNBUFFERED": "1"})
        with (attempt_dir / "train.log").open("w") as stream:
            subprocess.run(job["command"], cwd=work, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        validate_job_inputs(job, root)
        validate_job_inputs(job, work, full_bundle=False)
        validate_outputs(job, work)
        # Fresh metrics contain only time. Retain every historical non-time key.
        metric_name = next(name for name in job["files"] if "metrics_" in name)
        old_metric = attempt_dir / "before" / metric_name
        if old_metric.exists():
            payload = json.loads((work / metric_name).read_text())
            historical = json.loads(old_metric.read_text())
            payload.update({key: value for key, value in historical.items()
                            if key not in ("time", "best_params_time")})
            atomic_json(work / metric_name, payload)
        hashes = {}
        for name in job["files"]:
            destination = directory / state["snapshot"] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / name, destination)
            hashes[name] = digest(destination)
        validate_job_inputs(job, root)
        # Journal complete before/after images before the first canonical write.
        entries = {}
        for name, new_sha in hashes.items():
            before = attempt_dir / "before" / name
            old_sha = digest(before) if before.exists() else None
            current = digest(root / name) if (root / name).exists() else None
            if current != old_sha:
                raise ValueError(f"Canonical output changed during training: {name}")
            entries[name] = {"before_sha256": old_sha, "new_sha256": new_sha}
        atomic_json(journal_path, {"schema": 1, "root": str(root.resolve()),
                                   "state": "prepared", "entries": entries})
        # A later invocation restores all old images if killed before commit.
        for name in job["files"]:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + "." + attempt + ".publish.tmp")
            try:
                shutil.copy2(directory / state["snapshot"] / name, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        validate_job_inputs(job, root)
        state.update(state="success", artifacts=hashes)
    except BaseException as exc:
        if journal_path.exists():
            restore_publication(journal_path, root, allow_commit=False)
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic() - start,
                     finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, state)
    if journal_path.exists():
        restore_publication(journal_path, root)
    print(f"DONE {job['id']} ({state['elapsed_seconds']:.1f}s)", flush=True)


def main(argv=None):
    args = parse_args(argv)
    def run_plan():
        jobs = build_jobs(args)
        print(json.dumps({"mode": "execute" if args.execute else "dry-run", "jobs": [
            {key: value for key, value in job.items() if key != "split"} for job in jobs]}, indent=2))
        if args.execute:
            for job in jobs:
                run_job(job, args)
    if not args.execute:
        run_plan()
        return
    with worker_lock(ROOT, args.role, recover=args.recover_interrupted):
        recover_publications(args.campaign_dir.resolve(), ROOT, args.role)
        if args.recover_interrupted:
            print(json.dumps({"mode": "recovery-only", "role": args.role, "training_started": False}), flush=True)
            return
        run_plan()


if __name__ == "__main__":
    main()
