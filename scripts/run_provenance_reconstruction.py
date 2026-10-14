"""Run the independent 18-fit reconstruction queue, never the frozen comparison.

Each attempt gets fresh copies of the plan-pinned sources and inputs. Successful
resume requires intact evidence and outputs. No CPU/MPS training mode exists.
The outer runner is recorded separately and does not mutate the historical plan.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import signal
import subprocess
import sys
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    path = root / relative
    if path.resolve() != root.resolve() / relative:
        raise ValueError(f"Symlink path forbidden: {name}")
    return path


def verify_files(root, hashes):
    for name, expected in hashes.items():
        path = safe_path(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed file: {path}")


def validate_plan(root, plan):
    if plan.get("new_backend") != "cuda":
        raise ValueError("Only the independent CUDA reconstruction plan is supported")
    ids = set()
    for job in plan["jobs"]:
        if not re.fullmatch(r"[a-z0-9_]+", job["id"]) or job["id"] in ids:
            raise ValueError("Invalid or duplicate job id")
        ids.add(job["id"])
        if job["argv"][0] not in ("scripts/gnn_03_train_gin.py", "scripts/gnn_07_multitask.py"):
            raise ValueError("Unexpected trainer in reconstruction plan")
        if job["split"] not in ("random", "scaffold"):
            raise ValueError("Reconstruction must remain separate from time comparison")
        for name in job["inputs"]:
            if name not in plan["input_sha256"]:
                raise ValueError("Unpinned job input")
    for key in ("source_sha256", "input_sha256", "historical_metrics_sha256"):
        verify_files(root, plan[key])
    return {"jobs": len(ids), "source_files": len(plan["source_sha256"]),
            "input_files": len(plan["input_sha256"]), "status": "validated_no_training"}


@contextmanager
def queue_lock(run_dir):
    with (run_dir / ".queue.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Reconstruction queue is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU/MPS fallback is forbidden")
    torch.cuda.set_device(0)
    torch.ones(1, device="cuda").sum().item()
    return "cuda"


def cuda_environment(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    require_cuda()
    versions = {}
    for package in ("torch", "torch-geometric", "numpy", "scikit-learn", "rdkit"):
        versions[package] = importlib.metadata.version(package)
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return {"utc": now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "device": "cuda", "device_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": {"exit_code": smi.returncode, "stdout": smi.stdout, "stderr": smi.stderr}}


def child_run(workspace, job_path, environment_path, threads):
    """Same-process CUDA guard overrides the legacy multitask auto-selector."""
    job = json.loads(job_path.read_text())
    os.chdir(workspace)
    sys.path.insert(0, str(workspace / "scripts"))
    write_json(environment_path, cuda_environment(threads))
    import qsar_common
    qsar_common.select_device = require_cuda
    if job["argv"][0].endswith("gnn_03_train_gin.py"):
        argv = job["argv"]
        if "--device" not in argv or argv[argv.index("--device") + 1] != "cuda":
            raise ValueError("GIN reconstruction must explicitly require CUDA")
    sys.argv = list(job["argv"])
    runpy.run_path(str(workspace / job["argv"][0]), run_name="__main__")


def artifact_names(job):
    suffix = "" if job["seed"] == 42 else f"_seed{job['seed']}"
    split = job["split"]
    if job["argv"][0].endswith("gnn_03_train_gin.py"):
        return (f"results/gnn_metrics_egfr{suffix}.json",
                f"results/gnn_preds_egfr_{split}{suffix}.npz",
                f"results/gnn_models/egfr_gin_{split}{suffix}.pt")
    model = job["argv"][job["argv"].index("--model") + 1]
    stem = "mt_metrics" if model == "mt" else "pooled_st_metrics"
    return (f"results/multitask/{stem}{suffix}.json",
            f"results/multitask/{model}_preds_{split}{suffix}.npz",
            f"results/multitask/models/{model}_{split}{suffix}.pt")


def validate_checkpoint(workspace, job, checkpoint):
    """Load only our just-produced checkpoint; validate schema and architecture."""
    import numpy as np
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise ValueError("Checkpoint state_dict is missing")
    params = saved.get("args", {})
    for index in range(1, len(job["argv"]), 2):
        key = job["argv"][index].removeprefix("--").replace("-", "_")
        expected = job["argv"][index + 1]
        if str(params.get(key)) != expected:
            raise ValueError(f"Checkpoint parameter differs: {key}")
    if params.get("seed") != job["seed"] or params.get("split") != job["split"]:
        raise ValueError("Checkpoint seed/split mismatch")
    if not saved["state_dict"] or any(not torch.is_tensor(value) or not torch.isfinite(value).all()
                                      for value in saved["state_dict"].values()):
        raise ValueError("Checkpoint has empty or nonfinite weights")
    single = job["argv"][0].endswith("gnn_03_train_gin.py")
    expected_shape = () if single else (len(job["tags"]),)
    for key in ("y_mean", "y_std"):
        values = np.asarray(saved.get(key), dtype=float)
        if values.shape != expected_shape or not np.isfinite(values).all() or (key == "y_std" and (values <= 0).any()):
            raise ValueError(f"Invalid checkpoint {key}")
    if single:
        runtime = saved.get("runtime", {})
        if runtime.get("device") != "cuda" or runtime.get("seed") != job["seed"]:
            raise ValueError("Checkpoint CUDA runtime/seed mismatch")
    elif saved.get("tags") != job["tags"]:
        raise ValueError("Checkpoint task list mismatch")
    source = workspace / job["argv"][0]
    spec = importlib.util.spec_from_file_location("reconstruction_checkpoint_model", source)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(workspace / "scripts"))
    try:
        spec.loader.exec_module(module)
        if single:
            model = module.build_model(argparse.Namespace(**params))
        else:
            model = module.PooledGIN(len(job["tags"]), shared_head=params["model"] == "pooled",
                                     hidden=params["hidden"], num_layers=params["num_layers"], dropout=params["dropout"])
        model.load_state_dict(saved["state_dict"], strict=True)
    finally:
        sys.path.pop(0)
    return {"state_dict_strict": True, "finite_weights": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "args": params, "runtime": saved.get("runtime"),
            "runtime_note": "Legacy multitask runtime is independently recorded in environment.json."}


def accept_outputs(workspace, job):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    names = artifact_names(job)
    for name in names:
        if not safe_path(workspace, name).is_file() or not (workspace / name).stat().st_size:
            raise ValueError(f"Trainer did not produce required artifact: {name}")
    metrics = json.loads((workspace / names[0]).read_text())[job["split"]]
    checkpoint_validation = validate_checkpoint(workspace, job, workspace / names[2])
    single = job["argv"][0].endswith("gnn_03_train_gin.py")
    if single and metrics.get("device") != "cuda":
        raise ValueError("GIN metrics did not record CUDA")
    differences = {}
    with np.load(workspace / names[1], allow_pickle=False) as predictions:
        for tag in job["tags"]:
            key_suffix = "" if single else f"_{tag}"
            indices, truth, pred = [predictions[key + key_suffix]
                                    for key in ("test_idx", "y_true", "y_pred")]
            with np.load(workspace / f"data/processed/splits/{tag}_{job['split']}.npz") as split:
                if not np.array_equal(indices, split["test_idx"]):
                    raise ValueError(f"Prediction test membership differs for {tag}")
            with np.load(workspace / f"data/processed/{tag}_graphs.npz") as graphs:
                expected_truth = graphs["y"][indices]
            if (truth.shape != expected_truth.shape or pred.shape != truth.shape
                    or not np.isfinite(truth).all() or not np.isfinite(pred).all()
                    or not np.allclose(truth, expected_truth, rtol=1e-5, atol=1e-5)):
                raise ValueError(f"Invalid predictions/labels for {tag}")
            observed = {"r2": float(r2_score(truth, pred)),
                        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
                        "mae": float(mean_absolute_error(truth, pred))}
            stored = metrics if single else metrics[tag]
            for key, value in observed.items():
                if not np.isfinite(value) or not np.isclose(value, stored[key], rtol=1e-5, atol=1e-5):
                    raise ValueError(f"Prediction/metric mismatch: {tag}/{key}")
            historical = job["historical_r2"] if single else job["historical_r2"][tag]
            differences[tag] = {**observed, "historical_r2": historical,
                                "r2_difference": observed["r2"] - historical,
                                "interpretation": "Independent reconstruction; matching scores do not recover historical identity."}
    return {"outputs_sha256": {name: sha256(workspace / name) for name in names},
            "checkpoint_validation": checkpoint_validation,
            "comparisons": differences}


def resume_valid(job_dir, identity):
    receipt = job_dir / "complete.json"
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record["identity"] != identity:
        raise ValueError(f"Completed job identity changed: {job_dir.name}")
    attempt = safe_path(job_dir, record["attempt"])
    verify_files(attempt, record["evidence_sha256"])
    verify_files(attempt / "workspace", record["outputs_sha256"])
    verify_files(attempt / "workspace", record["execution_files_sha256"])
    return True


def execute_job(root, plan, job, run_dir, identity, threads):
    job_dir = safe_path(run_dir, job["id"])
    job_dir.mkdir(exist_ok=True)
    if resume_valid(job_dir, identity):
        print(f"Verified complete, skipped: {job['id']}", flush=True)
        return
    for status_path in job_dir.glob("attempt-*/status.json"):
        old = json.loads(status_path.read_text())
        if old.get("status") == "running":
            if not old.get("child_pid"):
                raise RuntimeError("Interrupted launch has no child PID; inspect before retry")
            try:
                os.kill(old["child_pid"], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f"Possibly surviving trainer PID {old['child_pid']}; inspect before retry")
    attempt = job_dir / ("attempt-" + uuid.uuid4().hex)
    workspace = attempt / "workspace"
    workspace.mkdir(parents=True)
    files = dict(plan["source_sha256"])
    files.update({name: plan["input_sha256"][name] for name in job["inputs"]})
    verify_files(root, files)
    for name in files:
        target = safe_path(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    verify_files(workspace, files)
    write_json(attempt / "job.json", job)
    write_json(attempt / "execution_files.json", files)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
               "--workspace", str(workspace), "--job-file", str(attempt / "job.json"),
               "--environment-file", str(attempt / "environment.json"), "--threads", str(threads)]
    state = {"job_id": job["id"], "identity": identity, "start_utc": now(),
             "status": "running", "command": command, "argv": job["argv"], "threads": threads}
    write_json(attempt / "status.json", state)
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = str(job["seed"])
    # The wrapper sets these itself; avoid setting interop threads twice in GIN.
    env.pop("QSAR_TORCH_THREADS", None)
    env.pop("QSAR_TORCH_INTEROP_THREADS", None)
    print(f"Starting: {job['id']} ({attempt.name})", flush=True)
    child = None
    try:
        with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen(command, cwd=workspace, env=env, stdout=stdout,
                                     stderr=stderr, start_new_session=True)
            state["child_pid"] = child.pid
            write_json(attempt / "status.json", state)
            state["exit_code"] = child.wait()
        if state["exit_code"] != 0:
            raise RuntimeError(f"Trainer failed with exit {state['exit_code']}; see {attempt}")
        verify_files(workspace, files)
        acceptance = accept_outputs(workspace, job)
        write_json(attempt / "acceptance.json", acceptance)
        state.update(status="complete", end_utc=now())
        write_json(attempt / "status.json", state)
        evidence = ["job.json", "execution_files.json", "environment.json", "stdout.log",
                    "stderr.log", "status.json", "acceptance.json"]
        write_json(job_dir / "complete.json", {"identity": identity, "attempt": attempt.name,
                   **acceptance, "execution_files_sha256": files,
                   "evidence_sha256": {name: sha256(attempt / name) for name in evidence}})
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(status="failed", end_utc=now(), error=str(error),
                     exit_code=None if child is None else child.returncode)
        write_json(attempt / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", type=Path, default=Path("results/provenance_reconstruction/plan.json"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--job", action="append", help="Select job ID; default all 18, serial")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--job-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--environment-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.child:
        child_run(args.workspace, args.job_file, args.environment_file, args.threads)
        return
    root = args.root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = json.loads(plan_path.read_text())
    validation = validate_plan(root, plan)
    if args.validate_only:
        print(json.dumps(validation))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required and must be isolated from source inputs/results")
    run_dir = args.run_dir.absolute()
    if run_dir.resolve() != run_dir:
        parser.error("--run-dir cannot traverse symlinks")
    if run_dir == root or root.is_relative_to(run_dir) or any(
            run_dir.is_relative_to(root / name) for name in ("results", "data", "scripts", "figures")):
        parser.error("--run-dir must not overlap primary artifacts")
    jobs = [job for job in plan["jobs"] if args.job is None or job["id"] in args.job]
    if args.job and set(args.job) != {job["id"] for job in jobs}:
        parser.error("Unknown job ID")
    identity = {"plan_sha256": sha256(plan_path), "runner_sha256": sha256(__file__),
                "threads": args.threads, "backend": "cuda"}
    run_dir.mkdir(parents=True, exist_ok=True)
    with queue_lock(run_dir):
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        for job in jobs:
            execute_job(root, plan, job, run_dir, identity, args.threads)
    print(f"All selected {len(jobs)} reconstruction jobs verified complete.")


if __name__ == "__main__":
    main()
