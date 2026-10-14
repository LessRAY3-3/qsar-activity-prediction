"""Queue safety tests use tiny subprocess fixtures, never actual training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "provenance_runner", Path(__file__).parents[1] / "scripts/run_provenance_reconstruction.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def queue(tmp_path):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/gnn_03_train_gin.py").write_text("raise RuntimeError('never train in tests')")
    (root / "data/processed/splits").mkdir(parents=True)
    np.savez(root / "data/processed/egfr_graphs.npz", y=np.array([1., 2., 3.]))
    np.savez(root / "data/processed/splits/egfr_random.npz", test_idx=np.array([0, 1, 2]))
    job = {"id": "egfr_gin_random_seed42", "seed": 42, "split": "random", "tags": ["egfr"],
           "argv": ["scripts/gnn_03_train_gin.py", "--device", "cuda"], "historical_r2": .5,
           "inputs": ["data/processed/egfr_graphs.npz", "data/processed/splits/egfr_random.npz"]}
    plan = {"new_backend": "cuda", "jobs": [job], "historical_metrics_sha256": {},
            "source_sha256": {"scripts/gnn_03_train_gin.py": runner.sha256(root / "scripts/gnn_03_train_gin.py")},
            "input_sha256": {name: runner.sha256(root / name) for name in job["inputs"]}}
    runs = tmp_path / "runs"
    runs.mkdir()
    return root, plan, job, runs


def write_outputs(workspace, job):
    metric, pred, checkpoint = runner.artifact_names(job)
    for name in (metric, pred, checkpoint):
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
    (workspace / metric).write_text(json.dumps({"random": {"device": "cuda", "r2": 1., "rmse": 0., "mae": 0.}}))
    np.savez(workspace / pred, test_idx=np.array([0, 1, 2]), y_true=np.array([1., 2., 3.]),
             y_pred=np.array([1., 2., 3.]))
    (workspace / checkpoint).write_bytes(b"test-only non-model fixture")


def test_cuda_unavailable_never_selects_cpu(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        runner.require_cuda()


def test_plan_hash_drift_and_time_jobs_rejected(queue):
    root, plan, job, _ = queue
    assert runner.validate_plan(root, plan)["jobs"] == 1
    job["split"] = "time"
    with pytest.raises(ValueError, match="separate"):
        runner.validate_plan(root, plan)
    job["split"] = "random"
    (root / job["inputs"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed file"):
        runner.validate_plan(root, plan)


def test_failed_subprocess_has_logs_no_receipt_and_fresh_retry(queue, monkeypatch):
    root, plan, job, runs = queue
    real_popen = subprocess.Popen
    def fail(command, **kwargs):
        return real_popen([sys.executable, "-c", "import sys;print('stdout marker');print('stderr marker',file=sys.stderr);sys.exit(17)"], **kwargs)
    monkeypatch.setattr(runner.subprocess, "Popen", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="exit 17"):
            runner.execute_job(root, plan, job, runs, {"plan": "fixed"}, 4)
    attempts = list((runs / job["id"]).glob("attempt-*"))
    assert len(attempts) == 2
    assert not (runs / job["id"] / "complete.json").exists()
    for attempt in attempts:
        status = json.loads((attempt / "status.json").read_text())
        assert status["exit_code"] == 17 and status["status"] == "failed"
        assert "stdout marker" in (attempt / "stdout.log").read_text()
        assert "stderr marker" in (attempt / "stderr.log").read_text()


def test_success_resume_checks_output_and_log_integrity(queue, monkeypatch):
    root, plan, job, runs = queue
    real_popen = subprocess.Popen
    calls = []
    def fixture_process(command, **kwargs):
        calls.append(command)
        workspace = Path(kwargs["cwd"])
        write_outputs(workspace, job)
        (workspace.parent / "environment.json").write_text('{"device":"cuda","test_fixture":true}')
        return real_popen([sys.executable, "-c", "print('simulated fixture completion')"], **kwargs)
    monkeypatch.setattr(runner.subprocess, "Popen", fixture_process)
    monkeypatch.setattr(runner, "validate_checkpoint", lambda *args: {"test_fixture": True})
    identity = {"plan": "fixed"}
    runner.execute_job(root, plan, job, runs, identity, 4)
    runner.execute_job(root, plan, job, runs, identity, 4)
    assert len(calls) == 1
    receipt = json.loads((runs / job["id"] / "complete.json").read_text())
    assert receipt["comparisons"]["egfr"]["r2_difference"] == .5
    with pytest.raises(ValueError, match="identity changed"):
        runner.resume_valid(runs / job["id"], {"plan": "other"})
    attempt = runs / job["id"] / receipt["attempt"]
    checkpoint = attempt / "workspace" / runner.artifact_names(job)[2]
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed file"):
        runner.resume_valid(runs / job["id"], identity)
    checkpoint.write_bytes(original)
    (attempt / "stdout.log").write_text("tampered")
    with pytest.raises(ValueError, match="changed file"):
        runner.resume_valid(runs / job["id"], identity)


def test_zero_exit_without_outputs_does_not_complete(queue, monkeypatch):
    root, plan, job, runs = queue
    real_popen = subprocess.Popen
    def empty_process(command, **kwargs):
        return real_popen([sys.executable, "-c", "print('no artifacts')"], **kwargs)
    monkeypatch.setattr(runner.subprocess, "Popen", empty_process)
    with pytest.raises(ValueError, match="required artifact"):
        runner.execute_job(root, plan, job, runs, {}, 4)
    assert not (runs / job["id"] / "complete.json").exists()


def test_metrics_and_membership_are_independently_validated(queue, monkeypatch):
    root, _, job, _ = queue
    write_outputs(root, job)
    monkeypatch.setattr(runner, "validate_checkpoint", lambda *args: {"test_fixture": True})
    assert runner.accept_outputs(root, job)["comparisons"]["egfr"]["r2"] == 1.
    names = runner.artifact_names(job)
    np.savez(root / names[1], test_idx=np.array([2, 1, 0]), y_true=np.array([3., 2., 1.]), y_pred=np.array([3., 2., 1.]))
    with pytest.raises(ValueError, match="membership"):
        runner.accept_outputs(root, job)
    write_outputs(root, job)
    (root / names[0]).write_text('{"random":{"device":"cuda","r2":0.2,"rmse":0,"mae":0}}')
    with pytest.raises(ValueError, match="metric mismatch"):
        runner.accept_outputs(root, job)


def test_symlink_job_directory_refused(queue):
    root, plan, job, runs = queue
    (runs / job["id"]).symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        runner.execute_job(root, plan, job, runs, {}, 4)


def test_live_orphan_child_refused(queue):
    import os
    root, plan, job, runs = queue
    attempt = runs / job["id"] / "attempt-existing"
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text(json.dumps({"status": "running", "child_pid": os.getpid()}))
    with pytest.raises(RuntimeError, match="surviving trainer"):
        runner.execute_job(root, plan, job, runs, {}, 4)


def test_queue_lock_prevents_concurrent_runner(queue):
    runs = queue[-1]
    with runner.queue_lock(runs):
        with pytest.raises(RuntimeError, match="already running"):
            with runner.queue_lock(runs):
                pytest.fail("second queue acquired lock")


def test_checkpoint_validates_real_schema_architecture_and_finite_weights(queue):
    import torch
    root, _, job, _ = queue
    source = root / job["argv"][0]
    source.write_text("import torch\ndef build_model(args):\n    return torch.nn.Linear(2, 1)\n")
    model = torch.nn.Linear(2, 1)
    checkpoint = root / "fixture.pt"
    payload = {"state_dict": model.state_dict(), "args": {"device": "cuda", "seed": 42, "split": "random"},
               "runtime": {"device": "cuda", "seed": 42}, "y_mean": 2., "y_std": 1.}
    torch.save(payload, checkpoint)
    result = runner.validate_checkpoint(root, job, checkpoint)
    assert result["state_dict_strict"] and result["parameter_count"] == 3
    payload["args"]["seed"] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="seed/split"):
        runner.validate_checkpoint(root, job, checkpoint)
    payload["args"]["seed"] = 42
    payload["state_dict"]["weight"][0, 0] = float("nan")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="nonfinite"):
        runner.validate_checkpoint(root, job, checkpoint)
    payload["state_dict"] = torch.nn.Linear(3, 1).state_dict()
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        runner.validate_checkpoint(root, job, checkpoint)
