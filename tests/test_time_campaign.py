"""Campaign safety tests use tiny frozen arrays and mocked training, never real fits."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from time_labels import LABEL_POLICY, dataset_signature, label_signature


@pytest.fixture
def campaign(load_script, tmp_path, monkeypatch):
    import torch
    worker = load_script("run_time_campaign")
    monkeypatch.setattr(worker, "build_checkpoint_model", lambda root, args: torch.nn.Linear(1, 1))
    root = tmp_path / "repo"
    sources = ("run_time_campaign", "time_labels", "qsar_common", "04_train_and_evaluate",
               "gnn_03_train_gin", "gnn_graph_dataset")
    (root / "scripts").mkdir(parents=True)
    for source in sources:
        (root / "scripts" / f"{source}.py").write_text("# frozen source\n")
    directory = root / "data" / "processed"
    (directory / "splits").mkdir(parents=True)
    smiles = np.array(["C", "CC", "CCC", "CCCC"])
    y = np.array([5., 6., 7., 8.], dtype=np.float32)
    split = dict(label_policy=LABEL_POLICY, smiles=smiles, y=y, original_y=y,
                 year=np.array([2000., 2010., 2020., 2021.]), train_idx=np.array([0]),
                 valid_idx=np.array([1]), test_idx=np.array([2, 3]), excluded_idx=np.array([], dtype=int),
                 cutoff_valid=2010, cutoff_test=2020, source_sha256="frozen raw",
                 dataset_signature=dataset_signature(smiles, y))
    split["label_id"] = label_signature(split)
    np.savez(directory / "splits" / "a2a_time.npz", **split)
    for kind in ("fingerprints", "graphs"):
        np.savez(directory / f"a2a_{kind}.npz", smiles=smiles, y=y)
    args = worker.parse_args(["--role", "gin", "--tags", "a2a", "--seeds", "42", "1",
                              "--campaign-dir", str(tmp_path / "campaign")])
    return worker, root, args


def write_outputs(job, root):
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true, y_pred = np.array([7., 8.]), np.array([6.9, 8.1])
    parameters = job["provenance"]["hyperparameters"]
    args = {**parameters, "tag": job["provenance"]["tag"], "model": "gin", "split": "time",
            "suffix": "" if parameters["seed"] == 42 else f"_seed{parameters['seed']}"}
    runtime = {"seed": parameters["seed"], "device": "cuda", "requested_device": "cuda"}
    for name in job["files"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if "_preds_" in name:
            np.savez(path, test_idx=job["split"]["test_idx"], y_true=y_true,
                     y_pred=y_pred, label_id=job["provenance"]["label_id"])
        elif "metrics_" in name:
            path.write_text(json.dumps({"time": {"label_id": job["provenance"]["label_id"],
                                                  "device": "cuda", "runtime": runtime,
                                                  "params": {k: v for k, v in args.items() if k != "split"},
                                                  "r2": r2_score(y_true, y_pred),
                                                  "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                                                  "mae": mean_absolute_error(y_true, y_pred)}}))
        else:
            torch.save({"state_dict": torch.nn.Linear(1, 1).state_dict(), "args": args,
                        "runtime": runtime, "label_id": job["provenance"]["label_id"],
                        "y_mean": 5., "y_std": 1.}, path)


    if job["provenance"]["role"] == "rf":
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        # Fitted metadata is sufficient for worker validation; no actual fit.
        model = RandomForestRegressor(random_state=42)
        model.estimators_ = [object()]
        model.time_label_id_ = job["provenance"]["label_id"]
        joblib.dump(model, next(root / name for name in job["files"] if name.endswith(".joblib")))
        metric = next(root / name for name in job["files"] if "metrics_" in name)
        payload = json.loads(metric.read_text())
        payload["best_params_time"] = model.get_params()
        metric.write_text(json.dumps(payload))


def test_plan_is_read_only_and_seed_suffixes(campaign):
    worker, root, args = campaign
    before = set(root.rglob("*"))
    jobs = worker.build_jobs(args, root)
    assert set(root.rglob("*")) == before
    assert not args.campaign_dir.exists()
    assert not args.execute
    assert [job["id"] for job in jobs] == ["gin_a2a_seed42", "gin_a2a_seed1"]
    assert "--suffix" not in jobs[0]["command"]
    assert jobs[1]["command"][-2:] == ["--suffix", "_seed1"]
    assert jobs[0]["provenance"]["hyperparameters"]["device"] == "cuda"


def test_plan_rejects_na_and_mismatched_cache(campaign):
    worker, root, args = campaign
    marker = root / "data/processed/splits/a2a_time_NA.json"
    marker.write_text('{"reason":"no dated labels"}')
    with pytest.raises(ValueError, match="NA marker"):
        worker.build_jobs(args, root)
    marker.unlink()
    np.savez(root / "data/processed/a2a_graphs.npz", smiles=np.array(["N", "CC", "CCC", "CCCC"]),
             y=np.array([5., 6., 7., 8.]))
    with pytest.raises(ValueError, match="row order"):
        worker.build_jobs(args, root)


def test_role_lock_refuses_second_owner_without_deleting(campaign):
    worker, root, _ = campaign
    with worker.worker_lock(root, "rf"):
        with pytest.raises(RuntimeError, match="lock exists"):
            with worker.worker_lock(root, "rf"):
                pass
        assert (root / ".time_campaign_rf.lock").exists()
    assert not (root / ".time_campaign_rf.lock").exists()


def test_success_snapshot_resume_and_tamper_guard(campaign, monkeypatch):
    worker, root, args = campaign
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)  # Existing historical outputs must be archived, never skipped.
    calls = []
    monkeypatch.setattr(worker, "git_info", lambda root: {"revision": "test", "dirty_status": ""})
    def train(*a, **k):
        calls.append(a)
        write_outputs(job, k["cwd"])
    monkeypatch.setattr(worker.subprocess, "run", train)
    worker.run_job(job, args, root)
    status_path = args.campaign_dir / job["id"] / "status.json"
    status = json.loads(status_path.read_text())
    assert status["state"] == "success"
    assert len(status["artifacts"]) == 3
    directory = status_path.parent
    assert (directory / Path(status["snapshot"]).parent / "before" / job["files"][0]).exists()
    worker.run_job(job, args, root)
    assert len(calls) == 1
    (directory / status["snapshot"] / job["files"][0]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="--rerun"):
        worker.run_job(job, args, root)
    args.rerun = True
    worker.run_job(job, args, root)
    assert len(calls) == 2


def test_changed_source_and_hyperparameters_invalidate_resume(campaign):
    worker, root, args = campaign
    original = worker.build_jobs(args, root)[0]["signature"]
    (root / "scripts/gnn_graph_dataset.py").write_text("# changed implementation\n")
    assert worker.build_jobs(args, root)[0]["signature"] != original
    source_changed = worker.build_jobs(args, root)[0]["signature"]
    args.device = "cpu"
    assert worker.build_jobs(args, root)[0]["signature"] != source_changed


def test_failed_process_records_failure_and_stops(campaign, monkeypatch):
    worker, root, args = campaign
    job = worker.build_jobs(args, root)[0]
    monkeypatch.setattr(worker, "git_info", lambda root: {})
    def train(*a, **k):
        k["stdout"].write("out of memory\n")
        raise subprocess.CalledProcessError(1, a[0])
    monkeypatch.setattr(worker.subprocess, "run", train)
    with pytest.raises(subprocess.CalledProcessError):
        worker.run_job(job, args, root)
    directory = args.campaign_dir / job["id"]
    status = json.loads((directory / "status.json").read_text())
    assert status["state"] == "failed"
    assert "out of memory" in (directory / status["log"]).read_text()
    with pytest.raises(RuntimeError, match="--rerun"):
        worker.run_job(job, args, root)


def test_outputs_reject_wrong_label_and_device(campaign):
    worker, root, args = campaign
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)
    worker.validate_outputs(job, root)
    metric = next(root / name for name in job["files"] if "metrics_" in name)
    payload = json.loads(metric.read_text())
    payload["time"]["device"] = "cpu"
    metric.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="device"):
        worker.validate_outputs(job, root)
    metric.write_text('{"time":{"label_id":"legacy","device":"cuda"}}')
    with pytest.raises(ValueError, match="label_id"):
        worker.validate_outputs(job, root)


def test_rf_one_job_ignores_gin_seed_list(campaign):
    worker, root, args = campaign
    args.role = "rf"
    jobs = worker.build_jobs(args, root)
    assert len(jobs) == 1
    assert jobs[0]["command"][-2:] == ["--split", "time"]
    assert jobs[0]["provenance"]["hyperparameters"]["device"] == "cpu"
    assert len(jobs[0]["files"]) == 4


def test_output_metrics_and_checkpoint_seed_weights_guards(campaign):
    import torch

    worker, root, args = campaign
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)
    metric = next(root / name for name in job["files"] if "metrics_" in name)
    payload = json.loads(metric.read_text())
    payload["time"]["r2"] += 0.01
    metric.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="r2"):
        worker.validate_outputs(job, root)
    write_outputs(job, root)
    checkpoint = next(root / name for name in job["files"] if name.endswith(".pt"))
    payload = torch.load(checkpoint, weights_only=True)
    payload["args"]["seed"] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="argument mismatch: seed"):
        worker.validate_outputs(job, root)
    payload["args"]["seed"] = 42
    payload["state_dict"] = {}
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="invalid weights"):
        worker.validate_outputs(job, root)
    checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        worker.validate_outputs(job, root)


def test_bundle_manifest_and_missing_git(campaign, monkeypatch):
    worker, root, args = campaign
    (root / "manifest.json").write_text('{"source":{"git_head":"frozen-commit"}}')
    job = worker.build_jobs(args, root)[0]
    assert job["provenance"]["bundle_manifest"]["source_git_head"] == "frozen-commit"
    def missing(*a, **kw):
        raise FileNotFoundError("git")
    monkeypatch.setattr(worker.subprocess, "run", missing)
    assert worker.git_info(root)["revision"] == "unavailable"


def test_rf_rejects_wrong_checkpoint_seed(campaign, monkeypatch):
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    worker, root, args = campaign
    args.role = "rf"
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)
    # No fit is performed: simulate fitted metadata to exercise validation.
    model = RandomForestRegressor(random_state=1)
    model.estimators_ = [object()]
    model.time_label_id_ = job["provenance"]["label_id"]
    monkeypatch.setattr(joblib, "load", lambda _: model)
    metric = next(root / name for name in job["files"] if "metrics_" in name)
    payload = json.loads(metric.read_text())
    payload["best_params_time"] = model.get_params()
    metric.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="label_id or seed"):
        worker.validate_outputs(job, root)
    model.random_state = 42
    payload["best_params_time"] = model.get_params()
    metric.write_text(json.dumps(payload))
    worker.validate_outputs(job, root)


@pytest.mark.parametrize("role", ["rf", "gin"])
@pytest.mark.parametrize("resume", [False, True])
def test_late_na_marker_blocks_execute_and_resume(campaign, monkeypatch, role, resume):
    worker, root, args = campaign
    args.role = role
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)
    original = {name: (root / name).read_bytes() for name in job["files"]}
    if resume:
        # Exercise the guard even when a previous job would be resumable.
        directory = args.campaign_dir / job["id"]
        directory.mkdir(parents=True)
        (directory / "status.json").write_text('{"state":"success"}')
        monkeypatch.setattr(worker, "can_resume", lambda *a: True)
    def unexpected(*a, **k):
        pytest.fail("A stale plan must not launch training or inspect git")
    monkeypatch.setattr(worker.subprocess, "run", unexpected)
    (root / "data/processed/splits/a2a_time_NA.json").write_text('{}')
    with pytest.raises(ValueError, match="NA marker"):
        worker.run_job(job, args, root)
    assert original == {name: (root / name).read_bytes() for name in job["files"]}


@pytest.mark.parametrize("role", ["rf", "gin"])
@pytest.mark.parametrize("failure", ["noop", "partial", "process", "na_root", "na_work"])
def test_fresh_attempt_cannot_accept_old_or_partial_outputs(campaign, monkeypatch, role, failure):
    worker, root, args = campaign
    args.role = role
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)
    original = {name: (root / name).read_bytes() for name in job["files"]}
    monkeypatch.setattr(worker, "git_info", lambda root: {})
    def train(*a, **k):
        work = k["cwd"]
        assert work != root
        assert all(not (work / name).exists() for name in job["files"])
        if failure == "noop":
            return
        write_outputs(job, work)
        if failure == "partial":
            (work / job["files"][-1]).unlink()
        elif failure == "process":
            raise subprocess.CalledProcessError(1, a[0])
        else:
            where = root if failure == "na_root" else work
            (where / "data/processed/splits/a2a_time_NA.json").write_text('{}')
    monkeypatch.setattr(worker.subprocess, "run", train)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        worker.run_job(job, args, root)
    status = json.loads((args.campaign_dir / job["id"] / "status.json").read_text())
    assert status["state"] == "failed"
    assert original == {name: (root / name).read_bytes() for name in job["files"]}


@pytest.mark.parametrize("change", ["input", "source", "manifest"])
@pytest.mark.parametrize("during_training", [False, True])
def test_recheck_current_inputs_and_source(campaign, monkeypatch, change, during_training):
    worker, root, args = campaign
    (root / "manifest.json").write_text('{"source":{"git_head":"frozen"}}')
    job = worker.build_jobs(args, root)[0]
    paths = {"input": next(iter(job["provenance"]["inputs"])),
             "source": "scripts/gnn_graph_dataset.py", "manifest": "manifest.json"}
    path = root / paths[change]
    monkeypatch.setattr(worker, "git_info", lambda root: {})
    def train(*a, **k):
        assert during_training
        write_outputs(job, k["cwd"])
        path.write_bytes(b"changed")
    monkeypatch.setattr(worker.subprocess, "run", train)
    if not during_training:
        path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        worker.run_job(job, args, root)
    assert all(not (root / name).exists() for name in job["files"])


@pytest.mark.parametrize("role", ["rf", "gin"])
def test_publication_preserves_non_time_metrics_and_rolls_back(campaign, monkeypatch, role):
    worker, root, args = campaign
    args.role = role
    job = worker.build_jobs(args, root)[0]
    write_outputs(job, root)
    metric = next(root / name for name in job["files"] if "metrics_" in name)
    historical = {"random": {"r2": 0.42}, "scaffold": {"r2": 0.24},
                  "best_params_random": {"seed": 42}, "tag": "a2a"}
    metric.write_text(json.dumps({**historical, "time": {"label_id": "old"}}))
    original = {name: (root / name).read_bytes() for name in job["files"]}
    monkeypatch.setattr(worker, "git_info", lambda root: {})
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: write_outputs(job, k["cwd"]))
    replace = worker.os.replace
    def fail_second_artifact(src, dst):
        if Path(dst) == root / job["files"][1]:
            raise OSError("publication interrupted")
        replace(src, dst)
    monkeypatch.setattr(worker.os, "replace", fail_second_artifact)
    with pytest.raises(OSError, match="publication interrupted"):
        worker.run_job(job, args, root)
    assert original == {name: (root / name).read_bytes() for name in job["files"]}
    monkeypatch.setattr(worker.os, "replace", replace)
    args.rerun = True
    worker.run_job(job, args, root)
    result = json.loads(metric.read_text())
    assert {key: result[key] for key in historical} == historical
    assert result["time"]["label_id"] == job["provenance"]["label_id"]
    worker.run_job(job, args, root)  # Snapshot hashes include the merged metrics.


@pytest.mark.parametrize("flag,value", [
    ("--epochs", "0"), ("--epochs", "-1"), ("--patience", "0"),
    ("--patience", "-1"), ("--min-epochs", "-1"), ("--min-epochs", "101"),
])
def test_invalid_gin_budget_rejected_before_planning(load_script, capsys, flag, value):
    worker = load_script("run_time_campaign")
    with pytest.raises(SystemExit) as exc:
        worker.parse_args(["--role", "gin", flag, value])
    assert exc.value.code == 2
    assert "training budget requires" in capsys.readouterr().err


def test_gin_budget_defaults_and_full_budget_propagate_to_execution(campaign, monkeypatch):
    worker, root, args = campaign
    defaults = worker.build_jobs(args, root)[0]
    expected_defaults = {"epochs": 100, "patience": 20, "min_epochs": 0}
    assert {key: defaults["provenance"]["hyperparameters"][key] for key in expected_defaults} == expected_defaults
    for key, value in expected_defaults.items():
        flag = "--" + key.replace("_", "-")
        assert defaults["command"][defaults["command"].index(flag) + 1] == str(value)

    args = worker.parse_args(["--role", "gin", "--tags", "a2a", "--seeds", "42",
                              "--epochs", "200", "--min-epochs", "60", "--patience", "40",
                              "--campaign-dir", str(args.campaign_dir)])
    job = worker.build_jobs(args, root)[0]
    expected = {"epochs": 200, "min_epochs": 60, "patience": 40}
    assert {key: job["provenance"]["hyperparameters"][key] for key in expected} == expected
    assert job["signature"] != defaults["signature"]
    assert job["provenance"]["inputs"] == defaults["provenance"]["inputs"]
    assert {key: value for key, value in job["provenance"]["hyperparameters"].items()
            if key not in expected} == {key: value for key, value in defaults["provenance"]["hyperparameters"].items()
                                       if key not in expected}
    monkeypatch.setattr(worker, "git_info", lambda root: {})
    calls = []
    def train(command, **kwargs):
        calls.append(command)
        for key, value in expected.items():
            flag = "--" + key.replace("_", "-")
            assert command[command.index(flag) + 1] == str(value)
        write_outputs(job, kwargs["cwd"])
    monkeypatch.setattr(worker.subprocess, "run", train)
    worker.run_job(job, args, root)
    status = json.loads((args.campaign_dir / job["id"] / "status.json").read_text())
    assert status["state"] == "success"
    assert status["provenance"]["command"] == calls[0][1:]
    assert not worker.can_resume(status, defaults, root, args.campaign_dir / job["id"])


@pytest.mark.parametrize("key,value", [("epochs", 200), ("patience", 40), ("min_epochs", 60)])
def test_each_gin_budget_parameter_invalidates_resume_signature(campaign, key, value):
    worker, root, args = campaign
    signature = worker.build_jobs(args, root)[0]["signature"]
    setattr(args, key, value)
    assert worker.build_jobs(args, root)[0]["signature"] != signature


def test_gin_budget_minimum_and_equal_floor_are_valid(load_script):
    worker = load_script("run_time_campaign")
    args = worker.parse_args(["--role", "gin", "--epochs", "1", "--patience", "1", "--min-epochs", "1"])
    assert (args.epochs, args.patience, args.min_epochs) == (1, 1, 1)


@pytest.mark.parametrize("name,value", [("QSAR_TORCH_THREADS", "4"), ("QSAR_TORCH_INTEROP_THREADS", "1")])
def test_explicit_torch_threads_recorded_and_invalidate_resume(campaign, monkeypatch, name, value):
    worker, root, args = campaign
    monkeypatch.delenv(name, raising=False)
    original = worker.build_jobs(args, root)[0]
    assert original["provenance"]["environment"]["variables"][name] is None
    monkeypatch.setenv(name, value)
    changed = worker.build_jobs(args, root)[0]
    assert changed["provenance"]["environment"]["variables"][name] == value
    assert changed["signature"] != original["signature"]
