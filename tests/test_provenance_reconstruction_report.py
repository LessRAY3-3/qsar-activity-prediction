"""Tiny untrained frozen models exercise CPU replay without any training."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("provenance_report", ROOT / "experiments/provenance_reconstruction_report.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)
runner = reporter.load_module(ROOT / "scripts/run_provenance_reconstruction.py", "test_return_runner")


def sources(root):
    (root / "scripts").mkdir(parents=True)
    for source in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(source, root / "scripts" / source.name)
    return {str(path.relative_to(root)): reporter.sha256(path) for path in (root / "scripts").glob("*.py")}


def fixture_job(root, model, seed=42, split="random", source_hashes=None):
    """Freeze untrained real GIN/pooled weights and reference trainer inference."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if source_hashes is None:
        source_hashes = sources(root)
    name = "scripts/gnn_03_train_gin.py" if model == "egfr_gin" else "scripts/gnn_07_multitask.py"
    tags = ["egfr"] if model == "egfr_gin" else ["egfr_full", "abl1", "vegfr2"]
    with reporter.frozen_imports(root, source_hashes):
        dataset_module = reporter.load_module(root / "scripts/gnn_graph_dataset.py", "fixture_graphs")
        module = reporter.load_module(root / name, "fixture_trainer")
        for task, tag in enumerate(tags):
            molecules = ["C", "CC", "CCC", "CO", "CCO", "CCN"]
            graphs = [dataset_module.smiles_to_graph(smiles) for smiles in molecules]
            directory = root / "data/processed"
            (directory / "splits").mkdir(parents=True, exist_ok=True)
            np.savez(directory / f"{tag}_graphs.npz",
                     node_feat=np.concatenate([g["node_feat"] for g in graphs]),
                     edge_index=np.concatenate([g["edge_index"] for g in graphs], axis=1),
                     edge_feat=np.concatenate([g["edge_feat"] for g in graphs]),
                     n_nodes=np.array([g["n_nodes"] for g in graphs]),
                     n_edges=np.array([g["n_edges"] for g in graphs]),
                     y=np.arange(1 + task, 7 + task, dtype=np.float32), smiles=np.array(molecules))
            np.savez(directory / "splits" / f"{tag}_{split}.npz",
                     train_idx=np.array([0, 1, 2]), valid_idx=np.array([3]), test_idx=np.array([4, 5]))
        suffix = "" if seed == 42 else f"_seed{seed}"
        argv = [name, "--model", "gin" if model == "egfr_gin" else model,
                "--split", split, "--seed", str(seed), "--hidden", "8", "--num-layers", "2"]
        if model == "egfr_gin":
            argv += ["--tag", "egfr", "--device", "cuda", "--suffix", suffix]
        else:
            argv += ["--tasks", "3", "--outdir", "results/multitask", "--figdir", "figures/multitask"]
        old_argv = sys.argv
        try:
            sys.argv = argv
            args = module.parse_args()
        finally:
            sys.argv = old_argv
        job = {"id": f"{model}_{split}_seed{seed}", "argv": argv, "tags": tags,
               "seed": seed, "split": split, "historical_r2": .5 if model == "egfr_gin" else dict.fromkeys(tags, .5),
               "inputs": [name for tag in tags for name in
                          (f"data/processed/{tag}_graphs.npz", f"data/processed/splits/{tag}_{split}.npz")]}
        means = np.array([2. + task for task in range(len(tags))])
        stds = np.full(len(tags), float(np.array([1, 2, 3], dtype=np.float32).std()))
        if model == "egfr_gin":
            network = module.build_model(args).eval()
            pooled = None
        else:
            network = module.PooledGIN(3, shared_head=model == "pooled", hidden=8, num_layers=2, dropout=args.dropout).eval()
            pooled = module.load_pool(tags)
        checkpoint = {"state_dict": network.state_dict(), "args": vars(args),
                      "y_mean": float(means[0]) if model == "egfr_gin" else means.tolist(),
                      "y_std": float(stds[0]) if model == "egfr_gin" else stds.tolist()}
        if model == "egfr_gin":
            # CUDA metadata exercises the returned schema; these test weights
            # were initialized on CPU and never optimized or claimed as a run.
            checkpoint["runtime"] = {"device": "cuda", "seed": seed, "test_fixture": True}
        else:
            checkpoint["tags"] = tags
        prediction_arrays, saved_metrics = {}, {}
        for task, tag in enumerate(tags):
            indices = np.array([4, 5])
            if model == "egfr_gin":
                ds = module.MoleculeGraphDataset(root / f"data/processed/{tag}_graphs.npz", indices)
                ds.y = (ds.y.copy() - means[task]) / stds[task]
                p, y = module.predict(network, DataLoader(ds, batch_size=512), "cpu")
                key_suffix = ""
            else:
                ds = module.PooledGraphDataset(pooled, indices + pooled["offsets"][task], means, stds)
                p, y = module.predict(network, DataLoader(ds, batch_size=512), "cpu", task)
                key_suffix = "_" + tag
            p = (p * stds[task] + means[task]).astype(np.float32)
            y = (y * stds[task] + means[task]).astype(np.float32)
            cell = module.metrics(y, p)
            if model == "egfr_gin":
                saved_metrics = {**cell, "device": "cuda"}
            else:
                saved_metrics[tag] = cell
            prediction_arrays.update({"test_idx" + key_suffix: indices, "y_true" + key_suffix: y, "y_pred" + key_suffix: p})
        metric_name, pred_name, checkpoint_name = runner.artifact_names(job)
        for name in (metric_name, pred_name, checkpoint_name):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / metric_name).write_text(json.dumps({split: saved_metrics}))
        np.savez(root / pred_name, **prediction_arrays)
        torch.save(checkpoint, root / checkpoint_name)
        return job, source_hashes


@pytest.mark.parametrize("model", ["egfr_gin", "mt", "pooled"])
def test_real_frozen_architecture_cpu_replay_and_normalization(tmp_path, model):
    job, hashes = fixture_job(tmp_path, model)
    rows, arrays = reporter.replay_job(tmp_path, job, runner, hashes)
    assert len(rows) == len(job["tags"]) and all(row["passed"] for row in rows)
    assert all(row["max_prediction_abs_error"] < 1e-5 for row in rows)
    assert set(arrays) == {key + "_" + tag for tag in job["tags"] for key in ("test_idx", "y_true", "y_pred")}
    checkpoint_path = tmp_path / runner.artifact_names(job)[2]
    saved = torch.load(checkpoint_path, weights_only=False)
    saved["y_mean"] = 99. if model == "egfr_gin" else [99., 3., 4.]
    torch.save(saved, checkpoint_path)
    with pytest.raises(ValueError, match="training labels"):
        reporter.replay_job(tmp_path, job, runner, hashes)


def test_prediction_drift_is_rejected_not_silently_relaxed(tmp_path):
    job, hashes = fixture_job(tmp_path, "egfr_gin")
    pred_path = tmp_path / runner.artifact_names(job)[1]
    with np.load(pred_path) as loaded:
        arrays = dict(loaded)
    arrays["y_pred"] += .01
    np.savez(pred_path, **arrays)
    rows, _ = reporter.replay_job(tmp_path, job, runner, hashes)
    assert not rows[0]["passed"] and not rows[0]["predictions_within_tolerance"]
    assert not rows[0]["stored_metrics_match_predictions"]


def test_full_18_receipt_manifest_cannot_omit_hashes(tmp_path):
    run_dir = tmp_path / "returns"
    run_dir.mkdir()
    template = tmp_path / "template"
    template_job, hashes = fixture_job(template, "egfr_gin")
    plan = {"new_backend": "cuda", "source_sha256": hashes, "input_sha256": {}, "jobs": []}
    for model in ("egfr_gin", "mt", "pooled"):
        for split in ("random", "scaffold"):
            for seed in (42, 1, 2):
                job = copy.deepcopy(template_job)
                job.update(id=f"{model}_{split}_seed{seed}", split=split, seed=seed)
                plan["jobs"].append(job)
    plan["input_sha256"] = {name: reporter.sha256(template / name) for name in template_job["inputs"]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    runner_path = ROOT / "scripts/run_provenance_reconstruction.py"
    identity = {"plan_sha256": reporter.sha256(plan_path), "runner_sha256": reporter.sha256(runner_path), "threads": 4, "backend": "cuda"}
    for job in plan["jobs"]:
        directory = run_dir / job["id"]
        (directory / "attempt-x/workspace").mkdir(parents=True)
        (directory / "complete.json").write_text(json.dumps({"identity": identity, "attempt": "attempt-x",
                                                            "evidence_sha256": {}, "outputs_sha256": {}, "execution_files_sha256": {}}))
    with pytest.raises(ValueError, match="Execution manifest differs"):
        reporter.verify_receipts(run_dir, plan_path, runner_path)


def test_report_refuses_existing_output_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        reporter.build_report(tmp_path / "missing", tmp_path / "missing.json", tmp_path / "runner.py", tmp_path)


def test_complete_18_job_tiny_untrained_cohort_replays_42_cells(tmp_path):
    run_dir = tmp_path / "returns"
    runner_path = ROOT / "scripts/run_provenance_reconstruction.py"
    plan = {"new_backend": "cuda", "jobs": [], "source_sha256": {}, "input_sha256": {}}
    entries = []
    for model in ("egfr_gin", "mt", "pooled"):
        for split in ("random", "scaffold"):
            for seed in (42, 1, 2):
                attempt = run_dir / f"{model}_{split}_seed{seed}" / "attempt-fixture"
                workspace = attempt / "workspace"
                job, hashes = fixture_job(workspace, model, seed, split)
                if plan["source_sha256"]:
                    assert plan["source_sha256"] == hashes
                plan["source_sha256"] = hashes
                plan["jobs"].append(job)
                plan["input_sha256"].update({name: reporter.sha256(workspace / name) for name in job["inputs"]})
                entries.append((attempt, workspace, job))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    identity = {"plan_sha256": reporter.sha256(plan_path), "runner_sha256": reporter.sha256(runner_path), "threads": 4, "backend": "cuda"}
    for attempt, workspace, job in entries:
        files = {**plan["source_sha256"], **{name: plan["input_sha256"][name] for name in job["inputs"]}}
        with reporter.frozen_imports(workspace, plan["source_sha256"]):
            acceptance = runner.accept_outputs(workspace, job)
        payloads = {"job.json": job, "execution_files.json": files, "acceptance.json": acceptance,
                    "status.json": {"status": "complete", "exit_code": 0, "identity": identity,
                                    "argv": job["argv"], "job_id": job["id"], "start_utc": "fixture-start", "end_utc": "fixture-end"},
                    "environment.json": {"device": "cuda", "device_name": "synthetic CUDA receipt schema, actual CPU-only fixture",
                                         "torch_threads": 4, "cuda_runtime": "test-fixture", "test_fixture": True}}
        for name, data in payloads.items():
            (attempt / name).write_text(json.dumps(data))
        (attempt / "stdout.log").write_text("Untrained tiny CPU fixture; no real CUDA run.\n")
        (attempt / "stderr.log").write_text("")
        receipt = {"identity": identity, "attempt": attempt.name, **acceptance, "execution_files_sha256": files,
                   "evidence_sha256": {name: reporter.sha256(attempt / name) for name in reporter.EVIDENCE_FILES}}
        (attempt.parent / "complete.json").write_text(json.dumps(receipt))
    result = reporter.build_report(run_dir, plan_path, runner_path, tmp_path / "report", cpu_threads=1)
    assert result["status"] == "accepted_independent_cpu_replay"
    assert result["rows_checked"] == result["rows_passed"] == 42
    assert len(result["jobs"]) == 18
    assert len(list((tmp_path / "report/predictions").glob("*.npz"))) == 18
    # The same valid receipt cannot conceal a modified log.
    (entries[0][0] / "stdout.log").write_text("altered")
    with pytest.raises(ValueError, match="changed file"):
        reporter.verify_receipts(run_dir, plan_path, runner_path)
