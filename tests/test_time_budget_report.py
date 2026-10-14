"""Budget-report guards: no partial/mixed study can become an accepted report."""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def reporter(load_experiment):
    return load_experiment("time_budget_report")


def put_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def make_study(tmp_path, reporter, seeds=(42, 1, 2), source="frozen", device="cpu", workers=4):
    study = tmp_path / "study"
    inputs = tmp_path / "inputs"
    files = {}
    for name in reporter.CODE_FILES | {
            name for tag in reporter.TAGS for name in
            (f"data/processed/{tag}_graphs.npz", f"data/processed/splits/{tag}_time.npz")}:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = reporter.digest(path)
    manifest = dict(role="gin", source={"dirty": False, "git_head": source},
                    files=[dict(path=k, sha256=v) for k, v in files.items()],
                    targets={tag: {"label_id": f"labels-{tag}"} for tag in reporter.TAGS})
    manifest_path = tmp_path / "manifest.json"
    put_json(manifest_path, manifest)
    mh = reporter.digest(manifest_path)
    env = {"variables": {"QSAR_TORCH_THREADS": "4", "QSAR_TORCH_INTEROP_THREADS": "1"},
           "packages": {"torch": "test"}, "platform": "test-host"}
    protocol = dict(arms=reporter.ARMS, seeds=list(seeds),
                    lanes=[list(lane) for lane in reporter.LANES], jobs=16 * len(seeds),
                    device=device, parallel_workers=workers, thread_environment=env["variables"],
                    manifest_sha256=mh, source_commit=source)
    lanes = []
    for arm in reporter.ARMS:
        for lane, tags in enumerate(reporter.LANES):
            root = study / arm / f"lane{lane}"
            put_json(root / "manifest.json", manifest)
            lane_state = dict(arm=arm, lane=lane, tags=list(tags), returncode=0)
            lanes.append(lane_state)
            put_json(root / "lane_status.json", lane_state)
            for tag in tags:
                for seed in seeds:
                    directory = root / f"results/time_campaign/gin_{tag}_seed{seed}"
                    p = dict(role="gin", tag=tag, label_id=f"labels-{tag}",
                             hyperparameters=reporter.expected_parameters(arm, seed, device),
                             bundle_manifest={"sha256": mh, "source_git_head": source},
                             command=["train.py"], environment=env,
                             inputs={k: files[k] for k in (f"data/processed/{tag}_graphs.npz",
                                                          f"data/processed/splits/{tag}_time.npz")},
                             code={k: files[k] for k in reporter.CODE_FILES})
                    artifacts = {}
                    for name in reporter.artifact_names(tag, seed):
                        for base in (root, directory / "attempt/outputs"):
                            path = base / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(name.encode())
                        artifacts[name] = reporter.digest(root / name)
                    put_json(directory / "status.json", dict(
                        state="success", signature=reporter.json_digest(p), provenance=p,
                        command=["python", "train.py"], host="M3", artifacts=artifacts,
                        snapshot="attempt/outputs", log="attempt/train.log"))
    put_json(study / "protocol.json", protocol)
    put_json(study / "study_status.json", dict(state="training_complete_pending_independent_audit", lanes=lanes))
    return study, inputs


@pytest.fixture
def study(tmp_path, reporter):
    return make_study(tmp_path, reporter)


def test_all_48_job_hashes_and_paired_configuration_are_required(study, reporter):
    jobs, evidence = reporter.inspect_study(*study)
    assert len(jobs) == 48
    assert len(evidence["input_sha256"]) == 16


@pytest.mark.parametrize("failure", ["missing", "running", "snapshot", "input", "parameters", "threads", "host"])
def test_partial_corrupt_or_mixed_study_is_rejected(study, reporter, failure):
    root, inputs = study
    path = root / "extended/lane0/results/time_campaign/gin_a2a_seed42/status.json"
    state = json.loads(path.read_text())
    if failure == "missing":
        path.unlink()
    elif failure == "snapshot":
        artifact = next(iter(state["artifacts"]))
        (path.parent / state["snapshot"] / artifact).write_bytes(b"corrupted")
    elif failure == "input":
        (inputs / "data/processed/a2a_graphs.npz").write_bytes(b"wrong version")
    else:
        if failure == "running":
            state["state"] = "running"
        elif failure == "parameters":
            state["provenance"]["hyperparameters"]["epochs"] = 999
        elif failure == "threads":
            state["provenance"]["environment"]["variables"]["QSAR_TORCH_THREADS"] = "8"
        else:
            state["host"] = "another machine"
        state["signature"] = reporter.json_digest(state["provenance"])
        put_json(path, state)
    with pytest.raises(ValueError):
        reporter.inspect_study(root, inputs)


def test_checkpoint_runtime_cpu_thread_guard(reporter):
    runtime = dict(seed=42, device="cpu", requested_device="cpu", torch_threads=4, torch_interop_threads=1)
    reporter.check_runtime(runtime, 42)
    with pytest.raises(ValueError, match="runtime"):
        reporter.check_runtime({**runtime, "torch_interop_threads": 2}, 42)


def test_pairing_uses_all_seeds_and_extended_minus_control(reporter):
    rows = []
    metrics = ("train_mse", "valid_rmse_std", "valid_rmse_pic50", "test_r2", "test_rmse",
               "test_mae", "best_epoch", "elapsed_seconds")
    for arm, value in (("control", 2.), ("extended", 1.)):
        for tag in reporter.TAGS:
            for seed in reporter.SEEDS:
                rows.append(dict(arm=arm, tag=tag, seed=seed, label_id=tag, common_flag=arm == "control",
                                 **dict.fromkeys(metrics, value)))
    frame = pd.DataFrame(rows)
    paired = reporter.paired_rows(frame)
    assert len(paired) == 24 and (paired.delta_train_mse == -1).all()
    assert (paired.delta_common_flag == -1).all()
    with pytest.raises(ValueError):
        reporter.paired_rows(frame.iloc[:-1])


def test_log_parser_ignores_test_scores_and_handles_rounded_numeric_difference(reporter):
    log = "warning\n epoch 10 train_mse=0.1234 valid_rmse=0.5001 lr=2.5e-04 *\nTEST R2=.8\n"
    assert reporter.logged_epochs(log) == {10: (.1234, .5001, .00025)}


@pytest.mark.parametrize("case", ["missing", "truncated", "missingstop", "below_minimum"])
def test_unproven_or_incomplete_epoch_budget_is_rejected(tmp_path, reporter, case):
    log = tmp_path / "train.log"
    if case == "truncated":
        log.write_text("epoch 10 train_mse=0.3 valid_rmse=0.7 lr=1e-3\nearly stop at epoch")
    elif case == "missingstop":
        log.write_text("epoch 50 train_mse=0.3 valid_rmse=0.7 lr=1e-3\nTEST R2=0.4\nsaved -> model")
    elif case == "below_minimum":
        log.write_text("early stop at epoch 50 (best epoch 10)")
    with pytest.raises(ValueError):
        reporter.completed_epochs(log, 10, reporter.ARMS["extended"])


def test_epoch_budget_accepts_valid_early_stop_or_full_maximum(tmp_path, reporter):
    log = tmp_path / "train.log"
    log.write_text("early stop at epoch 60 (best epoch 10)")
    assert reporter.completed_epochs(log, 10, reporter.ARMS["extended"]) == 60
    log.write_text("epoch 200 train_mse=0.3 valid_rmse=0.7 lr=1e-3")
    assert reporter.completed_epochs(log, 190, reporter.ARMS["extended"]) == 200


def test_replay_mismatch_fails_at_absolute_tolerance(reporter):
    reporter.close(np.array([1.]), np.array([1.000001]), "test")
    with pytest.raises(AssertionError):
        reporter.close(np.array([1.]), np.array([1.001]), "test")


def test_no_output_when_study_missing(tmp_path, reporter):
    output = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        reporter.main(["--study-root", str(tmp_path / "missing"), "--input-root", str(tmp_path),
                       "--out-dir", str(output)])
    assert exc.value.code == 1
    assert not output.exists()


@pytest.mark.parametrize("training_device", ["cpu", "cuda"])
def test_real_gin_checkpoint_replay_and_bad_normalization(tmp_path, reporter, load_script, training_device):
    import torch
    from gnn_graph_dataset import smiles_to_graph
    from time_labels import dataset_signature, label_signature, load_time_split

    trainer = load_script("gnn_03_train_gin")
    inputs, root = tmp_path / "inputs", tmp_path / "lane"
    processed = inputs / "data/processed"
    (processed / "splits").mkdir(parents=True)
    smiles = np.array(["C", "CC", "CCC", "O", "N", "CO"])
    y = np.array([6., 7., 8., 9., 7., 8.], dtype=np.float32)
    graphs = [smiles_to_graph(s) for s in smiles]
    graph_path = processed / "a2a_graphs.npz"
    np.savez(graph_path, smiles=smiles, scaffold_smiles=smiles, y=y,
             node_feat=np.concatenate([g["node_feat"] for g in graphs]),
             edge_index=np.concatenate([g["edge_index"] for g in graphs], axis=1),
             edge_feat=np.concatenate([g["edge_feat"] for g in graphs]),
             n_nodes=np.array([g["n_nodes"] for g in graphs]),
             n_edges=np.array([g["n_edges"] for g in graphs]))
    split = dict(smiles=smiles, year=np.array([2000, 2000, 2015, 2015, 2022, 2022]),
                 y=y, original_y=y, train_idx=np.array([0, 1]), valid_idx=np.array([2, 3]),
                 test_idx=np.array([4, 5]), excluded_idx=np.array([], dtype=int),
                 cutoff_valid=2010, cutoff_test=2020, label_policy="exact-dated-asof-v1",
                 source_sha256="synthetic", dataset_signature=dataset_signature(smiles, y))
    split["label_id"] = label_signature(split)
    np.savez(processed / "splits/a2a_time.npz", **split)
    parameters = reporter.expected_parameters("control", 42, training_device)
    args = {**parameters, "tag": "a2a", "model": "gin", "split": "time", "suffix": ""}
    model = trainer.build_model(Namespace(**args))
    mean, std = float(y[:2].mean()), float(y[:2].std())
    outputs = {}
    for pool in ("train", "valid", "test"):
        ds = trainer.MoleculeGraphDataset(graph_path, indices=split[f"{pool}_idx"])
        ds.y = (y - mean) / std
        outputs[pool] = trainer.predict(model, trainer.DataLoader(ds, batch_size=512), "cpu")
    trp, try_ = outputs["train"]
    vp, vy = outputs["valid"]
    tp, ty = (a * std + mean for a in outputs["test"])
    train_mse = float(np.mean((trp - try_)**2))
    names = reporter.artifact_names("a2a", 42)
    (root / "results/gnn_models").mkdir(parents=True)
    # CUDA case is synthetic metadata around a CPU-created checkpoint, not a GPU run.
    # This deliberately different build string must match training provenance only.
    training_torch = "2.14.0+cu126" if training_device == "cuda" else str(torch.__version__)
    runtime = dict(seed=42, device=training_device, requested_device=training_device, torch_threads=4,
                   torch_interop_threads=1, torch_version=training_torch)
    checkpoint = dict(state_dict=model.state_dict(), args=args, y_mean=mean, y_std=std,
                      runtime=runtime, label_id=split["label_id"])
    torch.save(checkpoint, root / names[0])
    np.savez(root / names[1], test_idx=split["test_idx"], y_true=ty, y_pred=tp, label_id=split["label_id"])
    metric = dict(params={k: v for k, v in args.items() if k != "split"},
                  train_mse_source="best_checkpoint_eval", runtime=runtime, device=training_device,
                  label_id=split["label_id"], train_mse=train_mse,
                  best_valid_rmse_std=float(np.sqrt(np.mean((vp-vy)**2))),
                  best_epoch=1, max_epochs=100, convergence_flag=train_mse > .5,
                  last_epoch_train_mse=train_mse, **trainer.metrics(ty, tp))
    put_json(root / names[2], {"time": metric})
    (root / "train.log").write_text("early stop at epoch 21 (best epoch 1)")
    job = dict(arm="control", tag="a2a", seed=42, root=root, directory=root,
               parameters=parameters, files=names, label_id=split["label_id"],
               status={"log": "train.log", "elapsed_seconds": 1.,
                       "provenance": {"environment": {"packages": {"torch": training_torch}}}})
    row, _ = reporter.replay_job(job, inputs, trainer, load_time_split)
    assert row["prediction_max_abs_error"] < 1e-5
    assert row["train_mse_abs_error"] < 1e-5
    assert row["device"] == training_device and row["replay_device"] == "cpu"
    job["status"]["provenance"]["environment"]["packages"]["torch"] = "wrong-training-build"
    with pytest.raises(ValueError, match="Runtime torch version"):
        reporter.replay_job(job, inputs, trainer, load_time_split)
    job["status"]["provenance"]["environment"]["packages"]["torch"] = training_torch
    checkpoint["y_mean"] += .01
    torch.save(checkpoint, root / names[0])
    with pytest.raises(ValueError, match="normalization"):
        reporter.replay_job(job, inputs, trainer, load_time_split)


@pytest.mark.parametrize("seeds", [(3, 4, 5, 6, 7), (42, 1, 2, 3, 4, 5, 6, 7)])
def test_declared_seed_cohort_counts(tmp_path, reporter, seeds):
    study, inputs = make_study(tmp_path, reporter, seeds)
    jobs, evidence = reporter.inspect_study(study, inputs, seeds)
    assert len(jobs) == 16 * len(seeds)
    assert evidence["protocol"]["seeds"] == list(seeds)
    with pytest.raises(ValueError, match="seed coverage"):
        reporter.inspect_study(study, inputs)


def test_merge_all_eight_seeds_with_new_wrapper_revision(tmp_path, reporter):
    pilot, inputs = make_study(tmp_path / "pilot", reporter)
    supplement, _ = make_study(tmp_path / "extra", reporter, (3, 4, 5, 6, 7), "new-wrapper")
    seeds = (42, 1, 2, 3, 4, 5, 6, 7)
    jobs, evidence = reporter.inspect_cohorts([pilot, supplement], inputs, seeds)
    assert len(jobs) == 128
    assert [e["protocol"]["source_commit"] for e in evidence["cohorts"]] == ["frozen", "new-wrapper"]
    assert evidence["cohorts"][0]["source_sha256"] == evidence["cohorts"][1]["source_sha256"]
    with pytest.raises(ValueError, match="expected seeds"):
        reporter.inspect_cohorts([pilot, supplement], inputs, seeds[:-1])
    with pytest.raises(ValueError, match="Duplicate study roots"):
        reporter.inspect_cohorts([pilot, pilot], inputs, seeds)


def test_overlapping_cohort_seeds_rejected(tmp_path, reporter):
    pilot, inputs = make_study(tmp_path / "pilot", reporter)
    supplement, _ = make_study(tmp_path / "extra", reporter, (2, 3, 4, 5, 6, 7))
    with pytest.raises(ValueError, match="Duplicate seeds"):
        reporter.inspect_cohorts([pilot, supplement], inputs, (42, 1, 2, 3, 4, 5, 6, 7))


@pytest.mark.parametrize("field", ["source_sha256", "input_sha256", "training_environment", "host", "label_id"])
def test_merge_rejects_cross_cohort_drift(tmp_path, reporter, monkeypatch, field):
    # Separate cohort audits can each pass, but merging requires identical evidence.
    def inspect(study, inputs, seeds, training_device):
        second = study.name == "extra"
        seed = 3 if second else 42
        evidence = dict(protocol={"seeds": [seed], "device": "cpu", "parallel_workers": 4, "thread_environment": {}}, source_sha256={"worker": "same"},
                        input_sha256={"graph": "same"}, training_environment={"threads": 4}, host="m3")
        if second and field != "label_id":
            evidence[field] = "different"
        jobs = [dict(arm=arm, tag=tag, seed=seed,
                     label_id="different" if second and field == "label_id" else tag)
                for arm in reporter.ARMS for tag in reporter.TAGS]
        return jobs, evidence
    monkeypatch.setattr(reporter, "inspect_study", inspect)
    with pytest.raises(ValueError, match="differs|versions differ"):
        reporter.inspect_cohorts([tmp_path / "pilot", tmp_path / "extra"], tmp_path, (42, 3))


@pytest.mark.parametrize("seeds,merged,device", [((3, 4, 5, 6, 7), False, "cpu"),
                                                ((42, 1, 2, 3, 4, 5, 6, 7), False, "cpu"),
                                                ((42, 1, 2, 3, 4, 5, 6, 7), True, "cpu"),
                                                ((42, 1, 2, 3, 4, 5, 6, 7), False, "cuda")])
def test_report_counts_pairs_and_ensembles_are_not_hardcoded(tmp_path, reporter, monkeypatch, seeds, merged, device):
    root, inputs = make_study(tmp_path, reporter, reporter.SEEDS if merged else seeds,
                              device=device, workers=1 if device == "cuda" else 4)
    supplement = None
    if merged:
        supplement, _ = make_study(tmp_path / "extra", reporter, (3, 4, 5, 6, 7), "new-wrapper")
    metrics = ("train_mse", "valid_rmse_std", "valid_rmse_pic50", "test_r2", "test_rmse",
               "test_mae", "best_epoch", "elapsed_seconds")
    def replay(job, *args):
        value = 2. if job["arm"] == "control" else 1.
        row = dict(arm=job["arm"], tag=job["tag"], seed=job["seed"], label_id=job["label_id"],
                   native_flag=False, common_flag=False, prediction_max_abs_error=0.,
                   train_mse_abs_error=0., valid_rmse_abs_error=0., **dict.fromkeys(metrics, value))
        pred = dict(test_idx=np.array([0, 1]), y_true=np.array([0., 1.]), y_pred=np.array([0., 1.]))
        return row, pred
    monkeypatch.setattr(reporter, "load_replay_modules", lambda *args: (None, None))
    monkeypatch.setattr(reporter, "replay_job", replay)
    # This test exercises count/aggregation; real checkpoint replay is tested above.
    output = tmp_path / "report"
    report = reporter.run(root, inputs, output, threads=1, seeds=seeds, supplement=supplement,
                          training_device=device)
    assert report["jobs_accepted"] == 16 * len(seeds)
    assert report["paired_rows"] == 8 * len(seeds)
    assert report["n_seeds"] == len(seeds)
    assert report["cohorts"] == (2 if merged else 1)
    assert report["factorial_cells_complete"] == 2
    assert report["training_device"] == device
    assert report["replay"]["device"] == "cpu"
    assert report["training_host"] == "M3"  # Synthetic fixture host, never inferred from CUDA.
    assert report["training_platform"] == "test-host"
    if device == "cuda":
        assert report["environment"] == "CUDA training (CPU replay)"
        assert report["cpu_cells_status"] == "not audited by this report"
    else:
        assert report["cuda_cells_status"] == "not audited by this report"
    assert report["training_prefix_comparison"]["pairs"] == 8 * len(seeds)
    targets = pd.read_csv(output / "targets.csv")
    assert len(targets) == 8 and (targets.n_seeds == len(seeds)).all()
    assert (targets.control_ensemble_test_r2 == 1.).all()
    assert (pd.read_csv(output / "paired.csv").delta_train_mse == -1.).all()
    old = (output / "report.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        reporter.run(root, inputs, output, threads=1, seeds=seeds)
    assert (output / "report.json").read_bytes() == old


@pytest.mark.parametrize("seeds", [(), (42, 42), (-1,), (True,), ("3",)])
def test_invalid_declared_seeds_rejected(reporter, seeds):
    with pytest.raises(ValueError, match="unique nonnegative"):
        reporter.validate_seeds(seeds)


@pytest.mark.parametrize("device,workers,accepted", [("cpu", 4, True), ("cpu", 1, False),
                                                    ("cuda", 1, True), ("cuda", 4, True),
                                                    ("cuda", 0, False), ("cuda", 5, False)])
def test_recorded_device_and_concurrency_are_required(tmp_path, reporter, device, workers, accepted):
    study, inputs = make_study(tmp_path, reporter, device=device, workers=workers)
    if accepted:
        jobs, _ = reporter.inspect_study(study, inputs, training_device=device)
        assert len(jobs) == 48 and all(j["parameters"]["device"] == device for j in jobs)
    else:
        with pytest.raises(ValueError, match="protocol/device/concurrency"):
            reporter.inspect_study(study, inputs, training_device=device)
    opposite = "cuda" if device == "cpu" else "cpu"
    with pytest.raises(ValueError, match="protocol/device/concurrency"):
        reporter.inspect_study(study, inputs, training_device=opposite)


@pytest.mark.parametrize("field,value", [("device", "cpu"), ("requested_device", "cpu"),
                                        ("torch_threads", 2), ("torch_interop_threads", 2)])
def test_cuda_runtime_rejects_cpu_fallback_and_thread_drift(reporter, field, value):
    runtime = dict(seed=42, device="cuda", requested_device="cuda", torch_threads=4,
                   torch_interop_threads=1, torch_version="2.14.0+cu126")
    reporter.check_runtime(runtime, 42, "cuda")
    with pytest.raises(ValueError, match="runtime"):
        reporter.check_runtime({**runtime, field: value}, 42, "cuda")


def test_cuda_cohorts_cannot_change_concurrency(tmp_path, reporter):
    pilot, inputs = make_study(tmp_path / "pilot", reporter, device="cuda", workers=1)
    supplement, _ = make_study(tmp_path / "extra", reporter, (3, 4, 5, 6, 7),
                               device="cuda", workers=2)
    with pytest.raises(ValueError, match="parallel_workers differs"):
        reporter.inspect_cohorts([pilot, supplement], inputs, (42, 1, 2, 3, 4, 5, 6, 7), "cuda")
