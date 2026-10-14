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


@pytest.fixture
def study(tmp_path, reporter):
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
    manifest = dict(role="gin", source={"dirty": False, "git_head": "frozen"},
                    files=[dict(path=k, sha256=v) for k, v in files.items()],
                    targets={tag: {"label_id": f"labels-{tag}"} for tag in reporter.TAGS})
    manifest_path = tmp_path / "manifest.json"
    put_json(manifest_path, manifest)
    mh = reporter.digest(manifest_path)
    env = {"variables": {"QSAR_TORCH_THREADS": "4", "QSAR_TORCH_INTEROP_THREADS": "1"},
           "packages": {"torch": "test"}, "platform": "test-host"}
    protocol = dict(arms=reporter.ARMS, seeds=list(reporter.SEEDS),
                    lanes=[list(lane) for lane in reporter.LANES], jobs=48,
                    device="cpu", parallel_workers=4, thread_environment=env["variables"],
                    manifest_sha256=mh, source_commit="frozen")
    lanes = []
    for arm in reporter.ARMS:
        for lane, tags in enumerate(reporter.LANES):
            root = study / arm / f"lane{lane}"
            put_json(root / "manifest.json", manifest)
            lane_state = dict(arm=arm, lane=lane, tags=list(tags), returncode=0)
            lanes.append(lane_state)
            put_json(root / "lane_status.json", lane_state)
            for tag in tags:
                for seed in reporter.SEEDS:
                    directory = root / f"results/time_campaign/gin_{tag}_seed{seed}"
                    p = dict(role="gin", tag=tag, label_id=f"labels-{tag}",
                             hyperparameters=reporter.expected_parameters(arm, seed),
                             bundle_manifest={"sha256": mh, "source_git_head": "frozen"},
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


def test_real_gin_checkpoint_replay_and_bad_normalization(tmp_path, reporter, load_script):
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
    parameters = reporter.expected_parameters("control", 42)
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
    runtime = dict(seed=42, device="cpu", requested_device="cpu", torch_threads=4,
                   torch_interop_threads=1, torch_version=str(torch.__version__))
    checkpoint = dict(state_dict=model.state_dict(), args=args, y_mean=mean, y_std=std,
                      runtime=runtime, label_id=split["label_id"])
    torch.save(checkpoint, root / names[0])
    np.savez(root / names[1], test_idx=split["test_idx"], y_true=ty, y_pred=tp, label_id=split["label_id"])
    metric = dict(params={k: v for k, v in args.items() if k != "split"},
                  train_mse_source="best_checkpoint_eval", runtime=runtime, device="cpu",
                  label_id=split["label_id"], train_mse=train_mse,
                  best_valid_rmse_std=float(np.sqrt(np.mean((vp-vy)**2))),
                  best_epoch=1, max_epochs=100, convergence_flag=train_mse > .5,
                  last_epoch_train_mse=train_mse, **trainer.metrics(ty, tp))
    put_json(root / names[2], {"time": metric})
    (root / "train.log").write_text("early stop at epoch 21 (best epoch 1)")
    job = dict(arm="control", tag="a2a", seed=42, root=root, directory=root,
               parameters=parameters, files=names, label_id=split["label_id"],
               status={"log": "train.log", "elapsed_seconds": 1.,
                       "provenance": {"environment": {"packages": {"torch": str(torch.__version__)}}}})
    row, _ = reporter.replay_job(job, inputs, trainer, load_time_split)
    assert row["prediction_max_abs_error"] < 1e-5
    assert row["train_mse_abs_error"] < 1e-5
    checkpoint["y_mean"] += .01
    torch.save(checkpoint, root / names[0])
    with pytest.raises(ValueError, match="normalization"):
        reporter.replay_job(job, inputs, trainer, load_time_split)
