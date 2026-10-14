"""Exercise campaign acceptance with real CPU training and serialized estimators.

The campaign's trainer process, model builder and checkpoint loader are not
stubbed. Six graphs keep this a small integration check, not a study rerun.
"""
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gnn_graph_dataset import smiles_to_graph
from time_labels import LABEL_POLICY, dataset_signature, label_signature


@pytest.fixture(scope="module")
def trained_campaign(tmp_path_factory, load_script):
    root = tmp_path_factory.mktemp("real-campaign")
    repository = Path(__file__).resolve().parents[1]
    shutil.copytree(repository / "scripts", root / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "campaign_history"))
    proc = root / "data/processed"
    (proc / "splits").mkdir(parents=True)
    smiles = np.array(["CC", "CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1"])
    original = np.arange(6, dtype=np.float32) + 5
    y = np.array([3., 5., 6., 7., 8., 10.], dtype=np.float32)
    graphs = [smiles_to_graph(s) for s in smiles]
    np.savez(proc / "a2a_graphs.npz", smiles=smiles, y=original,
             node_feat=np.concatenate([g["node_feat"] for g in graphs]),
             edge_index=np.concatenate([g["edge_index"] for g in graphs], axis=1),
             edge_feat=np.concatenate([g["edge_feat"] for g in graphs]),
             n_nodes=np.array([g["n_nodes"] for g in graphs]),
             n_edges=np.array([g["n_edges"] for g in graphs]),
             scaffold_smiles=smiles)
    split = dict(smiles=smiles, y=y, original_y=original,
                 train_idx=np.array([0, 1]), valid_idx=np.array([2, 3]),
                 test_idx=np.array([4, 5]), excluded_idx=np.array([], dtype=int),
                 year=np.array([2000, 2001, 2010, 2011, 2020, 2021]),
                 cutoff_valid=2010, cutoff_test=2020, source_sha256="integration-fixture",
                 label_policy=LABEL_POLICY, dataset_signature=dataset_signature(smiles, original))
    split["label_id"] = label_signature(split)
    np.savez(proc / "splits/a2a_time.npz", **split)
    worker = load_script("run_time_campaign")
    args = worker.parse_args(["--role", "gin", "--tags", "a2a", "--seeds", "42",
                              "--device", "cpu", "--epochs", "2", "--patience", "2",
                              "--campaign-dir", str(root / "campaign"), "--execute"])
    # These are process settings, not mocks of training, prediction or loading.
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("QSAR_TORCH_THREADS", "1")
        environment.setenv("QSAR_TORCH_INTEROP_THREADS", "1")
        job = worker.build_jobs(args, root)[0]
        worker.run_job(job, args, root)
        worker.run_job(job, args, root)  # Only a verified real result may resume.
    return worker, root, args, job


def test_real_cpu_process_checkpoint_and_predictions_are_accepted(trained_campaign):
    worker, root, args, job = trained_campaign
    status = json.loads((args.campaign_dir / job["id"] / "status.json").read_text())
    assert status["state"] == "success"
    assert worker.can_resume(status, job, root, args.campaign_dir / job["id"])
    worker.validate_outputs(job, root)
    checkpoint = torch.load(root / job["files"][0], map_location="cpu", weights_only=True)
    assert checkpoint["y_mean"] == 4.0 and checkpoint["y_std"] == 1.0
    assert checkpoint["runtime"]["device"] == "cpu"
    with np.load(root / job["files"][1]) as prediction:
        np.testing.assert_allclose(prediction["y_true"], [8., 10.], atol=1e-6)


@pytest.mark.parametrize("corruption", ["shape", "nan", "label", "scaling", "seed",
                                        "bad_mean", "bad_positive_std", "metric",
                                        "prediction_version", "runtime", "metric_label"])
def test_real_gin_checkpoint_corruption_rejected(trained_campaign, tmp_path, corruption):
    worker, original, _, job = trained_campaign
    shutil.copytree(original / "scripts", tmp_path / "scripts")
    shutil.copytree(original / "results", tmp_path / "results")
    path = tmp_path / job["files"][0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if corruption in ("shape", "nan"):
        key = next(k for k, v in checkpoint["state_dict"].items() if v.ndim == 2)
        value = checkpoint["state_dict"][key]
        checkpoint["state_dict"][key] = value[:1] if corruption == "shape" else torch.full_like(value, float("nan"))
    elif corruption == "label":
        checkpoint["label_id"] = "wrong-generation"
    elif corruption == "scaling":
        checkpoint["y_std"] = 0.0
    elif corruption == "bad_mean":
        checkpoint["y_mean"] = 999.0
    elif corruption == "bad_positive_std":
        checkpoint["y_std"] = 999.0
    elif corruption in ("metric", "metric_label"):
        metric = tmp_path / job["files"][2]
        payload = json.loads(metric.read_text())
        if corruption == "metric":
            payload["time"]["r2"] += 1.0
        else:
            payload["time"]["label_id"] = "wrong-generation"
        metric.write_text(json.dumps(payload))
    elif corruption == "prediction_version":
        prediction = tmp_path / job["files"][1]
        with np.load(prediction) as archive:
            payload = dict(archive)
        payload["label_id"] = "wrong-generation"
        np.savez(prediction, **payload)
    elif corruption == "runtime":
        checkpoint["runtime"]["device"] = "mps"
    else:
        checkpoint["args"]["seed"] = 1
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        worker.validate_outputs(job, tmp_path)


def test_real_fitted_rf_is_checked_and_wrong_generation_rejected(trained_campaign, tmp_path):
    worker, _, _, gin_job = trained_campaign
    split = gin_job["split"]
    label = str(split["label_id"])
    x = np.arange(12, dtype=float).reshape(6, 2)
    pool = np.r_[split["train_idx"], split["valid_idx"]]
    model = RandomForestRegressor(n_estimators=3, random_state=42).fit(x[pool], split["y"][pool])
    model.time_label_id_ = label
    y = split["y"][split["test_idx"]]
    pred = model.predict(x[split["test_idx"]])
    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    files = ["models/rf_time_split_a2a.joblib", "results/rf_preds_a2a_time.npz", "results/metrics_a2a.json"]
    job = {"files": files, "split": split,
           "provenance": {"role": "rf", "tag": "a2a", "label_id": label,
                          "hyperparameters": {"seed": 42, "device": "cpu"}}}
    np.savez(tmp_path / files[1], y_true=y, y_pred=pred, test_idx=split["test_idx"], label_id=label)
    (tmp_path / files[2]).write_text(json.dumps({"best_params_time": model.get_params(), "time": {
        "label_id": label, "runtime": {"seed": 42}, "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))), "mae": mean_absolute_error(y, pred)}}))
    joblib.dump(model, tmp_path / files[0])
    worker.validate_outputs(job, tmp_path)
    model.time_label_id_ = "wrong-generation"
    joblib.dump(model, tmp_path / files[0])
    with pytest.raises(ValueError, match="label_id"):
        worker.validate_outputs(job, tmp_path)
