"""gnn_03_train_gin --model: gin / gine / attentivefp train end-to-end.

A 6-molecule CSR graph cache + random split is hand-built inside tmp_path
(same npz format as gnn_02_build_graphs.py, featurized with
gnn_graph_dataset.smiles_to_graph), then the real main() runs per model
with epochs=2, batch=2, hidden=16, num_layers=2 on forced CPU. Asserts the
per-model artifact names, npz payload lengths, and the metrics json.

The default (no --model / --model gin) must keep the legacy names:
``{tag}_gin_random.pt`` (not ``{tag}_gin_gin_random.pt``), plain
``gnn_preds_{tag}_random.npz`` and ``gnn_metrics_{tag}.json``.
"""
import json
import sys

import numpy as np
import pytest
import torch

from gnn_graph_dataset import smiles_to_graph  # conftest puts scripts/ on sys.path

TAG = "tiny"
SMILES = ["CC", "CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1"]
# distinct values so r2 on the 2-sample test set is defined and finite
Y = np.array([5.5, 6.0, 6.5, 7.0, 7.5, 8.0], dtype=np.float32)


def test_time_training_uses_asof_labels_for_scaling_and_evaluation(tmp_path, monkeypatch, load_script):
    from time_labels import LABEL_POLICY, dataset_signature, label_signature
    _build_cache(tmp_path)
    proc = tmp_path / "data" / "processed"
    y = np.array([3., 5., 6., 7., 8., 10.], dtype=np.float32)
    sp = dict(train_idx=np.array([0, 1]), valid_idx=np.array([2, 3]), test_idx=np.array([4, 5]),
              excluded_idx=np.array([], dtype=int), smiles=np.array(SMILES), y=y, original_y=Y,
              year=np.array([2010, 2011, 2019, 2020, 2021, 2022]),
              label_policy=LABEL_POLICY, dataset_signature=dataset_signature(SMILES, Y),
              source_sha256="fixture", cutoff_valid=2019, cutoff_test=2021)
    sp["label_id"] = label_signature(sp)
    np.savez(proc / "splits" / f"{TAG}_time.npz", **sp)
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(mod, "select_device", lambda: "cpu")
    monkeypatch.setattr(sys, "argv", ["gin", "--tag", TAG, "--split", "time", "--epochs", "2",
                                     "--batch-size", "2", "--hidden", "16", "--num-layers", "2"])
    mod.main()
    pred = np.load(tmp_path / "results" / f"gnn_preds_{TAG}_time.npz")
    np.testing.assert_allclose(pred["y_true"], y[sp["test_idx"]], atol=1e-6)
    assert str(pred["label_id"]) == sp["label_id"]
    checkpoint = torch.load(tmp_path / "results" / "gnn_models" / f"{TAG}_gin_time.pt", weights_only=False)
    assert checkpoint["y_mean"] == 4.0 and checkpoint["y_std"] == 1.0


def _build_cache(base):
    """Write data/processed/{TAG}_graphs.npz + splits/{TAG}_random.npz (CSR)."""
    graphs = [smiles_to_graph(s) for s in SMILES]
    assert all(g is not None for g in graphs)
    proc = base / "data" / "processed"
    (proc / "splits").mkdir(parents=True)
    np.savez(
        proc / f"{TAG}_graphs.npz",
        node_feat=np.concatenate([g["node_feat"] for g in graphs]),
        edge_index=np.concatenate([g["edge_index"] for g in graphs], axis=1),
        edge_feat=np.concatenate([g["edge_feat"] for g in graphs]),
        n_nodes=np.array([g["n_nodes"] for g in graphs], dtype=np.int32),
        n_edges=np.array([g["n_edges"] for g in graphs], dtype=np.int32),
        y=Y,
        smiles=np.array(SMILES),
        scaffold_smiles=np.array(SMILES),
    )
    np.savez(proc / "splits" / f"{TAG}_random.npz",
             train_idx=np.array([0, 1]), valid_idx=np.array([2, 3]),
             test_idx=np.array([4, 5]))


def _run_main(tmp_path, monkeypatch, load_script, model_args):
    """Run the real gnn_03 main() against the tmp cache; returns results/."""
    _build_cache(tmp_path)
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv",
                        ["gnn_03_train_gin.py", "--tag", TAG, "--split", "random",
                         "--epochs", "2", "--batch-size", "2", "--hidden", "16",
                         "--num-layers", "2", *model_args])
    mod.main()
    return tmp_path / "results"


@pytest.mark.parametrize("model", ["gin", "gine", "attentivefp"])
def test_train_artifacts_per_model(tmp_path, monkeypatch, load_script, model):
    res = _run_main(tmp_path, monkeypatch, load_script, ["--model", model])
    infix = "" if model == "gin" else f"_{model}"

    pt = res / "gnn_models" / f"{TAG}_{model}_random.pt"
    preds = res / f"gnn_preds_{TAG}{infix}_random.npz"
    mj = res / f"gnn_metrics_{TAG}{infix}.json"
    assert pt.exists() and preds.exists() and mj.exists()
    # never-produced names: {tag}_gin_gin_*, un-infixed preds for non-gin models
    assert not (res / "gnn_models" / f"{TAG}_gin_gin_random.pt").exists()
    assert not (res / f"gnn_preds_{TAG}_gin_random.npz").exists()
    if model != "gin":
        assert not (res / f"gnn_preds_{TAG}_random.npz").exists()
        assert not (res / f"gnn_metrics_{TAG}.json").exists()

    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    assert ckpt["state_dict"] and ckpt["args"]["model"] == model

    d = np.load(preds)
    n_test = len(d["test_idx"])
    assert n_test == 2
    assert len(d["y_true"]) == len(d["y_pred"]) == n_test

    payload = json.loads(mj.read_text())
    assert set(payload) == {"random"}  # top-level keys stay split names
    m = payload["random"]
    assert {"r2", "rmse", "mae", "params", "best_epoch",
            "convergence_flag", "max_epochs", "train_mse",
            "device"} <= set(m)
    assert np.isfinite([m["r2"], m["rmse"], m["mae"]]).all()
    assert m["params"]["model"] == model
    # the json records the device the run actually trained on; both GPU
    # backends are mocked unavailable above, so it must be "cpu"
    assert m["device"] == "cpu"
    # a 2-epoch run can never satisfy best_epoch < 30% of max epochs (0.6)
    assert m["convergence_flag"] is False


def test_default_model_keeps_legacy_gin_names(tmp_path, monkeypatch, load_script):
    """No --model flag (or gin) must not double the model token in names."""
    res = _run_main(tmp_path, monkeypatch, load_script, [])

    assert (res / "gnn_models" / f"{TAG}_gin_random.pt").exists()
    assert not (res / "gnn_models" / f"{TAG}_gin_gin_random.pt").exists()
    assert (res / f"gnn_preds_{TAG}_random.npz").exists()
    assert not (res / f"gnn_preds_{TAG}_gin_random.npz").exists()
    assert (res / f"gnn_metrics_{TAG}.json").exists()
    assert not (res / f"gnn_metrics_{TAG}_gin.json").exists()


def test_convergence_flag_rule(load_script):
    """Both conditions required; anchored on the real stalled run (N3).

    abl1/scaffold seed 42 stopped at best_epoch 15/100 with train MSE
    0.5706 (flagged); its siblings ended at 93/97 with train MSE ~0.28-0.31
    (never flagged).
    """
    mod = load_script("gnn_03_train_gin")
    assert mod.convergence_flag(15, 100, 0.5706) is True   # stalled run
    assert mod.convergence_flag(15, 100, 0.6652) is True   # train MSE at best epoch
    assert mod.convergence_flag(93, 100, 0.2768) is False  # healthy sibling
    assert mod.convergence_flag(97, 100, 0.3092) is False  # healthy sibling
    assert mod.convergence_flag(15, 100, 0.30) is False    # early but well fit
    assert mod.convergence_flag(68, 100, 0.60) is False    # late but poorly fit
    assert mod.convergence_flag(29, 100, 0.51) is True     # boundary: 29 < 30
    assert mod.convergence_flag(30, 100, 0.99) is False    # boundary: 30 not < 30


def test_convergence_flag_fires_on_degenerate_early_stop(
        tmp_path, monkeypatch, load_script, capsys):
    """End-to-end: a run that never learns early-stops before 30% of max
    epochs with train MSE stuck high -> prominent WARNING + json flag."""
    _build_cache(tmp_path)
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # degenerate trainer: loss never decreases, the model never improves
    # its validation RMSE, so best stays at epoch 1 and patience=3 stops
    # at epoch 4 (4 < 30% of 20) with train MSE stuck at 0.9 (> 0.5)
    monkeypatch.setattr(mod, "run_epoch",
                        lambda model, loader, device, opt=None: 0.9)
    monkeypatch.setattr(sys, "argv",
                        ["gnn_03_train_gin.py", "--tag", TAG, "--split", "random",
                         "--epochs", "20", "--patience", "3", "--batch-size", "2",
                         "--hidden", "16", "--num-layers", "2"])
    mod.main()

    out = capsys.readouterr().out
    assert "early stop at epoch 4 (best epoch 1)" in out
    assert "WARNING: possible non-convergence" in out

    m = json.loads((tmp_path / "results" / f"gnn_metrics_{TAG}.json").read_text())
    m = m["random"]
    assert m["convergence_flag"] is True
    assert m["best_epoch"] == 1
    assert m["max_epochs"] == 20
    assert m["train_mse"] == pytest.approx(0.9)
    # same stub training path still records the device actually used
    assert m["device"] == "cpu"


def test_min_epochs_defers_early_stopping(tmp_path, monkeypatch, load_script, capsys):
    """--min-epochs suppresses a stop that would otherwise fire earlier."""
    _build_cache(tmp_path)
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "run_epoch",
                        lambda model, loader, device, opt=None: 0.9)
    monkeypatch.setattr(sys, "argv",
                        ["gnn_03_train_gin.py", "--tag", TAG, "--split", "random",
                         "--epochs", "20", "--patience", "3", "--min-epochs", "10",
                         "--batch-size", "2", "--hidden", "16", "--num-layers", "2"])
    mod.main()

    out = capsys.readouterr().out
    assert "early stopping deferred until epoch 10" in out
    # patience is satisfied from epoch 4 but the floor defers the stop
    assert "early stop at epoch 10 (best epoch 1)" in out
