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
    assert {"r2", "rmse", "mae", "params", "best_epoch"} <= set(m)
    assert np.isfinite([m["r2"], m["rmse"], m["mae"]]).all()
    assert m["params"]["model"] == model


def test_default_model_keeps_legacy_gin_names(tmp_path, monkeypatch, load_script):
    """No --model flag (or gin) must not double the model token in names."""
    res = _run_main(tmp_path, monkeypatch, load_script, [])

    assert (res / "gnn_models" / f"{TAG}_gin_random.pt").exists()
    assert not (res / "gnn_models" / f"{TAG}_gin_gin_random.pt").exists()
    assert (res / f"gnn_preds_{TAG}_random.npz").exists()
    assert not (res / f"gnn_preds_{TAG}_gin_random.npz").exists()
    assert (res / f"gnn_metrics_{TAG}.json").exists()
    assert not (res / f"gnn_metrics_{TAG}_gin.json").exists()
