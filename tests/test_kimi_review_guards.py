"""Regression cases for the 2026-10-15 independent review.

Corrupt inputs must fail before plots, accepted metrics, or datasets are replaced.
"""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from artifact_guards import (
    check_error_table,
    check_prediction,
    collect_seed_metrics,
    require_rf_params,
    rf_identity,
    training_scale,
)


def prediction_fixture():
    sp = dict(
        test_idx=np.array([1, 2]),
        smiles=np.array(["C", "CC", "CCC"]),
        scaffold_smiles=np.array(["", "", ""]),
    )
    y = np.array([5.0, 6.0, 7.0])
    pred = dict(
        test_idx=sp["test_idx"].copy(), y_true=y[sp["test_idx"]], y_pred=np.array([6.1, 7.2])
    )
    return sp, y, pred


@pytest.mark.parametrize("corruption", ["index", "labels", "nan", "shape"])
def test_prediction_corruption_rejected(corruption):
    sp, y, pred = prediction_fixture()
    if corruption == "index":
        pred["test_idx"] = pred["test_idx"][::-1]
    elif corruption == "labels":
        pred["y_true"] = pred["y_true"] + 0.1
    elif corruption == "nan":
        pred["y_pred"][0] = np.nan
    else:
        pred["y_pred"] = pred["y_pred"][:, None]
    with pytest.raises(ValueError):
        check_prediction(pred, sp, y)


def test_error_csv_same_length_wrong_order_rejected():
    sp, y, pred = prediction_fixture()
    df = pd.DataFrame(
        dict(
            smiles=["CC", "CCC"],
            scaffold=["", ""],
            y_true=[6.0, 7.0],
            err_rf=[0.1, 0.2],
            err_gin=[0.1, 0.2],
        )
    )
    check_error_table(df, sp, pred, pred, y)
    with pytest.raises(ValueError, match="smiles"):
        check_error_table(df.iloc[::-1], sp, pred, pred, y)
    df.loc[0, "err_rf"] = 0.3
    with pytest.raises(ValueError, match="err_rf"):
        check_error_table(df, sp, pred, pred, y)


def test_error_loader_rejects_predictions_agreeing_with_each_other_only(
    tmp_path, monkeypatch, load_script
):
    sp, y, pred = prediction_fixture()
    proc = tmp_path / "data/processed"
    (proc / "splits").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    np.savez(proc / "splits/demo_random.npz", **sp)
    np.savez(proc / "demo_fingerprints.npz", smiles=sp["smiles"], y=y)
    pred["test_idx"] = pred["test_idx"][::-1]
    for prefix in ("rf", "gnn"):
        np.savez(tmp_path / f"results/{prefix}_preds_demo_random.npz", **pred)
    mod = load_script("gnn_05_error_analysis")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    monkeypatch.setattr(mod, "TAG", "demo")
    with pytest.raises(ValueError, match="test_idx"):
        mod.analyse("random")
    assert not (tmp_path / "results/error_analysis_demo_random.csv").exists()


@pytest.mark.parametrize("mode", ["duplicate", "wrong_seed", "missing_seed"])
def test_seed_identity_rejected(tmp_path, mode):
    data = {s: dict(r2=0.5, rmse=1.0, params={"seed": 42}) for s in ["random", "scaffold"]}
    (tmp_path / "gnn_metrics_demo.json").write_text(json.dumps(data))
    if mode == "duplicate":
        (tmp_path / "gnn_metrics_demo_seed42.json").write_text(json.dumps(data))
    else:
        data["random"]["params"] = {"seed": 1} if mode == "wrong_seed" else {}
        (tmp_path / "gnn_metrics_demo.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed"):
        collect_seed_metrics(tmp_path, "demo")


def test_no_cross_split_rf_fallback():
    with pytest.raises(ValueError, match="best_params_scaffold"):
        require_rf_params({"best_params_random": {"n_estimators": 3}}, "scaffold")


@pytest.mark.parametrize("values", [[1.0, 1.0], [np.nan, 1.0], []])
def test_invalid_training_scale_rejected(values):
    with pytest.raises(ValueError):
        training_scale(np.array(values), np.arange(len(values)))


def test_partition_guard_survives_python_optimized():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    code = "from artifact_guards import check_partition; import numpy as np; check_partition(3,train=np.array([0,1]),test=np.array([1]))"
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        env={**os.environ, "PYTHONPATH": scripts},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "ValueError" in result.stderr


def test_default_rf_training_preserves_versioned_time(tmp_path, monkeypatch, load_script):
    from test_rf_time_split import _setup, _write_time_split

    mod, backup, path, splits, _ = _setup(tmp_path, monkeypatch, load_script)
    _write_time_split(splits, mod.TAG)
    label = str(np.load(splits / f"{mod.TAG}_time.npz")["label_id"])
    backup["time"] = dict(r2=0.123, label_id=label)
    backup["best_params_time"] = {"n_estimators": 17}
    path.write_text(json.dumps(backup))
    monkeypatch.setenv("QSAR_RF_JOBS", "1")
    monkeypatch.setattr(sys, "argv", ["rf"])
    mod.main()
    result = json.loads(path.read_text())
    assert result["time"] == backup["time"]
    assert result["best_params_time"] == backup["best_params_time"]
    assert result["random"]["r2"] != backup["random"]["r2"]


def test_stale_time_block_rejected_before_training(tmp_path, monkeypatch, load_script):
    from test_rf_time_split import _setup, _write_time_split

    mod, backup, path, splits, _ = _setup(tmp_path, monkeypatch, load_script)
    _write_time_split(splits, mod.TAG)
    backup["time"] = dict(r2=0.123, label_id="stale")
    path.write_text(json.dumps(backup))
    before = path.read_bytes()
    monkeypatch.setattr(mod, "run_split", lambda *a: pytest.fail("must not train"))
    monkeypatch.setattr(sys, "argv", ["rf"])
    with pytest.raises(ValueError, match="stale"):
        mod.main()
    assert path.read_bytes() == before


@pytest.mark.parametrize("versioned", [False, True])
def test_plot_rejects_stale_model(tmp_path, monkeypatch, load_script, versioned):
    from test_rf_time_split import _setup
    from sklearn.dummy import DummyRegressor

    mod, _, _, _, _ = _setup(tmp_path, monkeypatch, load_script)
    model = DummyRegressor().fit([[0], [1]], [2.0, 3.0])
    if versioned:
        model.qsar_identity_ = "wrong"
    Path(mod.MODEL_DIR).mkdir()
    joblib.dump(model, Path(mod.MODEL_DIR) / f"rf_random_split_{mod.TAG}.joblib")
    monkeypatch.setattr(mod, "scatter", lambda *a: pytest.fail("must not plot"))
    with pytest.raises(ValueError):
        mod.plot_saved_predictions(["random"])


def test_rf_identity_detects_same_labels_changed_features():
    X = np.zeros((3, 2))
    y = np.arange(3.0)
    smiles = np.array(["C", "CC", "CCC"])
    before = rf_identity(X, y, smiles, [0, 1], [2])
    X[0, 0] = 1
    assert before != rf_identity(X, y, smiles, [0, 1], [2])


def download_args():
    return Namespace(
        target_chembl_id="CHEMBL203",
        tag="demo",
        standard_type="IC50",
        units="nM",
        max_rows=0,
        page=2,
    )


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"activities": [], "page_meta": {"total_count": 3, "next": "next"}},
        {"activities": [{}], "page_meta": {"total_count": 3, "next": None}},
        {"activities": [], "page_meta": {}},
    ],
)
def test_download_failure_preserves_old_csv(tmp_path, monkeypatch, load_script, response):
    mod = load_script("01e_download_chembl")
    dest = tmp_path / "data.csv"
    dest.write_bytes(b"previous")
    monkeypatch.setattr(mod, "get_page", lambda *a: response)
    with pytest.raises((ValueError, RuntimeError)):
        mod.run_download(download_args(), str(dest))
    assert dest.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.part"))


def test_download_complete_replaces_old_csv(tmp_path, monkeypatch, load_script):
    mod = load_script("01e_download_chembl")
    dest = tmp_path / "data.csv"
    dest.write_bytes(b"previous")
    monkeypatch.setattr(
        mod,
        "get_page",
        lambda p: {
            "activities": [{"activity_id": p["assay_type"]}],
            "page_meta": {"total_count": 1, "next": None},
        },
    )
    mod.run_download(download_args(), str(dest))
    assert pd.read_csv(dest)["activity_id"].tolist() == ["B", "F"]


def test_esol_bad_cache_rejected(tmp_path, monkeypatch, load_script):
    mod = load_script("gnn_06_esol_prep")
    cache = tmp_path / "esol.csv"
    cache.write_bytes(b"truncated")
    monkeypatch.setattr(mod, "RAW_CSV", str(cache))
    with pytest.raises(ValueError, match="SHA256"):
        mod.download()
    assert cache.read_bytes() == b"truncated"


def test_stale_comparison_reference_rejected(tmp_path, monkeypatch, load_experiment):
    mod = load_experiment("time_split_summary")
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    data = {s: dict(r2=0.5, rmse=1.0, params={"seed": 42}) for s in ["random", "scaffold"]}
    (res / "gnn_metrics_demo.json").write_text(json.dumps(data))
    (res / "metrics_demo.json").write_text(json.dumps({s: {"r2": 0.4} for s in data}))
    table = pd.DataFrame([{"split": s, "rf_r2": 0.4, "gin_r2_mean": 0.5} for s in data])
    dest = res / "comparison_demo.csv"
    table.to_csv(dest, index=False)
    assert mod.comparison_reference("demo")["random"] == (0.4, 0.5)
    table.loc[0, "gin_r2_mean"] = 0.6
    table.to_csv(dest, index=False)
    with pytest.raises(ValueError, match="comparison CSV"):
        mod.comparison_reference("demo")


def test_new_campaign_bundle_cannot_omit_guard_dependency(tmp_path):
    import time_campaign_bundle as bundle
    from test_time_campaign_bundle import make_repo

    root = make_repo(tmp_path)
    (root / "scripts/gnn_03_train_gin.py").write_text(
        "from artifact_guards import training_scale\n"
    )
    manifest = bundle.make_manifest(root, "gin")
    with pytest.raises(ValueError, match="artifact_guards"):
        bundle.verify_manifest(manifest, root)


def test_learning_curve_missing_params_preserves_existing_output(
    tmp_path, monkeypatch, load_experiment
):
    from test_split_params import _write_tree

    _write_tree(tmp_path, with_scaffold_params=False)
    mod = load_experiment("learning_curve")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    dest = out / "learning_curve_tiny.csv"
    dest.write_bytes(b"accepted result")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "learning_curve",
            "--tag",
            "tiny",
            "--splits",
            "scaffold",
            "--out",
            str(out),
            "--fig-dir",
            str(tmp_path / "fig"),
        ],
    )
    with pytest.raises(ValueError, match="best_params_scaffold"):
        mod.main()
    assert dest.read_bytes() == b"accepted result"


def test_esol_partial_response_never_installed(tmp_path, monkeypatch, load_script):
    import io

    mod = load_script("gnn_06_esol_prep")
    path = tmp_path / "raw/esol.csv"
    monkeypatch.setattr(mod, "RAW_CSV", str(path))
    monkeypatch.setattr(mod.time, "sleep", lambda *a: None)
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"incomplete"))
    with pytest.raises(RuntimeError, match="verified ESOL"):
        mod.download()
    assert not path.exists()


def test_legacy_egfr_entry_uses_atomic_downloader(tmp_path, monkeypatch, load_script):
    import importlib

    generic = importlib.import_module("01e_download_chembl")
    legacy = load_script("01d_download_incremental")
    dest = tmp_path / "egfr.csv"
    dest.write_bytes(b"accepted")
    monkeypatch.setattr(legacy, "OUT_CSV", str(dest))
    monkeypatch.setattr(generic, "get_page", lambda *a: None)
    with pytest.raises(RuntimeError):
        legacy.main()
    assert dest.read_bytes() == b"accepted"
