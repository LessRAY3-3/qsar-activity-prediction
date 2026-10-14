"""Resource controls must constrain execution without silently changing it."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("raw,expected", [(None, -1), ("-1", -1), ("1", 1), ("8", 8)])
def test_rf_job_count(monkeypatch, load_script, raw, expected):
    mod = load_script("04_train_and_evaluate")
    if raw is None:
        monkeypatch.delenv("QSAR_RF_JOBS", raising=False)
    else:
        monkeypatch.setenv("QSAR_RF_JOBS", raw)
    assert mod.rf_jobs() == expected


@pytest.mark.parametrize("raw", ["", "0", "-2", "1.5", "eight"])
def test_invalid_rf_jobs_fail_before_data_or_outputs(monkeypatch, load_script, raw):
    mod = load_script("04_train_and_evaluate")
    monkeypatch.setenv("QSAR_RF_JOBS", raw)
    monkeypatch.setattr(sys, "argv", ["04_train_and_evaluate.py", "--split", "time"])
    monkeypatch.setattr(mod, "resolve_time_split", lambda: pytest.fail("loaded data"))
    with pytest.raises(ValueError, match="QSAR_RF_JOBS"):
        mod.main()


def test_rf_jobs_reaches_outer_grid_only(monkeypatch, load_script):
    mod = load_script("04_train_and_evaluate")
    monkeypatch.setenv("QSAR_RF_JOBS", "3")
    monkeypatch.delenv("QSAR_CV_BACKEND", raising=False)
    captured = {}

    class FakeGrid:
        def __init__(self, estimator, grid, **kwargs):
            captured.update(outer=kwargs["n_jobs"], inner=estimator.n_jobs, grid=grid)
            self.best_estimator_ = estimator
            self.best_params_, self.best_score_ = {}, 0.5

        def fit(self, X, y):
            captured["train_rows"] = len(y)

    monkeypatch.setattr(mod, "GridSearchCV", FakeGrid)
    monkeypatch.setattr(mod, "evaluate", lambda *args: ({"r2": 0.5}, np.array([1.])))
    monkeypatch.setattr(mod, "scatter", lambda *args: None)
    m, _, _ = mod.run_split(np.zeros((3, 2)), np.arange(3.), np.array(["C"] * 3),
                           np.array([0, 1]), np.array([2]), "time", "unused", "tiny")
    assert captured == {"outer": 3, "inner": 1, "grid": mod.PARAM_GRID, "train_rows": 2}
    assert m["runtime"]["rf_jobs"] == 3


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_unavailable_accelerator_fails_before_data(monkeypatch, load_script, requested):
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["gnn_03_train_gin.py", "--split", "time",
                                     "--device", requested])
    monkeypatch.setattr(mod.np, "load", lambda *a, **kw: pytest.fail("loaded data"))
    with pytest.raises(ValueError, match="no CPU fallback"):
        mod.main()


@pytest.mark.parametrize("requested", ["cpu", "cuda", "mps"])
def test_explicit_available_device_is_honored(monkeypatch, load_script, requested):
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert mod.resolve_device(requested) == requested


def test_auto_preserves_selection_and_runtime_provenance(monkeypatch, load_script):
    mod = load_script("gnn_03_train_gin")
    monkeypatch.setattr(mod, "select_device", lambda: "cpu")
    monkeypatch.setattr(sys, "argv", ["gnn_03_train_gin.py", "--split", "time"])
    assert mod.parse_args().device == "auto"
    assert mod.resolve_device("auto") == "cpu"
    runtime = mod.runtime_metadata(SimpleNamespace(device="auto", seed=2), "cpu")
    assert runtime["seed"] == 2 and runtime["device"] == "cpu"
    assert runtime["requested_device"] == "auto"
    assert runtime["torch_threads"] == torch.get_num_threads()
    assert runtime["torch_interop_threads"] == torch.get_num_interop_threads()
    assert runtime["torch_version"] == str(torch.__version__)
