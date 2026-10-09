"""04_train_and_evaluate records best_params_{split} for every split it runs.

Reuses the synthetic setup of test_rf_time_split (same tmp_path contract,
single-config grid): ``--split time`` must add ``best_params_time`` while
merging, and the default path must rewrite the file with BOTH
``best_params_random`` and ``best_params_scaffold``.
"""
import json
import sys

from test_rf_time_split import _setup, _write_time_split


def test_split_time_writes_best_params_time(tmp_path, monkeypatch, load_script):
    mod, backup, metrics_path, split_dir, _ = _setup(tmp_path, monkeypatch, load_script)
    _write_time_split(split_dir, mod.TAG)

    monkeypatch.setattr(sys, "argv", ["04_train_and_evaluate.py", "--split", "time"])
    assert mod.main() is None  # exit 0

    payload = json.loads(metrics_path.read_text())
    bp = payload["best_params_time"]
    assert isinstance(bp, dict)
    assert bp["n_estimators"] == 3          # shrunk grid from _setup
    assert bp["max_depth"] is None
    assert bp["min_samples_split"] == 2
    # the merge touches only time / best_params_time
    assert payload["random"] == backup["random"]
    assert payload["scaffold"] == backup["scaffold"]
    assert payload["best_params_random"] == backup["best_params_random"]
    assert "best_params_scaffold" not in payload  # default-path key, not merged here


def test_default_path_writes_best_params_for_both_splits(tmp_path, monkeypatch, load_script):
    mod, _, metrics_path, _, _ = _setup(tmp_path, monkeypatch, load_script)

    monkeypatch.setattr(sys, "argv", ["04_train_and_evaluate.py"])
    assert mod.main() is None  # exit 0

    payload = json.loads(metrics_path.read_text())
    for split in ("random", "scaffold"):
        bp = payload[f"best_params_{split}"]
        assert isinstance(bp, dict)
        assert bp["n_estimators"] == 3
        assert bp["max_depth"] is None
        assert bp["min_samples_split"] == 2
        assert -1.0 <= payload[split]["r2"] <= 1.0  # metrics written alongside
    assert payload["target"] == "Synthetic kinase"
