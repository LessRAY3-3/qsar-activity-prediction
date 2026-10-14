"""Behavioral regressions for PR #6 S5/S7 and latent statistical-tool issues."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _cell(tag="tiny", split="random", seed=42):
    return [{"tag": tag, "split": split, "model_pair": pair, "n_test": 3,
             "delta_r2_point": 0.1, "ci_lo": 0.01, "ci_hi": 0.2,
             "significant": True, "winner": "gin", "frac_boot_gin_better": 0.99,
             "delta_rmse_point": -0.1, "delta_rmse_ci_lo": -0.2,
             "delta_rmse_ci_hi": -0.01, "n_boot": 300, "boot_seed": seed}
            for pair in ("seed42", "mean3")]


def _summary(mod, seed=42):
    return pd.DataFrame([{**r, "pair": r["model_pair"], "d_r2": r["delta_r2_point"],
                          "d_rmse": r["delta_rmse_point"],
                          "d_rmse_ci_lo": r["delta_rmse_ci_lo"],
                          "d_rmse_ci_hi": r["delta_rmse_ci_hi"]}
                         for r in _cell(seed=seed)])


def test_summary_seed_guard_before_any_cell_write(tmp_path, monkeypatch, load_experiment):
    mod = load_experiment("paired_bootstrap")
    path = tmp_path / "summary_partial.csv"
    _summary(mod, seed=7).to_csv(path, index=False)
    cell = tmp_path / "tiny_random.csv"
    cell.write_text("sentinel\n")
    monkeypatch.setattr(mod, "cell_rows", lambda *a, **k: pytest.fail("must preflight first"))
    with pytest.raises(SystemExit, match="boot_seed"):
        mod.main(["--tags", "tiny", "--splits", "random", "--n-boot", "300",
                  "--out-dir", str(tmp_path)])
    assert cell.read_text() == "sentinel\n"
    with pytest.raises(SystemExit, match="boot_seed"):
        mod.merge_summary(_summary(mod), str(path), n_boot=300)
    got = mod.merge_summary(_summary(mod), str(path), n_boot=300, force=True)
    assert set(got.boot_seed) == {42}


def test_late_input_error_preserves_all_existing_cells(tmp_path, monkeypatch, load_experiment):
    mod = load_experiment("paired_bootstrap")
    paths = [tmp_path / f"{tag}_random.csv" for tag in ("tiny", "later")]
    for path in paths:
        pd.DataFrame(_cell()).to_csv(path, index=False)
    before = {p: p.read_bytes() for p in paths}

    def load(tag, split, **kwargs):
        if tag == "later":
            raise ValueError("RF labels differ")
        return _cell(tag, split)

    monkeypatch.setattr(mod, "cell_rows", load)
    with pytest.raises(ValueError, match="RF labels"):
        mod.main(["--tags", "tiny,later", "--splits", "random", "--n-boot", "300",
                  "--out-dir", str(tmp_path)])
    assert {p: p.read_bytes() for p in paths} == before
    assert not (tmp_path / "summary_partial.csv").exists()


@pytest.mark.parametrize("alias", [False, True])
def test_figure_routing_uses_physical_campaign_directory(tmp_path, monkeypatch,
                                                        load_experiment, alias):
    mod = load_experiment("paired_bootstrap")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    campaign = tmp_path / "results/significance"
    campaign.mkdir(parents=True)
    out = tmp_path / "output"
    if alias:
        out.symlink_to(campaign, target_is_directory=True)
    monkeypatch.setattr(mod, "cell_rows", lambda tag, split, **kw: _cell(tag, split))
    saved = []
    monkeypatch.setattr(mod, "make_figure", lambda rows, path, **kw: saved.append(Path(path)))
    mod.main(["--tags", "tiny", "--splits", "random", "--n-boot", "300",
              "--out-dir", str(out)])
    expected = tmp_path / "figures/significance" if alias else out
    assert saved == [expected / "ci_panel_partial.png"]


def test_same_training_pool_different_tests_is_not_verified(tmp_path, monkeypatch,
                                                           load_experiment):
    mod = load_experiment("xgboost_baseline")
    (tmp_path / "results").mkdir()
    (tmp_path / "results/metrics_tiny.json").write_text(json.dumps({"random": {"r2": 0.5}}))
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    kwargs = dict(tag="tiny", xgb_pools={"random": np.array([0, 1])},
                  rf_pools={"random": np.array([0, 1])}, test_sizes={"random": 1},
                  n_total=4, xgb_tests={"random": np.array([2])},
                  rf_tests={"random": np.array([3])},
                  rf_sources={"random": "results/rf_preds_tiny_random.npz:test_idx"})
    control = mod.load_rf_control(**kwargs)["random"]
    assert control["same_training_pool"] is True
    assert control["same_pool"] is False
    assert control["rf_test_sha256"] != control["xgb_test_sha256"]
    assert control["rf_pool_source"] == kwargs["rf_sources"]["random"]
    kwargs["rf_tests"] = {"random": np.array([2])}
    assert mod.load_rf_control(**kwargs)["random"]["same_pool"] is True
    kwargs["rf_tests"] = None
    assert mod.load_rf_control(**kwargs)["random"]["same_pool"] is False


def test_xgb_refit_rejects_unpaired_or_incomplete_inputs(tmp_path, load_experiment):
    mod = load_experiment("xgboost_paired")
    path = tmp_path / "split.npz"
    np.savez(path, train_idx=[0, 1], valid_idx=[2], test_idx=[3, 4])
    fp = {"y": np.arange(5.0)}
    with np.load(path) as sp:
        rf = {"test_idx": np.array([3, 4]), "y_true": np.array([3.0, 4.0])}
        assert mod.verify_inputs(fp, sp, rf)[0].tolist() == [0, 1, 2]
        with pytest.raises(ValueError, match="RF test_idx"):
            mod.verify_inputs(fp, sp, {**rf, "test_idx": np.array([4, 3])})
        with pytest.raises(ValueError, match="RF labels"):
            mod.verify_inputs(fp, sp, {**rf, "y_true": np.array([3.0, 5.0])})
        with pytest.raises(ValueError, match="partition"):
            mod.verify_inputs({"y": np.arange(6.0)}, sp, rf)


def test_afp_win_is_nominal_not_evidence_for_attention_component(load_experiment, capsys):
    mod = load_experiment("afp_panel")
    frame = pd.DataFrame([{"tag": "tiny", "split": "random", "rf_r2": 0.5,
                           "afp_r2_mean": 0.6, "afp_r2_std": 0.1,
                           "d_r2_mean": 0.1, "ci_lo": 0.02, "ci_hi": 0.2,
                           "significant": True}])
    mod.print_table(frame)
    out = capsys.readouterr().out
    assert "nominal per-cell" in out
    assert "uncorrected for multiple comparisons" in out
    assert "do not establish robustness" in out
    assert "real somewhere" not in out
