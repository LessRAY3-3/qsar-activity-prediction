"""Auxiliary experiments must not silently use all-year labels for time splits."""
import sys

import pytest

EXPERIMENTS = ("xgboost_baseline", "learning_curve", "y_randomization")


@pytest.mark.parametrize("name", EXPERIMENTS)
@pytest.mark.parametrize("splits", ["time", "random,time", "random, time ", "scafflod"])
def test_unsupported_split_fails_before_data_reads_or_output_changes(
    load_experiment, tmp_path, monkeypatch, capsys, name, splits
):
    experiment = load_experiment(name)
    out = tmp_path / "results"
    figs = tmp_path / "figures"
    out.mkdir()
    previous = out / "existing.csv"
    previous.write_bytes(b"previous accepted result\n")
    monkeypatch.setattr(sys, "argv", [name, "--splits", splits,
                                     "--out", str(out), "--fig-dir", str(figs)])
    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported temporal experiment must fail before loading cached labels")
    monkeypatch.setattr(experiment.np, "load", unexpected)
    with pytest.raises(SystemExit) as exc:
        experiment.main()
    assert exc.value.code == 2
    assert "time requires as-of labels" in capsys.readouterr().err
    assert previous.read_bytes() == b"previous accepted result\n"
    assert list(out.iterdir()) == [previous]
    assert not figs.exists()


@pytest.mark.parametrize("name", EXPERIMENTS)
@pytest.mark.parametrize("splits", [None, "random", "scaffold", "random,scaffold"])
def test_supported_cli_splits_remain_available(load_experiment, monkeypatch, name, splits):
    experiment = load_experiment(name)
    monkeypatch.setattr(sys, "argv", [name] + ([] if splits is None else ["--splits", splits]))
    assert experiment.parse_args().splits == (splits or "random,scaffold")
