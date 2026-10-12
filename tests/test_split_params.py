"""M2: the auxiliary analyses freeze RF hyperparameters per SPLIT.

Each aux script must read ``best_params_{split}`` out of
``results/metrics_{tag}.json`` (with a printed fallback to
``best_params_random`` when that split's key is absent) instead of silently
using the random-split winner everywhere.  Coverage:

* ``learning_curve.load_rf_params`` -- the one extracted loader, unit-tested
  for per-split selection, key filtering and the fallback note.
* ``fingerprint_ablation.main`` / ``y_randomization.main`` -- run end to end
  on a synthetic tree under tmp_path with distinct random/scaffold winners;
  the summary JSON's ``rf_params`` must record each split's own winner.
  (y_randomization's GIN arm is stubbed -- only the RF param path is under
  test, and the module is not allowed to train anything.)
* the committed fp_ablation / learning_curve / y_randomization summaries in
  results/ must still record params equal to metrics_{tag}.json, which is
  the evidence the review accepted for M2 (read-only check).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RF_KEYS = ("n_estimators", "max_depth", "min_samples_split")
# deliberately distinct per split so "always best_params_random" cannot pass
RANDOM_WINNER = {"n_estimators": 3, "max_depth": None, "min_samples_split": 2}
SCAFFOLD_WINNER = {"n_estimators": 7, "max_depth": 3, "min_samples_split": 5}


def _metrics_payload(with_scaffold=True):
    mj = {
        "random": {"split": "random", "r2": 0.5},
        "scaffold": {"split": "scaffold", "r2": 0.4},
        "best_params_random": dict(RANDOM_WINNER),
    }
    if with_scaffold:
        mj["best_params_scaffold"] = dict(SCAFFOLD_WINNER)
    return mj


def _write_tree(base, n=80, with_scaffold_params=True):
    """Synthetic fingerprints + split npz + metrics json under ``base``."""
    rng = np.random.default_rng(0)
    proc = base / "data" / "processed"
    split_dir = proc / "splits"
    split_dir.mkdir(parents=True)
    (base / "results").mkdir()
    smiles = np.array([f"CCO{i%7}" for i in range(n)])
    np.savez(proc / "tiny_fingerprints.npz",
             X=rng.integers(0, 2, size=(n, 2048)).astype(np.uint8),
             y=rng.normal(6.0, 1.0, n).astype(np.float32),
             smiles=smiles)
    splits = {
        "random": (np.arange(0, 48), np.arange(48, 64), np.arange(64, n)),
        "scaffold": (np.arange(0, 40), np.arange(40, 56), np.arange(56, n)),
    }
    for split, (tr, va, te) in splits.items():
        np.savez(split_dir / f"tiny_{split}.npz", train_idx=tr,
                 valid_idx=va, test_idx=te)
    (base / "results" / "metrics_tiny.json").write_text(
        json.dumps(_metrics_payload(with_scaffold_params)))
    return base


# ---------------------------------------------------------------- learning_curve

def test_load_rf_params_reads_that_split_not_random(load_experiment, capsys):
    mod = load_experiment("learning_curve")
    mj = _metrics_payload()
    mj["best_params_scaffold"]["n_jobs"] = 1          # extra keys are dropped

    assert mod.load_rf_params(mj, "random") == dict(RANDOM_WINNER)
    got = mod.load_rf_params(mj, "scaffold")
    assert got == dict(SCAFFOLD_WINNER)
    assert set(got) == set(RF_KEYS)                   # only the grid keys
    assert "not found" not in capsys.readouterr().out


def test_load_rf_params_falls_back_to_random_with_note(load_experiment, capsys):
    mod = load_experiment("learning_curve")
    got = mod.load_rf_params(_metrics_payload(with_scaffold=False), "time")
    assert got == dict(RANDOM_WINNER)
    out = capsys.readouterr().out
    assert "best_params_time not found; falling back to best_params_random" in out


# ---------------------------------------------------------------- fingerprint_ablation

def _run_fp_ablation(tmp_path, monkeypatch, load_experiment, with_scaffold_params,
                     splits="random,scaffold"):
    _write_tree(tmp_path, with_scaffold_params=with_scaffold_params)
    mod = load_experiment("fingerprint_ablation")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "fp_out"
    fig = tmp_path / "fp_fig"
    monkeypatch.setattr(sys, "argv", [
        "fingerprint_ablation.py", "--tag", "tiny",
        "--radii", "2", "--bits", "2048", "--seeds", "42",
        "--splits", splits, "--anchor-tol", "10",
        "--out", str(out), "--fig-dir", str(fig),
    ])
    assert mod.main() is None
    summary = json.loads((out / "summary_tiny.json").read_text())
    return mod, summary, out


def test_fp_ablation_records_per_split_best_params(tmp_path, monkeypatch,
                                                   load_experiment):
    _, summary, out = _run_fp_ablation(tmp_path, monkeypatch, load_experiment,
                                       with_scaffold_params=True)
    assert summary["rf_params"] == {
        "random": dict(RANDOM_WINNER),
        "scaffold": dict(SCAFFOLD_WINNER),
    }
    assert summary["anchor_failures"] == []
    assert (out / "fp_ablation_tiny.csv").exists()


def test_fp_ablation_falls_back_when_split_key_missing(tmp_path, monkeypatch,
                                                       load_experiment, capsys):
    _, summary, _ = _run_fp_ablation(tmp_path, monkeypatch, load_experiment,
                                     with_scaffold_params=False,
                                     splits="scaffold")
    assert summary["rf_params"] == {"scaffold": dict(RANDOM_WINNER)}
    out = capsys.readouterr().out
    assert "best_params_scaffold not found; falling back to best_params_random" in out


# ---------------------------------------------------------------- y_randomization

def _run_y_randomization(tmp_path, monkeypatch, load_experiment,
                         with_scaffold_params, splits="random,scaffold"):
    _write_tree(tmp_path, n=160, with_scaffold_params=with_scaffold_params)
    mod = load_experiment("y_randomization")
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    # stub the GIN arm: only the RF per-split params path is under test
    monkeypatch.setattr(
        mod, "gin_train_eval",
        lambda *args, **kwargs: ({"r2": 0.0, "rmse": 1.0, "mae": 0.8}, -1),
    )
    out = tmp_path / "yr_out"
    fig = tmp_path / "yr_fig"
    monkeypatch.setattr(sys, "argv", [
        "y_randomization.py", "--tag", "tiny", "--seeds", "42",
        "--splits", splits, "--out", str(out), "--fig-dir", str(fig),
    ])
    try:
        assert mod.main() is None
    except SystemExit as exc:
        # a shuffled-label R2 >= 0.2 verdict exits 1 AFTER writing outputs;
        # argparse errors (code 2) must not be swallowed
        assert exc.code == 1
    summary = json.loads((out / "summary_tiny.json").read_text())
    return summary, out


def test_y_randomization_records_per_split_best_params(tmp_path, monkeypatch,
                                                       load_experiment):
    summary, out = _run_y_randomization(tmp_path, monkeypatch, load_experiment,
                                        with_scaffold_params=True)
    assert summary["rf_params"] == {
        "random": dict(RANDOM_WINNER),
        "scaffold": dict(SCAFFOLD_WINNER),
    }
    assert (out / "y_randomization_tiny.csv").exists()


def test_y_randomization_falls_back_when_split_key_missing(tmp_path, monkeypatch,
                                                           load_experiment, capsys):
    summary, _ = _run_y_randomization(tmp_path, monkeypatch, load_experiment,
                                      with_scaffold_params=False,
                                      splits="scaffold")
    assert summary["rf_params"] == {"scaffold": dict(RANDOM_WINNER)}
    out = capsys.readouterr().out
    assert "best_params_scaffold not found; falling back to best_params_random" in out


# ---------------------------------------------------------------- committed results

def test_committed_summaries_match_metrics_best_params():
    """Read-only: every committed aux summary must record the params equal
    to that tag's metrics_{tag}.json best_params_{split} (the M2 evidence)."""
    campaigns = ("fp_ablation", "learning_curve", "y_randomization")
    n_checked = 0
    for campaign in campaigns:
        summary_dir = ROOT / "results" / campaign
        for sum_path in sorted(summary_dir.glob("summary_*.json")):
            payload = json.loads(sum_path.read_text())
            tag = payload["tag"]
            mj = json.loads(
                (ROOT / "results" / f"metrics_{tag}.json").read_text())
            rf_params = payload["rf_params"]
            assert {"random", "scaffold"} <= set(rf_params), sum_path.name
            for split, params in rf_params.items():
                bp = mj[f"best_params_{split}"]
                for key in RF_KEYS:
                    assert params[key] == bp[key], \
                        f"{campaign}/{sum_path.name}/{split}/{key}"
            n_checked += 1
    # 8 fp_ablation + 7 learning_curve + 7 y_randomization summaries
    assert n_checked >= 20, f"only {n_checked} summaries checked"
