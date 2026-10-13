"""fingerprint_ablation: fail-fast split guards before any training or write.

The stored CSV/summary/figure are rewritten only after the WHOLE grid has
finished, and two guards refuse unsafe runs before any output file is
opened (6c1e3bb, PR #5 Low):
  * an unanchorable --splits entry (e.g. time) is rejected up front --
    the old behaviour was a KeyError at the anchor AFTER the CSV had
    already been truncated and partly rewritten;
  * a split-subset run (e.g. --splits random only) never rewrites
    stored two-split outputs unless --force.

Tests drive main() on a tiny synthetic tree under tmp_path; the guards
fire before sklearn is imported, so no model is ever trained. Independent
of the notebook byte-identity test (that test fails on ANY script edit;
these pin the behaviour itself).
"""
import json
import sys

import numpy as np
import pytest

HEADER = "split,radius,n_bits,seed,r2,rmse,mae,train_seconds"


def _write_tree(tmp_path, tag="tiny"):
    """Minimal inputs main() reads before the guards: fingerprints npz,
    metrics json (headline + best_params), split npz for both splits."""
    proc = tmp_path / "data" / "processed"
    (proc / "splits").mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 40
    y = rng.uniform(3.0, 9.0, n)
    np.savez(proc / f"{tag}_fingerprints.npz", y=y,
             smiles=np.array(["CCO"] * n),
             X=np.zeros((n, 8), dtype=np.uint8))
    idx = np.arange(n)
    for s in ("random", "scaffold"):
        np.savez(proc / "splits" / f"{tag}_{s}.npz",
                 train_idx=idx[:24], valid_idx=idx[24:32], test_idx=idx[32:])
    results = tmp_path / "results"
    results.mkdir()
    (results / f"metrics_{tag}.json").write_text(json.dumps({
        "random": {"r2": 0.5},
        "scaffold": {"r2": 0.4},
        "best_params_random": {"n_estimators": 10, "max_depth": 4,
                               "min_samples_split": 2},
        "best_params_scaffold": {"n_estimators": 10, "max_depth": 4,
                                 "min_samples_split": 2},
    }))
    return tmp_path


def _stored_csv(path, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for s in splits:
        for r in (2, 3):
            for b in (1024, 2048):
                for seed in (42, 1, 2):
                    lines.append(f"{s},{r},{b},{seed},0.5,1.0,0.7,0.1")
    path.write_text("\n".join(lines) + "\n")


def test_split_subset_run_refuses_before_any_write(tmp_path, monkeypatch,
                                                   load_experiment):
    """Mutant killed: no coverage guard (old behaviour) -- a --splits random
    run would rewrite the stored two-split CSV/summary/figure, silently
    dropping the scaffold rows from the committed grid."""
    mod = load_experiment("fingerprint_ablation")
    _write_tree(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "results" / "fp_ablation"
    fig = tmp_path / "figures"
    csv_path = out / "fp_ablation_tiny.csv"
    _stored_csv(csv_path, ("random", "scaffold"))
    before = csv_path.read_text()
    sum_path = out / "summary_tiny.json"
    sum_path.write_text("{}")

    monkeypatch.setattr(sys, "argv", [
        "fingerprint_ablation.py", "--tag", "tiny", "--splits", "random",
        "--out", str(out), "--fig-dir", str(fig)])

    with pytest.raises(SystemExit, match="drop"):
        mod.main()

    assert csv_path.read_text() == before       # every output untouched
    assert sum_path.read_text() == "{}"
    assert not (fig / "fp_ablation_tiny.png").exists()


def test_unanchorable_split_rejected_before_any_write(tmp_path, monkeypatch,
                                                      load_experiment):
    """Mutant killed: the old KeyError-at-anchor path -- `time` was opened,
    the CSV truncated and partly rewritten before the failure."""
    mod = load_experiment("fingerprint_ablation")
    _write_tree(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "results" / "fp_ablation"
    fig = tmp_path / "figures"
    csv_path = out / "fp_ablation_tiny.csv"
    _stored_csv(csv_path, ("random", "scaffold"))
    before = csv_path.read_text()

    monkeypatch.setattr(sys, "argv", [
        "fingerprint_ablation.py", "--tag", "tiny", "--splits", "time",
        "--out", str(out), "--fig-dir", str(fig)])

    with pytest.raises(SystemExit, match="unsupported --splits"):
        mod.main()

    assert csv_path.read_text() == before       # refused before any write


def test_grid_subset_run_refuses_before_any_write(tmp_path, monkeypatch,
                                                  load_experiment):
    """A radius/bits/seed subset run never rewrites the stored full-grid
    outputs (6c1e3bb extension per PR #6: the coverage guard now checks
    all four grid dimensions, not splits only).

    Mutant killed: the splits-only guard -- --radii 2 (dropping the stored
    radius-3 rows) would rewrite the CSV with fewer rows."""
    mod = load_experiment("fingerprint_ablation")
    _write_tree(tmp_path)
    monkeypatch.setattr(mod, "BASE", str(tmp_path))
    out = tmp_path / "results" / "fp_ablation"
    fig = tmp_path / "figures"
    csv_path = out / "fp_ablation_tiny.csv"
    _stored_csv(csv_path, ("random", "scaffold"))
    before = csv_path.read_text()

    monkeypatch.setattr(sys, "argv", [
        "fingerprint_ablation.py", "--tag", "tiny",
        "--splits", "random,scaffold", "--radii", "2",
        "--out", str(out), "--fig-dir", str(fig)])

    with pytest.raises(SystemExit, match="radius"):
        mod.main()

    assert csv_path.read_text() == before       # every output untouched
