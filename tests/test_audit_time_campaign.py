"""Failure-oriented checks for the independent current-artifact auditor."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import audit_time_campaign as audit


def toy_labels():
    raw = pd.DataFrame({"canonical_smiles": ["raw-A", "raw-A", "raw-B", "raw-C"],
                        "document_chembl_id": ["d1", "d3", "d2", "d3"],
                        "standard_value": [100., 10000., 1000., 10.],
                        "standard_relation": ["=", "=", "=", "="],
                        "standard_units": ["nM"] * 4})
    years = pd.Series({"d1": 2000, "d2": 2015, "d3": 2022})
    split = dict(smiles=np.array(["A", "B", "C"]), year=np.array([2000, 2015, 2022]),
                 train_idx=np.array([0]), valid_idx=np.array([1]), test_idx=np.array([2]),
                 excluded_idx=np.array([], dtype=int), y=np.array([7., 6., 8.], dtype=np.float32),
                 cutoff_valid=2010, cutoff_test=2020)
    return raw, years, split


def test_raw_windows_exclude_future_measurements_and_do_not_renormalize_cache():
    raw, years, split = toy_labels()

    def parent(value):
        # Cached keys A/B/C must never be normalized a second time.
        assert value.startswith("raw-")
        return value.removeprefix("raw-")

    assert audit.rebuild_labels(raw, years, split, parent) == dict(train=1, valid=1, test=1, excluded=0)


def test_wrongly_excluded_eligible_row_is_detected():
    raw, years, split = toy_labels()
    split.update(train_idx=np.array([], dtype=int), excluded_idx=np.array([0]))
    split["y"][0] = np.nan
    with pytest.raises(AssertionError, match="membership"):
        audit.rebuild_labels(raw, years, split, lambda s: s.removeprefix("raw-"))


def test_future_contaminated_label_is_detected():
    raw, years, split = toy_labels()
    split["y"][0] = 6.0  # Median including 2022's observation would leak future data.
    with pytest.raises(AssertionError, match="labels"):
        audit.rebuild_labels(raw, years, split, lambda s: s.removeprefix("raw-"))


def test_prediction_label_version_cannot_be_mixed():
    split = dict(test_idx=np.array([0, 1]), y=np.array([1., 2.]), label_id="current")
    pred = dict(test_idx=np.array([0, 1]), y_true=np.array([1., 2.]),
                y_pred=np.array([1., 2.]), label_id="old")
    with pytest.raises(ValueError, match="version"):
        audit.check_prediction(pred, split, {"label_id": "current"})


def test_bootstrap_uses_requested_count_including_final_partial_chunk():
    y = np.arange(5., dtype=float)
    pred = y + 1
    result = audit.bootstrap(y, y, pred, 137, 7)
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(137):
        sample = y[rng.integers(0, len(y), size=len(y))]
        sst = np.sum((sample - sample.mean()) ** 2)
        if sst:
            draws.append(-len(y) / sst)
    np.testing.assert_allclose([result["ci_lo"], result["ci_hi"]],
                               np.percentile(draws, [2.5, 97.5]), rtol=0, atol=1e-12)
    assert result["frac_boot_gin_better"] == 0


def test_cli_missing_inputs_returns_failure_report(tmp_path):
    report = tmp_path / "audit.json"
    assert audit.main(["--root", str(tmp_path), "--report", str(report)]) == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert "FileNotFoundError" in payload["error"]


def test_data_scope_skips_unavailable_models_and_features_but_checks_splits(tmp_path, monkeypatch):
    model = "models/rf_time_split_a2a.joblib"
    feature = "data/processed/a2a_graphs.npz"
    split = "data/processed/splits/a2a_time.npz"
    for name in (split, "input_manifest.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")
    input_hash = hashlib.sha256(b"input").hexdigest()
    provenance = dict(inputs={feature: "missing-feature-hash", split: input_hash}, code={},
                      bundle_manifest={"sha256": input_hash})
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    jobs = [dict(tag=tag, role=role, seed=seed, state="success", provenance=provenance,
                 signature=signature, input_manifest="input_manifest.json")
            for tag in audit.TAGS for role, seeds in (("rf", (42,)), ("gin", (42, 1, 2)))
            for seed in seeds]
    acceptance = dict(jobs=jobs, integrated_files=[dict(path=model, sha256="missing-model-hash")],
                      analysis={"files": []})
    assert audit.check_provenance(tmp_path, acceptance, "data") == (0, sorted([model, feature]))
    with pytest.raises(FileNotFoundError):
        audit.check_provenance(tmp_path, acceptance, "full")
    (tmp_path / split).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="input SHA256"):
        audit.check_provenance(tmp_path, acceptance, "data")


def test_historical_archive_manifest_matches_every_snapshot():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "scripts/campaign_history/manifest.json").read_text())
    assert len(manifest["files"]) >= 13
    for record in manifest["files"]:
        path = root / record["archived_path"]
        assert path.suffix == ".txt"
        data = path.read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
