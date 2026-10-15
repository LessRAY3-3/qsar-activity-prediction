"""Check source attribution, time-window labels and clustered resampling."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load("source_heterogeneity")
rf = load("rf_core_confirmation")
verifier = load("verify_rf_core_confirmation")
gin_audit = load("audit_core_gin_artifacts")


def test_source_memberships_preserves_overlap_and_unknown():
    data = pd.DataFrame({"parent": ["a", "a", "a", "b"],
                         "document": ["D1", "D2", "D1", None]})
    assert source.source_memberships(data, ["b", "a"], "document") == [("UNKNOWN",), ("D1", "D2")]
    with pytest.raises(ValueError, match="without source"):
        source.source_memberships(data, ["c"], "document")


def test_time_records_exclude_future_train_and_old_test_measurements():
    data = pd.DataFrame({"parent": ["a"]*3+["b"]*3+["c"]*3,
                         "document_year": [2010, 2015, 2020]*3,
                         "pic50": [4., 5., 6.]*3})
    sp = {"smiles": np.array(["a", "b", "c"]), "train_idx": np.array([0]),
          "valid_idx": np.array([1]), "test_idx": np.array([2]),
          "cutoff_valid": 2015, "cutoff_test": 2020}
    for pool, expected in [("train", [4.]), ("valid", [4., 5.]), ("test", [6.])]:
        assert source.pool_records(data, sp, "time", pool).pic50.tolist() == expected
    assert len(source.pool_records(data, sp, "random", "train")) == 3


def test_labels_must_match_median_pic50_in_molecule_order():
    data = pd.DataFrame({"parent": ["a", "a", "b"], "pic50": [4., 8., 5.]})
    assert source.verify_labels(data, ["b", "a"], [5., 6.]) == 0
    for keys, labels in [(["a", "b"], [5., 6.]), (["a", "c"], [6., 5.])]:
        with pytest.raises(ValueError, match="reconstruction"):
            source.verify_labels(data, keys, labels)


def test_eligible_records_use_exact_labels_not_pchembl_presence():
    raw = pd.DataFrame({"standard_value": [1, 10, 100, 1000], "standard_relation": [" = ", ">", "=", "="],
                        "standard_units": ["nM", "nM", "uM", "nM"],
                        "canonical_smiles": ["C"]*4, "standard_type": ["IC50"]*4,
                        "assay_type": ["B"]*4, "document_chembl_id": ["D", "D", "D", None],
                        "pchembl_value": [np.nan]*4})
    years = pd.DataFrame({"document_chembl_id": ["D"], "document_year": [2020]})
    data = source.eligible_records(raw, years, {"C": "C"})
    assert data.pic50.tolist() == [9., 6.]
    assert data.document_chembl_id.tolist() == ["D", "UNKNOWN"]
    with pytest.raises(ValueError, match="conflicting"):
        source.eligible_records(raw, pd.concat([years, years.assign(document_year=2021)]), {"C": "C"})


@pytest.mark.parametrize("y", [[], [1], [2, 2, 2]])
def test_degenerate_r2_is_missing_not_zero(y):
    result = source.metrics(y, y, y)
    assert result["delta_r2"] is None
    assert result["rf_r2"] is None


def test_cluster_sufficient_statistics_match_expanded_molecule_draws():
    # Unequal document sizes are essential: resampling molecules or giving
    # each document a single averaged label would produce different answers.
    groups = np.array(["A", "A", "B", "C", "C", "C", "D", "E"])
    y = np.array([1., 3., 2., 3., 5., 9., 4., 6.])
    pred_rf = y + np.arange(8)*.07
    pred_gin = y + .2
    result = source.cluster_interval(y, pred_rf, pred_gin, groups, draws=1000)
    rng = np.random.default_rng(42)
    values = []
    # Same documented multinomial draw algorithm, explicit molecule expansion.
    for start in range(0, 1000, 256):
        for counts in rng.multinomial(5, np.full(5, .2), size=min(256, 1000-start)):
            indices = np.concatenate([np.tile(np.flatnonzero(groups == group), count)
                                      for group, count in zip(np.unique(groups), counts)])
            delta = source.metrics(y[indices], pred_rf[indices], pred_gin[indices])["delta_r2"]
            if delta is not None:
                values.append(delta)
    assert result["valid_draws"] == len(values)
    np.testing.assert_allclose([result["ci_low"], result["ci_high"]], np.quantile(values, [.025, .975]), atol=1e-12)


def test_tiny_or_degenerate_cluster_cohorts_have_no_interval():
    result = source.cluster_interval([1, 2], [1, 2], [2, 2], ["A", "B"])
    assert result["status"] == "too_few_documents"
    result = source.cluster_interval([1]*5, [1]*5, [2]*5, list("ABCDE"), draws=1000)
    assert result["status"] == "too_many_degenerate_draws"
    assert result["ci_low"] is None
    with pytest.raises(ValueError, match="exclusive"):
        source.cluster_interval([1], [1], [1], ["UNKNOWN"])


def test_rf_training_preserves_original_full_pool_order():
    from sklearn.model_selection import train_test_split
    fp = {"y": np.arange(20.), "smiles": np.array([str(i) for i in range(20)])}
    tr, te = train_test_split(np.arange(20), test_size=.2, random_state=42)
    sp = {"train_idx": tr[2:][::-1], "valid_idx": tr[:2], "test_idx": te, "smiles": fp["smiles"]}
    actual, _ = rf.training_indices(fp, sp, "random")
    np.testing.assert_array_equal(actual, tr)
    sp["test_idx"] = te[::-1]
    with pytest.raises(ValueError, match="pool differs"):
        rf.training_indices(fp, sp, "random")


def test_rf_input_manifest_rejects_partial_cohort(tmp_path):
    with pytest.raises(ValueError, match="cohort"):
        rf.verify_inputs(tmp_path, {"files": {}})


@pytest.mark.parametrize("field,value", [("status", "running"), ("input_manifest_sha256", "other"),
                                        ("environment_sha256", "other"), ("data_identity", "other"),
                                        ("parameters", {"random_state": 1})])
def test_rf_receipt_rejects_changed_identity(field, value):
    receipt = {"status": "complete", "input_manifest_sha256": "manifest", "environment_sha256": "env",
               "data_identity": "data", "parameters": {"random_state": 42}}
    verifier.validate_receipt(receipt, "manifest", "env", "data", {"random_state": 42})
    receipt[field] = value
    with pytest.raises(ValueError):
        verifier.validate_receipt(receipt, "manifest", "env", "data", {"random_state": 42})


@pytest.mark.parametrize("change", ["seed", "normalization", "label_id", "weights"])
def test_gin_audit_rejects_mismatched_checkpoint(change):
    import torch
    saved = {"args": {"tag": "a2a", "split": "time", "seed": 42, "model": "gin"},
             "y_mean": 6., "y_std": 1., "label_id": "labels", "state_dict": {"weight": torch.tensor([1.])}}
    gin_audit.check_checkpoint(saved, "a2a", "time", 42, 6., 1., "labels")
    if change == "seed":
        saved["args"]["seed"] = 1
    elif change == "normalization":
        saved["y_mean"] = 6.1
    elif change == "label_id":
        saved["label_id"] = "future"
    else:
        saved["state_dict"]["weight"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError):
        gin_audit.check_checkpoint(saved, "a2a", "time", 42, 6., 1., "labels")
