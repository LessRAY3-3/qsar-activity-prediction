"""Synthetic four-cell fixtures test strict completeness and contrast direction."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save_meta(root, metadata):
    (root / "report.json").write_text(json.dumps(metadata))


def write_diagnostics(root, metadata, frame, comparator):
    frame.to_csv(root / "diagnostics.csv", index=False)
    metadata["output_sha256"]["diagnostics.csv"] = comparator.digest(root / "diagnostics.csv")
    save_meta(root, metadata)


def fake_report(root, comparator, device, merged=False):
    root.mkdir()
    rows = []
    for arm in comparator.ARMS:
        for tag in comparator.TAGS:
            for index, seed in enumerate(comparator.SEEDS):
                # CPU budget .2 + index/100; CUDA budget .4 + index/50.
                offset = (0.5 if device == "cuda" else 0)
                if arm == "extended":
                    offset += (0.4 + index/50) if device == "cuda" else (0.2 + index/100)
                # Distinct flag counts per cell; boundary epoch 30 never flags.
                flagged_seeds = {("cpu", "control"): 1, ("cpu", "extended"): 2,
                                 ("cuda", "control"): 3, ("cuda", "extended"): 4}
                best_epoch = 29 if index < flagged_seeds[(device, arm)] else 30
                rows.append(dict(tag=tag, arm=arm, seed=seed, label_id=sha(tag), device=device,
                                 best_epoch=best_epoch, common_flag=best_epoch < 30,
                                 **{metric: (-2 if metric == "test_r2" else j + 1) + offset
                                    for j, metric in enumerate(comparator.METRICS)}))
    pd.DataFrame(rows).to_csv(root / "diagnostics.csv", index=False)
    for name in comparator.REPORT_FILES - {"diagnostics.csv"}:
        (root / name).write_text("fixture\n")
    env = {"platform": "macOS-arm64" if device == "cpu" else "Linux-x86_64",
           "packages": {"torch": "cpu-test" if device == "cpu" else "cuda-test"}}
    provenance = {"source_sha256": {name: sha(name) for name in comparator.CODE_FILES},
                  "input_sha256": {name: sha(name) for name in comparator.INPUT_FILES},
                  "host": device + "-host", "training_environment": env,
                  "protocol": {"device": device, "arms": comparator.ARMS,
                               "seeds": list(comparator.SEEDS), "jobs": 128}}
    if merged:
        cohorts = []
        for seeds in (comparator.SEEDS[:3], comparator.SEEDS[3:]):
            cohort = copy.deepcopy(provenance)
            cohort["protocol"].update(seeds=list(seeds), jobs=16 * len(seeds))
            cohorts.append(cohort)
        provenance["protocol"] = copy.deepcopy(cohorts[0]["protocol"])
        provenance["cohorts"] = cohorts
    metadata = {"schema": "m3-time-budget-report-v1", "status": "accepted", "training_device": device,
                "training_host": provenance["host"], "training_platform": env["platform"],
                "seeds": list(comparator.SEEDS), "n_seeds": 8, "jobs_accepted": 128,
                "paired_rows": 64, "targets": 8, "factorial_cells_complete": 2,
                "training_prefix_comparison": {
                    "pairs": 64, "matching_rounded_prefixes": 64 if device == "cpu" else 0,
                    "interpretation": "Only printed epochs and rounded train/validation/lr values are compared; no bitwise-equivalence requirement. Differences are reported, not suppressed or used to select runs."},
                **copy.deepcopy(comparator.OUTCOME_DEFINITIONS), "provenance": provenance,
                "output_sha256": {name: comparator.digest(root / name) for name in comparator.REPORT_FILES}}
    save_meta(root, metadata)
    return metadata


@pytest.fixture
def reports(tmp_path, load_experiment):
    comparator = load_experiment("time_environment_comparison")
    cpu, cuda = tmp_path / "cpu", tmp_path / "cuda"
    cpu_meta = fake_report(cpu, comparator, "cpu", merged=True)
    cuda_meta = fake_report(cuda, comparator, "cuda")
    return comparator, cpu, cuda, cpu_meta, cuda_meta, tmp_path / "comparison"


def test_full_four_cell_contrasts_and_descriptive_seed_summaries(reports):
    comparator, cpu, cuda, _, _, output = reports
    before = {str(path): path.read_bytes() for root in (cpu, cuda) for path in root.iterdir()}
    report = comparator.run(cpu, cuda, output)
    assert report["status"] == "accepted" and report["factorial_cells_complete"] == 4
    assert report["jobs_accepted"] == 256 and report["paired_target_seed_rows"] == 64
    cells = pd.read_csv(output / "cells.csv")
    assert list(cells.columns) == ["device", "arm", "jobs", "mean_train_mse",
                                   "mean_valid_rmse_std", "mean_test_r2", "common_flags"]
    assert list(cells[["device", "arm"]].itertuples(index=False, name=None)) == [
        ("cpu", "control"), ("cpu", "extended"), ("cuda", "control"), ("cuda", "extended")]
    assert cells.jobs.tolist() == [64] * 4
    np.testing.assert_allclose(cells.mean_train_mse, [1, 1.235, 1.5, 1.97])
    np.testing.assert_allclose(cells.mean_valid_rmse_std, [2, 2.235, 2.5, 2.97])
    np.testing.assert_allclose(cells.mean_test_r2, [-2, -1.765, -1.5, -1.03])
    assert cells.common_flags.tolist() == [8, 16, 24, 32]
    assert set(report["output_sha256"]) == {"cells.csv", "paired.csv", "targets.csv", "cross_target_means.csv"}
    assert "64 accepted" in report["cells_summary"]["aggregation"]
    assert "not a standard deviation" in report["cells_summary"]["columns"]["mean_valid_rmse_std"]
    pair = pd.read_csv(output / "paired.csv")
    target = pd.read_csv(output / "targets.csv")
    across = pd.read_csv(output / "cross_target_means.csv")
    assert len(pair) == 64 and len(target) == 8 and len(across) == 25
    for index, seed in enumerate(comparator.SEEDS):
        row = pair[(pair.tag == "a2a") & (pair.seed == seed)].iloc[0]
        for metric in comparator.METRICS:
            assert row[f"cpu_budget_{metric}"] == pytest.approx(.2 + index/100)
            assert row[f"cuda_budget_{metric}"] == pytest.approx(.4 + index/50)
            assert row[f"env_control_{metric}"] == pytest.approx(.5)
            assert row[f"env_extended_{metric}"] == pytest.approx(.7 + index/100)
            assert row[f"interaction_{metric}"] == pytest.approx(.2 + index/100)
    np.testing.assert_allclose(target.cpu_budget_train_mse_mean, .235)
    np.testing.assert_allclose(target.cpu_budget_train_mse_seed_sd, np.std(np.arange(8)/100, ddof=1))
    assert across.loc[across.contrast.eq("interaction") & across.metric.eq("test_r2"),
                      "mean_of_target_means"].iloc[0] == pytest.approx(.235)
    assert before == {str(path): path.read_bytes() for root in (cpu, cuda) for path in root.iterdir()}
    for name, expected in report["output_sha256"].items():
        assert comparator.digest(output / name) == expected
    assert "not data k-fold" in " ".join(report["limitations"])
    assert "EGFR" in " ".join(report["limitations"])
    assert not any("p_value" in name or "ci_" in name for name in target.columns)


@pytest.mark.parametrize("failure", ["missing_cuda", "not_accepted", "wrong_device", "partial_seeds",
                                    "partial_count", "outcomes", "source", "input", "inventory",
                                    "host", "cohort", "csv_hash"])
def test_missing_partial_or_incompatible_reports_fail_without_output(reports, failure):
    comparator, cpu, cuda, _, metadata, output = reports
    if failure == "missing_cuda":
        cuda = cuda.parent / "not-arrived"
    elif failure == "not_accepted":
        metadata["status"] = "pending"
    elif failure == "wrong_device":
        metadata["training_device"] = "cpu"
    elif failure == "partial_seeds":
        metadata["seeds"] = [42, 1, 2]
    elif failure == "partial_count":
        metadata["jobs_accepted"] = 48
    elif failure == "outcomes":
        metadata["primary_metrics"] = ["last-epoch dropout train MSE", "validation RMSE"]
    elif failure in ("source", "input", "inventory"):
        field = "source_sha256" if failure == "source" else "input_sha256"
        key = next(iter(metadata["provenance"][field]))
        if failure == "inventory":
            del metadata["provenance"][field][key]
        else:
            metadata["provenance"][field][key] = "0" * 64
    elif failure == "host":
        metadata["training_host"] = "unverified host"
    elif failure == "cohort":
        metadata["provenance"]["protocol"]["seeds"] = [42, 1, 2]
    elif failure == "csv_hash":
        (cuda / "paired.csv").write_bytes(b"changed")
    if failure != "missing_cuda":
        save_meta(cuda, metadata)
    with pytest.raises(ValueError):
        comparator.run(cpu, cuda, output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["missing_row", "duplicate", "wrong_seed", "label", "device", "nan", "negative", "r2",
                                     "missing_flags", "invalid_flags", "wrong_flags", "invalid_epoch"])
def test_rehashed_diagnostics_still_require_exact_matched_outcomes(reports, failure):
    comparator, cpu, cuda, _, metadata, output = reports
    frame = pd.read_csv(cuda / "diagnostics.csv")
    if failure == "missing_row":
        frame = frame.iloc[:-1]
    elif failure == "duplicate":
        frame.iloc[-1] = frame.iloc[0]
    elif failure == "wrong_seed":
        frame.loc[frame.seed == 7, "seed"] = 8
    elif failure == "label":
        frame.loc[frame.tag == "a2a", "label_id"] = sha("different labels")
    elif failure == "device":
        frame.loc[0, "device"] = "cpu"
    elif failure == "nan":
        frame.loc[0, "train_mse"] = np.nan
    elif failure == "negative":
        frame.loc[0, "test_rmse"] = -0.5
    elif failure == "r2":
        frame.loc[0, "test_r2"] = 1.5
    elif failure == "missing_flags":
        frame = frame.drop(columns="common_flag")
    elif failure == "invalid_flags":
        frame["common_flag"] = frame.common_flag.astype(str)
        frame.loc[0, "common_flag"] = "unknown"
    elif failure == "wrong_flags":
        frame.loc[0, "common_flag"] = not frame.loc[0, "common_flag"]
    elif failure == "invalid_epoch":
        frame["best_epoch"] = frame.best_epoch.astype(float)
        frame.loc[0, "best_epoch"] = 29.5
    write_diagnostics(cuda, metadata, frame, comparator)
    with pytest.raises(ValueError):
        comparator.run(cpu, cuda, output)
    assert not output.exists()


def test_existing_or_nested_output_is_never_overwritten(reports):
    comparator, cpu, cuda, _, _, output = reports
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_bytes(b"previous result")
    with pytest.raises(ValueError, match="already exists"):
        comparator.run(cpu, cuda, output)
    assert sentinel.read_bytes() == b"previous result"
    with pytest.raises(ValueError, match="separate"):
        comparator.run(cpu, cuda, cpu / "new-analysis")
    assert not (cpu / "new-analysis").exists()


@pytest.mark.parametrize("cuda_matching", [0, 17, 64])
def test_training_prefix_summaries_preserved_without_selecting_runs(reports, cuda_matching):
    comparator, cpu, cuda, cpu_metadata, cuda_metadata, output = reports
    cuda_metadata["training_prefix_comparison"]["matching_rounded_prefixes"] = cuda_matching
    save_meta(cuda, cuda_metadata)
    before = {str(path): path.read_bytes() for root in (cpu, cuda) for path in root.iterdir()}
    report = comparator.run(cpu, cuda, output)
    stored = json.loads((output / "report.json").read_text())
    assert stored["training_prefix_comparison"] == {
        "cpu": cpu_metadata["training_prefix_comparison"],
        "cuda": cuda_metadata["training_prefix_comparison"]}
    assert report["status"] == "accepted"
    assert report["jobs_accepted"] == 256
    assert report["paired_target_seed_rows"] == 64
    pairs = pd.read_csv(output / "paired.csv")
    assert set(zip(pairs.tag, pairs.seed)) == {(tag, seed) for tag in comparator.TAGS for seed in comparator.SEEDS}
    limitations = " ".join(report["limitations"])
    assert "Within-device" in limitations
    assert "run-to-run variation" in limitations
    assert "cannot be attributed solely to extending training" in limitations
    assert "interaction inherits this limitation" in limitations
    assert "Prefix mismatches do not exclude any declared runs" in limitations
    assert "Matching rounded prefixes do not establish bitwise-identical" in limitations
    assert before == {str(path): path.read_bytes() for root in (cpu, cuda) for path in root.iterdir()}
