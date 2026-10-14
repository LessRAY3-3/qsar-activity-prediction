"""Describe the complete CPU/CUDA x control/extended eight-seed budget study.

Requires two independently accepted time_budget_report directories. Missing or
unaccepted GPU results are an error, never a partial four-cell report. This
script only reads audited outcomes and writes to a new output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

SEEDS = (42, 1, 2, 3, 4, 5, 6, 7)
TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
ARMS = {"control": {"epochs": 100, "min_epochs": 0, "patience": 20},
        "extended": {"epochs": 200, "min_epochs": 60, "patience": 40}}
METRICS = ("train_mse", "valid_rmse_std", "test_r2", "test_rmse", "test_mae")
CONTRASTS = ("cpu_budget", "cuda_budget", "env_control", "env_extended", "interaction")
CODE_FILES = {"scripts/gnn_03_train_gin.py", "scripts/gnn_graph_dataset.py",
              "scripts/qsar_common.py", "scripts/run_time_campaign.py", "scripts/time_labels.py"}
INPUT_FILES = {name for tag in TAGS for name in (
    f"data/processed/{tag}_graphs.npz", f"data/processed/splits/{tag}_time.npz")}
REPORT_FILES = {"diagnostics.csv", "paired.csv", "targets.csv", "prefix_logs.csv"}
OUTCOME_DEFINITIONS = {
    "primary_metrics": ["best-checkpoint standardized train MSE", "validation RMSE"],
    "secondary_metrics": ["test R2", "test RMSE", "test MAE", "runtime"],
    "delta_convention": "extended minus control, paired by target and seed",
    "common_flag": "best_epoch < 30 and best-checkpoint standardized train MSE > 0.5",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_hash_inventory(values, names, label):
    require(isinstance(values, dict) and set(values) == names, f"Incomplete {label} hash inventory")
    require(all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in values.values()),
            f"Invalid {label} SHA256")


def load_report(root, device):
    """Recheck accepted report identity, integrity, coverage and outcome contract."""
    root = Path(root).resolve()
    require((root / "report.json").is_file(), f"Missing {device} accepted report: {root}")
    report = json.loads((root / "report.json").read_text())
    require(report.get("schema") == "m3-time-budget-report-v1" and report.get("status") == "accepted",
            f"{device} report is not an accepted budget audit")
    require(report.get("training_device") == device, f"{device} training_device differs")
    require(report.get("n_seeds") == 8 and len(report.get("seeds", [])) == 8
            and set(report["seeds"]) == set(SEEDS), f"{device} report must contain the eight declared seeds")
    require(report.get("jobs_accepted") == 128 and report.get("paired_rows") == 64
            and report.get("targets") == 8 and report.get("factorial_cells_complete") == 2,
            f"{device} report is not a complete two-cell cohort")
    for key, expected in OUTCOME_DEFINITIONS.items():
        require(report.get(key) == expected, f"{device} outcome definition differs: {key}")
    check_hash_inventory(report.get("output_sha256"), REPORT_FILES, f"{device} report output")
    for name, sha in report["output_sha256"].items():
        path = root / name
        require(path.is_file() and digest(path) == sha, f"{device} report CSV SHA256 mismatch: {name}")
    provenance = report["provenance"]
    check_hash_inventory(provenance.get("source_sha256"), CODE_FILES, f"{device} training source")
    check_hash_inventory(provenance.get("input_sha256"), INPUT_FILES, f"{device} input")
    require(report.get("training_host") == provenance["host"], f"{device} training host differs")
    require(report.get("training_platform") == provenance["training_environment"]["platform"],
            f"{device} training platform differs")
    cohorts = provenance.get("cohorts", [provenance])
    seen = set()
    for cohort in cohorts:
        for key in ("source_sha256", "input_sha256", "host", "training_environment"):
            require(cohort[key] == provenance[key], f"{device} mixed cohort {key}")
        protocol = cohort["protocol"]
        require(protocol["device"] == device and protocol["arms"] == ARMS,
                f"{device} cohort protocol differs")
        seeds = protocol["seeds"]
        require(len(seeds) == len(set(seeds)) and not seen.intersection(seeds)
                and protocol["jobs"] == 16 * len(seeds), f"{device} cohort seeds overlap or job count differs")
        seen.update(seeds)
    require(seen == set(SEEDS), f"{device} provenance has incomplete seed coverage")
    frame = pd.read_csv(root / "diagnostics.csv", dtype={"label_id": str, "tag": str, "arm": str})
    columns = {"arm", "tag", "seed", "label_id", "device", "best_epoch", "common_flag", *METRICS}
    require(columns <= set(frame.columns), f"{device} diagnostics missing required outcomes")
    require(len(frame) == 128 and not frame.duplicated(["arm", "tag", "seed"]).any(),
            f"{device} diagnostics contain missing/duplicate jobs")
    expected = {(arm, tag, seed) for arm in ARMS for tag in TAGS for seed in SEEDS}
    require(set(map(tuple, frame[["arm", "tag", "seed"]].to_numpy())) == expected,
            f"{device} diagnostics coverage differs")
    require(frame.device.eq(device).all(), f"{device} diagnostics training device differs")
    for tag in TAGS:
        labels = frame.loc[frame.tag == tag, "label_id"]
        require(labels.notna().all() and labels.nunique() == 1 and bool(labels.iloc[0]),
                f"{device}/{tag} has mixed/missing labels")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"{device} diagnostics contain non-finite outcomes")
    require((frame[[m for m in METRICS if m != "test_r2"]].to_numpy(dtype=float) >= 0).all(),
            f"{device} diagnostics contain negative squared/absolute errors")
    require((frame["test_r2"].to_numpy(dtype=float) <= 1 + 1e-12).all(),
            f"{device} diagnostics contain impossible R2 > 1")
    epochs = frame["best_epoch"].to_numpy(dtype=float)
    require(np.isfinite(epochs).all() and (epochs >= 1).all()
            and np.equal(epochs, np.floor(epochs)).all(), f"{device} diagnostics contain invalid best epochs")
    require(frame["common_flag"].map(lambda value: isinstance(value, (bool, np.bool_))).all(),
            f"{device} diagnostics common_flag must be boolean")
    expected_flags = frame.best_epoch.lt(30) & frame.train_mse.gt(0.5)
    require(frame.common_flag.eq(expected_flags).all(), f"{device} diagnostics common flags differ")
    return report, frame


def paired_contrasts(cpu, cuda):
    """One row per target/seed; preserve all four outcomes before differencing."""
    keys = ["tag", "seed"]
    cells = []
    for device, frame in (("cpu", cpu), ("cuda", cuda)):
        for arm in ARMS:
            cell = frame.loc[frame.arm == arm, [*keys, "label_id", *METRICS]].set_index(keys)
            cells.append(cell.add_prefix(f"{device}_{arm}_"))
    result = cells[0].join(cells[1:], how="inner").reset_index()
    require(len(result) == 64 and not result.duplicated(keys).any(), "Incomplete four-cell pairing")
    label_columns = [f"{device}_{arm}_label_id" for device in ("cpu", "cuda") for arm in ARMS]
    for column in label_columns[1:]:
        require(result[column].eq(result[label_columns[0]]).all(), "CPU/CUDA paired label versions differ")
    result["label_id"] = result[label_columns[0]]
    result = result.drop(columns=label_columns)
    for metric in METRICS:
        cc, ce = result[f"cpu_control_{metric}"], result[f"cpu_extended_{metric}"]
        gc, ge = result[f"cuda_control_{metric}"], result[f"cuda_extended_{metric}"]
        result[f"cpu_budget_{metric}"] = ce - cc
        result[f"cuda_budget_{metric}"] = ge - gc
        result[f"env_control_{metric}"] = gc - cc
        result[f"env_extended_{metric}"] = ge - ce
        result[f"interaction_{metric}"] = (ge - gc) - (ce - cc)
    return result.sort_values(keys).reset_index(drop=True)


def summarize_cells(cpu, cuda):
    """Four descriptive means over 8 targets x 8 seeds, with equal job weights.

    Completeness and label compatibility are checked before calling this helper.
    No cross-target dispersion, confidence interval or hypothesis test is added.
    """
    rows = []
    for device, frame in (("cpu", cpu), ("cuda", cuda)):
        for arm in ARMS:
            cell = frame.loc[frame.arm.eq(arm)]
            require(len(cell) == 64, f"{device}/{arm}: expected 64 jobs")
            rows.append({"device": device, "arm": arm, "jobs": len(cell),
                         "mean_train_mse": float(cell.train_mse.mean()),
                         "mean_valid_rmse_std": float(cell.valid_rmse_std.mean()),
                         "mean_test_r2": float(cell.test_r2.mean()),
                         "common_flags": int(cell.common_flag.sum())})
    return pd.DataFrame(rows)


def summarize(pairs):
    rows = []
    for tag in TAGS:
        subset = pairs[pairs.tag == tag]
        require(len(subset) == 8, f"{tag}: expected eight paired seeds")
        row = {"tag": tag, "n_seeds": 8}
        for contrast in CONTRASTS:
            for metric in METRICS:
                name = f"{contrast}_{metric}"
                row[f"{name}_mean"] = float(subset[name].mean())
                row[f"{name}_seed_sd"] = float(subset[name].std(ddof=1))
        rows.append(row)
    targets = pd.DataFrame(rows)
    # Equal target weights are descriptive only; overlapping datasets are not
    # independent experimental replications. No p-values or CIs are computed.
    across = pd.DataFrame([
        {"contrast": contrast, "metric": metric, "n_targets": 8,
         "mean_of_target_means": float(targets[f"{contrast}_{metric}_mean"].mean())}
        for contrast in CONTRASTS for metric in METRICS])
    return targets, across


def run(cpu_report, cuda_report, output):
    cpu_root, cuda_root, output = (Path(p).resolve() for p in (cpu_report, cuda_report, output))
    require(not output.exists(), "Output directory already exists; use a new comparison directory")
    require(cpu_root != cuda_root, "CPU and CUDA reports must be separate")
    for root in (cpu_root, cuda_root):
        require(not output.is_relative_to(root) and not root.is_relative_to(output),
                "Comparison output must be separate from both input report directories")
    cpu_meta, cpu = load_report(cpu_root, "cpu")
    cuda_meta, cuda = load_report(cuda_root, "cuda")
    for field in ("source_sha256", "input_sha256"):
        require(cpu_meta["provenance"][field] == cuda_meta["provenance"][field],
                f"CPU/CUDA {field} differs")
    pairs = paired_contrasts(cpu, cuda)
    targets, across = summarize(pairs)
    cells = summarize_cells(cpu, cuda)
    metadata = {
        "schema": "time-environment-comparison-v1", "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(), "factorial_cells_complete": 4,
        "jobs_accepted": 256, "paired_target_seed_rows": 64, "targets": 8, "seeds": list(SEEDS),
        "arms": ARMS, "primary_metrics": ["train_mse", "valid_rmse_std"],
        "secondary_metrics": ["test_r2", "test_rmse", "test_mae"],
        "cells_summary": {
            "file": "cells.csv",
            "aggregation": "Each cell is the arithmetic mean of all 64 accepted target/seed jobs (8 targets x 8 seeds), with equal target and seed weights; jobs are not weighted by molecule count.",
            "interpretation": "Descriptive four-cell outcomes only; no pooled cross-target SD, confidence intervals or p-values. Overlapping targets are not independent replications.",
            "columns": {"device": "Audited training device: cpu or cuda",
                        "arm": "Declared training budget: control or extended",
                        "jobs": "Number of accepted target/seed jobs, exactly 64 per cell",
                        "mean_train_mse": "Mean best-checkpoint train MSE in standardized label units",
                        "mean_valid_rmse_std": "Mean best-checkpoint validation RMSE in standardized label units; std denotes label standardization, not a standard deviation across jobs",
                        "mean_test_r2": "Mean held-out test R2, a secondary descriptive outcome",
                        "common_flags": "Count of jobs with best_epoch < 30 and best-checkpoint standardized train MSE > 0.5"}},
        "contrasts": {"cpu_budget": "CPU extended - CPU control",
                      "cuda_budget": "CUDA extended - CUDA control",
                      "env_control": "CUDA control - CPU control",
                      "env_extended": "CUDA extended - CPU extended",
                      "interaction": "(CUDA extended - CUDA control) - (CPU extended - CPU control)"},
        "direction": "Lower MSE/RMSE/MAE is better; higher test R2 is better. Interaction is a difference of descriptive budget contrasts, not a winner score.",
        "training_prefix_comparison": {
            "cpu": cpu_meta["training_prefix_comparison"],
            "cuda": cuda_meta["training_prefix_comparison"]},
        "selection_policy": "All declared tags/seeds retained; no flag-based groups or budget/seed selection; main results are unchanged.",
        "limitations": ["Environment contrasts combine OS, CPU/GPU architecture, kernels, libraries and RNG implementations; they are not pure hardware causal effects.",
                        "Equal seed integers across devices do not guarantee identical random-number streams or training trajectories.",
                        "Within-device control/extended trajectories can also differ at the same seed. When CUDA budget-arm logged prefixes do not match, the descriptive CUDA budget contrast includes run-to-run variation and cannot be attributed solely to extending training; the interaction inherits this limitation. Prefix mismatches do not exclude any declared runs.",
                        "Training-prefix summaries compare printed, rounded log values only. Matching rounded prefixes do not establish bitwise-identical trajectories or remove other limitations of causal attribution.",
                        "Eight seeds repeat optimization on the same frozen split; this is not data k-fold cross-validation.",
                        "EGFR and EGFR_FULL overlap; target averages are descriptive and not independent-target inference.",
                        "Across-seed SDs are descriptive; no p-values or confidence intervals are constructed.",
                        "Test outcomes are secondary and are not used to select budgets, seeds or report cohorts."],
        "source_sha256": cpu_meta["provenance"]["source_sha256"],
        "input_sha256": cpu_meta["provenance"]["input_sha256"],
        "inputs": {device: {"report_root": str(root), "report_sha256": digest(root / "report.json"),
                            "output_sha256": meta["output_sha256"], "provenance": meta["provenance"],
                            "training_device": meta["training_device"], "training_host": meta["training_host"],
                            "training_platform": meta["training_platform"]}
                   for device, root, meta in (("cpu", cpu_root, cpu_meta), ("cuda", cuda_root, cuda_meta))},
        "comparator_sha256": digest(Path(__file__)),
    }
    # All validation and calculations precede publication. Missing CUDA evidence
    # therefore cannot produce an apparent complete factorial report.
    output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(output / "cells.csv", index=False)
    pairs.to_csv(output / "paired.csv", index=False)
    targets.to_csv(output / "targets.csv", index=False)
    across.to_csv(output / "cross_target_means.csv", index=False)
    metadata["output_sha256"] = {name: digest(output / name) for name in
                                 ("cells.csv", "paired.csv", "targets.csv", "cross_target_means.csv")}
    (output / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", required=True, type=Path)
    parser.add_argument("--cuda-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.cpu_report, args.cuda_report, args.out_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"COMPARISON FAILED: {type(error).__name__}: {error}\n")
    print(json.dumps({key: report[key] for key in ("status", "factorial_cells_complete", "paired_target_seed_rows")}, indent=2))


if __name__ == "__main__":
    main()
