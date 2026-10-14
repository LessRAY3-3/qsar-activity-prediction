"""Audit CPU or CUDA training-budget cohorts with independent CPU replay.

python experiments/time_budget_report.py --study-root RETURNED/study \
    --input-root FROZEN_INPUT_ROOT --out-dir results/time_budget_sensitivity/m3

The input root supplies matching frozen scripts, eight graph caches and splits.
The study supplies protocol/status, eight lane manifests and all canonical and
snapshot outputs. Nothing is published until every expected job passes CPU replay. This
reports one environment of the planned environment x budget study; it never
selects seeds or replaces the main campaign. Outputs contain secondary test
metrics as well as the primary training/validation diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARMS = {"control": {"epochs": 100, "min_epochs": 0, "patience": 20},
        "extended": {"epochs": 200, "min_epochs": 60, "patience": 40}}
LANES = (("a2a", "egfr"), ("egfr_full", "hivpr"), ("herg", "mpro"), ("abl1", "vegfr2"))
TAGS = tuple(sorted(tag for lane in LANES for tag in lane))
SEEDS = (42, 1, 2)
CODE_FILES = {"scripts/gnn_03_train_gin.py", "scripts/gnn_graph_dataset.py",
              "scripts/qsar_common.py", "scripts/run_time_campaign.py", "scripts/time_labels.py"}
TOLERANCE = 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def inside(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts and bool(name.parts),
            f"Unsafe relative path: {relative}")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes root: {relative}")
    return path


def expected_parameters(arm, seed, training_device="cpu"):
    return {"hidden": 128, "num_layers": 4, "dropout": 0.2, "lr": 0.001,
            "batch_size": 128, "device": training_device, "seed": seed, **ARMS[arm]}


def artifact_names(tag, seed):
    suffix = "" if seed == 42 else f"_seed{seed}"
    return [f"results/gnn_models/{tag}_gin_time{suffix}.pt",
            f"results/gnn_preds_{tag}_time{suffix}.npz",
            f"results/gnn_metrics_{tag}{suffix}.json"]


def validate_seeds(seeds):
    seeds = tuple(seeds)
    require(bool(seeds) and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(seeds) == len(set(seeds)), "Seeds must be unique nonnegative integers")
    return seeds


def inspect_study(study, inputs, seeds=SEEDS, training_device="cpu"):
    """Verify coverage, protocol, manifests and both copies before replay."""
    require(training_device in ("cpu", "cuda"), "Unsupported training device")
    protocol = read_json(study / "protocol.json")
    cohort_seeds = validate_seeds(protocol["seeds"])
    if seeds is not None:
        require(set(cohort_seeds) == set(validate_seeds(seeds)), "Unexpected cohort seed coverage")
    count = len(ARMS) * len(TAGS) * len(cohort_seeds)
    require(protocol["arms"] == ARMS
            and protocol["lanes"] == [list(lane) for lane in LANES]
            and protocol["jobs"] == count and protocol["device"] == training_device
            and type(protocol["parallel_workers"]) is int
            and protocol["parallel_workers"] in ((4,) if training_device == "cpu" else (1, 2, 3, 4)),
            "Unexpected training study protocol/device/concurrency")
    require(protocol["thread_environment"].get("QSAR_TORCH_THREADS") == "4"
            and protocol["thread_environment"].get("QSAR_TORCH_INTEROP_THREADS") == "1",
            "Study protocol must request 4 intra-op / 1 inter-op host threads")
    top = read_json(study / "study_status.json")
    require(top["state"] == "training_complete_pending_independent_audit",
            "Study has not completed training")
    expected_lanes = {(arm, i) for arm in ARMS for i in range(4)}
    require(len(top["lanes"]) == 8 and {(x["arm"], x["lane"]) for x in top["lanes"]} == expected_lanes
            and all(x["returncode"] == 0 for x in top["lanes"]), "Incomplete/failed lane coverage")
    actual = set(study.glob("*/lane*/results/time_campaign/*/status.json"))
    expected = {study / arm / f"lane{i}/results/time_campaign/gin_{tag}_seed{seed}/status.json"
                for arm in ARMS for i, tags in enumerate(LANES) for tag in tags for seed in cohort_seeds}
    require(actual == expected, f"Expected exactly {count} job statuses, found {len(actual)}")
    require(not list(study.rglob(".time_campaign_*.lock")), "Returned study contains active worker locks")
    jobs, common_environment, common_host = [], None, None
    input_hashes, source_hashes = {}, {}
    for arm in ARMS:
        for lane, tags in enumerate(LANES):
            root = study / arm / f"lane{lane}"
            manifest = read_json(root / "manifest.json")
            mh = digest(root / "manifest.json")
            require(mh == protocol["manifest_sha256"], "Lane manifest differs from frozen protocol")
            require(manifest["role"] == "gin" and not manifest["source"]["dirty"]
                    and manifest["source"]["git_head"] == protocol["source_commit"],
                    "Wrong/dirty lane source revision")
            lane_state = read_json(root / "lane_status.json")
            require(lane_state["arm"] == arm and lane_state["lane"] == lane
                    and lane_state["tags"] == list(tags) and lane_state["returncode"] == 0,
                    "Lane status differs from protocol")
            manifest_files = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
            for tag in tags:
                for seed in cohort_seeds:
                    directory = root / f"results/time_campaign/gin_{tag}_seed{seed}"
                    status = read_json(directory / "status.json")
                    p = status["provenance"]
                    label = manifest["targets"][tag]["label_id"]
                    require(status["state"] == "success" and status["signature"] == json_digest(p),
                            f"{arm}/{tag}/{seed}: invalid success/signature")
                    require(p["role"] == "gin" and p["tag"] == tag and p["label_id"] == label
                            and p["hyperparameters"] == expected_parameters(arm, seed, training_device),
                            f"{arm}/{tag}/{seed}: parameters or labels differ")
                    require(p["bundle_manifest"] == {"sha256": mh, "source_git_head": protocol["source_commit"]},
                            "Job manifest provenance mismatch")
                    require(status["command"][1:] == p["command"], "Command provenance mismatch")
                    require(set(p["inputs"]) == {f"data/processed/{tag}_graphs.npz",
                                                f"data/processed/splits/{tag}_time.npz"}
                            and set(p["code"]) == CODE_FILES, "Incomplete source/input hash coverage")
                    for field, summary in (("inputs", input_hashes), ("code", source_hashes)):
                        for name, h in p[field].items():
                            require(manifest_files.get(name) == h and digest(inside(inputs, name)) == h,
                                    f"Frozen {field} SHA256 mismatch: {name}")
                            require(name not in summary or summary[name] == h, f"Mixed versions: {name}")
                            summary[name] = h
                    env = p["environment"]
                    require(env["variables"].get("QSAR_TORCH_THREADS") == "4"
                            and env["variables"].get("QSAR_TORCH_INTEROP_THREADS") == "1",
                            "Recorded worker thread configuration differs")
                    if common_environment is None:
                        common_environment, common_host = env, status["host"]
                    require(env == common_environment and status["host"] == common_host,
                            "Arms/jobs did not share one recorded training environment")
                    names = artifact_names(tag, seed)
                    require(set(status["artifacts"]) == set(names), "Incomplete/extra job artifacts")
                    snapshot = inside(directory, status["snapshot"])
                    for name in names:
                        for base in (root, snapshot):
                            require(digest(inside(base, name)) == status["artifacts"][name],
                                    f"Canonical/snapshot artifact hash mismatch: {arm}/{tag}/{seed}/{name}")
                    jobs.append(dict(arm=arm, tag=tag, seed=seed, lane=lane, root=root,
                                     directory=directory, status=status, parameters=p["hyperparameters"],
                                     label_id=label, files=names))
    return jobs, dict(protocol=protocol, training_environment=common_environment, host=common_host,
                      input_sha256=input_hashes, source_sha256=source_hashes)


def inspect_cohorts(studies, inputs, seeds=SEEDS, training_device="cpu"):
    """Require complete disjoint cohorts with identical training code, data and host.

    Bundle/coordinator revisions may differ; their full protocols remain recorded.
    The five worker/training source hashes, all graph/split hashes and recorded
    environment must match, regardless of the wrapper's newer Git revision.
    """
    seeds = validate_seeds(seeds)
    require(len(studies) == len(set(studies)), "Duplicate study roots")
    jobs, evidence, seen = [], [], set()
    for study in studies:
        cohort_jobs, cohort_evidence = inspect_study(study, inputs, seeds=None, training_device=training_device)
        cohort_seeds = set(cohort_evidence["protocol"]["seeds"])
        require(not seen.intersection(cohort_seeds), "Duplicate seeds across study cohorts")
        seen.update(cohort_seeds)
        if evidence:
            for field in ("device", "parallel_workers", "thread_environment"):
                require(cohort_evidence["protocol"][field] == evidence[0]["protocol"][field],
                        f"Cohort protocol {field} differs")
            for field in ("source_sha256", "input_sha256", "training_environment", "host"):
                require(cohort_evidence[field] == evidence[0][field], f"Cohort {field} differs")
        jobs.extend(cohort_jobs)
        evidence.append(cohort_evidence)
    require(seen == set(seeds), "Combined cohort seeds differ from expected seeds")
    expected = {(arm, tag, seed) for arm in ARMS for tag in TAGS for seed in seeds}
    keys = [(job["arm"], job["tag"], job["seed"]) for job in jobs]
    require(len(keys) == len(expected) and set(keys) == expected,
            "Combined jobs have duplicate or incomplete arm/target/seed coverage")
    labels = {}
    for job in jobs:
        require(job["tag"] not in labels or labels[job["tag"]] == job["label_id"],
                "Cohort label versions differ")
        labels[job["tag"]] = job["label_id"]
    # Retain the legacy single-cohort evidence shape and expose each merged protocol.
    summary = dict(evidence[0])
    if len(evidence) > 1:
        summary["cohorts"] = [dict(study_root=str(root), **item)
                              for root, item in zip(studies, evidence)]
    return jobs, summary


def load_replay_modules(inputs):
    sys.path.insert(0, str(inputs / "scripts"))
    spec = importlib.util.spec_from_file_location("budget_replay_trainer", inputs / "scripts/gnn_03_train_gin.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    # These are version/signature guards, not the training algorithm.
    from time_labels import load_time_split
    return trainer, load_time_split


def check_runtime(runtime, seed, training_device="cpu"):
    require(runtime.get("seed") == seed and runtime.get("device") == training_device
            and runtime.get("requested_device") == training_device
            and runtime.get("torch_threads") == 4 and runtime.get("torch_interop_threads") == 1,
            f"Checkpoint/metric runtime must record {training_device}, 4 intra-op and 1 inter-op threads")


def close(actual, expected, context, tolerance=TOLERANCE):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0, err_msg=context)


def completed_epochs(log, best, parameters):
    require(log.is_file(), f"Training log is required: {log}")
    text = log.read_text()
    maximum = parameters["epochs"]
    stops = re.findall(r"early stop at epoch (\d+) \(best epoch (\d+)\)", text)
    require(len(stops) <= 1, "Ambiguous repeated early-stop log")
    if stops:
        actual, logged_best = map(int, stops[0])
        require(logged_best == best and actual - best >= parameters["patience"],
                "Early-stop log differs from best checkpoint/patience")
    else:
        require(re.search(rf"epoch\s+{maximum}\s+train_mse=", text) is not None,
                "Training log has neither an early-stop record nor the maximum epoch")
        actual = maximum
    require(best <= actual <= maximum and actual >= parameters["min_epochs"],
            "Actual epoch count violates best/max/min epoch budget")
    return actual


def replay_job(job, inputs, trainer, load_split):
    import torch
    root, tag, seed = job["root"], job["tag"], job["seed"]
    checkpoint = torch.load(root / job["files"][0], weights_only=True, map_location="cpu")
    with np.load(root / job["files"][1], allow_pickle=False) as archive:
        pred = dict(archive)
    metric = read_json(root / job["files"][2])["time"]
    expected_args = {**job["parameters"], "tag": tag, "model": "gin", "split": "time",
                     "suffix": "" if seed == 42 else f"_seed{seed}"}
    require(checkpoint["args"] == expected_args, f"{tag}/{seed}: checkpoint arguments differ")
    require(metric["params"] == {k: v for k, v in expected_args.items() if k != "split"},
            f"{tag}/{seed}: metric parameters differ")
    require(metric.get("train_mse_source") == "best_checkpoint_eval", "Legacy last-epoch MSE is not acceptable")
    training_device = job["parameters"]["device"]
    require(training_device in ("cpu", "cuda"), "Unsupported checkpoint training device")
    check_runtime(checkpoint["runtime"], seed, training_device)
    check_runtime(metric["runtime"], seed, training_device)
    require(checkpoint["runtime"] == metric["runtime"], "Checkpoint/metric runtime differs")
    require(metric.get("device") == training_device, "Metric device differs")
    require(checkpoint["runtime"]["torch_version"] == job["status"]["provenance"]["environment"]["packages"]["torch"],
            "Runtime torch version differs from provenance")
    graph = inputs / f"data/processed/{tag}_graphs.npz"
    with np.load(graph, allow_pickle=False) as cache:
        split = load_split(inputs / "data/processed/splits", tag, cache["smiles"], cache["y"])
    require(split is not None, "No active temporal split")
    require(str(split["label_id"]) == checkpoint["label_id"] == metric["label_id"]
            == str(pred["label_id"]) == job["label_id"], "Mixed label versions")
    np.testing.assert_array_equal(pred["test_idx"], split["test_idx"])
    close(pred["y_true"], split["y"][split["test_idx"]], "Saved test labels", 1e-6)
    y = split["y"]
    mean, std = float(y[split["train_idx"]].mean()), float(y[split["train_idx"]].std())
    require(checkpoint["y_mean"] == mean and checkpoint["y_std"] == std and std > 0,
            "Training normalization differs from frozen train labels")
    model = trainer.build_model(argparse.Namespace(**checkpoint["args"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(all(torch.isfinite(value).all() for value in model.state_dict().values()), "Non-finite weights")
    model.eval()
    replay = {}
    for pool in ("train", "valid", "test"):
        ds = trainer.MoleculeGraphDataset(graph, indices=split[f"{pool}_idx"])
        ds.y = (y.copy() - mean) / std
        loader = trainer.DataLoader(ds, batch_size=512, shuffle=False)
        pp, yy = trainer.predict(model, loader, "cpu")
        require(np.isfinite(pp).all(), "Non-finite replay predictions")
        if pool == "train":
            replay["train_mse"] = float(np.mean((pp - yy) ** 2))
        elif pool == "valid":
            replay["valid_rmse_std"] = float(np.sqrt(np.mean((pp - yy) ** 2)))
        else:
            replay["prediction_max_abs_error"] = float(np.max(np.abs(pp * std + mean - pred["y_pred"])))
            close(pp * std + mean, pred["y_pred"], "Checkpoint test-prediction replay")
    close(replay["train_mse"], metric["train_mse"], "Best-checkpoint training MSE")
    close(replay["valid_rmse_std"], metric["best_valid_rmse_std"], "Best-checkpoint validation RMSE")
    scores = dict(test_r2=float(r2_score(pred["y_true"], pred["y_pred"])),
                  test_rmse=float(np.sqrt(mean_squared_error(pred["y_true"], pred["y_pred"]))),
                  test_mae=float(mean_absolute_error(pred["y_true"], pred["y_pred"])))
    for name, value in scores.items():
        close(value, metric[name.removeprefix("test_")], "Test metric recomputation", 1e-7)
    best, maximum = metric["best_epoch"], job["parameters"]["epochs"]
    require(isinstance(best, int) and 1 <= best <= maximum and metric["max_epochs"] == maximum,
            "Invalid best/max epoch")
    native = best < 0.3 * maximum and metric["train_mse"] > 0.5
    require(metric["convergence_flag"] == native, "Native flag differs from saved-checkpoint diagnostic")
    log = inside(job["directory"], job["status"]["log"])
    actual_epochs = completed_epochs(log, best, job["parameters"])
    row = dict(arm=job["arm"], tag=tag, seed=seed, label_id=job["label_id"],
               best_epoch=best, actual_epochs=actual_epochs, max_epochs=maximum,
               train_mse=replay["train_mse"], valid_rmse_std=replay["valid_rmse_std"],
               valid_rmse_pic50=replay["valid_rmse_std"] * std,
               native_flag=native, common_flag=bool(best < 30 and replay["train_mse"] > 0.5),
               last_epoch_train_mse=metric["last_epoch_train_mse"],
               prediction_max_abs_error=replay["prediction_max_abs_error"],
               train_mse_abs_error=abs(replay["train_mse"] - metric["train_mse"]),
               valid_rmse_abs_error=abs(replay["valid_rmse_std"] - metric["best_valid_rmse_std"]),
               elapsed_seconds=job["status"]["elapsed_seconds"], device=training_device, replay_device="cpu",
               torch_threads=4, torch_interop_threads=1, **scores)
    return row, pred


def paired_rows(frame, seeds=SEEDS):
    keys = ["tag", "seed"]
    seeds = validate_seeds(seeds)
    count = len(ARMS) * len(TAGS) * len(seeds)
    require(len(frame) == count and set(frame.arm) == set(ARMS)
            and not frame.duplicated(["arm", *keys]).any(), f"Expected {count} unique rows")
    expected = {(tag, seed) for tag in TAGS for seed in seeds}
    for arm in ARMS:
        require(set(map(tuple, frame.loc[frame.arm == arm, keys].to_numpy())) == expected,
                f"Incomplete {arm} cohort")
    control = frame[frame.arm == "control"].drop(columns="arm").set_index(keys).add_prefix("control_")
    extended = frame[frame.arm == "extended"].drop(columns="arm").set_index(keys).add_prefix("extended_")
    pair = control.join(extended, how="inner").reset_index()
    require((pair.control_label_id == pair.extended_label_id).all(), "Paired label versions differ")
    for metric in ("train_mse", "valid_rmse_std", "valid_rmse_pic50", "test_r2", "test_rmse",
                   "test_mae", "best_epoch", "elapsed_seconds"):
        pair[f"delta_{metric}"] = pair[f"extended_{metric}"] - pair[f"control_{metric}"]
    pair["delta_common_flag"] = pair.extended_common_flag.astype(int) - pair.control_common_flag.astype(int)
    return pair


def target_rows(frame, pairs, predictions, seeds=SEEDS):
    seeds = validate_seeds(seeds)
    rows = []
    for tag in TAGS:
        row = dict(tag=tag, n_seeds=len(seeds))
        for arm in ARMS:
            subset = frame[(frame.tag == tag) & (frame.arm == arm)]
            for metric in ("train_mse", "valid_rmse_std", "test_r2"):
                row[f"{arm}_{metric}_mean"] = float(subset[metric].mean())
            values = [predictions[(arm, tag, seed)] for seed in seeds]
            y = values[0]["y_true"]
            for item in values[1:]:
                np.testing.assert_array_equal(item["test_idx"], values[0]["test_idx"])
                close(item["y_true"], y, "Ensemble test-label alignment", 1e-6)
            ensemble = np.mean([v["y_pred"].astype(float) for v in values], axis=0)
            row[f"{arm}_ensemble_test_r2"] = float(r2_score(y.astype(float), ensemble))
        subset = pairs[pairs.tag == tag]
        for metric in ("train_mse", "valid_rmse_std", "test_r2"):
            row[f"delta_{metric}_mean"] = float(subset[f"delta_{metric}"].mean())
            row[f"delta_{metric}_seed_sd"] = float(subset[f"delta_{metric}"].std(ddof=1))
        row["delta_ensemble_test_r2"] = row["extended_ensemble_test_r2"] - row["control_ensemble_test_r2"]
        rows.append(row)
    return pd.DataFrame(rows)


def logged_epochs(text):
    pattern = r"^\s*epoch\s+(\d+)\s+train_mse=([-+.\deE]+)\s+valid_rmse=([-+.\deE]+)\s+lr=([-+.\deE]+)"
    return {int(epoch): tuple(map(float, (train, valid, lr)))
            for epoch, train, valid, lr in re.findall(pattern, text, re.MULTILINE)}


def prefix_rows(jobs, seeds=SEEDS):
    """Describe rounded logged trajectories; disagreement is not an audit failure."""
    logs = {}
    for job in jobs:
        path = inside(job["directory"], job["status"]["log"])
        logs[(job["arm"], job["tag"], job["seed"])] = logged_epochs(path.read_text()) if path.exists() else {}
    rows = []
    for tag in TAGS:
        for seed in validate_seeds(seeds):
            control, extended = (logs[(arm, tag, seed)] for arm in ARMS)
            horizon = min(max(control, default=0), max(extended, default=0))
            left = {e for e in control if e <= horizon}
            right = {e for e in extended if e <= horizon}
            shared = sorted(left & right)
            differences = np.asarray([np.abs(np.asarray(control[e]) - extended[e]) for e in shared])
            maximum = differences.max(axis=0) if shared else [np.nan] * 3
            rows.append(dict(tag=tag, seed=seed, common_logged_horizon=horizon,
                             compared_logged_epochs=len(shared),
                             missing_in_extended=",".join(map(str, sorted(left - right))),
                             missing_in_control=",".join(map(str, sorted(right - left))),
                             max_logged_train_mse_difference=maximum[0],
                             max_logged_valid_rmse_difference=maximum[1],
                             max_logged_lr_difference=maximum[2],
                             equal_rounded_log_prefix=bool(shared and left == right and not np.any(maximum))))
    return pd.DataFrame(rows)


def run(study, inputs, output, threads=4, *, supplement=None, seeds=SEEDS, training_device="cpu"):
    import torch
    study, inputs, output = (Path(p).resolve() for p in (study, inputs, output))
    require(not output.exists(), "Output directory already exists; use a new report directory")
    studies = [study] + ([Path(supplement).resolve()] if supplement is not None else [])
    seeds = validate_seeds(seeds)
    for root in studies:
        require(not output.is_relative_to(root) and not root.is_relative_to(output),
                "Report output must be separate from returned study")
    require(threads > 0, "Replay threads must be positive")
    jobs, evidence = inspect_cohorts(studies, inputs, seeds, training_device)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    trainer, load_split = load_replay_modules(inputs)
    rows, predictions = [], {}
    for job in jobs:
        row, pred = replay_job(job, inputs, trainer, load_split)
        rows.append(row)
        predictions[(job["arm"], job["tag"], job["seed"])] = pred
        print(f"PASS {job['arm']}/{job['tag']}/seed{job['seed']}: checkpoint replay", flush=True)
    frame = pd.DataFrame(rows).sort_values(["arm", "tag", "seed"])
    pairs = paired_rows(frame, seeds)
    targets = target_rows(frame, pairs, predictions, seeds)
    prefixes = prefix_rows(jobs, seeds)
    environment_label = "M3 CPU" if training_device == "cpu" else "CUDA training (CPU replay)"
    report = dict(schema="m3-time-budget-report-v1", status="accepted",
                  audited_at=datetime.now(timezone.utc).isoformat(), jobs_accepted=len(jobs),
                  seeds=list(seeds), n_seeds=len(seeds), cohorts=len(studies),
                  paired_rows=len(pairs), targets=len(TAGS), environment=environment_label,
                  training_device=training_device, training_host=evidence["host"],
                  training_platform=evidence["training_environment"].get("platform"),
                  factorial_cells_complete=2, planned_factorial_cells=4,
                  cpu_cells_status="accepted for declared seeds" if training_device == "cpu" else "not audited by this report",
                  cuda_cells_status="accepted for declared seeds" if training_device == "cuda" else "not audited by this report",
                  primary_metrics=["best-checkpoint standardized train MSE", "validation RMSE"],
                  secondary_metrics=["test R2", "test RMSE", "test MAE", "runtime"],
                  delta_convention="extended minus control, paired by target and seed",
                  selection_policy=f"All {len(jobs)} jobs retained; no test-based seed/budget selection or replacement of main results",
                  common_flag="best_epoch < 30 and best-checkpoint standardized train MSE > 0.5",
                  native_flag="best_epoch < 0.3 * each arm max_epochs and best-checkpoint train MSE > 0.5",
                  interpretation=["Flags are diagnostics, not proof of non-convergence or benefit from longer training",
                                  "Current report measures within one training environment budget contrasts only",
                                  "Future CPU/CUDA comparison combines OS, architecture, kernels and libraries; not a pure hardware causal effect",
                                  f"{len(seeds)} seeds and overlapping target sets do not provide independent target-level replications",
                                  "Completed cells refer only to this training environment and the declared audited seeds; other environment/seed cohorts are not implied",
                                  "Target seed SDs are descriptive, not confidence intervals; ensemble R2 differs from mean single-seed R2"],
                  replay=dict(device="cpu", torch_threads=threads, torch_interop_threads=1,
                              torch_version=str(torch.__version__), python=platform.python_version(),
                              platform=platform.platform(), absolute_tolerance=TOLERANCE,
                              max_test_prediction_abs_error=float(frame.prediction_max_abs_error.max()),
                              max_train_mse_abs_error=float(frame.train_mse_abs_error.max()),
                              max_valid_rmse_abs_error=float(frame.valid_rmse_abs_error.max())),
                  provenance=evidence, auditor_sha256=digest(Path(__file__)),
                  flags_by_arm={arm: dict(native=int(frame.loc[frame.arm == arm, "native_flag"].sum()),
                                          common=int(frame.loc[frame.arm == arm, "common_flag"].sum())) for arm in ARMS})
    report["training_prefix_comparison"] = dict(
        pairs=len(pairs), matching_rounded_prefixes=int(prefixes.equal_rounded_log_prefix.sum()),
        interpretation="Only printed epochs and rounded train/validation/lr values are compared; no bitwise-equivalence requirement. Differences are reported, not suppressed or used to select runs.")
    # No partial accepted report is written for a failed job or incomplete cohort.
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "diagnostics.csv", index=False)
    pairs.to_csv(output / "paired.csv", index=False)
    targets.to_csv(output / "targets.csv", index=False)
    prefixes.to_csv(output / "prefix_logs.csv", index=False)
    report["output_sha256"] = {name: digest(output / name) for name in
                               ("diagnostics.csv", "paired.csv", "targets.csv", "prefix_logs.csv")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--training-device", choices=("cpu", "cuda"), default="cpu",
                        help="Required recorded training device; model replay always runs on CPU")
    parser.add_argument("--supplement-root", type=Path, help="Optional disjoint cohort with identical training code/data/environment")
    parser.add_argument("--seeds", "--expected-seeds", nargs="+", type=int, default=list(SEEDS),
                        help="Expected seed union across roots; default 42 1 2. New cohort: 3 4 5 6 7; merged: 42 1 2 3 4 5 6 7")
    parser.add_argument("--out-dir", required=True, type=Path, help="New directory; existing reports are never overwritten")
    parser.add_argument("--threads", type=int, default=4, help="CPU replay threads; training is always verified as 4/1")
    args = parser.parse_args(argv)
    try:
        report = run(args.study_root, args.input_root, args.out_dir, args.threads,
                     supplement=args.supplement_root, seeds=args.seeds, training_device=args.training_device)
    except (ValueError, AssertionError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"AUDIT FAILED: {type(error).__name__}: {error}\n")
    print(json.dumps({k: report[k] for k in ("status", "jobs_accepted", "paired_rows", "flags_by_arm")}, indent=2))


if __name__ == "__main__":
    main()
