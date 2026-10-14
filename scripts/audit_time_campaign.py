"""Read-only audit of current temporal labels, predictions and bootstrap results.

This is a new, parameterized auditor, not the script originally used during
training. Historical coordinator snapshots live in scripts/campaign_history/.
Only the optional JSON report is written. No downloads, training, model loading,
or result regeneration occur. Requires raw data, restored time splits, current
results, and local Git history for the source commits in the acceptance record.
Default full scope also verifies saved model and feature-cache hashes; --scope
data explicitly skips those files so raw labels and published predictions can
be audited without downloading obsolete RF models. Neither scope replays weights.
Bootstrap uncertainty is conditional on the saved predictions.

Example: python scripts/audit_time_campaign.py --root . --report audit.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def git_bytes(root, revision, name):
    process = subprocess.run(["git", "-C", str(root), "show", f"{revision}:{name}"],
                             capture_output=True, check=False)
    require(process.returncode == 0,
            f"Cannot read historical source {revision}:{name}; local Git history is required")
    return process.stdout


def parent_function(root):
    # Reuse only the cleaning pipeline's chemical normalization, never the
    # production time splitter, label builder, prediction loader or bootstrap.
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("audit_clean_data", root / "scripts/02_clean_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = {}

    def parent(value):
        if not isinstance(value, str) or not value.strip():
            return None
        if value not in cache:
            cache[value] = module._parent_smiles(value)[0]
        return cache[value]

    return parent


def rebuild_labels(raw, years, split, parent):
    """Rebuild membership AND medians from raw records and saved cutoffs.

    Cached parent strings are used as-is. Rebuilding the candidate pools rather
    than iterating only stored retained rows also detects wrongly excluded rows.
    Cutoff selection itself is not independently re-optimized by this audit.
    """
    raw = raw.copy()
    raw["year"] = raw.document_chembl_id.map(years)
    raw["value"] = pd.to_numeric(raw.standard_value, errors="coerce")
    raw["parent"] = raw.canonical_smiles.map(parent)
    earliest = raw.groupby("parent").year.min().reindex(split["smiles"]).to_numpy()
    eligible = raw.loc[
        raw.standard_relation.fillna("").astype(str).str.strip().eq("=")
        & raw.standard_units.eq("nM") & np.isfinite(raw.value) & raw.value.gt(0)
        & np.isfinite(raw.year) & raw.parent.notna()
    ].copy()
    eligible["pic50"] = 9 - np.log10(eligible.value)
    cv, ct = int(split["cutoff_valid"]), int(split["cutoff_test"])
    require(cv < ct, "Temporal cutoffs must be increasing")
    masks = (earliest < cv, (earliest >= cv) & (earliest < ct), earliest >= ct)
    windows = (eligible.year < cv, eligible.year < ct, eligible.year >= ct)
    rebuilt = np.full(len(split["smiles"]), np.nan, dtype=np.float32)
    expected = {}
    for pool, mask, window in zip(("train", "valid", "test"), masks, windows):
        candidates = np.flatnonzero(mask)
        medians = eligible.loc[window].groupby("parent").pic50.median()
        values = medians.reindex(split["smiles"][candidates]).to_numpy()
        keep = np.isfinite(values)
        expected[pool] = candidates[keep]
        rebuilt[expected[pool]] = values[keep]
        np.testing.assert_array_equal(np.sort(split[f"{pool}_idx"]), expected[pool],
                                      err_msg=f"{pool}: raw-derived membership differs")
        np.testing.assert_array_equal(split["year"][expected[pool]], earliest[expected[pool]],
                                      err_msg=f"{pool}: first-publication year differs")
    expected["excluded"] = np.flatnonzero(~np.isfinite(rebuilt))
    np.testing.assert_array_equal(np.sort(split["excluded_idx"]), expected["excluded"],
                                  err_msg="Raw-derived exclusions differ")
    np.testing.assert_array_equal(rebuilt, split["y"], err_msg="Raw-derived labels differ")
    return {pool: len(indices) for pool, indices in expected.items()}


def check_prediction(pred, split, metric):
    np.testing.assert_array_equal(pred["test_idx"], split["test_idx"])
    np.testing.assert_allclose(pred["y_true"], split["y"][split["test_idx"]],
                               rtol=0, atol=1e-6)
    require(str(pred["label_id"]) == str(split["label_id"]) == metric["label_id"],
            "Prediction/metric label version differs from split")
    require(pred["y_true"].shape == pred["y_pred"].shape
            and np.isfinite(pred["y_pred"]).all(), "Invalid prediction shape/values")
    expected = [r2_score(pred["y_true"], pred["y_pred"]),
                np.sqrt(mean_squared_error(pred["y_true"], pred["y_pred"])),
                mean_absolute_error(pred["y_true"], pred["y_pred"])]
    np.testing.assert_allclose([metric[k] for k in ("r2", "rmse", "mae")], expected,
                               rtol=1e-7, atol=1e-8)


def bootstrap(y, rf, gin, n_boot, seed):
    """Independent SSE/SST algebra; does not import paired_bootstrap.py."""
    require(n_boot > 0, "n_boot must be positive")
    y, rf, gin = (np.asarray(v, dtype=np.float64) for v in (y, rf, gin))
    rng = np.random.default_rng(seed)
    r2_draws, rmse_draws = [], []
    for start in range(0, n_boot, 100):
        idx = rng.integers(0, len(y), size=(min(100, n_boot - start), len(y)))
        ys = y[idx]
        sst = np.sum((ys - ys.mean(axis=1, keepdims=True)) ** 2, axis=1)
        rf_sse = np.sum((ys - rf[idx]) ** 2, axis=1)
        gin_sse = np.sum((ys - gin[idx]) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (rf_sse - gin_sse) / sst
        rmse_delta = np.sqrt(gin_sse / len(y)) - np.sqrt(rf_sse / len(y))
        finite = np.isfinite(delta) & np.isfinite(rmse_delta)
        r2_draws.extend(delta[finite])
        rmse_draws.extend(rmse_delta[finite])
    require(len(r2_draws) > 0, "All bootstrap resamples are degenerate")
    ci = np.percentile(r2_draws, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_draws, [2.5, 97.5])
    return dict(d_r2=float(r2_score(y, gin) - r2_score(y, rf)),
                ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                frac_boot_gin_better=float(np.mean(np.asarray(r2_draws) > 0)),
                d_rmse=float(np.sqrt(np.mean((y - gin) ** 2))
                             - np.sqrt(np.mean((y - rf) ** 2))),
                d_rmse_ci_lo=float(rmse_ci[0]), d_rmse_ci_hi=float(rmse_ci[1]))


def check_provenance(root, acceptance, scope="full"):
    jobs = acceptance["jobs"]
    expected = {(tag, role, seed) for tag in TAGS
                for role, seeds in (("rf", (42,)), ("gin", (42, 1, 2))) for seed in seeds}
    keys = [(j["tag"], j["role"], j.get("seed", 42)) for j in jobs]
    require(len(keys) == len(expected) and set(keys) == expected, "Incomplete/duplicate job coverage")
    records = acceptance["integrated_files"] + acceptance["analysis"]["files"]
    skipped = set()
    verified = 0
    for record in records:
        if scope == "data" and record["path"].startswith(("models/", "results/gnn_models/")):
            skipped.add(record["path"])
            continue
        require(sha256(root / record["path"]) == record["sha256"],
                f"Current artifact SHA256 mismatch: {record['path']}")
        verified += 1
    for job in jobs:
        require(job["state"] == "success", "Non-successful accepted job")
        provenance = job["provenance"]
        signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        require(signature == job["signature"], "Job provenance signature mismatch")
        for name, digest in provenance["inputs"].items():
            if scope == "data" and not name.startswith("data/processed/splits/"):
                skipped.add(name)
                continue
            require(sha256(root / name) == digest, f"Training input SHA256 mismatch: {name}")
        for name, digest in provenance["code"].items():
            source = git_bytes(root, job["source_commit"], name)
            require(hashlib.sha256(source).hexdigest() == digest,
                    f"Historical training source SHA256 mismatch: {name}")
        require(sha256(root / job["input_manifest"]) == provenance["bundle_manifest"]["sha256"],
                "Job input manifest SHA256 mismatch")
    return verified, sorted(skipped)


def audit(root, report, scope="full"):
    acceptance = read_json(root / "results/time_split/campaign_acceptance.json")
    verified, skipped = check_provenance(root, acceptance, scope)
    report.update(artifact_hashes_verified=verified, skipped_model_or_feature_hashes=skipped)
    revision = acceptance["legacy_artifacts_git_revision"]
    report["non_time_baseline_revision"] = revision
    year_path = root / "data/raw/document_years.csv"
    docs = pd.read_csv(year_path)[["document_chembl_id", "document_year"]].drop_duplicates()
    require(not docs.document_chembl_id.duplicated().any(), "Conflicting document years")
    years = pd.to_numeric(docs.set_index("document_chembl_id").document_year, errors="coerce")
    sig = pd.read_csv(root / "results/significance/summary_time.csv").set_index(["tag", "pair"])
    summary = pd.read_csv(root / "results/time_split/summary.csv").set_index("tag")
    require(not summary.index.duplicated().any() and set(summary.index) == set(TAGS),
            "Unexpected time summary targets")
    require(not sig.index.duplicated().any()
            and set(sig.index) == {(tag, pair) for tag in TAGS for pair in ("seed42", "mean3")},
            "Unexpected bootstrap row coverage")
    parent = parent_function(root)
    report["targets"] = []
    for tag in TAGS:
        raw_path = root / f"data/raw/{tag}_activities.csv"
        with np.load(root / f"data/processed/splits/{tag}_time.npz") as archive:
            split = dict(archive)
        source_hash = hashlib.sha256(raw_path.read_bytes() + year_path.read_bytes()).hexdigest()
        require(source_hash == str(split["source_sha256"]), f"{tag}: raw source hash mismatch")
        counts = rebuild_labels(pd.read_csv(raw_path), years, split, parent)
        predictions = []
        for role, suffix in (("rf", ""), ("gnn", ""), ("gnn", "_seed1"), ("gnn", "_seed2")):
            with np.load(root / f"results/{role}_preds_{tag}_time{suffix}.npz") as archive:
                pred = dict(archive)
            metric_name = f"results/{'' if role == 'rf' else 'gnn_'}metrics_{tag}{suffix}.json"
            payload = read_json(root / metric_name)
            original = json.loads(git_bytes(root, revision, metric_name))
            def keep(d):
                return {k: v for k, v in d.items() if k not in ("time", "best_params_time")}
            require(keep(payload) == keep(original), f"{tag}: non-time metrics changed")
            check_prediction(pred, split, payload["time"])
            predictions.append(pred)
        y = predictions[0]["y_true"].astype(float)
        rf = predictions[0]["y_pred"].astype(float)
        gins = [p["y_pred"].astype(float) for p in predictions[1:]]
        ensemble = np.mean(gins, axis=0)
        n_boot_values = []
        for pair, pred in (("seed42", gins[0]), ("mean3", ensemble)):
            row = sig.loc[(tag, pair)]
            require(row.label_id == str(split["label_id"]), f"{tag}: stale bootstrap label_id")
            n_boot = int(row.n_boot)
            require(n_boot == row.n_boot and int(row.boot_seed) == row.boot_seed,
                    "Bootstrap count/seed must be integers")
            expected = bootstrap(y, rf, pred, n_boot, int(row.boot_seed))
            for key, value in expected.items():
                np.testing.assert_allclose(value, row[key], rtol=0, atol=1e-12,
                                           err_msg=f"{tag}/{pair}/{key}")
            significant = expected["ci_lo"] > 0 or expected["ci_hi"] < 0
            winner = "gin" if expected["ci_lo"] > 0 else "rf" if expected["ci_hi"] < 0 else "tie"
            require(bool(row.significant) == significant and row.winner == winner,
                    f"{tag}/{pair}: bootstrap verdict differs")
            cell = pd.read_csv(root / f"results/significance/{tag}_time.csv").set_index("model_pair")
            require(len(cell) == 2 and not cell.index.duplicated().any(), "Invalid per-cell coverage")
            cell_row = cell.loc[pair]
            for key, value in expected.items():
                cell_key = {"d_r2": "delta_r2_point", "d_rmse": "delta_rmse_point",
                            "d_rmse_ci_lo": "delta_rmse_ci_lo",
                            "d_rmse_ci_hi": "delta_rmse_ci_hi"}.get(key, key)
                np.testing.assert_allclose(cell_row[cell_key], value, rtol=0, atol=1e-12)
            for key in ("label_id", "significant", "winner", "n_boot", "boot_seed"):
                require(cell_row[key] == row[key], f"{tag}/{pair}: per-cell {key} differs")
            require(cell_row.tag == tag and cell_row.split == "time" and cell_row.n_test == len(y),
                    f"{tag}/{pair}: per-cell metadata differs")
            n_boot_values.append(n_boot)
        row = summary.loc[tag]
        require(row.label_id == str(split["label_id"]), f"{tag}: stale summary label_id")
        for pool in ("train", "valid", "test", "excluded"):
            require(int(row[f"n_{pool}"]) == counts[pool], f"{tag}: summary count differs")
        baseline = r2_score(y, np.full(len(y), split["y"].astype(float)[split["train_idx"]].mean()))
        for key, value in dict(constant_r2=baseline, rf_time_r2=r2_score(y, rf),
                               gin_time_r2_mean=np.mean([r2_score(p["y_true"], p["y_pred"])
                                                        for p in predictions[1:]])).items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-12)
        for key, source_key in (("d_r2_mean3", "d_r2"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi")):
            np.testing.assert_allclose(row[key], sig.loc[(tag, "mean3"), source_key], atol=1e-12, rtol=0)
        report["targets"].append(dict(tag=tag, label_id=str(split["label_id"]), counts=counts,
                                      prediction_files=4, bootstrap_counts=n_boot_values))
        print(f"PASS {tag}: raw labels, exclusions, 4 predictions, 2 bootstrap rows", file=sys.stderr)
    report.update(jobs_verified=32, prediction_files=32, bootstrap_rows=16,
                  labels=sum(sum(t["counts"][p] for p in ("train", "valid", "test"))
                             for t in report["targets"]),
                  excluded=sum(t["counts"]["excluded"] for t in report["targets"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, help="Write JSON audit report (otherwise stdout only)")
    parser.add_argument("--scope", choices=("full", "data"), default="full",
                        help="data skips model/feature-cache hashes; both audit labels, predictions and bootstrap")
    args = parser.parse_args(argv)
    report = dict(schema="independent-time-campaign-audit-v1", status="running",
                  audited_at=datetime.now(timezone.utc).isoformat(),
                  root=str(args.root.resolve()), auditor_sha256=sha256(Path(__file__)),
                  hash_scope=args.scope,
                  scope="raw-label/current-prediction/bootstrap audit; no checkpoint replay or retraining",
                  limitations=["Uses saved year cutoffs; does not reselect them",
                               "Reuses 02_clean_data parent normalization only",
                               "Bootstrap CIs condition on saved predictions; no seed/refit uncertainty"])
    try:
        audit(args.root.resolve(), report, args.scope)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
