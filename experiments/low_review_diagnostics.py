"""Reproduce PR8 Low statistical checks without fitting or modifying models.

Run from any directory: python experiments/low_review_diagnostics.py
Requires local raw a2a/document-year CSVs and restored frozen time inputs.
Writes separate diagnostic CSV/JSON outputs only. The historical RF comparison
uses the archived Git blob when available; it is not a training-seed experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from qsar_common import exact_relation_mask  # noqa: E402

_parent_smiles = importlib.import_module("02_clean_data")._parent_smiles
TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ad_diagnostics():
    rows = []
    sources = {}
    for tag in TAGS:
        for split in ("random", "scaffold"):
            path = ROOT / f"results/uncertainty/{tag}_{split}.csv"
            sources[str(path.relative_to(ROOT))] = sha(path)
            frame = pd.read_csv(path)
            frame = frame[frame.model.eq("rf")]
            row = {"tag": tag, "split": split}
            for label, mask in (("in", frame.max_sim.ge(.4)), ("out", frame.max_sim.lt(.4))):
                part = frame[mask]
                y, pred = part.y_true.to_numpy(), part.pred.to_numpy()
                variance = float(np.var(y)) if len(y) else np.nan
                mse = float(np.mean((pred - y) ** 2)) if len(y) else np.nan
                row.update({f"n_{label}": len(y), f"variance_{label}": variance,
                            f"rmse_{label}": float(np.sqrt(mse)),
                            f"mae_{label}": float(np.mean(np.abs(pred - y))) if len(y) else np.nan,
                            f"r2_{label}": 1 - mse / variance if len(y) >= 2 and variance > 0 else np.nan})
            rows.append(row)
    table = pd.DataFrame(rows)
    paired = table[table.r2_out.notna() & table.r2_in.notna()]
    summary = {"threshold": .4, "aggregation": "unweighted mean over the same scorable cells",
               "cells": len(table), "paired_scorable_cells": len(paired),
               "out_rmse_greater_cells": int((paired.rmse_out > paired.rmse_in).sum()),
               "mean_out_rmse": float(paired.rmse_out.mean()), "mean_in_rmse": float(paired.rmse_in.mean()),
               "mean_out_mae": float(paired.mae_out.mean()), "mean_in_mae": float(paired.mae_in.mean()),
               "limitations": "Subset R2 denominators differ; compare RMSE/MAE on paired cells too. Observational AD groups do not establish causality."}
    return table, summary, sources


def a2a_diagnostics():
    paths = [ROOT / "data/raw/a2a_activities.csv", ROOT / "data/raw/document_years.csv",
             ROOT / "data/processed/splits/a2a_time.npz"]
    acts, years = (pd.read_csv(p) for p in paths[:2])
    with np.load(paths[2]) as sp:
        keys = sp["smiles"][sp["test_idx"]].astype(str)
        y = sp["y"][sp["test_idx"]].astype(float)
        cutoff = int(sp["cutoff_test"])
        label_id = str(sp["label_id"])
        n_total = len(sp["smiles"])
    cache = {s: _parent_smiles(s)[0] for s in acts.canonical_smiles.dropna().unique()}
    acts["parent"] = acts.canonical_smiles.map(cache)
    acts["standard_value"] = pd.to_numeric(acts.standard_value, errors="coerce")
    acts = acts[exact_relation_mask(acts.standard_relation) & acts.standard_units.eq("nM")
                & np.isfinite(acts.standard_value) & acts.standard_value.gt(0)]
    acts = acts.merge(years, on="document_chembl_id", how="left", validate="many_to_one")
    acts = acts[acts.document_year.ge(cutoff) & acts.parent.isin(keys)].copy()
    acts["pic50"] = -np.log10(acts.standard_value * 1e-9)
    recomputed = acts.groupby("parent").pic50.median().reindex(keys).to_numpy()
    if not np.allclose(recomputed, y, atol=1e-6, rtol=0):
        raise ValueError("a2a raw-label reconstruction differs from frozen test labels")
    groups = acts.groupby(["document_year", "document_chembl_id"]).agg(
        activity_rows=("parent", "size"), test_parents=("parent", "nunique"),
        distinct_ic50_nm=("standard_value", "nunique"), assay_descriptions=("assay_description", "nunique")
    ).reset_index().sort_values("test_parents", ascending=False)
    groups["fraction_test"] = groups.test_parents / len(keys)
    largest = groups.iloc[0]
    source = acts[acts.document_chembl_id.eq(largest.document_chembl_id)]
    source_keys = set(source.parent)
    unique_docs = acts.groupby("parent").document_chembl_id.nunique()
    only = [k for k in source_keys if unique_docs[k] == 1]
    levels = source.standard_value.value_counts().sort_index()
    summary = {"label_id": label_id, "test_cutoff": cutoff, "n_total": n_total,
               "n_test": len(keys), "test_fraction": len(keys) / n_total,
               "raw_reconstruction_max_abs_error": float(np.max(np.abs(recomputed - y))),
               "largest_source_document": largest.document_chembl_id,
               "document_year": int(largest.document_year), "test_parents": len(source_keys),
               "fraction_test": len(source_keys) / len(keys), "activity_rows": len(source),
               "parents_with_no_other_eligible_document": len(only),
               "ic50_nm_activity_row_counts": {str(float(k)): int(v) for k, v in levels.items()},
               "assay_descriptions": sorted(source.assay_description.dropna().unique().tolist()),
               "frozen_labels_for_source_parents": {str(float(k)): int(v) for k, v in
                   pd.Series(y[np.isin(keys, list(source_keys))]).value_counts().sort_index().items()},
               "limitations": "Shared document and description are not proof of one laboratory batch; raw discrete IC50 values are source annotations, not independently verified precision. Parent medians can combine records/documents."}
    return groups, summary, {str(p.relative_to(ROOT)): sha(p) for p in paths}


def rf_evidence():
    current = pd.read_csv(ROOT / "results/time_split/summary.csv").set_index("tag").loc["herg"]
    report = {"published_rf_fits_per_target": 1, "published_rf_random_state": 42,
              "current_herg_ensemble_delta": float(current.d_r2_mean3),
              "review_five_random_states_r2_sd_claim": .006,
              "five_state_claim_status": "review-reported, not independently reproduced; no five-fit artifact cohort identified",
              "limitation": "Molecule bootstrap conditions on one RF fit. hERG input repair also retrained all three GIN seeds; before/after changes are not RF seed variance."}
    try:
        blob = subprocess.check_output(["git", "show", "920bc28:results/time_split/summary.csv"], cwd=ROOT, stderr=subprocess.DEVNULL)
        expected = next(x["sha256"] for x in json.loads((ROOT / "results/time_split/initial_campaign_acceptance.json").read_text())["analysis"]["files"] if x["path"] == "results/time_split/summary.csv")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("Initial summary blob differs from initial acceptance hash")
        initial = pd.read_csv(io.BytesIO(blob)).set_index("tag").loc["herg"]
        report.update(initial_summary_revision="920bc28", initial_summary_sha256=expected,
                      initial_herg_ensemble_delta=float(initial.d_r2_mean3),
                      initial_herg_rf_r2=float(initial.rf_time_r2), current_herg_rf_r2=float(current.rf_time_r2))
    except subprocess.CalledProcessError:
        report["initial_summary_status"] = "historical Git blob unavailable; fetch full history to reproduce"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/review_low_diagnostics")
    args = parser.parse_args()
    ad, ad_summary, sources = ad_diagnostics()
    groups, a2a, more_sources = a2a_diagnostics()
    sources.update(more_sources)
    for name in ("scripts/02_clean_data.py", "scripts/qsar_common.py",
                 "results/time_split/summary.csv",
                 "results/time_split/initial_campaign_acceptance.json"):
        sources[name] = sha(ROOT / name)
    package = ROOT / "results/time_split/inputs/splits.tar.gz"
    result = {"schema": "pr8-low-statistical-diagnostics-v1", "ad": ad_summary,
              "a2a_source_concentration": a2a, "rf_training_randomness": rf_evidence(),
              "time_input_package": {"bytes": package.stat().st_size, "mib": package.stat().st_size / 2**20, "sha256": sha(package)},
              "source_sha256": sources}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ad.to_csv(args.out_dir / "ad_subset_metrics.csv", index=False)
    groups.to_csv(args.out_dir / "a2a_source_groups.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
