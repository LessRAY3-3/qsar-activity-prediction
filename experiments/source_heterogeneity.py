"""Source sensitivity of frozen RF/three-seed GIN predictions, without fitting.

Documents are publication annotations, not unique assays or laboratory batches.
All 8 datasets x 3 splits are included. Source removal means removing test
molecules with ANY measurement from that document; it is not leave-source-out
training. This is exploratory robustness analysis, not external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import rdkit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from artifact_guards import atomic_json, check_partition, check_prediction  # noqa: E402
from qsar_common import exact_relation_mask  # noqa: E402
from time_labels import load_time_split, require_prediction_version  # noqa: E402

parent_smiles = importlib.import_module("02_clean_data")._parent_smiles
TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
UNKNOWN = "UNKNOWN"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible_records(raw, years, parents):
    data = raw.copy()
    data["standard_value"] = pd.to_numeric(data.standard_value, errors="coerce")
    keep = (exact_relation_mask(data.standard_relation) & data.standard_units.eq("nM")
            & np.isfinite(data.standard_value) & data.standard_value.gt(0)
            & data.canonical_smiles.fillna("").astype(str).str.strip().ne(""))
    data = data.loc[keep].copy()
    if not data.standard_type.eq("IC50").all() or not data.assay_type.isin(["B", "F"]).all():
        raise ValueError("unexpected endpoint or assay type in eligible frozen records")
    data["parent"] = data.canonical_smiles.map(parents)
    if data.parent.isna().any():
        raise ValueError("missing parent standardization")
    data["pic50"] = -np.log10(data.standard_value * 1e-9)
    data["document_chembl_id"] = data.document_chembl_id.fillna(UNKNOWN).astype(str)
    docs = years.dropna(subset=["document_chembl_id"]).copy()
    docs["document_year"] = pd.to_numeric(docs.document_year, errors="coerce")
    docs = docs[np.isfinite(docs.document_year)].drop_duplicates()
    if docs.document_chembl_id.duplicated().any():
        raise ValueError("conflicting document years")
    return data.merge(docs, on="document_chembl_id", how="left", validate="many_to_one")


def pool_records(data, split, kind, pool):
    result = data[data.parent.isin(split["smiles"][split[pool + "_idx"]])]
    if kind == "time":
        year = result.document_year
        if pool == "train":
            mask = year.lt(int(split["cutoff_valid"]))
        elif pool == "valid":
            mask = year.lt(int(split["cutoff_test"]))
        else:
            mask = year.ge(int(split["cutoff_test"]))
        result = result.loc[mask]
    return result.copy()


def verify_labels(data, keys, labels):
    actual = data.groupby("parent").pic50.median().reindex(keys).to_numpy(float)
    if not np.isfinite(actual).all() or not np.allclose(actual, labels, rtol=0, atol=1e-6):
        raise ValueError("raw source reconstruction differs from frozen labels")
    return float(np.max(np.abs(actual - labels)))


def metrics(y, rf, gin):
    y, rf, gin = (np.asarray(a, float) for a in (y, rf, gin))
    if any(a.shape != y.shape for a in (rf, gin)) or not all(np.isfinite(a).all() for a in (y, rf, gin)):
        raise ValueError("invalid metric inputs")
    out = {"n": len(y), "rf_r2": None, "gin_r2": None, "delta_r2": None,
           "rf_rmse": None, "gin_rmse": None, "rf_mae": None, "gin_mae": None}
    if not len(y):
        return out
    ss = float(np.sum((y-y.mean())**2))
    for name, pred in (("rf", rf), ("gin", gin)):
        out[f"{name}_rmse"] = float(np.sqrt(np.mean((pred-y)**2)))
        out[f"{name}_mae"] = float(np.mean(np.abs(pred-y)))
        if len(y) >= 2 and ss > 1e-12:
            out[f"{name}_r2"] = float(1-np.sum((pred-y)**2)/ss)
    if out["rf_r2"] is not None:
        out["delta_r2"] = out["gin_r2"]-out["rf_r2"]
    return out


def cluster_interval(y, rf, gin, documents, draws=10000, seed=42):
    """Paired document bootstrap, only molecules with exactly one known source.

    All molecules in a sampled document are repeated together. Variable cluster
    sizes imply variable molecule counts. Percentiles omit degenerate SST draws
    and report the omitted count. No interval with fewer than five documents.
    """
    y, rf, gin = (np.asarray(a, float) for a in (y, rf, gin))
    docs = np.asarray(documents, str)
    if len(docs) != len(y) or UNKNOWN in docs:
        raise ValueError("cluster bootstrap needs exclusive known sources")
    unique, inverse = np.unique(docs, return_inverse=True)
    g = len(unique)
    report = {"n_molecules": len(y), "n_documents": g, "draws": draws, "seed": seed,
              "ci_low": None, "ci_high": None, "valid_draws": 0,
              "status": "too_few_documents", "subset_metrics": metrics(y, rf, gin)}
    if g < 5:
        return report
    centered = y-y.mean()
    # Sufficient statistics give exactly the same R2 difference as expanding
    # molecule indices, without a B x N allocation.
    stats = np.stack([np.bincount(inverse, weights=a, minlength=g) for a in
                      (np.ones(len(y)), centered, centered**2, (rf-y)**2-(gin-y)**2)], axis=1)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, draws, 256):
        counts = rng.multinomial(g, np.full(g, 1/g), size=min(256, draws-start))
        n, sy, sy2, diff = (counts @ stats).T
        ss = sy2-sy**2/n
        valid = (n >= 2) & (ss > 1e-12)
        values.extend((diff[valid]/ss[valid]).tolist())
    report["valid_draws"] = len(values)
    if len(values) < .95*draws:
        report["status"] = "too_many_degenerate_draws"
        return report
    lo, hi = np.quantile(values, [.025, .975])
    report.update(status="exploratory_interval", ci_low=float(lo), ci_high=float(hi))
    return report


def source_memberships(data, keys, column):
    grouped = data.groupby("parent")[column].agg(lambda x: tuple(sorted(set(x.fillna(UNKNOWN).astype(str)))))
    result = grouped.reindex(keys)
    if result.isna().any():
        raise ValueError("test molecule without source records")
    return result.tolist()


def analyze_cell(root, tag, kind, data, out, sources, draws):
    def read_npz(rel):
        sources[rel] = sha(root / rel)
        with np.load(root / rel) as z:
            return dict(z)

    fp = read_npz(f"data/processed/{tag}_fingerprints.npz")
    split = read_npz(f"data/processed/splits/{tag}_{kind}.npz")
    if not np.array_equal(fp["smiles"], split["smiles"]):
        raise ValueError("feature/split molecule order mismatch")
    if kind == "time":
        split = load_time_split(root / "data/processed/splits", tag, fp["smiles"], fp["y"])
        y_all = split["y"]
    else:
        check_partition(len(fp["y"]), **{k: split[k] for k in ("train_idx", "valid_idx", "test_idx")})
        y_all = fp["y"]
    pool_data = {pool: pool_records(data, split, kind, pool) for pool in ("train", "valid", "test")}
    errors = {pool: verify_labels(part, split["smiles"][split[pool+"_idx"]],
                                  y_all[split[pool+"_idx"]]) for pool, part in pool_data.items()}
    rf = read_npz(f"results/rf_preds_{tag}_{kind}.npz")
    gin = [read_npz(f"results/gnn_preds_{tag}_{kind}{suffix}.npz") for suffix in ("", "_seed1", "_seed2")]
    for pred in [rf, *gin]:
        check_prediction(pred, split, y_all)
        if kind == "time":
            require_prediction_version(pred, split)
    idx = split["test_idx"]
    keys = split["smiles"][idx]
    y, rf_pred = y_all[idx].astype(float), rf["y_pred"].astype(float)
    gin_pred = np.mean([p["y_pred"].astype(float) for p in gin], axis=0)
    test = pool_data["test"]
    docs = source_memberships(test, keys, "document_chembl_id")
    types = source_memberships(test, keys, "assay_type")
    single = np.asarray([len(d) == 1 and d[0] != UNKNOWN for d in docs])
    cluster = cluster_interval(y[single], rf_pred[single], gin_pred[single],
                               [d[0] for d, keep in zip(docs, single) if keep], draws)
    molecule_bootstrap = cluster_interval(y[single], rf_pred[single], gin_pred[single],
                                          [f"molecule_{i}" for i in range(int(single.sum()))], draws)
    molecule_bootstrap["resampling_unit"] = "molecule, on exactly the same exclusive-source subset"
    molecule_bootstrap["n_resampling_units"] = molecule_bootstrap.pop("n_documents")
    all_docs = sorted({d for group in docs for d in group if d != UNKNOWN})
    counts = {d: sum(d in group for group in docs) for d in all_docs}
    ranked = sorted(all_docs, key=lambda d: (-counts[d], d))
    source_rows, exclusion_rows = [], []
    baseline = metrics(y, rf_pred, gin_pred)
    for rank, doc in enumerate(ranked):
        mask = np.asarray([doc in group for group in docs])
        row = {"tag": tag, "split": kind, "document": doc, "membership_n": int(mask.sum()),
               "fraction_test": float(mask.mean()), "activity_rows": int(test.document_chembl_id.eq(doc).sum())}
        # Tiny source-specific R2 is not interpreted; counts remain visible.
        row.update(metrics(y[mask], rf_pred[mask], gin_pred[mask]) if mask.sum() >= 20 else {"n": int(mask.sum())})
        source_rows.append(row)
        if rank < 5:
            reduced = metrics(y[~mask], rf_pred[~mask], gin_pred[~mask])
            exclusion_rows.append({"tag": tag, "split": kind, "rank_by_size": rank+1,
                                   "removed_document": doc, "removed_n": int(mask.sum()),
                                   "removed_fraction": float(mask.mean()), **reduced})
    assay_rows = []
    labels = np.asarray(["+".join(t) for t in types])
    for label in sorted(set(labels)):
        mask = labels == label
        assay_rows.append({"tag": tag, "split": kind, "assay_type_set": label,
                           **metrics(y[mask], rf_pred[mask], gin_pred[mask])})
    train_docs = set(pool_data["train"].document_chembl_id)-{UNKNOWN}
    rf_train_docs = train_docs if kind == "time" else train_docs | (set(pool_data["valid"].document_chembl_id)-{UNKNOWN})
    medians = test.groupby(["parent", "document_chembl_id"]).pic50.median()
    known = medians[medians.index.get_level_values(1) != UNKNOWN]
    ranges = known.groupby(level=0).agg(lambda x: x.max()-x.min())
    n_docs = known.groupby(level=0).size()
    multi_ranges = ranges[n_docs.ge(2)]
    table = pd.DataFrame({"test_idx": idx, "parent_smiles": keys, "y_true": y,
                          "rf_pred": rf_pred, "gin_mean3_pred": gin_pred,
                          "documents_json": [json.dumps(d) for d in docs],
                          "assay_types_json": [json.dumps(t) for t in types],
                          "single_known_document": single})
    table.to_csv(out / f"{tag}_{kind}_molecules.csv", index=False)
    summary = {"tag": tag, "split": kind, **baseline,
               "label_reconstruction_max_abs_error": max(errors.values()),
               "known_documents": len(all_docs), "single_source_test_n": int(single.sum()),
               "single_source_fraction": float(single.mean()),
               "unknown_source_test_n": sum(UNKNOWN in d for d in docs),
               "multiple_source_test_n": sum(len(d) > 1 for d in docs),
               "largest_document": ranked[0] if ranked else None,
               "largest_document_fraction": counts[ranked[0]]/len(y) if ranked else None,
               "test_n_sharing_document_with_gin_train": sum(bool(set(d) & train_docs) for d in docs),
               "test_n_sharing_document_with_rf_train": sum(bool(set(d) & rf_train_docs) for d in docs),
               "multi_document_parent_n": len(multi_ranges),
               "cross_document_median_pic50_range_median": float(multi_ranges.median()) if len(multi_ranges) else None,
               "cross_document_median_pic50_range_gt1_n": int(multi_ranges.gt(1).sum()),
               "cluster_bootstrap": cluster,
               "molecule_bootstrap_same_subset": molecule_bootstrap,
               "largest_document_excluded": exclusion_rows[0] if exclusion_rows else None}
    return summary, source_rows, exclusion_rows, assay_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/source_heterogeneity")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.draws < 1000:
        parser.error("draws must be >=1000")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sources = {}
    for rel in ("experiments/source_heterogeneity.py", "scripts/02_clean_data.py",
                "scripts/qsar_common.py", "scripts/artifact_guards.py", "scripts/time_labels.py",
                "data/raw/document_years.csv"):
        sources[rel] = sha(ROOT / rel)
    years = pd.read_csv(ROOT / "data/raw/document_years.csv")
    parents, cells, documents, exclusions, assays = {}, [], [], [], []
    for tag in TAGS:
        rel = f"data/raw/{tag}_activities.csv"
        sources[rel] = sha(ROOT / rel)
        raw = pd.read_csv(ROOT / rel)
        for smi in raw.canonical_smiles.dropna().unique():
            if smi not in parents:
                parents[smi] = parent_smiles(smi)[0]
        data = eligible_records(raw, years, parents)
        for kind in ("random", "scaffold", "time"):
            summary, doc, exc, assay = analyze_cell(ROOT, tag, kind, data, args.out_dir, sources, args.draws)
            cells.append(summary)
            documents.extend(doc)
            exclusions.extend(exc)
            assays.extend(assay)
            print(f"{tag}/{kind}: n={summary['n']} documents={summary['known_documents']} delta={summary['delta_r2']:.6f}", flush=True)
    if any(sha(ROOT / rel) != digest for rel, digest in sources.items()):
        raise ValueError("an input changed during analysis")
    pd.DataFrame(documents).to_csv(args.out_dir / "documents.csv", index=False)
    pd.DataFrame(exclusions).to_csv(args.out_dir / "top5_document_exclusions.csv", index=False)
    pd.DataFrame(assays).to_csv(args.out_dir / "assay_type_metrics.csv", index=False)
    atomic_json(args.out_dir / "report.json", {
        "schema": "qsar-source-heterogeneity-v1", "cells": cells, "source_sha256": sources,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "rdkit": rdkit.__version__},
        "protocol": {"models": "frozen RF vs arithmetic mean of GIN seeds 42,1,2",
                     "selection": "all eight tags, all three splits; top five documents by membership count, lexical tie break",
                     "bootstrap": "10000 paired document-cluster draws by default, seed42, exclusive-known-source subset, >=5 documents",
                     "inference": "exploratory, conditional on frozen fitted models; no multiplicity adjustment or confirmatory significance claims"},
        "limitations": ["No unique assay ID, laboratory ID, species or mutant harmonization is inferred from descriptions.",
                        "Document groups overlap when a parent has multiple sources; exclusion removes every parent linked to that document.",
                        "Cluster intervals exclude mixed/unknown-source parents; their estimand differs from the full cohort.",
                        "Test-source removal does not refit models or remove training sources, and is not leave-document-out validation.",
                        "Datasets overlap (EGFR subset/full); 24 cells are not independent studies.",
                        "Assay B/F strata and source-associated effects are descriptive and do not establish causal batch effects."],
        "outputs_sha256": {p.name: sha(p) for p in sorted(args.out_dir.glob("*.csv"))},
    })


if __name__ == "__main__":
    main()
