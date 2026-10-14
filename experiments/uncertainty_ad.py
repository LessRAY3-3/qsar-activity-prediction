"""P8 phase 0: prediction uncertainty + applicability domain (AD) analysis.

Question: how much of RF/GIN's test error is *knowable before* the assay --
(1) does model-reported uncertainty track realised error, (2) do molecules
far from the training set (low max Tanimoto similarity) drive the errors,
and (3) does filtering a virtual screen by AD / uncertainty lift precision?
Analysis only: every artifact (RF forest, GIN checkpoints, cached
fingerprints/graphs) already exists; nothing here is retrained and no
trained prediction artifact is changed. AD-only refresh updates derived CSVs,
summaries and plots after checking their prediction/label alignment.

Run:
  python experiments/uncertainty_ad.py                 # --model gin (default)
  python experiments/uncertainty_ad.py --model gine    # all 8 tags have a
                                                       # full 3-seed gine
                                                       # checkpoint set;
                                                       # cells still skip
                                                       # cleanly if a .pt set
                                                       # is absent or partial
                                                       # (recorded as RF-only,
                                                       # n_gnn_seeds)

Inputs (per tag x split):
  results/rf_preds_{tag}_{split}.npz / models/rf_{split}_split_{tag}.joblib
  results/gnn_models/{tag}_{gin|gine}_{split}[_seedN].pt  3-seed checkpoints
  results/gnn_preds_{tag}[_gin]_{split}*.npz            cross-check only
  data/processed/{tag}_fingerprints.npz, {tag}_graphs.npz
  data/processed/splits/{tag}_{split}.npz

Uncertainty definitions (both in original pIC50 units):
  RF  = std over the forest's 500 individual tree predictions on X_test
        (the ensemble mean is asserted equal to the stored rf_preds y_pred)
  GNN = deep ensemble: the 3 seed checkpoints each predict the test graphs
        (MoleculeGraphDataset + gnn_03.predict, batch 512, MPS-if-available),
        predictions are de-normalised with each checkpoint's y_mean/y_std,
        then the per-molecule std across the 3 seeds (population std).

AD = per test molecule, max Morgan r=2/2048 Tanimoto similarity against the
model's fitted molecules (RF: train + valid; GNN: train) (qsar_common's generator, ExplicitBitVect +
BulkTanimotoSimilarity, one molecule at a time -- never a full n_test x
n_train matrix).

Virtual screening (story closer): the test set plays the library; RF ranks
it. Three strategies x precision@100/@200 (k capped at library size):
  rf_top      whole library ranked by predicted pIC50
  ad_tXX      keep max_sim >= threshold only (0.30/0.35/0.40/0.45/0.50),
              coverage + precision reported per threshold
  unc_lowstd  rank by RF tree-std ascending (most confident first)
True hit = y_true >= 7.0 (primary) and y_true in the top 5% of the library
(second口径). Separately, the "model flags a hit" set {pred >= 7.0} is
scored for precision/recall under three library conditions: unfiltered,
AD >= 0.40, and the confident half (std <= median).

Diagnostics: quintile-of-predicted-std binning vs observed RMSE (calibration
slope/intercept + Spearman of std vs |error|), |error| vs max_sim scatter
with binned means, and R2/MAE inside vs outside AD (threshold 0.4).

Outputs:
  results/uncertainty/{tag}_{split}.csv        per molecule x model:
                                               model, test_idx, smiles,
                                               y_true, pred, std, max_sim,
                                               abs_err
  results/uncertainty/summary_{tag}.json       calibration, AD in/out,
                                               screening tables (both splits)
  figures/uncertainty/calib_{tag}_{split}.png
  figures/uncertainty/ad_error_{tag}_{split}.png
  figures/uncertainty/vs_screen_panel_{split}.png   8 tag subplots, one
                                               panel per split
(--model gine appends "_gine" to all output names so a future gine run
cannot overwrite the gin results.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qsar_common import get_morgan_generator, select_device  # noqa: E402

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
SPLITS = ("random", "scaffold")
SEED_SUFFIXES = ("", "_seed1", "_seed2")
AD_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50)
AD_PRIMARY = 0.40
HIT_THRESHOLD = 7.0
TOPK = (100, 200)
TOP_FRAC = 0.05
N_QUINTILES = 5
BATCH_SIZE = 512
STRATEGY_ORDER = ["rf_top"] + [f"ad_t{t:.2f}" for t in AD_THRESHOLDS] + ["unc_lowstd"]


# ---------------------------------------------------------------- metrics

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    if sst <= 0 or len(y_true) < 2:
        return float("nan")
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / sst


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------- fingerprints / AD

def morgan_fingerprints(smiles_list, cache=None):
    """Morgan r=2/2048 ExplicitBitVect per SMILES; None when RDKit fails.

    `cache` (dict) is shared across the two splits of a tag so each unique
    SMILES is fingerprinted once per run.
    """
    from rdkit import Chem

    gen = get_morgan_generator()
    out = []
    for smi in smiles_list:
        if cache is not None and smi in cache:
            out.append(cache[smi])
            continue
        mol = Chem.MolFromSmiles(smi)
        fp = gen.GetFingerprint(mol) if mol is not None else None
        if cache is not None:
            cache[smi] = fp
        out.append(fp)
    return out


def max_similarities(query_fps, ref_fps):
    """Max Tanimoto similarity of each query fp against all reference fps.

    np.nan when the query failed to fingerprint or the reference set is
    empty; every finite value lies in [0, 1].
    """
    from rdkit import DataStructs

    sims = np.full(len(query_fps), np.nan, dtype=np.float64)
    refs = [f for f in ref_fps if f is not None]
    for i, q in enumerate(query_fps):
        if q is None or not refs:
            continue
        sims[i] = max(DataStructs.BulkTanimotoSimilarity(q, refs))
    return sims


# ---------------------------------------------------------------- model loads

def rf_tree_predict(rf, X):
    """(ensemble mean, inter-tree std) over the forest's individual trees."""
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)   # cast once, not per tree
    tree_preds = np.stack([est.predict(X) for est in rf.estimators_])
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def gnn_model_path(tag, split, model_kind, seed_suffix):
    return os.path.join(BASE, "results", "gnn_models",
                        f"{tag}_{model_kind}_{split}{seed_suffix}.pt")


def gnn_preds_path(tag, split, model_kind, seed_suffix):
    if model_kind == "gin":   # gnn_03 writes no model_kind in the preds name
        return os.path.join(BASE, "results",
                            f"gnn_preds_{tag}_{split}{seed_suffix}.npz")
    return os.path.join(BASE, "results",
                        f"gnn_preds_{tag}_{model_kind}_{split}{seed_suffix}.npz")


def available_seed_suffixes(tag, split, model_kind):
    return [s for s in SEED_SUFFIXES
            if os.path.exists(gnn_model_path(tag, split, model_kind, s))]


def gnn_ensemble_predict(tag, split, model_kind, test_idx, batch_size=BATCH_SIZE):
    """(n_seeds, n_test) predictions in original pIC50 units from the
    stored checkpoints -- inference only, no training.

    Each checkpoint's stored test_idx (via its preds npz, when present) is
    asserted equal to `test_idx`; predictions are de-normalised with the
    checkpoint's own y_mean/y_std.
    """
    import torch
    from torch_geometric.loader import DataLoader

    import gnn_03_train_gin as gnn03
    from gnn_graph_dataset import MoleculeGraphDataset

    predict_fn = gnn03.predict
    model_cls = getattr(gnn03, "MODELS", {}).get(model_kind)
    if model_cls is None:   # pre-refactor gnn_03: GIN only, GINE elsewhere
        if model_kind == "gine":
            from gnn_fairness import GINERegressor as model_cls
        else:
            model_cls = gnn03.GINRegressor

    device = select_device()
    graphs = os.path.join(BASE, "data", "processed", f"{tag}_graphs.npz")
    ds = MoleculeGraphDataset(graphs, indices=test_idx)
    loader = DataLoader(ds, batch_size=batch_size)

    stack = []
    for suf in SEED_SUFFIXES:
        ckpt_path = gnn_model_path(tag, split, model_kind, suf)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ref_path = gnn_preds_path(tag, split, model_kind, suf)
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            if not np.array_equal(ref["test_idx"], test_idx):
                raise ValueError(f"{tag}/{split}{suf}: preds test_idx != "
                                 f"split test_idx")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = model_cls(a["hidden"], a["num_layers"], a["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        pred_std, _ = predict_fn(model, loader, device)
        stack.append(np.asarray(pred_std, dtype=np.float64) * ckpt["y_std"]
                     + ckpt["y_mean"])
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return np.stack(stack)


# ---------------------------------------------------------------- diagnostics

def quintile_calibration(y_true, pred, std, n_bins=N_QUINTILES):
    """Equal-count bins of predicted std vs realised RMSE + OLS fit.

    Returns bin means/counts, the slope/intercept of bin RMSE on bin mean
    std, Spearman(std, |error|) as a monotonicity check, and whether the
    bin RMSEs are non-decreasing in bin std.
    """
    from scipy.stats import spearmanr

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    abs_err = np.abs(y - p)
    order = np.argsort(s, kind="stable")
    bins = np.array_split(order, n_bins)
    bin_std = [float(s[b].mean()) for b in bins]
    bin_rmse = [rmse(y[b], p[b]) for b in bins]
    bin_count = [int(len(b)) for b in bins]

    if np.ptp(bin_std) > 0:
        slope, intercept = np.polyfit(bin_std, bin_rmse, 1)
        slope, intercept = float(slope), float(intercept)
    else:
        slope = intercept = float("nan")
    if np.ptp(s) > 0 and np.ptp(abs_err) > 0:
        rho, _ = spearmanr(s, abs_err)
        rho = float(rho)
    else:
        rho = float("nan")
    diffs = np.diff(bin_rmse)
    return {
        "n_bins": n_bins,
        "bin_mean_std": bin_std,
        "bin_rmse": bin_rmse,
        "bin_count": bin_count,
        "slope": slope,
        "intercept": intercept,
        "spearman_std_vs_abs_err": rho,
        "rmse_monotone_in_std": bool(np.all(diffs >= -1e-12)),
        "std_median": float(np.median(s)),
        "std_mean": float(np.mean(s)),
    }


def ad_metrics(y_true, pred, max_sim, threshold=AD_PRIMARY):
    """R2/RMSE/MAE inside (max_sim >= threshold) vs outside the AD."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    sim = np.asarray(max_sim, dtype=np.float64)
    finite = np.isfinite(sim)
    in_mask = finite & (sim >= threshold)
    out_mask = finite & (sim < threshold)

    def _block(mask):
        if mask.sum() < 2:
            return {"n": int(mask.sum()), "r2": None, "rmse": None, "mae": None}
        return {"n": int(mask.sum()), "r2": r2(y[mask], p[mask]),
                "rmse": rmse(y[mask], p[mask]), "mae": mae(y[mask], p[mask])}

    return {
        "threshold": float(threshold),
        "coverage_in": float(in_mask.sum()) / len(y),
        "n_nan_sim": int((~finite).sum()),
        "in": _block(in_mask),
        "out": _block(out_mask),
    }


# ---------------------------------------------------------------- screening

def _pr(selected, hits):
    """(precision, recall) of `selected` indices against boolean `hits`.

    Recall always divides by ALL true hits in the library, so a filter that
    discards hits shows up as lost recall instead of vanishing.
    """
    selected = np.asarray(selected, dtype=np.int64)
    n_hits = int(hits.sum())
    if len(selected) == 0:
        return float("nan"), float("nan")
    got = int(hits[selected].sum())
    prec = got / len(selected)
    rec = got / n_hits if n_hits else float("nan")
    return float(prec), float(rec)


def screening_summary(pred, std, max_sim, y_true, ad_thresholds=AD_THRESHOLDS,
                      hit_threshold=HIT_THRESHOLD, topk=TOPK,
                      top_frac=TOP_FRAC):
    """RF-ranked virtual-screening tables for one test library.

    Strategies (candidate sets, all ranked by the strategy's key):
      rf_top / ad_tXX / unc_lowstd -- see module docstring. Metrics per
      strategy: coverage, precision/recall at each k (k capped at the
      candidate count), under both hit definitions. Plus the
      model-flagged set {pred >= hit_threshold} scored unfiltered,
      AD-filtered per threshold, and on the confident half (std <= median).
    """
    p = np.asarray(pred, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    sim = np.asarray(max_sim, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    n = len(y)
    finite = np.isfinite(sim)

    is_hit = y >= hit_threshold
    n_top = max(1, int(np.ceil(top_frac * n)))
    is_hit5 = np.zeros(n, dtype=bool)
    is_hit5[np.argsort(-y, kind="stable")[:n_top]] = True

    candidates = {"rf_top": np.argsort(-p, kind="stable")}
    for t in ad_thresholds:
        idx = np.flatnonzero(finite & (sim >= t))
        candidates[f"ad_t{t:.2f}"] = idx[np.argsort(-p[idx], kind="stable")]
    candidates["unc_lowstd"] = np.argsort(s, kind="stable")

    strategies = {}
    names = ["rf_top"] if "rf_top" in candidates else []
    names += sorted(n for n in candidates if n.startswith("ad_t"))
    names += ["unc_lowstd"] if "unc_lowstd" in candidates else []
    for name in names:
        order = candidates[name]
        row = {"n_candidates": int(len(order)),
               "coverage": float(len(order)) / n,
               "precision_at": {}, "recall_at": {},
               "precision_at_top5": {}, "recall_at_top5": {}}
        for k in topk:
            k_eff = min(int(k), len(order))
            top = order[:k_eff]
            key = str(k)
            row["precision_at"][key], row["recall_at"][key] = _pr(top, is_hit)
            (row["precision_at_top5"][key],
             row["recall_at_top5"][key]) = _pr(top, is_hit5)
        strategies[name] = row

    flagged = {}
    conditions = {"all": np.ones(n, dtype=bool)}
    for t in ad_thresholds:
        conditions[f"ad_t{t:.2f}"] = finite & (sim >= t)
    conditions["unc_median"] = s <= np.nanmedian(s)
    for cond_name, mask in conditions.items():
        sel = np.flatnonzero(mask & (p >= hit_threshold))
        prec, rec = _pr(sel, is_hit)
        prec5, rec5 = _pr(sel, is_hit5)
        flagged[cond_name] = {
            "coverage_library": float(mask.sum()) / n,
            "n_flagged": int(len(sel)),
            "precision": prec, "recall": rec,
            "precision_top5": prec5, "recall_top5": rec5,
        }

    return {
        "n_library": int(n),
        "hit_threshold": float(hit_threshold),
        "n_hits": int(is_hit.sum()),
        "n_hits_top5": int(is_hit5.sum()),
        "strategies": strategies,
        "flagged_pred_ge_threshold": flagged,
    }


# ---------------------------------------------------------------- figures

def plot_calibration(tag, split, calib, fig_path, model_kind="gin"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for model, color in (("rf", "tab:orange"), ("gin", "tab:green")):
        c = calib.get(model)
        if c is None:
            continue
        display_model = model_kind.upper() if model == "gin" else model.upper()
        ax.plot(c["bin_mean_std"], c["bin_rmse"], marker="o", color=color,
                label=f"{display_model}  slope={c['slope']:.2f}, "
                      f"ρ={c['spearman_std_vs_abs_err']:.2f}")
    lo = min(min(calib[m]["bin_mean_std"]) for m in calib
             if calib.get(m) is not None)
    hi = max(max(calib[m]["bin_mean_std"]) for m in calib
             if calib.get(m) is not None)
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.5",
            linewidth=1, label="perfect calibration (y = x)")
    ax.set_xlabel("predicted std (quintile bin mean)")
    ax.set_ylabel("observed RMSE")
    ax.set_title(f"{tag}/{split}: uncertainty calibration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_ad_error(tag, split, y_true, preds, max_sim, ad_thr, fig_path, model_kind="gin"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for model, pred, color in preds:
        sim = max_sim[model] if isinstance(max_sim, dict) else max_sim
        finite = np.isfinite(sim)
        abs_err = np.abs(np.asarray(y_true, float) - np.asarray(pred, float))
        ax.scatter(sim[finite], abs_err[finite], s=8, alpha=0.3,
                   color=color, linewidths=0,
                   label=f"{model.upper()} molecules" if model == "rf" else None)
        # binned mean line over 10 quantile bins of max_sim
        idx = np.flatnonzero(finite)
        groups = [b for b in np.array_split(idx[np.argsort(sim[idx])], 10)
                  if len(b)]
        means_x = [sim[b].mean() for b in groups]
        means_y = [abs_err[b].mean() for b in groups]
        display_model = model_kind.upper() if model == "gin" else model.upper()
        ax.plot(means_x, means_y, marker="s", color=color, linewidth=2,
                label=f"{display_model} binned |error| mean")
    ax.axvline(ad_thr, color="black", linestyle="--", linewidth=1,
               label=f"AD threshold {ad_thr}")
    ax.set_xlabel("max Tanimoto similarity to fitted molecules")
    ax.set_ylabel("|error| (pIC50)")
    ax.set_title(f"{tag}/{split}: error vs applicability domain")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_vs_screen_panel(split, screen_by_tag, fig_path):
    """One panel per tag: precision@k of every screening strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    tags = [t for t in TAGS if t in screen_by_tag]
    if not tags:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.5), sharey=True)
    axes = axes.ravel()
    width = 0.2
    x = np.arange(len(STRATEGY_ORDER))
    for ax, tag in zip(axes, tags):
        scr = screen_by_tag[tag]["strategies"]
        series = [
            ("precision_at", "100", "tab:blue", "", "P@100 (y_true ≥ 7)"),
            ("precision_at", "200", "tab:orange", "", "P@200 (y_true ≥ 7)"),
            ("precision_at_top5", "100", "tab:blue", "//", "P@100 (top-5%)"),
            ("precision_at_top5", "200", "tab:orange", "//", "P@200 (top-5%)"),
        ]
        for j, (field, k, color, hatch, label) in enumerate(series):
            vals = [scr.get(name, {}).get(field, {}).get(k, np.nan)
                    for name in STRATEGY_ORDER]
            ax.bar(x + (j - 1.5) * width, vals, width, color=color,
                   hatch=hatch, edgecolor="white", linewidth=0.4,
                   label=label if tag == tags[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_ORDER, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{tag}  (n={screen_by_tag[tag]['n_library']})",
                     fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(tags):]:
        ax.axis("off")
    fig.suptitle(f"Virtual screening on the test set -- {split} split "
                 f"(RF ranking, hit = true pIC50 ≥ 7 / top 5%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ---------------------------------------------------------------- json helper

def jsonable(obj):
    """Recursively cast numpy scalars; non-finite floats become None."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def model_ad_similarities(smiles, split, fp_cache=None, *, split_name):
    """RF fits the whole non-test pool; the GNN fits train only."""
    if split_name not in SPLITS:
        raise ValueError("AD supports random/scaffold only; time uses a different RF fit pool")
    query = morgan_fingerprints(list(smiles[split["test_idx"]]), cache=fp_cache)
    pools = {"rf": np.concatenate([split["train_idx"], split["valid_idx"]]),
             "gin": split["train_idx"]}
    return {model: max_similarities(query, morgan_fingerprints(list(smiles[idx]), cache=fp_cache))
            for model, idx in pools.items()}


def redraw_stored_figures(tag, split, model_kind, out_dir, fig_dir):
    """Render saved calibration/AD values only; do not run inference or write data.

    The stored schema uses the legacy key 'gin' for either GNN architecture;
    summary.model_kind identifies the architecture used for display labels.
    """
    suffix = "" if model_kind == "gin" else f"_{model_kind}"
    with open(os.path.join(out_dir, f"summary_{tag}{suffix}.json")) as fh:
        summary = json.load(fh)
    if summary.get("tag") != tag or summary.get("model_kind") != model_kind:
        raise ValueError("stored summary identifies a different campaign")
    cell = summary["splits"][split]
    frame = pd.read_csv(os.path.join(out_dir, f"{tag}_{split}{suffix}.csv"))
    rf = frame[frame["model"] == "rf"]
    if len(rf) != cell["n_test"] or rf["test_idx"].duplicated().any():
        raise ValueError("stored RF rows do not match summary test count")
    preds, sims = [], {}
    for model, color in (("rf", "tab:orange"), ("gin", "tab:green")):
        rows = frame[frame["model"] == model]
        if rows.empty:
            if model in cell["calibration"]:
                raise ValueError("stored CSV is missing a calibration model")
            continue
        if (not np.array_equal(rows["test_idx"], rf["test_idx"])
                or not np.array_equal(rows["y_true"], rf["y_true"])):
            raise ValueError("stored model rows do not align with RF indices/labels")
        preds.append((model, rows["pred"].to_numpy(), color))
        sims[model] = rows["max_sim"].to_numpy()
    paths = [os.path.join(fig_dir, f"calib_{tag}_{split}{suffix}.png"),
             os.path.join(fig_dir, f"ad_error_{tag}_{split}{suffix}.png")]
    plot_calibration(tag, split, cell["calibration"], paths[0], model_kind=model_kind)
    plot_ad_error(tag, split, rf["y_true"].to_numpy(), preds, sims,
                  summary["ad_threshold"], paths[1], model_kind=model_kind)
    return paths


def refresh_stored_ad(tag, split, model_kind, out_dir, fig_dir, fp_cache=None):
    """Recompute AD from saved predictions/std; no inference or retraining."""
    suffix = "" if model_kind == "gin" else f"_{model_kind}"
    csv_path = os.path.join(out_dir, f"{tag}_{split}{suffix}.csv")
    summary_path = os.path.join(out_dir, f"summary_{tag}{suffix}.json")
    if split not in SPLITS:
        raise ValueError("AD refresh supports random/scaffold only")
    original_text = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame = pd.read_csv(csv_path)
    with open(summary_path) as fh:
        summary = json.load(fh)
    if summary.get("tag") != tag or summary.get("model_kind") != model_kind:
        raise ValueError("stored uncertainty summary identifies a different campaign")
    cell = summary["splits"][split]
    models = set(frame["model"])
    if not models <= {"rf", "gin"} or "rf" not in models:
        raise ValueError("stored uncertainty rows must contain RF and optional GNN only")
    with np.load(os.path.join(BASE, "data", "processed", "splits", f"{tag}_{split}.npz")) as sp:
        with np.load(os.path.join(BASE, "data", "processed", f"{tag}_fingerprints.npz")) as fp:
            test_idx = sp["test_idx"]
            for model in models:
                part = frame[frame["model"] == model]
                if not np.array_equal(part["test_idx"], test_idx):
                    raise ValueError("stored uncertainty rows differ from split")
                if not np.array_equal(part["smiles"], fp["smiles"][test_idx]):
                    raise ValueError("stored uncertainty structures differ from cache")
                if not np.allclose(part["y_true"], fp["y"][test_idx], rtol=0, atol=1e-4):
                    raise ValueError("stored uncertainty labels differ from cache")
                values = part[["y_true", "pred", "std", "abs_err"]].to_numpy()
                if not np.isfinite(values).all() or (part["std"] < 0).any():
                    raise ValueError("invalid stored uncertainty values")
                if not np.allclose(part["abs_err"], abs(part["y_true"] - part["pred"]), rtol=0, atol=1e-8):
                    raise ValueError("stored uncertainty errors are stale")
            # Validate saved predictions before expensive fingerprint comparisons or writes.
            with np.load(os.path.join(BASE, "results", f"rf_preds_{tag}_{split}.npz")) as rf:
                part = frame[frame["model"] == "rf"]
                if (not np.array_equal(rf["test_idx"], test_idx)
                        or not np.allclose(part["y_true"], rf["y_true"], rtol=0, atol=1e-4)
                        or not np.allclose(part["pred"], rf["y_pred"], rtol=0, atol=1e-8)):
                    raise ValueError("stored RF predictions are stale or misaligned")
            if "gin" in models:
                stack = []
                part = frame[frame["model"] == "gin"]
                for seed_suffix in SEED_SUFFIXES:
                    with np.load(gnn_preds_path(tag, split, model_kind, seed_suffix)) as pred:
                        if (not np.array_equal(pred["test_idx"], test_idx)
                                or not np.allclose(part["y_true"], pred["y_true"], rtol=0, atol=1e-4)):
                            raise ValueError("stored GNN predictions are misaligned")
                        stack.append(np.asarray(pred["y_pred"], dtype=float))
                # The saved CSV was inferred separately; permit float32/device roundoff.
                stack = np.stack(stack)
                if (not np.allclose(part["pred"], stack.mean(axis=0), rtol=0, atol=1e-5)
                        or not np.allclose(part["std"], stack.std(axis=0), rtol=0, atol=1e-5)):
                    raise ValueError("stored GNN predictions or uncertainty are stale")
            sims = model_ad_similarities(fp["smiles"], sp, fp_cache, split_name=split)
    plot_inputs = []
    for model, color in (("rf", "tab:orange"), ("gin", "tab:green")):
        mask = frame["model"] == model
        if not mask.any():
            continue
        frame.loc[mask, "max_sim"] = sims[model]
        part = frame[mask]
        cell["ad"][model] = ad_metrics(part["y_true"].to_numpy(), part["pred"].to_numpy(), sims[model])
        plot_inputs.append((model, part["pred"].to_numpy(), color))
    rf_rows = frame[frame["model"] == "rf"]
    y = rf_rows["y_true"].to_numpy()
    cell["screening"] = screening_summary(rf_rows["pred"].to_numpy(), rf_rows["std"].to_numpy(), sims["rf"], y)
    cell["ad_reference"] = {"rf": "train+valid", "gin": "train"}
    # Preserve the exact prediction/std text; refreshing AD must not churn it.
    original_text["max_sim"] = frame["max_sim"].map(str)
    original_text.to_csv(csv_path, index=False)
    plot_ad_error(tag, split, y, plot_inputs, sims, AD_PRIMARY,
                  os.path.join(fig_dir, f"ad_error_{tag}_{split}{suffix}.png"), model_kind=model_kind)
    return cell


# ---------------------------------------------------------------- cell

def process_cell(tag, split, model_kind, out_dir, fig_dir, fp_cache=None,
                 batch_size=BATCH_SIZE):
    """Run one (tag, split): CSV + two figures + summary dict."""
    if split not in SPLITS:
        raise ValueError("AD supports random/scaffold only")
    t0 = time.time()
    sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                              f"{tag}_{split}.npz"))
    test_idx = sp["test_idx"]
    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{tag}_fingerprints.npz"))
    smiles = fp["smiles"]

    rf_npz = np.load(os.path.join(BASE, "results",
                                  f"rf_preds_{tag}_{split}.npz"))
    if not np.array_equal(rf_npz["test_idx"], test_idx):
        raise ValueError(f"{tag}/{split}: rf test_idx != split test_idx")
    y_true = np.asarray(rf_npz["y_true"], dtype=np.float64)
    rf_pred = np.asarray(rf_npz["y_pred"], dtype=np.float64)

    # ---- RF uncertainty (tree-wise)
    import joblib
    rf = joblib.load(os.path.join(BASE, "models",
                                  f"rf_{split}_split_{tag}.joblib"))
    tree_mean, rf_std = rf_tree_predict(rf, fp["X"][test_idx])
    tree_diff = float(np.abs(tree_mean - rf_pred).max())
    if tree_diff > 1e-6:
        print(f"  [{tag}/{split}] warning: tree-mean vs stored rf pred "
              f"max|diff|={tree_diff:.2e}")

    # ---- GNN deep ensemble (skips when checkpoints are missing)
    # A partial checkpoint set is treated like an absent one: skip the
    # GNN rows for this cell and record the actual seed count, rather
    # than mixing a 1- or 2-seed ensemble into a 3-seed campaign (and
    # gnn_ensemble_predict would raise on the missing .pt anyway).
    seeds = available_seed_suffixes(tag, split, model_kind)
    gin_pred, gin_std, gnn_diff = None, None, None
    if len(seeds) == len(SEED_SUFFIXES):
        stack = gnn_ensemble_predict(tag, split, model_kind, test_idx,
                                     batch_size=batch_size)
        gin_pred, gin_std = stack.mean(axis=0), stack.std(axis=0)
        refs = [np.load(gnn_preds_path(tag, split, model_kind, s))
                for s in seeds
                if os.path.exists(gnn_preds_path(tag, split, model_kind, s))]
        if len(refs) == len(seeds):
            ref_stack = np.stack([np.asarray(r["y_pred"], dtype=np.float64)
                                  for r in refs])
            gnn_diff = float(np.abs(stack - ref_stack).max())
            if gnn_diff > 5e-3:
                print(f"  [{tag}/{split}] warning: ensemble vs stored preds "
                      f"max|diff|={gnn_diff:.2e}")
    elif seeds:
        print(f"  [{tag}/{split}] only {len(seeds)}/{len(SEED_SUFFIXES)} "
              f"{model_kind} checkpoints -> GNN rows omitted (RF-only cell; "
              f"partial seed set recorded as n_gnn_seeds={len(seeds)})")
    else:
        print(f"  [{tag}/{split}] no {model_kind} checkpoints -> "
              f"GNN rows omitted (RF-only cell)")

    # Each model's actual fitted molecule pool defines its AD.
    max_sim = model_ad_similarities(smiles, sp, fp_cache, split_name=split)

    # ---- per-molecule CSV
    frames = []
    for model, pred, std in (("rf", rf_pred, rf_std),
                             ("gin", gin_pred, gin_std)):
        if pred is None:
            continue
        frames.append(pd.DataFrame({
            "model": model,
            "test_idx": test_idx,
            "smiles": smiles[test_idx],
            "y_true": y_true,
            "pred": pred,
            "std": std,
            "max_sim": max_sim[model],
            "abs_err": np.abs(y_true - pred),
        }))
    suffix = "" if model_kind == "gin" else f"_{model_kind}"
    csv_path = os.path.join(out_dir, f"{tag}_{split}{suffix}.csv")
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)

    # ---- diagnostics
    calib = {}
    ad = {}
    for model, pred in (("rf", rf_pred), ("gin", gin_pred)):
        if pred is None:
            continue
        calib[model] = quintile_calibration(y_true, pred, std=(rf_std if model == "rf"
                                                               else gin_std))
        ad[model] = ad_metrics(y_true, pred, max_sim[model], AD_PRIMARY)

    plot_calibration(tag, split, calib,
                     os.path.join(fig_dir, f"calib_{tag}_{split}{suffix}.png"), model_kind=model_kind)
    scatter_inputs = [("rf", rf_pred, "tab:orange")]
    if gin_pred is not None:
        scatter_inputs.append(("gin", gin_pred, "tab:green"))
    plot_ad_error(tag, split, y_true, scatter_inputs, max_sim,
                  AD_PRIMARY,
                  os.path.join(fig_dir, f"ad_error_{tag}_{split}{suffix}.png"), model_kind=model_kind)

    screening = screening_summary(rf_pred, rf_std, max_sim["rf"], y_true)
    print(f"  [{tag}/{split}] n={len(y_true)}  "
          f"std med rf={np.median(rf_std):.3f}"
          + (f" gin={np.median(gin_std):.3f}" if gin_std is not None else "")
          + f"  AD>=0.4 coverage={ad['rf']['coverage_in']:.2f}  "
          f"P@100 full={screening['strategies']['rf_top']['precision_at']['100']:.3f} "
          f"ad0.40="
          f"{screening['strategies']['ad_t0.40']['precision_at']['100']:.3f}  "
          f"({time.time() - t0:.0f}s)")

    return {
        "ad_reference": {"rf": "train+valid", "gin": "train"},
        "n_test": int(len(y_true)),
        "rf_tree_mean_max_diff": tree_diff,
        "gnn_ensemble_max_diff": gnn_diff,
        "n_gnn_seeds": len(seeds),
        "calibration": calib,
        "ad": ad,
        "screening": screening,
        "csv": os.path.relpath(csv_path, BASE),
    }


# ---------------------------------------------------------------- report

def print_conclusion(tag_summaries, model_kind):
    print(f"\n=== Uncertainty + AD ({model_kind}) ===")
    for tag, splits in tag_summaries.items():
        for split, cell in splits.items():
            cal = cell["calibration"]
            ad = cell["ad"]["rf"]
            rf_cal = cal.get("rf")
            gin_cal = cal.get("gin")
            txt = (f"  {tag}/{split:<9} calib slope rf="
                   f"{rf_cal['slope']:+.2f} (rho={rf_cal['spearman_std_vs_abs_err']:+.2f}, "
                   f"monotone={rf_cal['rmse_monotone_in_std']})")
            if gin_cal is not None:
                txt += (f" | gin={gin_cal['slope']:+.2f} "
                        f"(rho={gin_cal['spearman_std_vs_abs_err']:+.2f}, "
                        f"monotone={gin_cal['rmse_monotone_in_std']})")
            rin, rout = ad["in"], ad["out"]
            txt += (f" | AD-in cov={ad['coverage_in']:.2f} "
                    f"R2 {rin['r2']:.3f} vs out {rout['r2']:.3f}"
                    if rin["r2"] is not None and rout["r2"] is not None
                    else f" | AD-in cov={ad['coverage_in']:.2f}")
            print(txt)

    # aggregate screening deltas: rf_top vs AD 0.40 vs uncertainty
    for split in SPLITS:
        cells = [(tag, ss[split]) for tag, ss in tag_summaries.items()
                 if split in ss]
        if not cells:
            continue
        base, ad_hit, unc = [], [], []
        for tag, cell in cells:
            st = cell["screening"]["strategies"]
            base.append(st["rf_top"]["precision_at"]["100"])
            ad_hit.append(st["ad_t0.40"]["precision_at"]["100"])
            unc.append(st["unc_lowstd"]["precision_at"]["100"])
        print(f"  [{split}] mean P@100 (hit y>=7): rf_top={np.nanmean(base):.3f}"
              f"  AD>=0.40={np.nanmean(ad_hit):.3f}  "
              f"unc_lowstd={np.nanmean(unc):.3f}  "
              f"(AD lift {np.nanmean(ad_hit) - np.nanmean(base):+.3f})")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["gin", "gine"], default="gin",
                   help="GNN checkpoints to build the deep ensemble from")
    p.add_argument("--tags", default=",".join(TAGS))
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    refresh = p.add_mutually_exclusive_group()
    refresh.add_argument("--refresh-ad-only", action="store_true",
                         help="reuse stored predictions/std and refresh AD/screening only")
    refresh.add_argument("--refresh-figures-only", action="store_true",
                         help="redraw calibration/AD figures from stored CSV/JSON only")
    p.add_argument("--out-dir", default=os.path.join(BASE, "results",
                                                     "uncertainty"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures",
                                                     "uncertainty"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not tags or not splits or len(set(tags)) != len(tags) or len(set(splits)) != len(splits):
        raise ValueError("tags and splits must be nonempty without duplicates")
    if not set(splits) <= set(SPLITS):
        raise ValueError("AD supports random/scaffold only")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    model_kind = args.model
    suffix = "" if model_kind == "gin" else f"_{model_kind}"

    if args.refresh_figures_only:
        for tag in tags:
            for split in splits:
                paths = redraw_stored_figures(tag, split, model_kind, args.out_dir, args.fig_dir)
                print(f"  figures only -> {', '.join(paths)}")
        return

    if not args.refresh_ad_only and not any(available_seed_suffixes(t, s, model_kind)
               for t in tags for s in splits):
        print(f"no {model_kind} checkpoints under results/gnn_models "
              f"({tags[0]}... x {splits}) - nothing to analyse, exiting 0")
        return

    out_dir = os.path.abspath(args.out_dir)
    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    tag_summaries = {}
    screen_by_split = {s: {} for s in splits}
    fp_cache = {}
    for tag in tags:
        per_split = {}
        for split in splits:
            if args.refresh_ad_only:
                cell = refresh_stored_ad(tag, split, model_kind, out_dir, fig_dir, fp_cache)
            else:
                cell = process_cell(tag, split, model_kind, out_dir, fig_dir,
                                    fp_cache=fp_cache, batch_size=args.batch_size)
            per_split[split] = cell
            screen_by_split[split][tag] = cell["screening"]
        tag_summaries[tag] = per_split
        json_path = os.path.join(out_dir, f"summary_{tag}{suffix}.json")
        payload = {"tag": tag, "model_kind": model_kind,
                   "ad_threshold": AD_PRIMARY, "splits": {}}
        if os.path.exists(json_path):
            with open(json_path) as fh:
                previous = json.load(fh)
            if previous.get("tag") != tag or previous.get("model_kind") != model_kind:
                raise ValueError("existing summary identifies a different campaign")
            payload.update(previous)
        payload["splits"].update(per_split)
        with open(json_path, "w") as fh:
            json.dump(jsonable(payload), fh, indent=2)
        print(f"  summary -> {os.path.relpath(json_path, BASE)}")

    for split in splits:
        partial = "" if set(tags) == set(TAGS) else "_partial"
        fig_path = os.path.join(fig_dir, f"vs_screen_panel_{split}{suffix}{partial}.png")
        made = plot_vs_screen_panel(split, screen_by_split[split], fig_path)
        if made:
            print(f"  figure  -> {os.path.relpath(made, BASE)}")

    print_conclusion(tag_summaries, model_kind)


if __name__ == "__main__":
    main()
