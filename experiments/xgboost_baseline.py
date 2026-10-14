"""XGBoost baseline: is the traditional-ML ceiling higher than RF suggests?

Identical evaluation contract as the headline RF: same persisted
2048-bit Morgan r=2 fingerprints (the headline featurization), same
persisted test set, and the same training molecules - the full non-test
pool (train + valid; the RF cross-validates, so it never carved out a
validation set, while the GNN's 10% valid split is a subset of the RF's
training rows). Fixed hyperparameters, 3 seeds x 2 splits. The RF
control is read from results/metrics_{TAG}.json (the headline
full-pool RF); the RF training pool behind the same_pool fingerprint is
rederived from the RF's own predictions artifact
(results/rf_preds_{TAG}_{SPLIT}.npz test_idx), not from the same split
file the XGB pool uses, so the check compares two independent artifacts.
Learning-curve RFs are never used as controls - they fit at n_train far
below the pool, so they are not same-pool either.

Note: XGBoost runs on CPU (tree_method="hist") - deterministic for a
fixed seed, and fast enough at this scale. libomp is required on macOS:
    brew install libomp

Outputs:
  results/xgboost/xgboost_{TAG}.csv  one row per run
  results/xgboost/summary_{TAG}.json (embeds the RF control values and
    the XGB/RF pool fingerprints behind the same_pool verdict)
  figures/xgboost/xgboost_{TAG}.png
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
HEADER = "split,seed,r2,rmse,mae,train_seconds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--splits", default="random,scaffold",
                   help="comma-separated random/scaffold only; time labels are not supported")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--out", default=os.path.join(BASE, "results", "xgboost"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "xgboost"))
    args = p.parse_args()
    if any(split.strip() not in ("random", "scaffold") for split in args.splits.split(",")):
        p.error("--splits supports only random/scaffold; time requires as-of labels "
                "and is not implemented in this experiment")
    return args


def pool_fingerprint(idx):
    """SHA-256 over a pool's sorted row indices (int64 bytes).

    Order-independent: train+valid concatenated in any order hashes the
    same, so the fingerprint identifies the pool's membership, not its
    layout. Returns None when the pool cannot be rebuilt (idx is None).
    """
    if idx is None:
        return None
    arr = np.sort(np.asarray(idx, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_rf_test_idx(tag, split):
    """Test indices from the RF's own predictions artifact, or None.

    results/rf_preds_{tag}_{split}.npz is written by the run that FIT the
    headline RF, so its test_idx -- not the split file's -- is the
    independent ground truth for which rows the RF never trained on.
    Rebuilding the RF pool from the same split npz the XGB pool uses would
    make the same_pool check vacuous for random/scaffold (the two pools
    could never disagree).
    """
    path = os.path.join(BASE, "results", f"rf_preds_{tag}_{split}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["test_idx"]


def rf_train_pool(split, sp, n_total, rf_test_idx=None):
    """Rows the headline RF trains on for `split`.

    random/scaffold: the complement of the RF's own test set -- taken from
    rf_test_idx (results/rf_preds_{tag}_{split}.npz, see
    load_rf_test_idx) when available, falling back to the split file's
    test_idx only when the preds artifact is missing (main() warns: the
    fallback is self-referential and cannot disagree with the XGB pool).
    04 fits everything outside the test fold (5-fold CV, no validation
    carve-out), so the complement reproduces the RF's training rows.
    time: train_idx only - the time RF is fit on the training window and
    never sees the valid carve-out, so the XGB train+valid pool is a
    different (larger) pool and must not be labelled same-pool.
    """
    if split == "time":
        return np.asarray(sp["train_idx"])
    test = np.asarray(sp["test_idx"] if rf_test_idx is None else rf_test_idx)
    return np.setdiff1d(np.arange(n_total), test)


def load_rf_control(tag, xgb_pools, rf_pools, test_sizes, n_total,
                    xgb_tests=None, rf_tests=None, rf_sources=None):
    """Verify both training and test membership against independent RF artifacts.

    Missing test indices or a size-only training match cannot establish the
    evaluation contract. Such controls retain their score but fail same_pool.
    ``test_sizes`` and ``n_total`` remain accepted for historical callers.
    """
    mj = json.load(open(os.path.join(BASE, "results", f"metrics_{tag}.json")))
    out = {}
    for split in sorted(xgb_pools):
        if split not in mj:
            raise SystemExit(f"results/metrics_{tag}.json has no '{split}' entry")
        xgb_idx = np.asarray(xgb_pools[split])
        rf_idx = None if rf_pools is None else rf_pools.get(split)
        xgb_fp, rf_fp = pool_fingerprint(xgb_idx), pool_fingerprint(rf_idx)
        xgb_test_fp = pool_fingerprint((xgb_tests or {}).get(split))
        rf_test_fp = pool_fingerprint((rf_tests or {}).get(split))
        train_match = rf_fp is not None and xgb_fp == rf_fp
        test_match = rf_test_fp is not None and xgb_test_fp == rf_test_fp
        same_pool = train_match and test_match
        source = (rf_sources or {}).get(split, "unrecorded RF pool source")
        if rf_fp is None:
            print(f"WARNING: [{tag}/{split}] RF pool fingerprint unavailable; "
                  "row counts cannot verify same-pool")
        verdict = ("train and test fingerprints match" if same_pool else
                   "NOT same-pool: training/test fingerprint mismatch or unavailable")
        out[split] = {"r2_mean": float(mj[split]["r2"]), "r2_std": None,
                      "same_pool": same_pool, "same_training_pool": train_match,
                      "same_test_set": test_match,
                      "xgb_n_train": int(len(xgb_idx)),
                      "xgb_pool_sha256": xgb_fp, "rf_pool_sha256": rf_fp,
                      "xgb_test_sha256": xgb_test_fp, "rf_test_sha256": rf_test_fp,
                      "rf_pool_source": source,
                      "source": f"headline metrics json ({verdict}); {source}"}
    return out


def fit_pool(sp):
    """XGBoost training rows: train + valid = every non-test molecule.

    Matches the headline RF's training set exactly (it trains on the
    complement of the test set and never splits off a validation set);
    train_idx alone would be the GNN's 10%-carved pool instead.
    """
    parts = [sp["train_idx"]]
    if "valid_idx" in sp.files:
        parts.append(sp["valid_idx"])
    return np.concatenate(parts)


def make_figure(csv_path, fig_path, tag, control):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    splits = sorted(df["split"].unique())
    x = np.arange(len(splits))
    w = 0.35
    xgb_means = [df[df["split"] == s]["r2"].mean() for s in splits]
    # ddof=0: population std, the repo-wide convention (JSON summaries use
    # np.std); pandas' default ddof=1 would not match the committed numbers
    xgb_stds = [df[df["split"] == s]["r2"].std(ddof=0) for s in splits]
    rf_means = [control[s]["r2_mean"] for s in splits]
    rf_stds = [control[s]["r2_std"] or 0.0 for s in splits]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, xgb_means, w, yerr=xgb_stds, capsize=4,
           label="XGBoost", color="tab:orange")
    ax.bar(x + w / 2, rf_means, w, yerr=rf_stds, capsize=4,
           label="RF (control)", color="tab:blue")
    ax.set_xticks(x, splits)
    ax.set_ylabel("test R2 (3 seeds, mean +/- std)")
    ax.set_title(f"XGBoost vs RF - {tag}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out, f"xgboost_{args.tag}.csv")
    fig_path = os.path.join(args.fig_dir, f"xgboost_{args.tag}.png")
    sum_path = os.path.join(args.out, f"summary_{args.tag}.json")

    try:
        import xgboost as xgb
    except Exception as e:  # ImportError, or libomp load failure on macOS
        print(f"ERROR: xgboost unavailable: {e}\n"
              "On macOS: brew install libomp && pip install xgboost")
        raise SystemExit(2)

    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{args.tag}_fingerprints.npz"))
    X, y = fp["X"], fp["y"]
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)

    # load every split up front so the RF control can be pool-checked
    # against the RF's OWN test set (rf_preds npz), not just size
    fit_idx, test_idx, rf_idx, rf_tests, rf_sources = {}, {}, {}, {}, {}
    for split in splits:
        sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                                  f"{args.tag}_{split}.npz"))
        fit_idx[split] = fit_pool(sp)
        test_idx[split] = sp["test_idx"]
        rf_test = (None if split == "time"
                   else load_rf_test_idx(args.tag, split))
        if split != "time" and rf_test is None:
            print(f"WARNING: [{args.tag}/{split}] results/rf_preds_"
                  f"{args.tag}_{split}.npz not found - RF pool rebuilt from "
                  "the split file (self-referential: it cannot disagree "
                  "with the XGB pool)")
        rf_tests[split] = rf_test
        rf_sources[split] = (f"results/rf_preds_{args.tag}_{split}.npz:test_idx"
                             if rf_test is not None else "split-file fallback (unverified)")
        rf_idx[split] = rf_train_pool(split, sp, len(X), rf_test_idx=rf_test)

    control = load_rf_control(
        args.tag,
        fit_idx,
        rf_idx,
        {s: len(test_idx[s]) for s in splits},
        len(X), xgb_tests=test_idx, rf_tests=rf_tests, rf_sources=rf_sources,
    )
    print(f"[{args.tag}] RF control: "
          + json.dumps({k: v["r2_mean"] for k, v in control.items()}))
    for split, c in control.items():
        print(f"[{args.tag}/{split}] same_pool={c['same_pool']} "
              f"xgb_sha={(c['xgb_pool_sha256'] or 'none')[:12]} "
              f"rf_sha={(c['rf_pool_sha256'] or 'none')[:12]}")
        if not c["same_pool"]:
            print(f"WARNING: [{args.tag}/{split}] control pool mismatch: "
                  f"{c['source']}")

    rows = []
    csv_f = open(csv_path, "w")
    csv_f.write(HEADER + "\n")

    def emit(row):
        rows.append(row)
        csv_f.write(",".join(str(row[k]) for k in HEADER.split(",")) + "\n")
        csv_f.flush()

    print(f"[{args.tag}] seeds={seeds} splits={splits} "
          f"n_est={args.n_estimators} lr={args.lr} depth={args.max_depth} "
          f"device=cpu/hist (xgboost {xgb.__version__})")
    for split in splits:
        tr, te = fit_idx[split], test_idx[split]
        print(f"[{args.tag}/{split}] fit_pool={len(tr)} test={len(te)}")
        for seed in seeds:
            t0 = time.time()
            model = xgb.XGBRegressor(
                n_estimators=args.n_estimators, learning_rate=args.lr,
                max_depth=args.max_depth, random_state=seed,
                tree_method="hist", n_jobs=args.n_jobs)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            np.savez_compressed(os.path.join(args.out,
                                f"preds_{args.tag}_{split}_seed{seed}.npz"),
                                test_idx=te, y_true=y[te], y_pred=pred,
                                seed=seed, xgboost_version=xgb.__version__)
            m = {"r2": float(r2_score(y[te], pred)),
                 "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
                 "mae": float(mean_absolute_error(y[te], pred))}
            dt = time.time() - t0
            print(f"  seed={seed} R2={m['r2']:.4f} RMSE={m['rmse']:.4f} ({dt:.0f}s)")
            emit({"split": split, "seed": seed, **m,
                  "train_seconds": round(dt, 1)})
    csv_f.close()

    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r["r2"])
    summary = {f"{s}": {"r2_mean": float(np.mean(v)),
                        "r2_std": float(np.std(v)),
                        "rmse_mean": float(np.mean([r["rmse"] for r in rows
                                                    if r["split"] == s])),
                        "delta_vs_rf_control": (float(np.mean(v))
                                                - control[s]["r2_mean"]),
                        "runs": len(v)}
               for s, v in sorted(by_split.items())}
    json.dump({"tag": args.tag, "seeds": seeds,
               "xgb_params": {"n_estimators": args.n_estimators,
                              "learning_rate": args.lr,
                              "max_depth": args.max_depth,
                              "tree_method": "hist", "device": "cpu"},
               "xgboost_version": xgb.__version__,
               "fit_pool": {s: int(len(fit_idx[s])) for s in splits},
               "n_test": {s: int(len(test_idx[s])) for s in splits},
               "rf_control": control, "by_split": summary},
              open(sum_path, "w"), indent=2)
    print(f"summary -> {os.path.relpath(sum_path, BASE)}")

    make_figure(csv_path, fig_path, args.tag, control)
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")
    print(f"csv     -> {os.path.relpath(csv_path, BASE)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
