"""XGBoost baseline: is the traditional-ML ceiling higher than RF suggests?

Identical evaluation contract as the headline RF: same persisted
2048-bit Morgan r=2 fingerprints (the headline featurization), same
persisted test set, and the same training molecules - the full non-test
pool (train + valid; the RF cross-validates, so it never carved out a
validation set, while the GNN's 10% valid split is a subset of the RF's
training rows). Fixed hyperparameters, 3 seeds x 2 splits. The RF
control is read from results/metrics_{TAG}.json (the headline
full-pool RF); learning-curve RFs are never used as controls - they fit
at n_train far below the pool, so they are not same-pool either.

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
    p.add_argument("--splits", default="random,scaffold")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--out", default=os.path.join(BASE, "results", "xgboost"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "xgboost"))
    return p.parse_args()


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


def rf_train_pool(split, sp, n_total):
    """Rows the headline RF trains on for `split`, rebuilt from the split npz.

    random/scaffold: the complement of the test set - 04 fits everything
    outside the test fold (5-fold CV, no validation carve-out), so
    train+valid from the split file reproduces it row for row.
    time: train_idx only - the time RF is fit on the training window and
    never sees the valid carve-out, so the XGB train+valid pool is a
    different (larger) pool and must not be labelled same-pool.
    """
    if split == "time":
        return np.asarray(sp["train_idx"])
    return np.setdiff1d(np.arange(n_total), np.asarray(sp["test_idx"]))


def load_rf_control(tag, xgb_pools, rf_pools, test_sizes, n_total):
    """Headline RF control per split + a fingerprint-based same_pool verdict.

    Learning-curve RFs are deliberately never used as a control: they fit
    at n_train <= n_max rows - often far below the pool - so they are not
    the headline RF even when a large n_max looks comparable.

    same_pool is True only when the XGB fit pool and the RF's training
    pool contain exactly the same row indices: both are hashed with
    pool_fingerprint and the verdict is hash equality, so a pool with the
    right row count but different membership is flagged as not same-pool.
    When a fingerprint cannot be computed (rf_pools[split] is None), the
    check falls back to comparing row counts and prints a warning.

    Args:
        xgb_pools: {split: index array XGBoost fits on (train + valid)}
        rf_pools: {split: index array the headline RF trains on, or None
            when it cannot be rebuilt from the split npz}
        test_sizes: {split: test-set size}, used only by the size fallback
        n_total: total fingerprint rows
    """
    mj = json.load(open(os.path.join(BASE, "results", f"metrics_{tag}.json")))
    out = {}
    for split in sorted(xgb_pools):
        if split not in mj:
            raise SystemExit(f"results/metrics_{tag}.json has no '{split}' entry")
        xgb_idx = np.asarray(xgb_pools[split])
        rf_idx = None if rf_pools is None else rf_pools.get(split)
        xgb_fp = pool_fingerprint(xgb_idx)
        rf_fp = pool_fingerprint(rf_idx)
        if rf_fp is not None:
            same_pool = xgb_fp == rf_fp
            if same_pool:
                source = "headline metrics json (pool fingerprints match)"
            else:
                source = (f"pool sha256 mismatch: XGB {len(xgb_idx)} rows "
                          f"({xgb_fp[:12]}...) vs RF {len(rf_idx)} rows "
                          f"({rf_fp[:12]}...) - NOT same-pool")
                if split == "time":
                    source += (" (time RF trains train_idx only; XGB fits "
                               "train+valid)")
        else:
            rf_n = n_total - test_sizes[split]
            print(f"WARNING: [{tag}/{split}] RF pool fingerprint unavailable - "
                  f"falling back to row-count comparison (XGB {len(xgb_idx)} vs "
                  f"RF-expected {rf_n})")
            same_pool = int(len(xgb_idx)) == rf_n
            source = (f"headline metrics json (size-only fallback: RF trains "
                      f"{rf_n} rows vs XGB {len(xgb_idx)} rows"
                      f"{'' if same_pool else ' - NOT same-pool'})")
        out[split] = {"r2_mean": float(mj[split]["r2"]),
                      "r2_std": None,
                      "same_pool": same_pool,
                      "xgb_n_train": int(len(xgb_idx)),
                      "xgb_pool_sha256": xgb_fp,
                      "rf_pool_sha256": rf_fp,
                      "source": source}
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
    # against a rebuilt RF training pool (composition, not just size)
    fit_idx, test_idx, rf_idx = {}, {}, {}
    for split in splits:
        sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                                  f"{args.tag}_{split}.npz"))
        fit_idx[split] = fit_pool(sp)
        test_idx[split] = sp["test_idx"]
        rf_idx[split] = rf_train_pool(split, sp, len(X))

    control = load_rf_control(
        args.tag,
        fit_idx,
        rf_idx,
        {s: len(test_idx[s]) for s in splits},
        len(X),
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
                tree_method="hist", n_jobs=-1)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
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
