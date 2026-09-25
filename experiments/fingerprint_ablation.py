"""Fingerprint ablation: how sensitive is the RF conclusion to the Morgan
fingerprint parameters?

Grid: radius {2, 3} x n_bits {1024, 2048}. RF hyperparameters are frozen
to the original grid-search winner and the FULL headline train pool
(train_idx + valid_idx = the 80% the headline RF trained on) is used with
the persisted test set. Seeds vary RF's random_state.

Anchor: radius=2 / 2048 bits must reproduce the headline metrics
(0.747 random / 0.562 scaffold) because it IS the headline featurization,
read from the persisted cache written by scripts/03_featurize.py. The
other three cells featurize on the fly with the identical RDKit call.

Outputs:
  results/fp_ablation/fp_ablation_{TAG}.csv   one row per run
  results/fp_ablation/summary_{TAG}.json
  figures/fp_ablation/fp_ablation_{TAG}.png
Exit code 1 + no figure if the anchor fails (caller must stop and report).
"""
import argparse
import json
import os
import time

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
HEADER = "split,radius,n_bits,seed,r2,rmse,mae,train_seconds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--radii", default="2,3")
    p.add_argument("--bits", default="1024,2048")
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--splits", default="random,scaffold")
    p.add_argument("--anchor-tol", type=float, default=0.01)
    p.add_argument("--out", default=os.path.join(BASE, "results", "fp_ablation"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "fp_ablation"))
    return p.parse_args()


def morgan_matrix(smiles, radius, n_bits):
    """Identical featurization code path as scripts/03_featurize.py."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        X[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
    return X


def make_figure(csv_path, fig_path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    splits = sorted(df["split"].unique())
    configs = sorted({(r, b) for r, b in zip(df["radius"], df["n_bits"])},
                     key=lambda t: (t[0], t[1]))
    labels = [f"r={r}/{b}b" for r, b in configs]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, len(splits), figsize=(4.5 * len(splits) + 2, 4.5),
                             sharey=True)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        means, stds = [], []
        for r, b in configs:
            g = df[(df["split"] == split) & (df["radius"] == r) & (df["n_bits"] == b)]
            means.append(g["r2"].mean())
            stds.append(g["r2"].std())
        ax.bar(x, means, yerr=stds, capsize=4,
               color=["tab:blue" if c == (2, 2048) else "tab:cyan" for c in configs])
        ax.set_xticks(x, labels, rotation=20)
        ax.set_title(f"{split} split")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("test R2 (3 seeds, mean +/- std)")
    fig.suptitle(f"Fingerprint ablation - {tag} (anchor = r=2/2048b)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    radii = [int(r) for r in args.radii.split(",")]
    bits_list = [int(b) for b in args.bits.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    csv_path = os.path.join(args.out, f"fp_ablation_{args.tag}.csv")
    fig_path = os.path.join(args.fig_dir, f"fp_ablation_{args.tag}.png")
    sum_path = os.path.join(args.out, f"summary_{args.tag}.json")

    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{args.tag}_fingerprints.npz"))
    y, smiles = fp["y"], fp["smiles"]
    mj = json.load(open(os.path.join(BASE, "results", f"metrics_{args.tag}.json")))
    bp = mj["best_params_random"]
    rf_params = {k: bp[k] for k in ("n_estimators", "max_depth", "min_samples_split")}
    headline = {k: mj[k]["r2"] for k in ("random", "scaffold")}

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)

    rows = []
    csv_f = open(csv_path, "w")
    csv_f.write(HEADER + "\n")

    def emit(row):
        rows.append(row)
        csv_f.write(",".join(str(row[k]) for k in HEADER.split(",")) + "\n")
        csv_f.flush()

    anchor_fail = []
    print(f"[{args.tag}] radii={radii} bits={bits_list} seeds={seeds}")
    for split in splits:
        sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                                  f"{args.tag}_{split}.npz"))
        pool = np.concatenate([sp["train_idx"], sp["valid_idx"]])
        test_idx = sp["test_idx"]
        print(f"[{args.tag}/{split}] pool={len(pool)} test={len(test_idx)}")
        for radius in radii:
            for n_bits in bits_list:
                t0 = time.time()
                if radius == 2 and n_bits == 2048:
                    X = fp["X"]  # persisted headline featurization
                else:
                    X = morgan_matrix(smiles, radius, n_bits)
                featurize_s = time.time() - t0
                for seed in seeds:
                    t0 = time.time()
                    rf = RandomForestRegressor(random_state=seed, n_jobs=-1,
                                               **rf_params)
                    rf.fit(X[pool], y[pool])
                    pred = rf.predict(X[test_idx])
                    m = {"r2": float(r2_score(y[test_idx], pred)),
                         "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
                         "mae": float(mean_absolute_error(y[test_idx], pred))}
                    dt = time.time() - t0
                    print(f"  r={radius} bits={n_bits} seed={seed} "
                          f"R2={m['r2']:.4f} (fit {dt:.0f}s, feat {featurize_s:.0f}s)")
                    emit({"split": split, "radius": radius, "n_bits": n_bits,
                          "seed": seed, **m, "train_seconds": round(dt, 1)})
                if radius == 2 and n_bits == 2048:
                    cell = [r["r2"] for r in rows
                            if r["split"] == split and r["radius"] == 2
                            and r["n_bits"] == 2048]
                    mean_r2 = float(np.mean(cell))
                    if abs(mean_r2 - headline[split]) > args.anchor_tol:
                        anchor_fail.append(
                            f"{split}: anchor {mean_r2:.4f} vs headline "
                            f"{headline[split]:.4f}")
                        print(f"  ANCHOR FAIL {split}: {mean_r2:.4f} vs "
                              f"{headline[split]:.4f}")
    csv_f.close()

    by_cell = {}
    for r in rows:
        key = f"{r['split']}/r={r['radius']}/bits={r['n_bits']}"
        by_cell.setdefault(key, []).append(r["r2"])
    anchor_means = {k.split("/")[0]: float(np.mean(v))
                    for k, v in by_cell.items()
                    if k.endswith("r=2/bits=2048")}
    summary = {}
    for k, v in sorted(by_cell.items()):
        split = k.split("/")[0]
        summary[k] = {"r2_mean": float(np.mean(v)),
                      "r2_std": float(np.std(v)),
                      "delta_vs_anchor": (float(np.mean(v)) - anchor_means[split])
                      if split in anchor_means else None,
                      "runs": len(v)}
    json.dump({"tag": args.tag, "radii": radii, "bits": bits_list, "seeds": seeds,
               "rf_params": rf_params, "headline_r2": headline,
               "anchor_failures": anchor_fail, "by_cell": summary},
              open(sum_path, "w"), indent=2)
    print(f"summary -> {os.path.relpath(sum_path, BASE)}")

    if anchor_fail:
        print("ANCHOR FAILED -> " + "; ".join(anchor_fail))
        raise SystemExit(1)
    make_figure(csv_path, fig_path, args.tag)
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")
    print(f"csv     -> {os.path.relpath(csv_path, BASE)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
