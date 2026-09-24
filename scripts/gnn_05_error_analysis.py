"""GNN step 5: error analysis - do RF and GIN fail on the SAME molecules?

Questions answered (per split):
  1. If we rank test molecules by absolute error, are the two models'
     worst-K sets the same molecules?
        - large overlap  -> the molecules themselves are hard/noisy
                            (data problem, not a model problem)
        - small overlap  -> the models fail on different chemotypes
                            (complementary failure modes)
  2. Is |error| correlated across models per molecule (Spearman)?
  3. Which scaffolds are hard for BOTH models vs hard for only one?

Inputs : results/rf_preds_{TAG}_{split}.npz, results/gnn_preds_{TAG}_{split}.npz
         (produced by gnn_04_compare.py / gnn_03_train_gin.py)
Outputs: results/error_analysis_{TAG}_{split}.csv  (per-molecule errors)
         figures/error_analysis_{TAG}_{split}.png   (|err| RF vs |err| GIN,
                                                      coloured by scaffold)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except ImportError:  # repo has no scipy dependency; implement a tiny fallback
    HAVE_SCIPY = False

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")
TOP_K = 50
N_COLOR_SCAFFOLDS = 8


def _spearman(a, b):
    if HAVE_SCIPY:
        return float(spearmanr(a, b).statistic)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load(split):
    d = np.load(os.path.join(BASE, "data", "processed", "splits", f"{TAG}_{split}.npz"))
    scaf = d["scaffold_smiles"][d["test_idx"]]
    rf = np.load(os.path.join(BASE, "results", f"rf_preds_{TAG}_{split}.npz"))
    gin = np.load(os.path.join(BASE, "results", f"gnn_preds_{TAG}_{split}.npz"))
    assert np.array_equal(rf["test_idx"], gin["test_idx"])
    smiles = d["smiles"][d["test_idx"]]
    return smiles, scaf, rf["y_true"], rf["y_pred"], gin["y_pred"]


def analyse(split):
    smiles, scaf, y, rf_pred, gin_pred = load(split)
    err_rf = np.abs(rf_pred - y)
    err_gin = np.abs(gin_pred - y)

    # --- 1 & 2: overlap of worst-K and per-molecule error correlation ---
    worst_rf = set(np.argsort(-err_rf)[:TOP_K].tolist())
    worst_gin = set(np.argsort(-err_gin)[:TOP_K].tolist())
    overlap = len(worst_rf & worst_gin)
    # random expectation: K^2 / N
    expected = TOP_K ** 2 / len(y)
    rho = _spearman(err_rf, err_gin)
    print(f"\n=== {TAG} / {split} split ===")
    print(f"  worst-{TOP_K} overlap: {overlap} molecules "
          f"(random expectation ~{expected:.0f}, max {TOP_K})")
    print(f"  Spearman corr of |error|: {rho:.3f}")

    # --- 3: per-scaffold difficulty ---
    rows = []
    for i in range(len(y)):
        rows.append({
            "smiles": smiles[i], "scaffold": scaf[i],
            "y_true": float(y[i]),
            "err_rf": float(err_rf[i]), "err_gin": float(err_gin[i]),
            "worst_rf": int(i in worst_rf), "worst_gin": int(i in worst_gin),
        })
    import csv
    csv_path = os.path.join(BASE, "results", f"error_analysis_{TAG}_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    uniq = np.unique(scaf)
    scaf_rows = []
    for s in uniq:
        m = scaf == s
        if m.sum() < 5:  # skip singletons for the printed summary
            continue
        scaf_rows.append((s, int(m.sum()), float(err_rf[m].mean()), float(err_gin[m].mean())))
    scaf_rows.sort(key=lambda r: -(r[2] + r[3]))
    print(f"  scaffolds with >=5 test mols, hardest by combined mean |err|:")
    print(f"    {'scaffold':<28} {'n':>4} {'mean|err| RF':>13} {'mean|err| GIN':>14}")
    for s, n, er, eg in scaf_rows[:10]:
        print(f"    {s:<28.26} {n:>4} {er:>13.3f} {eg:>14.3f}")

    # --- figure: |err| vs |err|, coloured by scaffold ---
    uniq, counts = np.unique(scaf, return_counts=True)
    order = np.argsort(-counts)
    cmap = plt.get_cmap("tab10")
    color_of = {uniq[i]: (cmap(r % 10) if r < N_COLOR_SCAFFOLDS else (0.75, 0.75, 0.75, 0.5))
                for r, i in enumerate(order)}
    colors = np.array([color_of[s] for s in scaf])

    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(err_rf, err_gin, s=16, c=colors, edgecolors="none", alpha=0.85)
    lim = max(err_rf.max(), err_gin.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="same error")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("|error| RF (Morgan FP)")
    ax.set_ylabel("|error| GIN (molecular graph)")
    ax.set_title(f"{TAG.upper()} {split} test: {overlap}/{TOP_K} shared worst molecules, "
                 f"Spearman $\\rho$={rho:.2f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(BASE, "figures", f"error_analysis_{TAG}_{split}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved -> {os.path.relpath(csv_path, BASE)}, {os.path.relpath(out, BASE)}")


def main():
    for split in ("random", "scaffold"):
        analyse(split)


if __name__ == "__main__":
    main()
