"""P8 time-split final summary: 3-split table + collapse panel (analysis only).

Question: with all 8 targets now trained on the publication-year split, how
does time-split performance compare with the random/scaffold campaign
numbers, and is the RF-vs-GIN delta on the time split separable from
resampling noise? Nothing is trained here -- every number is read from an
artifact that already exists on disk.

Inputs (per tag):
  data/processed/splits/{tag}_time.npz               sizes + test years
  results/metrics_{tag}.json                         RF "time" r2/rmse
  results/gnn_metrics_{tag}{,_seed1,_seed2}.json     GIN "time" r2 (3 seeds)
  results/comparison_{tag}.csv                       RF/GIN random+scaffold R2
  results/significance/summary_time.csv              paired bootstrap (mean3)

Outputs:
  results/time_split/summary.csv           one row per tag; test_frac is
                                           reported as-is (a2a 0.62 and
                                           mpro 0.51 exceed 0.5 because the
                                           test set cumulates the newest
                                           years to reach ~20%)
  figures/time_split/panel_3splits.png     2 facets (RF | GIN), 3 bars per
                                           tag: random / scaffold / time
Console: headline conclusions.

Caveats documented in the log/report, not in the numbers: egfr_full RF
time R2 = -0.891 and herg -0.034 are genuine temporal-drift signals, not
bugs.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
SPLITS = ("random", "scaffold", "time")
SEED_SUFFIXES = ("", "_seed1", "_seed2")

SUMMARY_COLS = [
    "tag",
    "test_year_start", "test_year_end",
    "n_total", "n_train", "n_valid", "n_test", "test_frac",
    "rf_time_r2", "rf_time_rmse",
    "gin_time_r2_mean", "gin_time_r2_std",
    "d_r2_mean3", "ci_lo", "ci_hi", "significant", "winner", "p_one_sided",
    "rf_random_r2", "gin_random_r2_mean",
    "rf_scaffold_r2", "gin_scaffold_r2_mean",
]


# ---------------------------------------------------------------- loading

def _load_json(name):
    with open(os.path.join(BASE, "results", name)) as fh:
        return json.load(fh)


def time_split_stats(tag):
    """Sizes, test fraction and test-year span of {tag}_time.npz."""
    sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                              f"{tag}_time.npz"))
    n_total = int(len(sp["year"]))
    n_train = int(len(sp["train_idx"]))
    n_valid = int(len(sp["valid_idx"]))
    n_test = int(len(sp["test_idx"]))
    if n_train + n_valid + n_test != n_total:
        raise ValueError(f"{tag}: train/valid/test do not sum to n_total")
    years = np.asarray(sp["year"], dtype=np.float64)[sp["test_idx"]]
    if np.isnan(years).any():
        raise ValueError(f"{tag}: test set has molecules without a year")
    return {
        "n_total": n_total, "n_train": n_train, "n_valid": n_valid,
        "n_test": n_test, "test_frac": n_test / n_total,
        "test_year_start": int(years.min()),
        "test_year_end": int(years.max()),
    }


def gin_time_r2(tag):
    """(mean, population std) of the 3-seed GIN time R2 (gnn_04 convention)."""
    vals = np.array([_load_json(f"gnn_metrics_{tag}{suf}.json")["time"]["r2"]
                     for suf in SEED_SUFFIXES], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def comparison_reference(tag):
    """{(split): (rf_r2, gin_r2_mean)} for the random/scaffold campaign."""
    df = pd.read_csv(os.path.join(BASE, "results", f"comparison_{tag}.csv"))
    out = {}
    for split in ("random", "scaffold"):
        row = df.loc[df["split"] == split]
        if len(row) != 1:
            raise ValueError(f"{tag}: expected one comparison row for {split}")
        out[split] = (float(row["rf_r2"].iloc[0]),
                      float(row["gin_r2_mean"].iloc[0]))
    return out


def time_significance():
    """mean3 rows of the paired bootstrap as a tag-indexed frame."""
    df = pd.read_csv(os.path.join(BASE, "results", "significance",
                                  "summary_time.csv"))
    sub = df.loc[df["pair"] == "mean3"].copy()
    if len(sub) != len(TAGS) or set(sub["tag"]) != set(TAGS):
        raise ValueError("significance/summary_time.csv does not cover all tags")
    return sub.set_index("tag")


# ---------------------------------------------------------------- table

def build_summary():
    sig = time_significance()
    rows = []
    for tag in TAGS:
        stats = time_split_stats(tag)
        rf = _load_json(f"metrics_{tag}.json")["time"]
        gin_mean, gin_std = gin_time_r2(tag)
        ref = comparison_reference(tag)
        s = sig.loc[tag]
        rows.append({
            "tag": tag,
            **stats,
            "rf_time_r2": float(rf["r2"]),
            "rf_time_rmse": float(rf["rmse"]),
            "gin_time_r2_mean": gin_mean,
            "gin_time_r2_std": gin_std,
            "d_r2_mean3": float(s["d_r2"]),
            "ci_lo": float(s["ci_lo"]),
            "ci_hi": float(s["ci_hi"]),
            "significant": bool(s["significant"]),
            "winner": s["winner"],
            "p_one_sided": float(s["p_one_sided"]),
            "rf_random_r2": ref["random"][0],
            "gin_random_r2_mean": ref["random"][1],
            "rf_scaffold_r2": ref["scaffold"][0],
            "gin_scaffold_r2_mean": ref["scaffold"][1],
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


# ---------------------------------------------------------------- figure

def make_panel(df, fig_path):
    """2 facets (RF | GIN), 3 bars per tag: random / scaffold / time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    x = np.arange(len(df))
    width = 0.26
    alpha = {"random": 0.5, "scaffold": 0.75, "time": 1.0}
    panels = (("rf", "RF", "tab:red", "rf_{split}_r2"),
              ("gin", "GIN (3-seed mean)", "tab:blue", "gin_{split}_r2_mean"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    for ax, (_, title, color, pattern) in zip(axes, panels):
        for i, split in enumerate(SPLITS):
            vals = df[pattern.format(split=split)].to_numpy(dtype=np.float64)
            ax.bar(x + (i - 1) * width, vals, width * 0.92,
                   color=color, alpha=alpha[split],
                   edgecolor="black", linewidth=0.4)
            if split == "time":                    # annotate the star split
                for xi, v in zip(x + (i - 1) * width, vals):
                    ax.annotate(f"{v:+.2f}", (xi, v),
                                textcoords="offset points",
                                xytext=(0, 3 if v >= 0 else -10),
                                ha="center", fontsize=6.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["tag"], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("test R²")
    axes[0].set_ylim(min(-1.0, float(df[["rf_time_r2",
                                         "gin_time_r2_mean"]].to_numpy().min())
                         - 0.15),
                     max(1.0, float(df[["rf_random_r2",
                                        "gin_random_r2_mean"]].to_numpy().max())
                         + 0.1))
    handles = [Patch(facecolor="0.55", alpha=alpha[s], label=s)
               for s in SPLITS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, title="bar shade / position (hue = panel model)",
               title_fontsize=8)
    fig.suptitle("Time-split collapse: test R² under random, scaffold and "
                 "time splits (8 targets)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def print_conclusion(df):
    """Console verdict: collapse magnitude, sign flips, CI tally, caveats."""
    print("\n=== P8 time-split final summary (8 targets) ===")
    print("  RF  time R² range : "
          f"[{df['rf_time_r2'].min():+.3f}, {df['rf_time_r2'].max():+.3f}]")
    print("  GIN time R² range : "
          f"[{df['gin_time_r2_mean'].min():+.3f}, "
          f"{df['gin_time_r2_mean'].max():+.3f}]")

    df = df.copy()
    df["rf_drop"] = df["rf_time_r2"] - df["rf_random_r2"]
    df["gin_drop"] = df["gin_time_r2_mean"] - df["gin_random_r2_mean"]
    print(f"  mean ΔR² (time − random): RF {df['rf_drop'].mean():+.3f}, "
          f"GIN {df['gin_drop'].mean():+.3f}")
    rf_neg = df.loc[df["rf_time_r2"] < 0, "tag"].tolist()
    gin_neg = df.loc[df["gin_time_r2_mean"] < 0, "tag"].tolist()
    print(f"  negative time R²: RF {rf_neg or 'none'}; GIN {gin_neg or 'none'}")

    sig = df[df["significant"]]
    rf_wins = sig.loc[sig["winner"] == "rf", "tag"].tolist()
    gin_wins = sig.loc[sig["winner"] == "gin", "tag"].tolist()
    print(f"  paired bootstrap (mean3): {len(sig)}/8 CIs exclude 0 | "
          f"GIN ahead: {gin_wins or 'none'}; RF ahead: {rf_wins or 'none'}")

    big = df.loc[df["test_frac"] > 0.5, ["tag", "test_frac"]]
    for r in big.itertuples():
        print(f"  caveat: {r.tag} test_frac={r.test_frac:.2f} > 0.5 "
              "(newest-year data concentration under the cumulative-20% rule)")
    print("  note: extreme negatives (egfr_full RF -0.891, herg -0.034) are "
          "genuine temporal drift, not bugs")


def main():
    df = build_summary()
    out_csv = os.path.join(BASE, "results", "time_split", "summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    fig_path = os.path.join(BASE, "figures", "time_split", "panel_3splits.png")
    make_panel(df, fig_path)
    print_conclusion(df)
    print(f"\nsummary -> {os.path.relpath(out_csv, BASE)} ({len(df)} rows)")
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")


if __name__ == "__main__":
    main()
