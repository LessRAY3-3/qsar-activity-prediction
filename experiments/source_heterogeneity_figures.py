"""Plot paired molecule/document intervals on the SAME exclusive-source subset."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "results/source_heterogeneity/report.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.9), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("random", "scaffold", "time")):
        cells = [c for c in data["cells"] if c["split"] == split]
        for y, cell in enumerate(cells):
            for key, shift, color, label in (
                ("molecule_bootstrap_same_subset", -.13, "#2171b5", "Molecule bootstrap"),
                ("cluster_bootstrap", .13, "#d95f02", "Document bootstrap"),
            ):
                entry = cell[key]
                if entry["ci_low"] is None:
                    continue
                point = entry["subset_metrics"]["delta_r2"]
                ax.hlines(y+shift, entry["ci_low"], entry["ci_high"], color=color, lw=1.6)
                ax.plot(point, y+shift, "o", color=color, ms=4, label=label if y == 0 else None)
        ax.set_title(split.capitalize(), fontsize=12)
        ax.axvline(0, color=".4", lw=.8, ls="--")
        ax.set_yticks(range(len(cells)), [c["tag"] for c in cells])
        ax.set_ylim(7.6, -.6)
        ax.grid(axis="x", alpha=.16)
        ax.set_xlabel("GIN mean-prediction R² − RF R²")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, fontsize=9)
    fig.suptitle("Source sensitivity: paired 95% intervals on identical molecule subsets\n"
                 "10,000 draws; single known-document parents only; exploratory, unadjusted", fontsize=12)
    folder = ROOT / "figures/source_heterogeneity"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "paired_source_intervals.png", dpi=200)
    fig.savefig(folder / "paired_source_intervals.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
