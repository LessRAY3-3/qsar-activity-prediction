"""Scaling / crossover analysis: does GIN overtake RF at larger n?

Reads results/learning_curve/learning_curve_{TAG}.csv (columns
split,n_train,seed,model,r2,...; model in {rf, gin}; 3 paired seeds) and,
per (split, model), fits the saturating power law

    R2(n) = R2_inf - a * n^(-b)      (a > 0, b > 0)

to the 3-seed mean curve with scipy.optimize.curve_fit (analytic jacobian).
Bounds are enforced (a, b strictly positive); p0 is multi-start over a
small grid of initial exponents and the solution with the lowest
residual sum of squares is kept, with maxfev large enough to converge
from any start.  Parameters and goodness of fit (R2 and RMSE of the fit
residuals) are printed; fits whose asymptote is not identified (b ~ 0
ridge, or R2_inf > 1) additionally print a note and carry it into the
JSON, since their extrapolation is unreliable.

Crossover estimate: the gap gin(n) - rf(n) between the two fitted mean
curves is scanned on a geometric grid over [max observed n, 10 x max
observed n]; every sign change is refined with scipy.optimize.brentq and
the smallest root is n*.  If GIN is already ahead at the largest
observed n, the scan is repeated below it so a crossover inside the
observed range is still found.  The same fit+root procedure is repeated
for each seed independently (per-seed fit); the distribution of
per-seed n* values gives the CI.  Seeds whose fit fails or whose curves
have no root on the interval are skipped, and the success rate is
reported.

If the interval contains no zero we do not project a crossover: we
report the measured paired gap at the largest observed n together with
the extrapolated gap at 2x and 4x that n, and flag the split
"no crossover projected".

Outputs:
  results/learning_curve/scaling_{TAG}.json  fits, n* + CI or no-crossover
  figures/learning_curve/scaling_{TAG}.png   mean+-std scatter, fits, n*
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS = ("rf", "gin")
COLORS = {"rf": "tab:blue", "gin": "tab:orange"}
B0_GRID = (0.3, 0.8, 1.5)               # multi-start initial exponents
MAXFEV = 20000                          # ample: SSR converges well below this
GRID_MULT = 10.0                        # scan n* up to 10x max observed n
GRID_POINTS = 1000
GAP_MULTS = (2, 4)                      # extrapolation points when no n*


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--csv", default=None,
                   help="input csv (default: results/learning_curve/"
                        "learning_curve_{tag}.csv)")
    p.add_argument("--out", default=os.path.join(BASE, "results", "learning_curve"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "learning_curve"))
    return p.parse_args()


def power_law(n, r2_inf, a, b):
    return r2_inf - a * np.power(n, -b)


def power_law_jac(n, r2_inf, a, b):
    t = np.power(n, -b)
    return np.column_stack([np.ones_like(n), -t, a * t * np.log(n)])


def fit_notes(fit):
    """Flag fits whose extrapolated asymptote is not trustworthy."""
    notes = []
    if fit["b"] < 0.05:
        notes.append("b<0.05: no visible saturation in the observed range, "
                     "R2_inf/a/b lie on a near-degenerate ridge (R2_inf >> 1); "
                     "asymptote not identified")
    if fit["r2_inf"] > 1.0:
        notes.append("R2_inf>1: fitted asymptote exceeds perfect R2")
    return notes


def fit_power_law(n, y):
    """Fit R2_inf - a*n^-b with a>0, b>0; multi-start p0. None on failure."""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(y.max() - y.min())
    best = None
    for b0 in B0_GRID:
        r2_inf0 = float(y.max()) + max(0.02, 0.25 * span)
        a0 = (r2_inf0 - float(y.min())) * float(n.min()) ** b0
        try:
            popt, pcov = curve_fit(
                power_law, n, y, p0=[r2_inf0, a0, b0], jac=power_law_jac,
                bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
                maxfev=MAXFEV)
        except (RuntimeError, ValueError):
            continue
        resid = y - power_law(n, *popt)
        ss_res = float(np.sum(resid ** 2))
        if best is None or ss_res < best["ss_res"]:
            with np.errstate(invalid="ignore"):
                perr = np.sqrt(np.diag(pcov))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            best = {
                "r2_inf": float(popt[0]), "a": float(popt[1]), "b": float(popt[2]),
                "se_r2_inf": float(perr[0]), "se_a": float(perr[1]),
                "se_b": float(perr[2]),
                "fit_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "fit_rmse": float(np.sqrt(ss_res / len(y))),
                "ss_res": ss_res,
            }
    return best


def json_safe(obj):
    """Replace non-finite floats with None so the payload is strict JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def gap_fn(n, rf_fit, gin_fit):
    return (power_law(n, gin_fit["r2_inf"], gin_fit["a"], gin_fit["b"])
            - power_law(n, rf_fit["r2_inf"], rf_fit["a"], rf_fit["b"]))


def find_crossover(rf_fit, gin_fit, n_lo, n_hi):
    """Smallest zero of gin(n)-rf(n) on [n_lo, n_hi], or None."""
    grid = np.geomspace(n_lo, n_hi, GRID_POINTS)
    gap = gap_fn(grid, rf_fit, gin_fit)
    for i in range(len(grid) - 1):
        if gap[i] == 0.0:
            return float(grid[i])
        if gap[i] * gap[i + 1] < 0.0:
            root = brentq(lambda n: gap_fn(n, rf_fit, gin_fit),
                          grid[i], grid[i + 1], xtol=1e-6, rtol=1e-10)
            return float(root)
    return None


def fit_label(fit):
    return (f"R2_inf={fit['r2_inf']:+.4f}  a={fit['a']:.4g}  b={fit['b']:.4f}  "
            f"(fit R2={fit['fit_r2']:.4f}, RMSE={fit['fit_rmse']:.4f})")


def analyse_split(df, split, seeds, tag):
    """Fit mean curves, estimate n* (mean + per-seed CI), gaps. Returns dict."""
    sub = df[df["split"] == split]
    n_max = int(sub["n_train"].max())
    n_lo, n_hi = n_max, GRID_MULT * n_max

    means = {}
    fits = {}
    for model in MODELS:
        g = (sub[sub["model"] == model].groupby("n_train")["r2"]
             .agg(["mean", "std"]).sort_index())
        means[model] = g
        fits[model] = fit_power_law(g.index.values, g["mean"].values)
        if fits[model] is None:
            raise RuntimeError(f"[{tag}/{split}] {model} mean-curve fit failed")
        print(f"[{tag}/{split}] {model.upper():3s} fit: {fit_label(fits[model])}")
        for note in fit_notes(fits[model]):
            print(f"[{tag}/{split}] {model.upper():3s} note: {note}")

    # measured paired gap at the largest observed n
    piv = sub.pivot_table(index=["n_train", "seed"], columns="model", values="r2")
    tail = piv.xs(n_max, level="n_train").dropna()
    gaps = tail["gin"] - tail["rf"]
    measured = {"n": n_max, "gap_mean": float(gaps.mean()),
                "gap_std": float(gaps.std(ddof=0)), "n_pairs": int(len(gaps))}
    print(f"[{tag}/{split}] measured gap at n={n_max} (paired, {len(gaps)} seeds): "
          f"{measured['gap_mean']:+.4f} +/- {measured['gap_std']:.4f}")

    # extrapolated gap from the fitted mean curves
    extrap = {f"{m}x": float(gap_fn(m * n_max, fits["rf"], fits["gin"]))
              for m in GAP_MULTS}
    print(f"[{tag}/{split}] extrapolated gap: "
          + "  ".join(f"{m}x({m * n_max})={extrap[f'{m}x']:+.4f}"
                      for m in GAP_MULTS))

    # mean-curve crossover: primary scan on [max observed n, 10x max n]
    n_star = find_crossover(fits["rf"], fits["gin"], n_lo, n_hi)
    scan_lo, scan_hi, star_note = n_lo, n_hi, None
    if n_star is None and gap_fn(n_max, fits["rf"], fits["gin"]) > 0.0:
        # GIN already ahead at the largest observed n -> look below it too
        n_min = int(sub["n_train"].min())
        n_star = find_crossover(fits["rf"], fits["gin"], n_min, n_max)
        if n_star is not None:
            scan_lo, scan_hi = n_min, float(n_max)
            star_note = "GIN already ahead at max observed n; " \
                        "root lies within the observed range"
        else:
            star_note = "GIN already ahead across the whole observed range"

    # per-seed fits -> distribution of per-seed n* (same interval as the mean)
    per_seed_n, skipped = [], []
    for seed in seeds:
        ss = sub[sub["seed"] == seed]
        s_fits, failed = {}, None
        for model in MODELS:
            m = ss[ss["model"] == model].sort_values("n_train")
            s_fits[model] = fit_power_law(m["n_train"].values, m["r2"].values)
            if s_fits[model] is None:
                failed = f"fit failure ({model})"
                break
        if failed:
            skipped.append({"seed": int(seed), "reason": failed})
            continue
        ns = find_crossover(s_fits["rf"], s_fits["gin"], scan_lo, scan_hi)
        if ns is None:
            skipped.append({"seed": int(seed),
                            "reason": f"no root in [{scan_lo:g}, {scan_hi:.0f}]"})
        else:
            per_seed_n.append(ns)

    total = len(seeds)
    per_seed = {
        "total": total, "success": len(per_seed_n),
        "success_rate": len(per_seed_n) / total if total else 0.0,
        "n_star_values": [round(v, 1) for v in per_seed_n],
        "skipped": skipped,
        "n_star_mean": float(np.mean(per_seed_n)) if per_seed_n else None,
        "n_star_std": float(np.std(per_seed_n, ddof=1)) if len(per_seed_n) > 1 else None,
        "ci95_percentile": ([float(v) for v in np.percentile(per_seed_n, [2.5, 97.5])]
                            if len(per_seed_n) >= 2 else None),
    }
    if len(per_seed_n) >= 2:
        ci = per_seed["ci95_percentile"]
        print(f"[{tag}/{split}] per-seed n*: {per_seed['n_star_values']} -> "
              f"mean {per_seed['n_star_mean']:.0f} +/- {per_seed['n_star_std']:.0f}"
              f", 95% CI [{ci[0]:.0f}, {ci[1]:.0f}]  "
              f"({per_seed['success']}/{total} succeeded)")
    elif per_seed_n:
        print(f"[{tag}/{split}] per-seed n*: {per_seed['n_star_values']} -> "
              f"1/{total} succeeded (CI unavailable, needs >= 2 seeds)")
    else:
        reasons = ", ".join(f"seed={s['seed']} {s['reason']}" for s in skipped)
        print(f"[{tag}/{split}] per-seed n*: 0/{total} succeeded (skipped: {reasons})")

    if n_star is None:
        extra = f" -- {star_note}" if star_note else ""
        print(f"[{tag}/{split}] crossover: no crossover projected in "
              f"[{scan_lo:g}, {scan_hi:.0f}]{extra}")
    else:
        loc = f" -- {star_note}" if star_note else ""
        print(f"[{tag}/{split}] crossover n* = {n_star:.0f} "
              f"(scan interval [{scan_lo:g}, {scan_hi:.0f}]){loc}")

    return {
        "n_observed": [int(v) for v in means["rf"].index],
        "n_max_observed": n_max,
        "seeds": [int(s) for s in seeds],
        "fits": {m: {**fits[m], "notes": fit_notes(fits[m])} for m in MODELS},
        "gap_at_n_max": measured,
        "extrapolated_gap": extrap,
        "crossover": {
            "scan_interval": [scan_lo, scan_hi],
            "n_star": n_star,
            "no_crossover_projected": n_star is None,
            "note": star_note,
            "per_seed": per_seed,
        },
        "_means": means,          # kept for the figure, dropped before json
        "_fits": fits,
    }


def make_figure(tag, results, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(results)
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 4.8),
                             sharey=True)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        d = results[split]
        n_max, n_star = d["n_max_observed"], d["crossover"]["n_star"]
        x_max = n_star * 1.15 if n_star is not None else GAP_MULTS[-1] * n_max * 1.15
        for model in MODELS:
            g = d["_means"][model]
            ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt="o", ms=4.5,
                        color=COLORS[model], capsize=2,
                        label=f"{model.upper()} mean$\\pm$std")
            fit = d["_fits"][model]
            n_line = np.geomspace(g.index.min(), x_max, 300)
            ax.plot(n_line, power_law(n_line, fit["r2_inf"], fit["a"], fit["b"]),
                    color=COLORS[model], lw=1.6, ls="-",
                    label=f"{model.upper()} fit")
        if n_star is not None:
            ax.axvline(n_star, ls="--", c="k", lw=1.2)
            info = f"n* $\\approx$ {n_star:.0f}"
        else:
            info = ("no crossover projected\n"
                    f"gap@{n_max} = {d['gap_at_n_max']['gap_mean']:+.4f}\n"
                    + "  ".join(f"{m}x: {d['extrapolated_gap'][f'{m}x']:+.4f}"
                                for m in GAP_MULTS))
        ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6",
                          alpha=0.9))
        ax.set_xlim(d["_means"]["rf"].index.min() * 0.9, x_max)
        ax.set_title(f"{split} split")
        ax.set_xlabel("training molecules")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("test R2")
    fig.suptitle(f"Scaling / crossover - {tag}  "
                 r"(fit: $R^2(n)=R^2_\infty - a\,n^{-b}$)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    tag = args.tag
    csv_path = args.csv or os.path.join(args.out, f"learning_curve_{tag}.csv")
    json_path = os.path.join(args.out, f"scaling_{tag}.json")
    fig_path = os.path.join(args.fig_dir, f"scaling_{tag}.png")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    seeds = sorted(int(s) for s in df["seed"].unique())
    splits = sorted(df["split"].unique())
    print(f"[{tag}] csv={os.path.relpath(csv_path, BASE)} "
          f"rows={len(df)} splits={splits} seeds={seeds}")

    results, plain = {}, {}
    for split in splits:
        results[split] = analyse_split(df, split, seeds, tag)
        plain[split] = {k: v for k, v in results[split].items()
                        if not k.startswith("_")}

    payload = {
        "tag": tag,
        "csv": os.path.relpath(csv_path, BASE),
        "model": "R2(n) = R2_inf - a * n^(-b), a>0, b>0 (curve_fit, multi-start p0)",
        "crossover_scan": f"[max observed n, {GRID_MULT:g}x max observed n], brentq",
        "splits": plain,
    }
    json.dump(json_safe(payload), open(json_path, "w"), indent=2)
    make_figure(tag, results, fig_path)
    print(f"json    -> {os.path.relpath(json_path, BASE)}")
    print(f"figure  -> {os.path.relpath(fig_path, BASE)}")


if __name__ == "__main__":
    main()
