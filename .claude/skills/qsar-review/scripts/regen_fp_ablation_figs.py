# 2026-10-11 数据刷新战役用：fp ablation 缺失补图（复用 experiments/fingerprint_ablation.py 绘图路径）；原 .scratch/regen_fp_ablation_figs.py。
"""Regenerate missing figures/fp_ablation/fp_ablation_{tag}.png from
results/fp_ablation/fp_ablation_{tag}.csv, reusing the exact plotting code
path (experiments/fingerprint_ablation.py:make_figure) so style/dpi match
the existing figures.

Import-safe: module import only defines helpers (the 2026-10-11 review found
the original version writing into figures/ at import time). All reads/writes
happen in main(), guarded by ``if __name__ == "__main__"``.

Flags:
  --out DIR    figure output directory (default figures/fp_ablation);
               results/, data/ and models/ are refused
  --dry-run    list what would be regenerated, write nothing
  --force      regenerate figures that already exist (default: skip them)

Usage:
    python .claude/skills/qsar-review/scripts/regen_fp_ablation_figs.py --dry-run
    python .claude/skills/qsar-review/scripts/regen_fp_ablation_figs.py --force
"""
import argparse
import importlib.util
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES_DIR = os.path.join(BASE, "results", "fp_ablation")
DEFAULT_FIG_DIR = os.path.join(BASE, "figures", "fp_ablation")
PROTECTED_DIRS = tuple(
    os.path.join(BASE, d) for d in ("results", "data", "models")
)


def is_protected(path):
    """True when path is inside results/, data/ or models/."""
    return any(path == d or path.startswith(d + os.sep) for d in PROTECTED_DIRS)


def load_plot_module():
    """Import experiments/fingerprint_ablation.py (guarded; no side effects)."""
    spec = importlib.util.spec_from_file_location(
        "fingerprint_ablation",
        os.path.join(BASE, "experiments", "fingerprint_ablation.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fingerprint_ablation"] = mod
    spec.loader.exec_module(mod)
    return mod


def list_tags():
    return sorted(
        f[len("summary_"):-len(".json")]
        for f in os.listdir(RES_DIR)
        if f.startswith("summary_") and f.endswith(".json")
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_FIG_DIR,
                   help="figure output directory (default: figures/fp_ablation)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the regeneration plan; write nothing")
    p.add_argument("--force", action="store_true",
                   help="also regenerate figures that already exist")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fig_dir = os.path.abspath(args.out)
    if is_protected(fig_dir):
        raise SystemExit(f"--out {fig_dir} points into a protected directory; refusing")

    tags = list_tags()
    plan, missing_csv = [], []
    for tag in tags:
        png = os.path.join(fig_dir, f"fp_ablation_{tag}.png")
        csv = os.path.join(RES_DIR, f"fp_ablation_{tag}.csv")
        if not os.path.exists(csv):
            missing_csv.append(tag)
            continue
        if os.path.exists(png) and not args.force:
            print(f"[skip] {tag}: figure exists (use --force to refresh)")
            continue
        plan.append(tag)

    if missing_csv:
        print(f"[MISSING CSV] {missing_csv}")
    if args.dry_run:
        print(f"dry-run: would regenerate {len(plan)} figure(s) into {fig_dir}: {plan}")
        return 0

    mod = load_plot_module()
    os.makedirs(fig_dir, exist_ok=True)
    for tag in plan:
        csv = os.path.join(RES_DIR, f"fp_ablation_{tag}.csv")
        png = os.path.join(fig_dir, f"fp_ablation_{tag}.png")
        mod.make_figure(csv, png, tag)
        print(f"[done] {tag} -> {os.path.relpath(png, BASE)}")
    print(f"tags={len(tags)} regenerated={len(plan)}: {plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
