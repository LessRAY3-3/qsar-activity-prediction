# 2026-10-11 数据刷新战役用：fp ablation 缺失补图（复用 experiments/fingerprint_ablation.py 绘图路径）；原 .scratch/regen_fp_ablation_figs.py。
"""One-off: regenerate missing figures/fp_ablation/fp_ablation_{tag}.png
from results/fp_ablation/fp_ablation_{tag}.csv, reusing the exact plotting
code path (experiments/fingerprint_ablation.py:make_figure) so style/dpi
match the existing 6 figures.
"""
import importlib.util
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
spec = importlib.util.spec_from_file_location(
    "fingerprint_ablation",
    os.path.join(BASE, "experiments", "fingerprint_ablation.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules["fingerprint_ablation"] = mod
spec.loader.exec_module(mod)

res_dir = os.path.join(BASE, "results", "fp_ablation")
fig_dir = os.path.join(BASE, "figures", "fp_ablation")
os.makedirs(fig_dir, exist_ok=True)

tags = sorted(
    f[len("summary_"):-len(".json")]
    for f in os.listdir(res_dir)
    if f.startswith("summary_") and f.endswith(".json")
)
missing = []
for tag in tags:
    png = os.path.join(fig_dir, f"fp_ablation_{tag}.png")
    csv = os.path.join(res_dir, f"fp_ablation_{tag}.csv")
    if os.path.exists(png):
        print(f"[skip] {tag}: figure exists")
        continue
    if not os.path.exists(csv):
        print(f"[MISSING CSV] {tag}")
        continue
    missing.append(tag)

for tag in missing:
    csv = os.path.join(res_dir, f"fp_ablation_{tag}.csv")
    png = os.path.join(fig_dir, f"fp_ablation_{tag}.png")
    mod.make_figure(csv, png, tag)
    print(f"[done] {tag} -> {os.path.relpath(png, BASE)}")

print(f"tags={len(tags)} regenerated={len(missing)}: {missing}")
