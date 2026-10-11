# 2026-10-11 数据刷新战役用：README 表格 1-4 独立复算校验（逐格比对 results/ 源文件）；原 .scratch/verify_readme_tables.py。
"""Final-batch spot check: independently recompute README tables 1-4 from
results/ source files and compare cell-by-cell.

Table 1 : eight-target panel   -- metrics_{tag}.json + gnn_metrics_{tag}[_seed1|_seed2].json
                                   + significance/summary.csv (mean3 rows)
Table 2 : time split           -- time_split/summary.csv; delta/CI from
                                   significance/summary_time.csv (mean3 rows);
                                   GIN ens recomputed from gnn_preds_{tag}_time*.npz
Table 3 : XGBoost delta        -- xgboost/summary_{tag}.json r2_mean minus the
                                   refreshed metrics_{tag}.json control
Table 4 : multitask 6 rows     -- multitask/summary.csv (delta vs refreshed RF)
         + fp ablation anchors -- fp_ablation/summary_{tag}.json by_cell
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TAGS = ["a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2"]
TOL = 0.0005001

readme = open(os.path.join(BASE, "README.md"), encoding="utf-8").read()
lines = readme.splitlines()

results = []  # (table, cell_id, displayed, computed, ok)


def add(table, cell, disp, comp, ok=None):
    if isinstance(disp, (int, float)) and isinstance(comp, (int, float)):
        ok = abs(float(disp) - float(comp)) <= TOL
    results.append((table, cell, disp, comp, bool(ok)))


def find_table(header_fragment):
    for i, ln in enumerate(lines):
        if ln.startswith("|") and header_fragment in ln:
            rows = []
            j = i + 2  # skip separator
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            return rows
    raise SystemExit(f"table not found: {header_fragment}")


def num(s):
    """parse '+0.621±0.009' / '−0.089' / '0.710' -> float (strip markdown)."""
    s = s.replace("**", "").replace("−", "-").replace("+", "").strip()
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0))


def split_pm(s):
    s = s.replace("**", "").replace("−", "-")
    a, b = s.split("±")
    return float(a.strip().lstrip("+")), float(b.strip())


def r3(x):
    return round(x + 0.0, 3)


def fmt3(x):
    return f"{x:+.3f}" if False else f"{x:.3f}"


def gin_stats(tag):
    vals = []
    for suf in ("", "_seed1", "_seed2"):
        d = json.load(open(f"{BASE}/results/gnn_metrics_{tag}{suf}.json"))
        assert d["random"]["params"]["model"] == "gin", tag
        vals.append(d)
    out = {}
    for split in ("random", "scaffold"):
        r2 = [v[split]["r2"] for v in vals]
        out[split] = (float(np.mean(r2)), float(np.std(r2, ddof=0)))
    return out


# ---------------------------------------------------------------- table 1
sig_rows = {}
with open(f"{BASE}/results/significance/summary.csv") as f:
    hdr = f.readline().strip().split(",")
    for ln in f:
        r = dict(zip(hdr, ln.strip().split(",")))
        if r["pair"] == "mean3":
            sig_rows[(r["tag"], r["split"])] = r

t1 = find_table("| Tag | Target (ChEMBL id) | Family | n | RF rand | GIN rand |")
assert len(t1) == 8, len(t1)
for row in t1:
    tag = row[0].strip("`")
    assert tag in TAGS, tag
    mj = json.load(open(f"{BASE}/results/metrics_{tag}.json"))
    gs = gin_stats(tag)
    n_total = None
    for ln in open(f"{BASE}/results/time_split/summary.csv"):
        parts = ln.strip().split(",")
        if parts[0] == tag:
            n_total = int(parts[3])
    add("T1", f"{tag}/n", num(row[3].replace(",", "")), n_total)
    for k, split in ((4, "random"), (8, "scaffold")):
        rf = mj[split]["r2"]
        gin_mean, gin_std = gs[split]
        srow = sig_rows[(tag, split)]
        winner = ("RF" if srow["winner"] == "rf"
                  else "GIN" if srow["winner"] == "gin" else "ns")
        if srow["significant"] == "False":
            winner = "ns"
        add("T1", f"{tag}/{split}/RF", num(row[k]), rf)
        d_mean, d_std = split_pm(row[k + 1])
        add("T1", f"{tag}/{split}/GIN_mean", d_mean, gin_mean)
        add("T1", f"{tag}/{split}/GIN_std", d_std, gin_std)
        add("T1", f"{tag}/{split}/delta", num(row[k + 2]), gin_mean - rf)
        disp_sig = row[k + 3].replace("**", "").strip()
        add("T1", f"{tag}/{split}/sig", disp_sig, winner,
            ok=disp_sig == winner)

# ---------------------------------------------------------------- table 2
ts = {}
with open(f"{BASE}/results/time_split/summary.csv") as f:
    hdr = f.readline().strip().split(",")
    for ln in f:
        r = dict(zip(hdr, ln.strip().split(",")))
        ts[r["tag"]] = r

st_rows = {}
with open(f"{BASE}/results/significance/summary_time.csv") as f:
    hdr = f.readline().strip().split(",")
    for ln in f:
        r = dict(zip(hdr, ln.strip().split(",")))
        if r["pair"] == "mean3":
            st_rows[r["tag"]] = r


def gin_ens_time(tag):
    preds = []
    y = None
    for suf in ("", "_seed1", "_seed2"):
        d = np.load(f"{BASE}/results/gnn_preds_{tag}_time{suf}.npz")
        preds.append(d["y_pred"])
        y = d["y_true"]
    ens = np.mean(preds, axis=0)
    ss_res = float(np.sum((y - ens) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


t2 = find_table("| Tag | RF rand | GIN rand | RF time | GIN time | GIN ens |")
assert len(t2) == 8, len(t2)
for row in t2:
    tag = row[0].strip("`")
    s = ts[tag]
    st = st_rows[tag]
    add("T2", f"{tag}/RF_rand", num(row[1]), float(s["rf_random_r2"]))
    add("T2", f"{tag}/GIN_rand_mean", num(row[2]),
        float(s["gin_random_r2_mean"]))
    add("T2", f"{tag}/RF_time", num(row[3]), float(s["rf_time_r2"]))
    d_mean, d_std = split_pm(row[4])
    add("T2", f"{tag}/GIN_time_mean", d_mean, float(s["gin_time_r2_mean"]))
    add("T2", f"{tag}/GIN_time_std", d_std, float(s["gin_time_r2_std"]))
    add("T2", f"{tag}/GIN_ens", num(row[5]), gin_ens_time(tag))
    delta_disp = row[6].replace("**", "")
    add("T2", f"{tag}/delta", num(delta_disp), float(st["d_r2"]))
    # delta consistency: ens - rf_time
    add("T2", f"{tag}/delta_vs_ens-rf", float(st["d_r2"]),
        gin_ens_time(tag) - float(s["rf_time_r2"]))
    bold = "**" in row[6]
    ci_sig = st["significant"] == "True"
    add("T2", f"{tag}/bold==sig", bold, ci_sig, ok=bold == ci_sig)
    m = re.match(r"\[(.+?), (.+?)\]", row[7].replace("**", ""))
    add("T2", f"{tag}/ci_lo", num(m.group(1)), float(st["ci_lo"]))
    add("T2", f"{tag}/ci_hi", num(m.group(2)), float(st["ci_hi"]))

# ---------------------------------------------------------------- table 3
t3 = find_table("| Tag | Δ rand | Δ scaf |")
assert len(t3) == 8, len(t3)
xgb_rows = 0
for row in t3:
    tag = row[0].strip("`")
    sj = json.load(open(f"{BASE}/results/xgboost/summary_{tag}.json"))
    mj = json.load(open(f"{BASE}/results/metrics_{tag}.json"))
    for k, split in ((1, "random"), (2, "scaffold")):
        xgb_rows += 1
        xgb_r2 = sj["by_split"][split]["r2_mean"]
        # control as stored vs refreshed metrics
        ctrl = sj["rf_control"][split]["r2_mean"]
        add("T3", f"{tag}/{split}/ctrl==metrics", ctrl, mj[split]["r2"],
            ok=abs(ctrl - mj[split]["r2"]) <= TOL)
        add("T3", f"{tag}/{split}/delta", num(row[k]), xgb_r2 - mj[split]["r2"])
        add("T3", f"{tag}/{split}/delta_stored", num(row[k]),
            sj["by_split"][split]["delta_vs_rf_control"])

# ---------------------------------------------------------------- table 4a
mt = {}
with open(f"{BASE}/results/multitask/summary.csv") as f:
    hdr = f.readline().strip().split(",")
    for ln in f:
        r = dict(zip(hdr, ln.strip().split(",")))
        mt[(r["task"], r["split"], r["method"])] = r

t4 = find_table("| Task | Split | ST-RF | ST-GIN | pooled | MT-GIN |")
assert len(t4) == 6, len(t4)
for row in t4:
    task, split = row[0].strip("`"), row[1]
    mj = json.load(open(f"{BASE}/results/metrics_{task}.json"))
    st_rf = mj[split]["r2"]
    add("T4mt", f"{task}/{split}/st_rf", num(row[2]), st_rf)
    for k, method in ((3, "st_gin"), (4, "pooled_st"), (5, "mt_gin")):
        rec = mt[(task, split, method)]
        r2 = float(rec["r2_mean"])
        std = float(rec["r2_std"])
        d_mean, d_std = split_pm(row[k].split("(")[0].strip())
        add("T4mt", f"{task}/{split}/{method}/mean", d_mean, r2)
        add("T4mt", f"{task}/{split}/{method}/std", d_std, std)
        m = re.search(r"\((.+?)\)", row[k])
        add("T4mt", f"{task}/{split}/{method}/delta", num(m.group(1)), r2 - st_rf)

# ---------------------------------------------------------------- table 4b
fp = {}
for tag in TAGS:
    fp[tag] = json.load(open(f"{BASE}/results/fp_ablation/summary_{tag}.json"))

t5 = find_table("| Tag | random anchor R² | random Δ range | scaffold anchor R² |")
assert len(t5) == 8, len(t5)
for row in t5:
    tag = row[0].strip("`")
    bc = fp[tag]["by_cell"]
    for k, split in ((1, "random"), (3, "scaffold")):
        anchor = bc[f"{split}/r=2/bits=2048"]["r2_mean"]
        add("T4fp", f"{tag}/{split}/anchor", num(row[k]), anchor)
        deltas = [bc[f"{split}/r={r}/bits={b}"]["delta_vs_anchor"]
                  for (r, b) in ((2, 1024), (3, 1024), (3, 2048))]
        lo, hi = min(deltas), max(deltas)
        parts = [p for p in row[k + 1].replace("**", "").split("…")]
        add("T4fp", f"{tag}/{split}/range_lo", num(parts[0]), lo)
        add("T4fp", f"{tag}/{split}/range_hi", num(parts[1]), hi)

# ---------------------------------------------------------------- report

by_tab = defaultdict(lambda: [0, 0, []])
for table, cell, disp, comp, ok in results:
    by_tab[table][0] += 1
    if not ok:
        by_tab[table][1] += 1
        by_tab[table][2].append((cell, disp, comp))

print(f"total cells: {len(results)}, mismatches: {sum(v[1] for v in by_tab.values())}")
for t in sorted(by_tab):
    n, bad, items = by_tab[t]
    print(f"  {t:6s} cells={n:3d} mismatch={bad}")
    for cell, disp, comp in items:
        print(f"      FAIL {cell}: readme={disp!r} source={comp!r}")
