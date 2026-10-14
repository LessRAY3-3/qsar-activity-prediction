# 2026-10-11 数据刷新战役用：README 表格逐格独立复算校验（results/ 源文件比对；含 fairness、数据规模、EGFR/BACE 明细、GIN 对照）；原 .scratch/verify_readme_tables.py。
"""Independently recompute README tables from results/ source files and
compare cell-by-cell; exit non-zero when any cell mismatches.

Tables covered:
  T1 : eight-target panel   -- metrics_{tag}.json + gnn_metrics_{tag}[_seed1|_seed2].json
                                   + significance/summary.csv (mean3 rows)
                                   + recomputed seed-deletion conditional column
  T2 : time split           -- time_split/summary.csv; delta/CI from
                                   significance/summary_time.csv (mean3 rows);
                                   GIN ens recomputed from gnn_preds_{tag}_time*.npz
  T3 : XGBoost delta        -- xgboost/summary_{tag}.json r2_mean minus the
                                   refreshed metrics_{tag}.json control
  T4 : multitask 6 rows     -- multitask/summary.csv (delta vs refreshed RF)
         + fp ablation anchors -- fp_ablation/summary_{tag}.json by_cell
  T5 : EGFR detail table    -- metrics_egfr.json (R2/RMSE/MAE per split)
  T6 : BACE detail table    -- metrics_bace.json
  T7 : GIN vs RF (EGFR)     -- metrics_egfr.json + gnn_metrics_egfr{,_seed1,_seed2}.json
  T8 : fairness campaign    -- fairness/summary_{vegfr2,herg,abl1}.json
                                   (RF row cross-checked against metrics_{tag}.json;
                                    GINE rows verified from the same file)
  T10: CPU/CUDA four cells -- accepted budget reports + independent diagnostics aggregation
  T9 : dataset sizes        -- row counts of data/processed/{tag}_pic50_clean.csv

Known limitations (not machine-checked here):
  * The uncertainty-screening section, the GINE panel verdict bullets and the
    AttentiveFP section are prose + figures, not markdown tables - their
    numbers are NOT verified by this script (parse-fragile; verify manually).
  * Narrative sentences outside tables (Δ discussions, ranges, counts in
    text) are not verified either; only table cells are.
  * Tolerance is 0.0005001, i.e. agreement at the README's 3-decimal
    rounding.

Flags:
  --out PATH    write a JSON report (totals, per-table counts, failures);
                results/, data/, models/ and figures/ are refused
  --dry-run     report mismatches but always exit 0 (exploration mode)

Usage:
    python .claude/skills/qsar-review/scripts/verify_readme_tables.py
    python .claude/skills/qsar-review/scripts/verify_readme_tables.py --out /tmp/report.json
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TAGS = ["a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2"]
FAIRNESS_TAGS = ["vegfr2", "herg", "abl1"]
TOL = 0.0005001

lines = []      # README lines, bound in main()
results = []    # (table, cell_id, displayed, computed, ok); reset in main()


def read_csv_rows(path):
    """DictReader over a results CSV - quoted fields (e.g. time-split
    gin_best_epochs "6,3,11") must not shift later columns the way a naive
    str.split(",") does."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_protected(path):
    """True when path is inside results/, figures/, models/ or data/.

    Paths are compared after os.path.realpath + lowercasing so that
    case-insensitive filesystems (macOS) cannot be bypassed with e.g.
    ``--out Results/x``, and ``..``/symlink tricks are resolved first.
    """
    p = os.path.realpath(path).lower()
    protected = (
        os.path.realpath(os.path.join(BASE, d)).lower()
        for d in ("results", "figures", "models", "data"))
    return any(p == d or p.startswith(d + os.sep) for d in protected)


def add(table, cell, disp, comp, ok=None):
    if isinstance(disp, (int, float)) and isinstance(comp, (int, float)):
        ok = abs(float(disp) - float(comp)) <= TOL
    results.append((table, cell, disp, comp, bool(ok)))


def find_table(header_fragment, occurrence=1):
    seen = 0
    for i, ln in enumerate(lines):
        if ln.startswith("|") and header_fragment in ln:
            seen += 1
            if seen != occurrence:
                continue
            rows = []
            j = i + 2  # skip separator
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            return rows
    raise SystemExit(f"table not found: {header_fragment} (occurrence {occurrence})")


def num(s):
    """parse '+0.621±0.009' / '−0.089' / '0.710' -> float (strip markdown)."""
    s = s.replace("**", "").replace("−", "-").replace("+", "").strip()
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0))


def split_pm(s):
    s = s.replace("**", "").replace("−", "-")
    a, b = s.split("±")
    return float(a.strip().lstrip("+")), float(b.strip())


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


def gin_seed42(tag, split):
    d = json.load(open(f"{BASE}/results/gnn_metrics_{tag}.json"))
    return d[split]["r2"], d[split]["rmse"]


# ---------------------------------------------------------------- table 1
def check_t1():
    sig_rows = {}
    for r in read_csv_rows(f"{BASE}/results/significance/summary.csv"):
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
        for r in read_csv_rows(f"{BASE}/results/time_split/summary.csv"):
            if r["tag"] == tag:
                n_total = int(r["n_total"])
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


def check_t2():
    ts = {}
    for r in read_csv_rows(f"{BASE}/results/time_split/summary.csv"):
        ts[r["tag"]] = r

    st_rows = {}
    for r in read_csv_rows(f"{BASE}/results/significance/summary_time.csv"):
        if r["pair"] == "mean3":
            st_rows[r["tag"]] = r

    t2 = find_table("| Tag | RF rand | GIN rand | Constant | RF time | GIN time | GIN ens |")
    assert len(t2) == 8, len(t2)
    for row in t2:
        tag = row[0].strip("`")
        s = ts[tag]
        st = st_rows[tag]
        add("T2", f"{tag}/RF_rand", num(row[1]), float(s["rf_random_r2"]))
        add("T2", f"{tag}/GIN_rand_mean", num(row[2]),
            float(s["gin_random_r2_mean"]))
        add("T2", f"{tag}/constant", num(row[3]), float(s["constant_r2"]))
        add("T2", f"{tag}/RF_time", num(row[4]), float(s["rf_time_r2"]))
        d_mean, d_std = split_pm(row[5])
        add("T2", f"{tag}/GIN_time_mean", d_mean, float(s["gin_time_r2_mean"]))
        add("T2", f"{tag}/GIN_time_std", d_std, float(s["gin_time_r2_std"]))
        add("T2", f"{tag}/GIN_ens", num(row[6]), gin_ens_time(tag))
        delta_disp = row[7].replace("**", "")
        add("T2", f"{tag}/delta", num(delta_disp), float(st["d_r2"]))
        # delta consistency: ens - rf_time
        add("T2", f"{tag}/delta_vs_ens-rf", float(st["d_r2"]),
            gin_ens_time(tag) - float(s["rf_time_r2"]))
        bold = "**" in row[7]
        ci_sig = st["significant"] == "True"
        add("T2", f"{tag}/bold==sig", bold, ci_sig, ok=bold == ci_sig)
        m = re.match(r"\[(.+?), (.+?)\]", row[8].replace("**", ""))
        add("T2", f"{tag}/ci_lo", num(m.group(1)), float(st["ci_lo"]))
        add("T2", f"{tag}/ci_hi", num(m.group(2)), float(st["ci_hi"]))


# ---------------------------------------------------------------- table 3
def check_t3():
    t3 = find_table("| Tag | Δ rand | Δ scaf |")
    assert len(t3) == 8, len(t3)
    for row in t3:
        tag = row[0].strip("`")
        sj = json.load(open(f"{BASE}/results/xgboost/summary_{tag}.json"))
        mj = json.load(open(f"{BASE}/results/metrics_{tag}.json"))
        for k, split in ((1, "random"), (2, "scaffold")):
            xgb_r2 = sj["by_split"][split]["r2_mean"]
            # control as stored vs refreshed metrics
            ctrl = sj["rf_control"][split]["r2_mean"]
            add("T3", f"{tag}/{split}/ctrl==metrics", ctrl, mj[split]["r2"],
                ok=abs(ctrl - mj[split]["r2"]) <= TOL)
            add("T3", f"{tag}/{split}/delta", num(row[k]), xgb_r2 - mj[split]["r2"])
            add("T3", f"{tag}/{split}/delta_stored", num(row[k]),
                sj["by_split"][split]["delta_vs_rf_control"])


# ---------------------------------------------------------------- table 4a
def check_t4():
    mt = {}
    for r in read_csv_rows(f"{BASE}/results/multitask/summary.csv"):
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

    # fp ablation anchors
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


# ---------------------------------------------------------------- T5/T6: per-split detail tables
def check_detail(tag, table_id):
    mj = json.load(open(f"{BASE}/results/metrics_{tag}.json"))
    rows = find_table("| Split | R2 | RMSE | MAE |",
                      occurrence=1 if tag == "egfr" else 2)
    assert len(rows) == 2, len(rows)
    for row in rows:
        split = row[0].strip()
        add(table_id, f"{tag}/{split}/r2", num(row[1]), mj[split]["r2"])
        add(table_id, f"{tag}/{split}/rmse", num(row[2]), mj[split]["rmse"])
        add(table_id, f"{tag}/{split}/mae", num(row[3]), mj[split]["mae"])


# ---------------------------------------------------------------- T7: GIN vs RF (EGFR)
def check_t7():
    rows = find_table("| Split | RF R2 / RMSE | GIN R2 / RMSE (seed 42) | "
                      "GIN R2 mean +- std (3 seeds) |")
    assert len(rows) == 2, len(rows)
    mj = json.load(open(f"{BASE}/results/metrics_egfr.json"))
    gs = gin_stats("egfr")
    for row in rows:
        split = row[0].strip()
        rf_r2, rf_rmse = [float(x) for x in row[1].split("/")]
        add("T7", f"RF/{split}/r2", rf_r2, mj[split]["r2"])
        add("T7", f"RF/{split}/rmse", rf_rmse, mj[split]["rmse"])
        g42_r2, g42_rmse = gin_seed42("egfr", split)
        s42_r2, s42_rmse = [float(x) for x in row[2].split("/")]
        add("T7", f"GIN42/{split}/r2", s42_r2, g42_r2)
        add("T7", f"GIN42/{split}/rmse", s42_rmse, g42_rmse)
        m_mean, m_std = split_pm(row[3])
        add("T7", f"GIN3/{split}/mean", m_mean, gs[split][0])
        add("T7", f"GIN3/{split}/std", m_std, gs[split][1])


# ---------------------------------------------------------------- T8: fairness campaign
def check_t8():
    rows = find_table("| Target | Model | random R2 | Δ rand | scaffold R2 | Δ scaf |")
    model_key = {"baseline GIN": "baseline_gin", "tuned GIN": "tuned_gin",
                 "GINE": "gine"}
    target = None
    for row in rows:
        if row[0].strip():
            m = re.match(r"`(\w+)`", row[0])
            assert m, row[0]
            target = m.group(1)
            assert target in FAIRNESS_TAGS, target
        model = row[1].strip()
        fj = json.load(open(f"{BASE}/results/fairness/summary_{target}.json"))
        if model == "RF":
            mj = json.load(open(f"{BASE}/results/metrics_{target}.json"))
            add("T8", f"{target}/rf==metrics/random",
                fj["rf"]["random"]["r2"], mj["random"]["r2"],
                ok=abs(fj["rf"]["random"]["r2"] - mj["random"]["r2"]) <= TOL)
            add("T8", f"{target}/rf==metrics/scaffold",
                fj["rf"]["scaffold"]["r2"], mj["scaffold"]["r2"],
                ok=abs(fj["rf"]["scaffold"]["r2"] - mj["scaffold"]["r2"]) <= TOL)
            add("T8", f"{target}/RF/random", num(row[2]), mj["random"]["r2"])
            add("T8", f"{target}/RF/scaffold", num(row[4]), mj["scaffold"]["r2"])
            continue
        key = model_key[model]
        for k, split in ((2, "random"), (4, "scaffold")):
            res = fj["results"][key][split]
            mean, std = split_pm(row[k])
            add("T8", f"{target}/{key}/{split}/mean", mean, res["r2_mean"])
            add("T8", f"{target}/{key}/{split}/std", std, res["r2_std"])
            add("T8", f"{target}/{key}/{split}/delta",
                num(row[k + 1]), fj["delta_r2_gin_minus_rf"][key][split])


# ---------------------------------------------------------------- T9: dataset sizes
def check_t9():
    import pandas as pd
    rows = find_table("| Dataset tag | Target | ChEMBL id | Unique compounds | Source |")
    assert len(rows) == 9, len(rows)
    for row in rows:
        m = re.match(r"`(\w+)`", row[0])
        assert m, row[0]
        tag = m.group(1)
        n = len(pd.read_csv(f"{BASE}/data/processed/{tag}_pic50_clean.csv"))
        add("T9", f"{tag}/n", num(row[3].replace(",", "")), n)



def check_conditional_column():
    """Recompute the GIN table's seed-sensitivity labels, not a prose-log lookup."""
    path = os.path.join(os.path.dirname(__file__), "single_seed_verdicts.py")
    spec = importlib.util.spec_from_file_location("review_seed_verdicts", path)
    seed_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_check)
    bootstrap = seed_check.bootstrap_module(BASE)
    rows = find_table("| Tag | Target (ChEMBL id) | Family | n | RF rand | GIN rand |")
    for row in rows:
        tag = row[0].strip("`")
        conditional = [short for split, short in (("random", "rand"), ("scaffold", "scaf"))
                       if seed_check.cell_sensitivity(BASE, tag, split,
                                                      bootstrap=bootstrap)["conditional"]]
        expected = "+".join(conditional) or "—"
        displayed = row[12].strip()
        add("T1_conditional", tag, displayed, expected, ok=displayed == expected)


def check_environment_table():
    """Verify the optional four-cell table directly from audited job diagnostics.

    Reuse the report loader's identity/integrity/coverage contract, but calculate
    the five numeric columns independently; cells.csv is never an input. An
    absent table contributes no checks, while a present table without accepted
    CPU and CUDA reports fails rather than claiming that twenty cells passed.
    """
    header = "| Environment | Budget | Jobs | Train MSE | Validation RMSE | Test R² | Flags |"
    if not any(line.startswith("|") and header in line for line in lines):
        print("T10: four-cell environment table not present; no checks performed")
        return
    rows = find_table(header)
    expected_cells = {(device, budget) for device in ("CPU", "CUDA")
                      for budget in ("original", "extended")}
    if (len(rows) != 4 or any(len(row) != 7 for row in rows)
            or {(row[0], row[1]) for row in rows} != expected_cells):
        raise ValueError("T10: expected exactly CPU/CUDA x original/extended rows")
    # Resolve the implementation beside this published skill, not from an
    # alternate BASE used to inspect a candidate repository's input files.
    repository = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    path = os.path.join(repository, "experiments", "time_environment_comparison.py")
    spec = importlib.util.spec_from_file_location("readme_environment_loader", path)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    loaded = {}
    metadata = {}
    for device, directory in (("CPU", "m3_eight_seed"), ("CUDA", "cuda_eight_seed")):
        root = os.path.join(BASE, "results", "time_budget_sensitivity", directory, "analysis")
        metadata[device], loaded[device] = comparator.load_report(root, device.lower())
    for field in ("source_sha256", "input_sha256"):
        if metadata["CPU"]["provenance"][field] != metadata["CUDA"]["provenance"][field]:
            raise ValueError(f"T10: CPU/CUDA {field} differs")
    for tag in TAGS:
        labels = [frame.loc[frame.tag.eq(tag), "label_id"].iloc[0]
                  for frame in loaded.values()]
        if labels[0] != labels[1]:
            raise ValueError(f"T10: CPU/CUDA {tag} label versions differ")
    computed = []
    for row in rows:
        device, budget = row[:2]
        arm = "control" if budget == "original" else "extended"
        frame = loaded[device]
        selected = frame.loc[frame.arm.eq(arm)]
        values = (len(selected), float(np.mean(selected.train_mse.to_numpy(dtype=float))),
                  float(np.mean(selected.valid_rmse_std.to_numpy(dtype=float))),
                  float(np.mean(selected.test_r2.to_numpy(dtype=float))),
                  int(np.count_nonzero(selected.common_flag.to_numpy(dtype=bool))))
        for field, displayed, value in zip(("jobs", "train_mse", "valid_rmse_std", "test_r2", "flags"),
                                           row[2:], values):
            computed.append((f"{device}/{budget}/{field}", num(displayed), value))
    for cell, displayed, value in computed:
        add("T10_environment", cell, displayed, value)


def run_checks():
    check_t1()
    check_t2()
    check_t3()
    check_t4()
    check_detail("egfr", "T5")
    check_detail("bace", "T6")
    check_t7()
    check_t8()
    check_t9()
    check_conditional_column()
    check_environment_table()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="optional JSON report path")
    p.add_argument("--dry-run", action="store_true",
                   help="report mismatches but do not fail the exit code")
    return p.parse_args(argv)


def main(argv=None):
    global lines, results
    args = parse_args(argv)
    lines = []
    results = []  # reset: repeated main() calls must not accumulate cells
    if args.out and is_protected(args.out):
        raise SystemExit(f"--out {os.path.abspath(args.out)} points into a "
                         "protected directory (results/ figures/ models/ data/); refusing")
    readme = open(os.path.join(BASE, "README.md"), encoding="utf-8").read()
    lines = readme.splitlines()
    run_checks()

    by_tab = defaultdict(lambda: [0, 0, []])
    for table, cell, disp, comp, ok in results:
        by_tab[table][0] += 1
        if not ok:
            by_tab[table][1] += 1
            by_tab[table][2].append((cell, disp, comp))

    n_bad = sum(v[1] for v in by_tab.values())
    print(f"total cells: {len(results)}, mismatches: {n_bad}")
    for t in sorted(by_tab):
        n, bad, items = by_tab[t]
        print(f"  {t:6s} cells={n:3d} mismatch={bad}")
        for cell, disp, comp in items:
            print(f"      FAIL {cell}: readme={disp!r} source={comp!r}")

    if args.out:
        report = {
            "total_cells": len(results),
            "mismatches": n_bad,
            "tables": {t: {"cells": v[0], "mismatches": v[1],
                           "failures": [{"cell": c, "readme": d, "source": s}
                                        for c, d, s in v[2]]}
                       for t, v in sorted(by_tab.items())},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"report -> {args.out}")

    if n_bad and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
