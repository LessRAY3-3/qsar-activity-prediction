# 2026-10-11 数据刷新战役用：egfr_full H1 记录级对账（历史/新拉取多集匹配、清洗 diff、censoring 规则对比）；原 .scratch/h1_sensitivity/reconcile_h1.py。
"""H1 record-level reconciliation for egfr_full (historical vs fresh pull).

Steps:
  1. Multiset-match the HISTORICAL raw pull (--old-raw) against the fresh
     pull (--new-raw) on molecule_chembl_id + canonical_smiles +
     standard_value + assay_type + document_chembl_id, and report the
     standard_relation of matched rows -> the record-level censored share
     of the historical data.
  2. Re-run the PRE-H1 02_clean_data.py logic on the new pull (no relation
     filter - the H1 fix landed after this experiment) and diff per-SMILES
     pIC50 against the clean CSV passed via --clean.
  3. Per-SMILES censoring status after the cleaning filters: a SMILES is
     "censored-only" if none of its surviving records has standard_relation "=".
  4. Compare the two candidate inclusion rules: standard_relation == "=" vs
     pchembl_value.notna() (rows kept, compounds kept, and the differing cases).

Post-refresh input warning (2026-10-11 data refresh, tag
`data-refresh-2026-10-11`): the hard-coded paths this script originally
used were clobbered by the refresh. `data/raw/egfr_full_activities.csv` is
now the REFRESHED pull (identical rows to egfr_full_v2_activities.csv), so
the default old/new comparison is degenerate and the committed
`data/processed/egfr_full_pic50_clean.csv` is the refreshed clean CSV, not
the one the experiment diffed against. A faithful rerun must pass the
pre-refresh inputs explicitly:
    --old-raw <historical pull, e.g. the artifacts-v1 release raw CSV>
    --clean .scratch/h1_sensitivity/egfr_full_pic50_clean_OLD.csv
--dry-run resolves and validates all inputs (existence, rows,
standard_relation presence, old==new degeneracy) and writes nothing.

Fixes since the 2026-10-11 review: explicit --old-raw/--new-raw/--clean/
--out/--dry-run flags; the record-match no longer raises KeyError when BOTH
pulls carry standard_relation (the merged frame suffixes it to
standard_relation_x/_y); missing standard_relation in the analysis pull
fails with a clear message instead of a groupby KeyError.

Outputs into --out (default .scratch/h1_sensitivity/):
  reconcile_summary.json, label_diff_per_compound.csv,
  censored_only_smiles.csv, filter_strategy_diff_examples.csv

Usage:
    python .claude/skills/qsar-review/scripts/reconcile_h1.py --dry-run
    python .claude/skills/qsar-review/scripts/reconcile_h1.py \
        --old-raw /path/to/historical_pull.csv \
        --clean .scratch/h1_sensitivity/egfr_full_pic50_clean_OLD.csv \
        --out /tmp/h1_reconcile
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_OUT = os.path.join(BASE, ".scratch", "h1_sensitivity")
KEY = ["molecule_chembl_id", "canonical_smiles", "standard_value",
       "assay_type", "document_chembl_id"]
PR1_CLAIM = {"censored_only": 2718, "total": 13497, "pct": 20.1}


def canon_value(v):
    """Canonical string form of standard_value so '500000' == '500000.0'."""
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return str(v)


def clean_like_02(df):
    """Mirror the PRE-H1 scripts/02_clean_data.py (no relation filter)."""
    df = df[df["standard_units"] == "nM"].copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    df = df[df["standard_value"] > 0].copy()
    df["pic50"] = -np.log10(df["standard_value"] * 1e-9)
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-raw", default=os.path.join(BASE, "data", "raw",
                                                     "egfr_full_activities.csv"),
                   help="historical raw pull (pre-refresh; without "
                        "standard_relation). Default is now the REFRESHED pull - "
                        "pass the pre-refresh file for a faithful reconciliation.")
    p.add_argument("--new-raw", default=os.path.join(BASE, "data", "raw",
                                                     "egfr_full_v2_activities.csv"),
                   help="fresh pull the campaign downloaded (default: egfr_full_v2)")
    p.add_argument("--clean", default=os.path.join(BASE, "data", "processed",
                                                   "egfr_full_pic50_clean.csv"),
                   help="committed clean CSV to diff labels against (pre-refresh "
                        "snapshot: .scratch/h1_sensitivity/egfr_full_pic50_clean_OLD.csv)")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and validate inputs, print the plan, write nothing")
    return p.parse_args(argv)


def check_inputs(args):
    """Existence/shape report; returns True when the run would be meaningful."""
    ok = True
    for name, path in (("--old-raw", args.old_raw), ("--new-raw", args.new_raw),
                       ("--clean", args.clean)):
        if not os.path.exists(path):
            print(f"  MISSING {name}: {path}")
            ok = False
            continue
        df = pd.read_csv(path, nrows=5)
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        has_rel = "standard_relation" in df.columns
        print(f"  {name}: {path} ({n} rows, standard_relation={'yes' if has_rel else 'NO'})")
    if ok:
        old_ids = pd.read_csv(args.old_raw, usecols=["molecule_chembl_id"])
        new_ids = pd.read_csv(args.new_raw, usecols=["molecule_chembl_id"])
        if len(old_ids) == len(new_ids):
            same = pd.read_csv(args.old_raw).equals(pd.read_csv(args.new_raw))
            if same:
                print("  WARNING: --old-raw and --new-raw are identical files - "
                      "the reconciliation would be degenerate (both are the "
                      "post-refresh pull). Pass the pre-refresh historical pull "
                      "via --old-raw (e.g. artifacts-v1 release raw).")
                ok = False
        if args.clean == os.path.join(BASE, "data", "processed",
                                      "egfr_full_pic50_clean.csv"):
            print("  NOTE: --clean is the committed (refreshed) clean CSV; label "
                  "diffs are only meaningful against the pre-refresh snapshot "
                  ".scratch/h1_sensitivity/egfr_full_pic50_clean_OLD.csv")
    return ok


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out)

    if args.dry_run:
        print(f"dry-run inputs for {os.path.basename(__file__)}:")
        ok = check_inputs(args)
        print(f"  would write: reconcile_summary.json, label_diff_per_compound.csv, "
              f"censored_only_smiles.csv, filter_strategy_diff_examples.csv -> {out_dir}")
        return 0 if ok else 1

    for name, path in (("--old-raw", args.old_raw), ("--new-raw", args.new_raw),
                       ("--clean", args.clean)):
        if not os.path.exists(path):
            raise SystemExit(f"{name} not found: {path} (see --dry-run / docstring "
                             "for the pre-refresh inputs)")
    os.makedirs(out_dir, exist_ok=True)

    old = pd.read_csv(args.old_raw)
    new = pd.read_csv(args.new_raw)
    if "standard_relation" not in new.columns:
        raise SystemExit(f"--new-raw {args.new_raw} has no standard_relation column; "
                         "the analysis pull must be a relation-aware ChEMBL download")
    summary = {"old_raw_rows": int(len(old)), "new_raw_rows": int(len(new)),
               "old_raw": args.old_raw, "new_raw": args.new_raw,
               "clean_csv": args.clean}

    # ---------- 1. record-level multiset match ----------
    for c in ["molecule_chembl_id", "canonical_smiles", "assay_type", "document_chembl_id"]:
        new[c] = new[c].astype(object).where(new[c].notna(), None)
        old[c] = old[c].astype(object).where(old[c].notna(), None)
    new["standard_value"] = new["standard_value"].map(canon_value)
    old["standard_value"] = old["standard_value"].map(canon_value)
    old_has_rel = "standard_relation" in old.columns

    # occurrence index within each key -> true 1:1 multiset pairing
    def occ(d):
        return d.assign(_occ=d.groupby(KEY, dropna=False).cumcount())
    new_o, old_o = occ(new), occ(old)
    m = old_o.merge(new_o, on=KEY + ["_occ"], how="left", indicator=True)
    matched = m[m["_merge"] == "both"].copy()
    old_unmatched = m[m["_merge"] == "left_only"]
    new_only = new_o.merge(old_o[KEY + ["_occ"]], on=KEY + ["_occ"], how="left",
                           indicator=True)

    # relation comes from the fresh pull; when both sides carry it the merge
    # suffixes the columns (this used to raise KeyError on matched[...])
    if "standard_relation" in matched.columns:
        rel = matched["standard_relation"].fillna("<null>")
    elif "standard_relation_y" in matched.columns:
        rel = matched["standard_relation_y"].fillna("<null>")
    elif "standard_relation_x" in matched.columns:
        rel = matched["standard_relation_x"].fillna("<null>")
    elif old_has_rel:
        raise SystemExit("merged frame lost standard_relation unexpectedly")
    else:
        raise SystemExit("matched rows carry no standard_relation on either side")
    rel_counts = rel.value_counts().to_dict()
    summary["record_match"] = {
        "old_rows": int(len(old)),
        "old_matched": int(len(matched)),
        "old_unmatched": int(len(old_unmatched)),
        "new_only_rows": int((new_only["_merge"] == "left_only").sum()),
        "relation_source": "new pull",
        "relation_of_matched": {str(k): int(v) for k, v in sorted(rel_counts.items())},
        "censored_share_of_matched": round(float((rel != "=").mean()), 4),
        "censored_share_excluding_null_relation": round(
            float(rel.isin([">", "<", ">=", "<="]).mean()), 4),
    }

    # ---------- 2/3. clean the new pull, per-SMILES status ----------
    nclean = clean_like_02(new)
    summary["cleaning_new"] = {
        "rows_after_unit_nm": int((new["standard_units"] == "nM").sum()),
        "rows_after_clean": int(len(nclean)),
    }
    g = nclean.groupby("canonical_smiles")
    per = g.agg(pic50_new=("pic50", "median"),
                n_records=("pic50", "size"),
                n_exact=("standard_relation", lambda s: int((s == "=").sum())),
                molecule_chembl_id_new=("molecule_chembl_id", "first"))
    per["relations"] = g["standard_relation"].agg(lambda s: "|".join(sorted({str(x) for x in s})))
    per["has_exact"] = per["n_exact"] > 0
    per = per.reset_index()
    n_comp = len(per)
    cens = per[~per["has_exact"]]
    summary["censored_only"] = {
        "n_compounds": int(n_comp),
        "n_censored_only": int(len(cens)),
        "pct": round(100 * len(cens) / n_comp, 2),
        "pr1_claim": PR1_CLAIM,
    }

    # label reproduction: new-clean (no filter) vs the supplied clean CSV
    old_clean = pd.read_csv(args.clean)
    j = old_clean.merge(per, on="canonical_smiles", how="outer", indicator=True)
    both = j[j["_merge"] == "both"].copy()
    both["delta"] = both["pic50_new"] - both["pic50"]
    changed = both[both["delta"].abs() > 1e-9]
    summary["label_reproduction"] = {
        "committed_compounds": int(len(old_clean)),
        "new_compounds": int(n_comp),
        "in_both": int(len(both)),
        "committed_only": int((j["_merge"] == "left_only").sum()),
        "new_only": int((j["_merge"] == "right_only").sum()),
        "labels_differ": int(len(changed)),
        "max_abs_delta": float(both["delta"].abs().max()) if len(both) else 0.0,
        "mean_abs_delta_of_changed":
            float(changed["delta"].abs().mean()) if len(changed) else 0.0,
    }

    # what labels would become under the "=" fix (exact records only)
    exact = nclean[nclean["standard_relation"] == "="]
    exact_lab = (exact.groupby("canonical_smiles")["pic50"].median()
                 .rename("pic50_exact_only"))
    both2 = both.merge(exact_lab, on="canonical_smiles", how="left")
    shift = (both2["pic50_exact_only"] - both2["pic50"]).abs()
    would_change = both2[both2["pic50_exact_only"].notna() & (shift > 1e-9)]
    gone = both2[both2["pic50_exact_only"].isna()]  # no exact record at all
    summary["label_if_fixed"] = {
        "would_change": int(len(would_change)),
        "mean_abs_shift": float((would_change["pic50_exact_only"]
                                 - would_change["pic50"]).abs().mean())
            if len(would_change) else 0.0,
        "max_abs_shift": float((would_change["pic50_exact_only"]
                                - would_change["pic50"]).abs().max())
            if len(would_change) else 0.0,
        "gone_censored_only": int(len(gone)),
    }

    # ---------- 4. the two inclusion rules ----------
    def rule_report(mask, name):
        kept = nclean[mask]
        summary[f"rule_{name}"] = {
            "rows_kept": int(len(kept)),
            "rows_dropped": int(len(nclean) - len(kept)),
            "compounds_kept": int(kept["canonical_smiles"].nunique()),
        }

    rule_report(nclean["standard_relation"] == "=", "relation_eq")
    rule_report(nclean["pchembl_value"].notna(), "pchembl_notna")
    only_eq = nclean[(nclean["standard_relation"] == "=")
                     & (nclean["pchembl_value"].isna())].copy()
    only_pc = nclean[(nclean["standard_relation"] != "=")
                     & (nclean["pchembl_value"].notna())].copy()
    summary["rule_diff"] = {
        "eq_but_no_pchembl_rows": int(len(only_eq)),
        "eq_but_no_pchembl_compounds": int(only_eq["canonical_smiles"].nunique()),
        "pchembl_but_not_eq_rows": int(len(only_pc)),
        "pchembl_but_not_eq_compounds": int(only_pc["canonical_smiles"].nunique()),
    }
    cols = ["activity_id", "molecule_chembl_id", "canonical_smiles", "standard_value",
            "standard_relation", "data_validity_comment", "pchembl_value",
            "assay_type", "document_chembl_id"]
    present = [c for c in cols if c in only_eq.columns]
    only_eq[present].head(20).to_csv(
        os.path.join(out_dir, "filter_strategy_diff_examples.csv"), index=False)

    # ---------- outputs ----------
    per_out = both[["canonical_smiles", "pic50", "pic50_new", "delta",
                    "n_measurements", "n_records", "n_exact", "relations",
                    "has_exact", "molecule_chembl_id"]].rename(
        columns={"pic50": "pic50_committed", "n_measurements": "n_committed"})
    per_out = per_out.merge(exact_lab, on="canonical_smiles", how="left")
    per_out.to_csv(os.path.join(out_dir, "label_diff_per_compound.csv"), index=False)
    cens[["canonical_smiles"]].to_csv(
        os.path.join(out_dir, "censored_only_smiles.csv"), index=False)

    with open(os.path.join(out_dir, "reconcile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
