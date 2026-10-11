# 2026-10-11 数据刷新战役用：egfr_full H1 记录级对账（历史/新拉取多集匹配、清洗 diff、censoring 规则对比）；原 .scratch/h1_sensitivity/reconcile_h1.py。
"""H1 record-level reconciliation for egfr_full.

Steps:
  1. Multiset-match the historical raw pull (data/raw/egfr_full_activities.csv,
     no relation columns) against the fresh pull (egfr_full_v2_activities.csv)
     on molecule_chembl_id + canonical_smiles + standard_value + assay_type +
     document_chembl_id, and report the standard_relation of matched rows ->
     the record-level censored share of the *historical* data.
  2. Re-run 02_clean_data.py logic on the new pull (no relation filter, exactly
     as the pipeline does today) and diff per-SMILES pIC50 against the committed
     clean CSV.
  3. Per-SMILES censoring status after the cleaning filters: a SMILES is
     "censored-only" if none of its surviving records has standard_relation "=".
  4. Compare the two candidate inclusion rules: standard_relation == "=" vs
     pchembl_value.notna() (rows kept, compounds kept, and the differing cases).

Outputs into .scratch/h1_sensitivity/:
  reconcile_summary.json, label_diff_per_compound.csv, censored_only_smiles.csv,
  filter_strategy_diff_examples.csv
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.join(BASE, ".scratch", "h1_sensitivity")
OLD_RAW = os.path.join(BASE, "data", "raw", "egfr_full_activities.csv")
NEW_RAW = os.path.join(BASE, "data", "raw", "egfr_full_v2_activities.csv")
OLD_CLEAN = os.path.join(BASE, "data", "processed", "egfr_full_pic50_clean.csv")

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
    """Mirror scripts/02_clean_data.py exactly (still no relation filter)."""
    df = df[df["standard_units"] == "nM"].copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    df = df[df["standard_value"] > 0].copy()
    df["pic50"] = -np.log10(df["standard_value"] * 1e-9)
    return df


def main():
    old = pd.read_csv(OLD_RAW)
    new = pd.read_csv(NEW_RAW)
    summary = {"old_raw_rows": int(len(old)), "new_raw_rows": int(len(new))}

    # ---------- 1. record-level multiset match ----------
    for c in ["molecule_chembl_id", "canonical_smiles", "assay_type", "document_chembl_id"]:
        new[c] = new[c].astype(object).where(new[c].notna(), None)
        old[c] = old[c].astype(object).where(old[c].notna(), None)
    new["standard_value"] = new["standard_value"].map(canon_value)
    old["standard_value"] = old["standard_value"].map(canon_value)
    # occurrence index within each key -> true 1:1 multiset pairing
    def occ(d):
        return d.assign(_occ=d.groupby(KEY, dropna=False).cumcount())
    new_o, old_o = occ(new), occ(old)
    m = old_o.merge(new_o, on=KEY + ["_occ"], how="left", indicator=True)
    matched = m[m["_merge"] == "both"].copy()
    old_unmatched = m[m["_merge"] == "left_only"]
    new_only = new_o.merge(old_o[KEY + ["_occ"]], on=KEY + ["_occ"], how="left",
                           indicator=True)

    rel = matched["standard_relation"].fillna("<null>")
    rel_counts = rel.value_counts().to_dict()
    summary["record_match"] = {
        "old_rows": int(len(old)),
        "old_matched": int(len(matched)),
        "old_unmatched": int(len(old_unmatched)),
        "new_only_rows": int((new_only["_merge"] == "left_only").sum()),
        "relation_of_matched": {str(k): int(v) for k, v in sorted(rel_counts.items())},
        "censored_share_of_matched": round(float((rel != "=").mean()), 4),
        "censored_share_excluding_null_relation": round(
            float(matched["standard_relation"].isin([">", "<", ">=", "<="]).mean()), 4),
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

    # label reproduction: new-clean (no filter) vs committed clean
    old_clean = pd.read_csv(OLD_CLEAN)
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
        "max_abs_delta": float(both["delta"].abs().max()),
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
        "mean_abs_shift": float((would_change["pic50_exact_only"] - would_change["pic50"]).abs().mean())
            if len(would_change) else 0.0,
        "max_abs_shift": float((would_change["pic50_exact_only"] - would_change["pic50"]).abs().max())
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
    only_eq = nclean[(nclean["standard_relation"] == "=") & (nclean["pchembl_value"].isna())]
    only_pc = nclean[(nclean["standard_relation"] != "=") & (nclean["pchembl_value"].notna())]
    summary["rule_diff"] = {
        "eq_but_no_pchembl_rows": int(len(only_eq)),
        "eq_but_no_pchembl_compounds": int(only_eq["canonical_smiles"].nunique()),
        "pchembl_but_not_eq_rows": int(len(only_pc)),
        "pchembl_but_not_eq_compounds": int(only_pc["canonical_smiles"].nunique()),
    }
    cols = ["activity_id", "molecule_chembl_id", "canonical_smiles", "standard_value",
            "standard_relation", "data_validity_comment", "pchembl_value",
            "assay_type", "document_chembl_id"]
    only_eq[cols].head(20).to_csv(os.path.join(OUT, "filter_strategy_diff_examples.csv"),
                                   index=False)

    # ---------- outputs ----------
    per_out = both[["canonical_smiles", "pic50", "pic50_new", "delta",
                    "n_measurements", "n_records", "n_exact", "relations",
                    "has_exact", "molecule_chembl_id"]].rename(
        columns={"pic50": "pic50_committed", "n_measurements": "n_committed"})
    per_out = per_out.merge(exact_lab, on="canonical_smiles", how="left")
    per_out.to_csv(os.path.join(OUT, "label_diff_per_compound.csv"), index=False)
    cens[["canonical_smiles"]].to_csv(os.path.join(OUT, "censored_only_smiles.csv"),
                                      index=False)

    with open(os.path.join(OUT, "reconcile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
