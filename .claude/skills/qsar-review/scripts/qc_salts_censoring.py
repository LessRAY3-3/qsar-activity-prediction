"""QC: salt forms, parent-level duplicates and censored activities (audit only).

What / why:
  1. ChEMBL ``canonical_smiles`` keeps counter-ions ("Cl.", "Br.", "[Na+]" ...).
     ``02_clean_data.py`` deduplicates on the raw SMILES string, so a free base
     and its hydrochloride are two "unique compounds" with two labels, their
     fingerprints/graphs carry the counter-ion atoms, and the same parent can
     sit on both sides of a split (an analogue leak the scaffold split cannot
     see because the scaffold is identical).  This script strips every SMILES
     to its largest organic fragment (RDKit LargestFragmentChooser), counts
     how many rows collapse onto an already-present parent, and - when the
     persisted split indices exist - how many parents straddle train/test.
  2. ChEMBL stores censored measurements ("IC50 > 10000 nM") with a
     ``standard_relation`` of ">" / "<" / ">=" / "<=".  The downloaders do not
     pull that column, so such rows enter the regression as exact values.
     If the raw CSV carries ``standard_relation`` (only once the downloader
     is extended to request it) the censored rows are counted directly;
     otherwise the exact-value striations at 10^3/10^4/10^5 nM are reported
     as a proxy.  ``chembl_censoring_counts.py`` in this folder quantifies
     censoring straight from the ChEMBL API instead.

Nothing is filtered or rewritten - like ``07_qc_checks.py`` the audit is the
deliverable.  Output: console, plus a JSON summary when --out is given.

Usage (from anywhere; the repo root is found by walking up to
scripts/qsar_common.py, or pass --repo):
    QSAR_TAG=egfr python .claude/skills/qsar-review/scripts/qc_salts_censoring.py
    python .claude/skills/qsar-review/scripts/qc_salts_censoring.py --tag herg --out /tmp/herg.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


def find_repo_root(start):
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "scripts", "qsar_common.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("repo root not found (no scripts/qsar_common.py above "
                             f"{start}); pass --repo")
        d = parent


# module-level paths are (re)bound in main() from --repo/--tag
BASE = TAG = CLEAN_CSV = RAW_CSV = SPLIT_DIR = None

CENSOR_RELATIONS = (">", "<", ">=", "<=", "~")
STRIATION_NM = (1000.0, 10000.0, 100000.0)  # exact pIC50 6 / 5 / 4

_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def parent_smiles(smi):
    """Canonical SMILES of the largest fragment (None if RDKit cannot parse)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(_CHOOSER.choose(mol))


def check_salts(df):
    smiles = df["canonical_smiles"].astype(str).to_numpy()
    multi = np.array(["." in s for s in smiles])
    parents = np.array([parent_smiles(s) or s for s in smiles], dtype=object)
    counts = pd.Series(parents).value_counts()
    dup_parents = counts[counts > 1]
    n_rows_in_dups = int(dup_parents.sum())
    spread = []
    for p in dup_parents.index:
        vals = df.loc[parents == p, "pic50"].to_numpy()
        spread.append(float(vals.max() - vals.min()))
    ions = pd.Series(
        [frag for s in smiles[multi] for frag in s.split(".") if len(frag) <= 12]
    ).value_counts()

    print("== 1. salt forms / parent-level duplicates ==")
    print(f"  {int(multi.sum())} / {len(df)} SMILES ({multi.mean():.1%}) contain more "
          "than one fragment (counter-ion / solvate)")
    if len(ions):
        print("  most common small fragments: "
              + ", ".join(f"{k} x{v}" for k, v in ions.head(6).items()))
    print(f"  {len(dup_parents)} parents appear under >1 SMILES "
          f"({n_rows_in_dups} rows -> {len(df) - n_rows_in_dups + len(dup_parents)} "
          "rows after parent-level dedup)")
    if spread:
        print(f"  pIC50 spread inside those parent groups: median {np.median(spread):.2f}, "
              f"max {np.max(spread):.2f}")
    return parents, {
        "n_rows": int(len(df)),
        "n_multifragment": int(multi.sum()),
        "n_parents_duplicated": int(len(dup_parents)),
        "n_rows_in_duplicated_parents": n_rows_in_dups,
        "n_rows_after_parent_dedup": int(len(df) - n_rows_in_dups + len(dup_parents)),
        "pic50_spread_median": float(np.median(spread)) if spread else None,
        "pic50_spread_max": float(np.max(spread)) if spread else None,
        "top_fragments": {str(k): int(v) for k, v in ions.head(10).items()},
    }


def check_split_leakage(parents):
    """Parents with at least one member in train AND one in test, per split."""
    print("\n== 2. parent-level train/test leakage ==")
    out = {}
    for split in ("random", "scaffold", "time"):
        path = os.path.join(SPLIT_DIR, f"{TAG}_{split}.npz")
        if not os.path.exists(path):
            print(f"  {split:9s} no split file ({os.path.relpath(path, BASE)}), skipped")
            continue
        d = np.load(path)
        if len(d["smiles"]) != len(parents):
            print(f"  {split:9s} split file has {len(d['smiles'])} molecules, "
                  f"clean CSV has {len(parents)} - skipped")
            continue
        train = set(parents[d["train_idx"]])
        if "valid_idx" in d.files:
            train |= set(parents[d["valid_idx"]])
        test_parents = parents[d["test_idx"]]
        leaked = np.array([p in train for p in test_parents])
        n_leak = int(leaked.sum())
        print(f"  {split:9s} {n_leak} / {len(test_parents)} test molecules "
              f"({n_leak / len(test_parents):.1%}) share a parent with a train molecule")
        out[split] = {"n_test": int(len(test_parents)), "n_test_with_parent_in_train": n_leak}
    return out


def check_censoring():
    print("\n== 3. censored activities ==")
    if not os.path.exists(RAW_CSV):
        print(f"  raw CSV missing ({os.path.relpath(RAW_CSV, BASE)}); run "
              "scripts/fetch_artifacts.py --raw or re-download, skipped")
        return {"raw_available": False}
    raw = pd.read_csv(RAW_CSV)
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    raw = raw.dropna(subset=["standard_value"])
    raw = raw[raw["standard_value"] > 0]
    out = {"raw_available": True, "n_raw_rows": int(len(raw))}

    if "standard_relation" in raw.columns:
        rel = raw["standard_relation"].astype(str).str.strip()
        censored = rel.isin(CENSOR_RELATIONS)
        out["standard_relation_available"] = True
        out["n_censored"] = int(censored.sum())
        out["by_relation"] = {str(k): int(v) for k, v in rel.value_counts().items()}
        print(f"  {int(censored.sum())} / {len(raw)} raw rows ({censored.mean():.1%}) carry a "
              f"censored standard_relation ({', '.join(out['by_relation'])})")
    else:
        out["standard_relation_available"] = False
        print("  standard_relation column not in the raw CSV (older download) - "
              "censored rows cannot be identified directly; proxy below")

    striations = {}
    for nm in STRIATION_NM:
        n = int((raw["standard_value"] == nm).sum())
        striations[f"{nm:g}"] = n
    total = sum(striations.values())
    out["exact_value_rows"] = striations
    print(f"  rows at exactly {', '.join(f'{k} nM' for k in striations)}: "
          f"{', '.join(str(v) for v in striations.values())} "
          f"(= {total / len(raw):.1%} of raw rows; typical 'inactive above X' pins)")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--repo", default=None, help="repo root (default: auto-detect)")
    p.add_argument("--out", default=None, help="optional JSON summary path")
    return p.parse_args(argv)


def main(argv=None):
    global BASE, TAG, CLEAN_CSV, RAW_CSV, SPLIT_DIR
    args = parse_args(argv)
    BASE = args.repo or find_repo_root(os.path.dirname(__file__))
    TAG = args.tag
    CLEAN_CSV = os.path.join(BASE, "data", "processed", f"{TAG}_pic50_clean.csv")
    RAW_CSV = os.path.join(BASE, "data", "raw", f"{TAG}_activities.csv")
    SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")

    df = pd.read_csv(CLEAN_CSV)
    print(f"[{TAG}] {len(df)} cleaned compounds")
    parents, salts = check_salts(df)
    leakage = check_split_leakage(parents)
    censoring = check_censoring()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"tag": TAG, "salts": salts, "split_leakage": leakage,
                       "censoring": censoring}, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
