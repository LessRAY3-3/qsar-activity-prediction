"""QC: parent-level salts, split leakage and censored activities (audit only).

What / why:
  1. Since review finding M1, ``02_clean_data.py`` standardizes every record
     to ``parent_smiles`` (RDKit LargestFragmentChooser + Uncharger) and
     deduplicates on it, so the committed clean CSV is one row per parent:
     single-fragment, unique, and assigned to exactly one split. This script
     audits the committed ``parent_smiles`` column directly - multi-fragment
     parents (must be 0), duplicate parents (must be 0), and cross-split
     parent overlap (must be 0) - and recomputes parents from the provenance
     ``canonical_smiles`` variant with the same standardization to confirm
     the stored column. It also reports provenance-side numbers (counter-ions
     still visible in ``canonical_smiles``) as context, clearly labelled:
     the 2026-10-10 version audited the provenance variant itself, whose
     ">1 fragment" headline counted counter-ions that 02 already strips and
     whose duplicate/leakage checks were 0 by construction.
  2. ChEMBL stores censored measurements ("IC50 > 10000 nM") with a
     ``standard_relation`` of ">" / "<" / ">=" / "<=".  Since the 2026-10-11
     refresh the downloaders request that column and 02 keeps only "=" rows.
     This script counts removed rows with 02's own inclusion rule, verbatim:
     the pool is filtered exactly like 02 (unit nM, non-missing non-empty
     SMILES, value > 0) and a row counts as censored/removed iff
     ``standard_relation != "="`` (no stripping; empty/other values such as
     ">>" are removed by 02 and are counted here too).  When the column is
     absent (older/bace-style download) the exact-value striations at
     10^3/10^4/10^5 nM are reported as a proxy.  ``chembl_censoring_counts.py``
     in this folder additionally audits censoring at the parent level
     (every modelled parent keeps >= 1 "=" record) and quantifies upstream
     censoring straight from the ChEMBL API.

Exit code: the "must be 0" invariants are real checks - multi-fragment
parent_smiles, duplicated parent_smiles, and cross-split parent overlap each
make the script exit 1 when violated (skipped splits/inputs do not).  A
clean CSV without a ``parent_smiles`` column (bace, which predates the M1
parent standardization) skips the parent-level checks with a printed note
instead of crashing, and does not fail on its own.

Nothing is filtered or rewritten - like ``07_qc_checks.py`` the audit is the
deliverable.  Output: console, plus a JSON summary when --out is given
(``results/``, ``data/``, ``models/`` and ``figures/`` are refused as --out).

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

CENSOR_RELATIONS = (">", "<", ">=", "<=", "~")  # reported breakdown only
STRIATION_NM = (1000.0, 10000.0, 100000.0)  # exact pIC50 6 / 5 / 4


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


_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def parent_smiles(smi):
    """Parent SMILES via largest fragment + Uncharge (same as 02's _parent_smiles)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(_UNCHARGER.uncharge(_CHOOSER.choose(mol)))


def check_salts(df):
    if "parent_smiles" not in df.columns:
        print("== 1. parent_smiles / salt forms ==")
        print(f"  no parent_smiles column in {os.path.relpath(CLEAN_CSV, BASE)} - "
              "parent-level salt audit skipped. This clean CSV predates the M1 "
              "parent standardization (bace: 02's parent handling was added for "
              "the ChEMBL panel; bace keeps the original download format), so "
              "there is no stored parent column to audit. Multi-fragment and "
              "duplicate checks below are vacuous - rerun 02 on this tag if a "
              "parent-level audit is needed.")
        return None, {"n_rows": int(len(df)), "parent_column_present": False,
                      "skipped": "no parent_smiles column (pre-M1 clean CSV)"}
    parents = df["parent_smiles"].astype(str).to_numpy()
    prov = df["canonical_smiles"].astype(str).to_numpy()

    multi_parent = np.array(["." in s for s in parents])
    multi_prov = np.array(["." in s for s in prov])
    recomputed = np.array([parent_smiles(s) or s for s in prov], dtype=object)
    mismatch = recomputed != parents
    counts = pd.Series(parents).value_counts()
    dup_parents = counts[counts > 1]
    n_rows_in_dups = int(dup_parents.sum())
    spread = []
    for p in dup_parents.index:
        vals = df.loc[parents == p, "pic50"].to_numpy()
        spread.append(float(vals.max() - vals.min()))
    ions = pd.Series(
        [frag for s in prov[multi_prov] for frag in s.split(".") if len(frag) <= 12]
    ).value_counts()

    print("== 1. parent_smiles / salt forms ==")
    print(f"  {int(multi_parent.sum())} / {len(df)} modelled parent_smiles contain more "
          "than one fragment (must be 0: 02 strips counter-ions before dedup)")
    print(f"    provenance canonical_smiles (first variant per parent): "
          f"{int(multi_prov.sum())} contain '.' - counter-ions visible in provenance, "
          "already stripped from parent_smiles")
    if len(ions):
        print("  most common small fragments in provenance: "
              + ", ".join(f"{k} x{v}" for k, v in ions.head(6).items()))
    print(f"  {len(dup_parents)} duplicated parent_smiles (must be 0: 02 dedups by "
          f"parent; rows would collapse to "
          f"{len(df) - n_rows_in_dups + len(dup_parents)} on dedup)")
    if spread:
        print(f"  pIC50 spread inside those parent groups: median {np.median(spread):.2f}, "
              f"max {np.max(spread):.2f}")
    print(f"  recomputed parent from provenance canonical_smiles (largest fragment + "
          f"Uncharger, same fn as 02): {int(mismatch.sum())} / {len(df)} mismatches")
    if mismatch.any():
        for i in np.flatnonzero(mismatch)[:5]:
            print(f"    stored={parents[i]!r} recomputed={recomputed[i]!r}")
    return parents, {
        "parent_column_present": True,
        "n_rows": int(len(df)),
        "n_parent_multifragment": int(multi_parent.sum()),
        "n_provenance_multifragment": int(multi_prov.sum()),
        "n_parents_duplicated": int(len(dup_parents)),
        "n_rows_in_duplicated_parents": n_rows_in_dups,
        "n_rows_after_parent_dedup": int(len(df) - n_rows_in_dups + len(dup_parents)),
        "n_provenance_parent_mismatch": int(mismatch.sum()),
        "pic50_spread_median": float(np.median(spread)) if spread else None,
        "pic50_spread_max": float(np.max(spread)) if spread else None,
        "top_fragments": {str(k): int(v) for k, v in ions.head(10).items()},
    }


def check_split_leakage(parents):
    """Cross-split parent overlap. Must be 0: each parent is exactly one clean-CSV row."""
    print("\n== 2. parent-level train/test leakage ==")
    out = {}
    if parents is None:
        print("  skipped: no parent_smiles column to check against (see section 1)")
        return out
    for split in ("random", "scaffold", "time"):
        path = os.path.join(SPLIT_DIR, f"{TAG}_{split}.npz")
        if not os.path.exists(path):
            print(f"  {split:9s} no split file ({os.path.relpath(path, BASE)}), skipped")
            continue
        d = np.load(path)
        smiles = np.asarray(d["smiles"], dtype=object)
        if len(smiles) != len(parents):
            print(f"  {split:9s} split file has {len(smiles)} molecules, "
                  f"clean CSV has {len(parents)} - skipped")
            continue
        order_mismatch = int(np.sum(smiles != parents))
        if order_mismatch:
            print(f"  {split:9s} WARNING: split smiles differ from clean parent_smiles "
                  f"in {order_mismatch} / {len(parents)} positions - leakage numbers "
                  "are not interpretable (split file predates the clean CSV?)")
            continue
        train = set(parents[d["train_idx"]])
        if "valid_idx" in d.files:
            train |= set(parents[d["valid_idx"]])
        test_parents = parents[d["test_idx"]]
        leaked = np.array([p in train for p in test_parents])
        n_leak = int(leaked.sum())
        print(f"  {split:9s} {n_leak} / {len(test_parents)} test parents also in train "
              "(expected 0: rows are unique parents assigned to one split; non-zero "
              "means split/clean misalignment)")
        out[split] = {"n_test": int(len(test_parents)),
                      "n_test_with_parent_in_train": n_leak,
                      "split_matches_clean_order": True}
    return out


def check_censoring():
    print("\n== 3. censored activities in the raw pull ==")
    if not os.path.exists(RAW_CSV):
        print(f"  raw CSV missing ({os.path.relpath(RAW_CSV, BASE)}); run "
              "scripts/fetch_artifacts.py --raw or re-download, skipped")
        return {"raw_available": False}
    raw = pd.read_csv(RAW_CSV)
    # mirror 02's record pool exactly (unit / value / SMILES filters) so the
    # removed-row count below is directly comparable to 02's own printout
    if "standard_units" in raw.columns:
        raw = raw[raw["standard_units"] == "nM"].copy()
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    raw = raw.dropna(subset=["standard_value", "canonical_smiles"])
    raw = raw[raw["canonical_smiles"].str.strip() != ""]
    raw = raw[raw["standard_value"] > 0].copy()
    out = {"raw_available": True, "n_raw_rows": int(len(raw))}

    if "standard_relation" in raw.columns:
        # 02's inclusion rule, verbatim: keep only standard_relation == "="
        # (no stripping). Everything else - including empty and odd values
        # like ">>" - is removed by 02 and counts as censored/removed here.
        removed = raw["standard_relation"] != "="
        out["standard_relation_available"] = True
        out["n_removed_by_relation_rule"] = int(removed.sum())
        out["by_relation"] = {("(empty)" if pd.isna(k) else str(k)): int(v)
                              for k, v in raw.loc[removed, "standard_relation"]
                              .value_counts(dropna=False).items()}
        n_bounds = int(raw["standard_relation"].isin(CENSOR_RELATIONS).sum())
        out["n_censored_bounds"] = n_bounds  # >, <, >=, <=, ~ subset of removed
        bounds = {k: v for k, v in out["by_relation"].items() if k in CENSOR_RELATIONS}
        other = {k: v for k, v in out["by_relation"].items() if k not in CENSOR_RELATIONS}
        print(f"  {int(removed.sum())} / {len(raw)} raw rows ({removed.mean():.1%}) removed by "
              f"02's rule (standard_relation != '='), on 02's own unit/value/SMILES pool; "
              "these never become labels")
        print(f"    classic censored bounds: "
              f"{', '.join(f'{k} x{v}' for k, v in bounds.items()) or 'none'}"
              + (f"; other non-'=' values: "
                 f"{', '.join(f'{k} x{v}' for k, v in other.items())}"
                 if other else ""))
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

    if args.out and is_protected(args.out):
        raise SystemExit(f"--out {os.path.abspath(args.out)} points into a "
                         "protected directory (results/ figures/ models/ data/); refusing")

    df = pd.read_csv(CLEAN_CSV)
    print(f"[{TAG}] {len(df)} cleaned compounds")
    parents, salts = check_salts(df)
    leakage = check_split_leakage(parents)
    censoring = check_censoring()

    # the "must be 0" invariants are real checks: any violation exits 1
    violations = []
    if salts.get("parent_column_present"):
        if salts["n_parent_multifragment"]:
            violations.append(
                f"{salts['n_parent_multifragment']} multi-fragment parent_smiles (must be 0)")
        if salts["n_parents_duplicated"]:
            violations.append(
                f"{salts['n_parents_duplicated']} duplicated parent_smiles (must be 0)")
    for split, info in leakage.items():
        if info.get("split_matches_clean_order") and info["n_test_with_parent_in_train"]:
            violations.append(
                f"{split}: {info['n_test_with_parent_in_train']} test parents also in train "
                "(expected 0)")
    if violations:
        print("\nFAIL - 'must be 0' invariants violated:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("\nOK - all checked 'must be 0' invariants hold")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"tag": TAG, "salts": salts, "split_leakage": leakage,
                       "censoring": censoring, "violations": violations}, f, indent=2)
        print(f"saved -> {args.out}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
