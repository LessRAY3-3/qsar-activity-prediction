"""Quantify censored IC50 records ("> 10000 nM" etc.) from ChEMBL and the raw pulls.

Why: ChEMBL stores bounds like "IC50 > 10000 nM" with ``standard_relation``
">" / "<" / ">=" / "<=", which are not exact measurements. Since the
2026-10-11 data refresh the downloaders (scripts/01d, scripts/01e) *do*
request ``standard_relation``, and ``02_clean_data.py`` keeps only
``standard_relation == "="`` rows before aggregating labels (review finding
H1), so censored records no longer reach training. This script quantifies
how much censoring exists upstream (live API) and audits that the filter
did what it claims (local raw CSVs).

Modes
  counts  (default)  one limit=1 probe per relation on the live ChEMBL API
                     -> total_count per target and assay type (B/F, as the
                     downloaders use)
  --overlap TAG      offline raw-CSV audit (no network): recompute
                     ``parent_smiles`` for ``data/raw/{TAG}_activities.csv``
                     with the exact standardization from 02_clean_data.py
                     (largest fragment + Uncharger), then
                     (a) assert every modelled parent in
                         ``data/processed/{TAG}_pic50_clean.csv`` keeps at
                         least one "=" record in the raw pull (post-H1
                         invariant),
                     (b) count parents with mixed exact + censored records,
                     (c) count raw parents with no "=" record at all - the
                         true "censored-only" compounds, which are absent
                         from the clean CSV by construction. (The 2026-10-10
                         version of this script counted "ONLY censored"
                         *inside* the clean CSV, which is 0 by design, and
                         matched provenance ``canonical_smiles`` rather than
                         ``parent_smiles``.)

Network: only the counts mode; polite single-threaded paging (0.3 s sleep,
8 retries like 01e). ChEMBL keeps growing, so counts drift slowly between
releases. ``--overlap`` reads only local files.

Usage:
    python .claude/skills/qsar-review/scripts/chembl_censoring_counts.py
    python .claude/skills/qsar-review/scripts/chembl_censoring_counts.py \
        --targets egfr_full=CHEMBL203 --overlap egfr_full
    python .claude/skills/qsar-review/scripts/chembl_censoring_counts.py \
        --overlap a2a --skip-counts          # offline parent-level audit
"""
import argparse
import importlib.util
import os
import time

import requests

API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
CENSORED = (">", "<", ">=", "<=")
PANEL = {
    "a2a": "CHEMBL251", "abl1": "CHEMBL1862", "egfr_full": "CHEMBL203",
    "herg": "CHEMBL240", "hivpr": "CHEMBL243", "mpro": "CHEMBL4523582",
    "vegfr2": "CHEMBL279",
}


def find_repo_root(start):
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "scripts", "qsar_common.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit("repo root not found; pass --repo")
        d = parent


def get(params, tries=8):
    for t in range(1, tries + 1):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code} (try {t})", flush=True)
        except Exception as e:  # network hiccup: same policy as 01e
            print(f"    {e} (try {t})", flush=True)
        time.sleep(min(8 * t, 45))
    raise SystemExit(f"ChEMBL API unreachable for {params}")


def count(**params):
    return get(dict(params, limit=1))["page_meta"]["total_count"]


def base(chembl_id, assay_type):
    return {"target_chembl_id": chembl_id, "standard_type": "IC50",
            "standard_units": "nM", "assay_type": assay_type}


def print_counts(targets):
    print(f"{'tag':10s} {'assay':5s} {'total':>7s} {'censored':>9s} {'share':>6s} "
          f"{'censored w/ pChEMBL':>20s}")
    for tag, cid in targets.items():
        for at in ("B", "F"):
            b = base(cid, at)
            tot = count(**b)
            cens = sum(count(**b, standard_relation=r) for r in CENSORED)
            cens_pc = sum(count(**b, standard_relation=r, pchembl_value__isnull="false")
                          for r in CENSORED)
            share = f"{cens / tot:.1%}" if tot else "n/a"
            print(f"{tag:10s} {at:5s} {tot:7d} {cens:9d} {share:>6s} {cens_pc:20d}", flush=True)
            time.sleep(0.3)


def load_parent_fn(repo):
    """Import 02_clean_data._parent_smiles so standardization matches exactly."""
    path = os.path.join(repo, "scripts", "02_clean_data.py")
    spec = importlib.util.spec_from_file_location("qsar_02_clean_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._parent_smiles


def replicate_parent_fn():
    """Fallback: copy of 02's parent standardization (largest fragment + Uncharger)."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    frag = rdMolStandardize.LargestFragmentChooser()
    unch = rdMolStandardize.Uncharger()

    def parent(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, 0, False
        return (Chem.MolToSmiles(unch.uncharge(frag.choose(mol))),
                len(Chem.GetMolFrags(mol)), True)

    return parent


def overlap(tag, repo):
    """Offline parent-level censoring audit on the raw pull + clean CSV."""
    import pandas as pd

    clean_path = os.path.join(repo, "data", "processed", f"{tag}_pic50_clean.csv")
    raw_path = os.path.join(repo, "data", "raw", f"{tag}_activities.csv")
    if not os.path.exists(raw_path):
        raise SystemExit(f"raw pull missing: {raw_path}\n"
                         "--overlap audits local raw CSVs; fetch them first "
                         "(scripts/fetch_artifacts.py --raw or a downloader).")
    clean = pd.read_csv(clean_path)
    if "parent_smiles" not in clean.columns:
        raise SystemExit(f"{clean_path} has no parent_smiles column (pre-M1 clean CSV?)")

    try:
        parent_fn = load_parent_fn(repo)
        src = "imported scripts/02_clean_data.py::_parent_smiles"
    except Exception as exc:  # pragma: no cover - only if 02 is broken/unimportable
        parent_fn = replicate_parent_fn()
        src = f"replicated 02's standardization (import failed: {exc})"

    # mirror 02's record pool exactly: unit / value / SMILES filters, then relations
    raw = pd.read_csv(raw_path)
    raw = raw[raw["standard_units"] == "nM"].copy()
    raw["standard_value"] = pd.to_numeric(raw["standard_value"], errors="coerce")
    raw = raw.dropna(subset=["standard_value", "canonical_smiles"])
    raw = raw[raw["canonical_smiles"].str.strip() != ""]
    raw = raw[raw["standard_value"] > 0].copy()
    if "standard_relation" not in raw.columns:
        raise SystemExit(f"{raw_path} has no standard_relation column; "
                         "cannot audit censoring (old bace-style download?)")

    parent_map = {smi: parent_fn(smi)[0] for smi in raw["canonical_smiles"].unique()}
    raw["_parent"] = raw["canonical_smiles"].map(parent_map)
    rel = raw["standard_relation"].fillna("").astype(str).str.strip()
    raw["_exact"] = rel == "="        # 02 keeps stripped equals; missing values are non-exact
    raw["_nonexact"] = ~raw["_exact"]

    grp = raw.groupby("_parent")
    n_exact = grp["_exact"].sum()
    n_nonexact = grp["_nonexact"].sum()
    n_exact_rows = int(raw["_exact"].sum())

    modelled = clean["parent_smiles"].astype(str)
    mset = set(modelled)

    print(f"\n[{tag}] raw pool: {len(raw)} rows after 02's unit/value/SMILES filters "
          f"({n_exact_rows} with standard_relation '='); "
          f"{len(clean)} modelled parents in the clean CSV")
    print(f"  parent standardization: {src}")

    # (a) post-H1 invariant: every modelled parent keeps >= 1 exact record,
    #     and n_measurements must equal that exact-record count.
    missing = sorted(p for p in mset if p not in n_exact.index)
    no_exact = sorted(p for p in mset if p in n_exact.index and n_exact[p] < 1)
    if missing or no_exact:
        raise SystemExit(
            f"[{tag}] FAIL: {len(missing)} modelled parents absent from the raw pool, "
            f"{len(no_exact)} modelled parents without any '=' record; "
            f"examples: {(missing + no_exact)[:10]}")
    stored = dict(zip(modelled, clean["n_measurements"]))
    off = [p for p in mset if int(n_exact[p]) != int(stored[p])]
    if off:
        raise SystemExit(f"[{tag}] FAIL: n_measurements != raw '=' record count for "
                         f"{len(off)} parents; e.g. {off[:10]}")
    print(f"  PASS: all {len(mset)} modelled parents have >= 1 standard_relation '=' "
          "record in the raw pull, and n_measurements equals the '=' record count")

    # (b) parents with mixed exact + censored records
    mixed = [p for p in mset if n_nonexact.get(p, 0) >= 1]
    print(f"  {len(mixed)} / {len(mset)} modelled parents ({len(mixed) / len(mset):.1%}) "
          "carry mixed exact + censored records")
    if mixed:
        sub = raw[raw["_parent"].isin(mixed) & raw["_nonexact"]]
        vc = sub["standard_relation"].fillna("(empty)").value_counts()
        print("    their non-exact records by relation: "
              + ", ".join(f"{k!r} x{v}" for k, v in vc.items()))

    # (c) raw parents with no "=" record at all -> dropped by the H1 filter
    censored_only = [p for p in n_exact.index if n_exact[p] < 1]
    print(f"  {len(censored_only)} raw parents have no '=' record at all "
          "(dropped by the H1 filter; these are the true censored-only compounds "
          "and are absent from the clean CSV by construction)")
    return {
        "tag": tag,
        "raw_rows_after_pool_filters": int(len(raw)),
        "n_exact_rows": n_exact_rows,
        "n_modelled_parents": int(len(mset)),
        "n_modelled_without_exact": 0,
        "n_measurements_mismatches": 0,
        "n_modelled_mixed_exact_censored": len(mixed),
        "n_raw_parents_censored_only": len(censored_only),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", default=None,
                   help="comma list tag=CHEMBLID (default: the 7 ChEMBL panel targets)")
    p.add_argument("--overlap", default=None, metavar="TAG",
                   help="offline audit: recompute parent_smiles on data/raw/{TAG}_"
                        "activities.csv with 02's standardization, assert every "
                        "modelled parent keeps >= 1 '=' record, count mixed "
                        "exact+censored parents")
    p.add_argument("--skip-counts", action="store_true",
                   help="skip the live ChEMBL API counts (only --overlap runs)")
    p.add_argument("--repo", default=None)
    args = p.parse_args()

    targets = PANEL
    if args.targets:
        targets = dict(kv.split("=", 1) for kv in args.targets.split(","))
    if not args.skip_counts:
        print_counts(targets)
    if args.overlap:
        repo = args.repo or find_repo_root(os.path.dirname(__file__))
        overlap(args.overlap, repo)
    elif args.skip_counts:
        raise SystemExit("--skip-counts without --overlap: nothing to do")


if __name__ == "__main__":
    main()
