"""Quantify censored IC50 records ("> 10000 nM" etc.) straight from ChEMBL.

Why: the downloaders (scripts/01d, scripts/01e) never request
``standard_relation``, so a record stored as "IC50 > 10000 nM" reaches
02_clean_data.py as an exact 10000 nM (pIC50 5.0).  ChEMBL leaves
``pchembl_value`` empty for every censored record, so the pIC50-vs-pChEMBL
"100% agreement" check in 02 is computed on uncensored rows only and cannot
see the problem.

Modes
  counts  (default)  one limit=1 probe per relation -> total_count, per
                     target and assay type (B/F, as the downloaders use)
  --overlap TAG      additionally page through the censored records of one
                     target and report how many compounds of
                     data/processed/{TAG}_pic50_clean.csv carry at least one
                     censored measurement, and how many carry ONLY censored
                     ones (their training label is a detection limit, not a
                     measurement)

Network: polite single-threaded paging (0.3 s sleep, 8 retries like 01e).
ChEMBL keeps growing, so counts drift slowly between releases.

Usage:
    python .claude/skills/qsar-review/scripts/chembl_censoring_counts.py
    python .claude/skills/qsar-review/scripts/chembl_censoring_counts.py \
        --targets egfr_full=CHEMBL203 --overlap egfr_full
"""
import argparse
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


def censored_smiles(chembl_id, page=500):
    out = []
    for at in ("B", "F"):
        for rel in CENSORED:
            offset = 0
            while True:
                js = get(dict(base(chembl_id, at), standard_relation=rel,
                              limit=page, offset=offset))
                batch = js.get("activities", [])
                out += [a.get("canonical_smiles") for a in batch if a.get("canonical_smiles")]
                offset += len(batch)
                if not batch or not js.get("page_meta", {}).get("next"):
                    break
                time.sleep(0.3)
    return out


def overlap(tag, chembl_id, repo):
    import pandas as pd

    clean = pd.read_csv(os.path.join(repo, "data", "processed", f"{tag}_pic50_clean.csv"))
    smis = censored_smiles(chembl_id)
    cens = pd.Series(smis).value_counts()
    hit = clean["canonical_smiles"].isin(cens.index)
    only = hit & (clean["canonical_smiles"].map(cens).fillna(0) >= clean["n_measurements"])
    print(f"\n[{tag}] {len(smis)} censored records fetched")
    print(f"  {int(hit.sum())} / {len(clean)} cleaned compounds ({hit.mean():.1%}) "
          "have >= 1 censored measurement")
    print(f"  {int(only.sum())} / {len(clean)} ({only.mean():.1%}) have ONLY censored "
          "measurements (label = a detection limit)")
    if only.any():
        vc = clean.loc[only, "pic50"].round(2).value_counts().head(5)
        print("  most common labels among those: "
              + ", ".join(f"pIC50 {k:g} x{v}" for k, v in vc.items()))
    print("  note: counts use today's ChEMBL; the committed raw pull may predate a few records")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", default=None,
                   help="comma list tag=CHEMBLID (default: the 7 ChEMBL panel targets)")
    p.add_argument("--overlap", default=None, metavar="TAG",
                   help="also page through censored records for TAG and match the clean CSV")
    p.add_argument("--repo", default=None)
    args = p.parse_args()

    targets = PANEL
    if args.targets:
        targets = dict(kv.split("=", 1) for kv in args.targets.split(","))
    print_counts(targets)
    if args.overlap:
        if args.overlap not in targets:
            raise SystemExit(f"--overlap {args.overlap} is not in --targets")
        repo = args.repo or find_repo_root(os.path.dirname(__file__))
        overlap(args.overlap, targets[args.overlap], repo)


if __name__ == "__main__":
    main()
