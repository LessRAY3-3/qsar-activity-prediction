"""Step 1 (generic): download ChEMBL activity data for ANY target.

Generalises 01d_download_incremental.py (hardcoded CHEMBL203 / EGFR, 8000-row
early stop) into a CLI-driven downloader for multi-target campaigns.

Design tradeoffs (inherited from 01d, still deliberate here):
  * API is intermittently 500 (bad backend nodes) -> retry each page x8.
  * Offset windows can be poisoned (a specific record crashes the serializer)
    -> after 8 failed tries, SKIP that window and keep going; partial data is
    acceptable. >6 skipped windows -> abandon that assay_type entirely.
  * Rows were lost when the script died mid-run -> append+flush every page.
  * Two passes (assay_type=B first, then F): separate result sets dodge the
    poisoned record and give broader assay coverage. Offsets accumulate
    independently within each pass.
  * --max-rows 0 means "pull everything" (no early stop); a positive value
    reproduces 01d's early-stop behaviour for quick smoke tests.

A limit=1 probe up front prints page_meta.total_count so the expected row
count is known before the long crawl starts.

Output: data/raw/{tag}_activities.csv  (deleted and rebuilt on every run)
"""
import argparse
import os
import time

import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MAX_TRIES_PER_PAGE = 8
MAX_SKIPPED_WINDOWS = 6   # abandon this assay_type beyond this many skips
SLEEP_BETWEEN_PAGES = 0.3

COLUMNS = [
    "molecule_chembl_id", "canonical_smiles", "standard_type",
    "standard_value", "standard_units", "standard_relation",
    "data_validity_comment", "pchembl_value", "assay_type",
    "assay_description", "target_pref_name", "bao_label", "document_chembl_id",
    "activity_id",
]

ASSAY_TYPES = ["B", "F"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-chembl-id", required=True,
                   help="ChEMBL target id, e.g. CHEMBL251 (A2A receptor)")
    p.add_argument("--tag", required=True,
                   help="dataset tag; output -> data/raw/{tag}_activities.csv")
    p.add_argument("--standard-type", default="IC50")
    p.add_argument("--units", default="nM")
    p.add_argument("--max-rows", type=int, default=0,
                   help="early stop once this many raw rows are collected; 0 = pull all")
    p.add_argument("--page", type=int, default=500, help="rows per API page")
    return p.parse_args()


def base_params(args):
    return {"target_chembl_id": args.target_chembl_id,
            "standard_type": args.standard_type,
            "standard_units": args.units}


def get_page(params, max_tries=MAX_TRIES_PER_PAGE):
    for t in range(1, max_tries + 1):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            print(f"    HTTP {r.status_code} (try {t})", flush=True)
        except Exception as e:
            print(f"    {e} (try {t})", flush=True)
        time.sleep(min(8 * t, 45))
    return None


def probe_total_count(args):
    """One limit=1 query to learn the expected row count up front."""
    params = dict(base_params(args), limit=1, offset=0)
    js = get_page(params)
    if js is None:
        print("  probe failed; total_count unknown, crawling blindly.", flush=True)
        return None
    total = js.get("page_meta", {}).get("total_count")
    print(f"  expected rows (total_count, all assay types): {total}", flush=True)
    return total


def fetch_assay_type(args, assay_type, out_csv, collected):
    """Crawl one assay_type; offset accumulates within this pass only.

    Returns (rows_collected_in_this_pass, skipped_windows, api_total_count).
    """
    offset, skipped, pass_rows, api_total = 0, [], 0, None
    while True:
        if args.max_rows > 0 and collected + pass_rows >= args.max_rows:
            break
        params = dict(base_params(args), assay_type=assay_type,
                      limit=args.page, offset=offset)
        js = get_page(params)
        if js is None:
            print(f"  SKIP window [{offset}, {offset + args.page})", flush=True)
            skipped.append((offset, offset + args.page))
            offset += args.page
            if len(skipped) > MAX_SKIPPED_WINDOWS:
                print("  too many skips, moving on.", flush=True)
                break
            continue
        batch = js.get("activities", [])
        if not batch:
            break
        if api_total is None:
            api_total = js.get("page_meta", {}).get("total_count")
        df = pd.DataFrame([{c: a.get(c) for c in COLUMNS} for a in batch])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        pass_rows += len(batch)
        print(f"  {assay_type}: +{len(batch)} (total {collected + pass_rows}, offset {offset})",
              flush=True)
        offset += len(batch)
        if not js.get("page_meta", {}).get("next"):
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return pass_rows, skipped, api_total


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"{args.tag}_activities.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    print(f"Target {args.target_chembl_id} | {args.standard_type}/{args.units} "
          f"| page={args.page} | max_rows={'all' if args.max_rows == 0 else args.max_rows}",
          flush=True)
    print("Probing expected row count ...", flush=True)
    probe_total_count(args)

    collected, all_skipped, coverage = 0, [], {}
    for at in ASSAY_TYPES:
        print(f"Fetching assay_type={at} ...", flush=True)
        pass_rows, sk, api_total = fetch_assay_type(args, at, out_csv, collected)
        collected += pass_rows
        all_skipped += sk
        coverage[at] = {"rows": pass_rows, "api_total_count": api_total}
        if args.max_rows > 0 and collected >= args.max_rows:
            break

    print(f"\nDONE: {collected} raw rows -> {out_csv}", flush=True)
    print(f"skipped windows: {all_skipped}", flush=True)
    print("assay_type coverage:", flush=True)
    for at, info in coverage.items():
        exp = info["api_total_count"]
        got = info["rows"]
        frac = f"{got}/{exp} ({got / exp:.1%})" if exp else f"{got}/?"
        print(f"  {at}: {got} rows collected, API total_count={exp}  [{frac}]", flush=True)


if __name__ == "__main__":
    main()
