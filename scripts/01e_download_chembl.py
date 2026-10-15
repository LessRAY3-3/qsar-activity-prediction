"""Step 1 (generic): download ChEMBL activity data for ANY target.

Generalises 01d_download_incremental.py (hardcoded CHEMBL203 / EGFR, 8000-row
early stop) into a CLI-driven downloader for multi-target campaigns.

Each page is retried. Missing pages, inconsistent pagination or changed totals
abort the download. A complete crawl (or an explicitly requested --max-rows
cap) is staged beside the destination, then atomically replaces it. This is
a fresh download, not a resume operation; failures leave the previous CSV intact.
"""
import argparse
import os
import time
import tempfile

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
    args = p.parse_args()
    if args.page <= 0 or args.max_rows < 0:
        p.error('--page must be positive and --max-rows nonnegative')
    return args


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
            raise RuntimeError(f"{assay_type}: page at offset {offset} failed after retries")
        batch = js.get("activities")
        meta = js.get("page_meta", {})
        total = meta.get("total_count")
        if not isinstance(batch, list) or type(total) is not int or total < 0:
            raise ValueError("ChEMBL response lacks activities/total_count")
        if api_total is None:
            api_total = total
        if total != api_total:
            raise ValueError("ChEMBL total_count changed during crawl; retry a fresh download")
        end = offset + len(batch)
        if end > total or (not batch and offset < total):
            raise ValueError("ChEMBL empty/truncated or oversized page before reported end")
        if bool(meta.get("next")) != (end < total):
            raise ValueError("ChEMBL pagination disagrees with total_count")
        if not batch:
            break
        # An explicit cap is intentional sampling, never a completeness claim.
        if args.max_rows:
            batch = batch[:args.max_rows - collected - pass_rows]
        df = pd.DataFrame([{c: a.get(c) for c in COLUMNS} for a in batch])
        df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0, index=False)
        pass_rows += len(batch)
        print(f"  {assay_type}: +{len(batch)} (total {collected + pass_rows}, offset {offset})",
              flush=True)
        offset += len(batch)
        if not js.get("page_meta", {}).get("next"):
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return pass_rows, skipped, api_total


def run_download(args, out_csv):
    """Stage a fresh crawl and replace the destination only after validation."""
    directory = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(out_csv) + '.', suffix='.part',
                                     dir=directory)
    os.close(fd)
    try:
        collected, coverage = 0, {}
        for at in ASSAY_TYPES:
            count, skipped, total = fetch_assay_type(args, at, temporary, collected)
            if skipped:
                raise RuntimeError("incomplete crawl: skipped API windows")
            collected += count
            coverage[at] = {"rows": count, "api_total_count": total}
            if args.max_rows > 0 and collected >= args.max_rows:
                break
        if collected == 0:
            raise ValueError("no activities downloaded; preserving previous CSV")
        os.replace(temporary, out_csv)
        print(f"DONE: {collected} rows -> {out_csv}; coverage={coverage}; "
              f"requested_max_rows={args.max_rows}", flush=True)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    args = parse_args()
    run_download(args, os.path.join(OUT_DIR, f"{args.tag}_activities.csv"))


if __name__ == "__main__":
    main()
