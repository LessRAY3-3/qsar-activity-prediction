"""Step 1 (incremental): resume-friendly EGFR download with skip-ahead.

Learnings from earlier runs:
  * API is intermittently 500 (bad backend nodes) -> retry each page.
  * Offset ~2500 is consistently poisoned (a specific record seems to crash
    the serializer) -> after 8 failed tries, SKIP that 500-row window and
    keep going; partial data is fine (we need 1000+ compounds, not all).
  * Rows were lost when the script died mid-run -> append+flush every page.

Fetches assay_type=B first, then F (separate result sets also dodge the
poisoned record), stops early once TARGET_ROWS raw rows are collected.
"""
import os, time
import pandas as pd
import requests

TARGET_ID = "CHEMBL203"
API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_CSV = os.path.join(OUT_DIR, "egfr_chembl_activities.csv")
TARGET_ROWS = 8000          # early stop: plenty for 1000+ unique compounds
PAGE = 500
MAX_TRIES_PER_PAGE = 8

COLUMNS = [
    "molecule_chembl_id", "canonical_smiles", "standard_type",
    "standard_value", "standard_units", "pchembl_value", "assay_type",
    "assay_description", "target_pref_name", "bao_label", "document_chembl_id",
]


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


def fetch_assay_type(assay_type, total_rows):
    offset, skipped = 0, []
    while total_rows < TARGET_ROWS:
        params = {"target_chembl_id": TARGET_ID, "standard_type": "IC50",
                  "standard_units": "nM", "assay_type": assay_type,
                  "limit": PAGE, "offset": offset}
        js = get_page(params)
        if js is None:
            print(f"  SKIP window [{offset}, {offset + PAGE})", flush=True)
            skipped.append((offset, offset + PAGE))
            offset += PAGE
            if len(skipped) > 6:
                print("  too many skips, moving on.", flush=True)
                break
            continue
        batch = js.get("activities", [])
        if not batch:
            break
        df = pd.DataFrame([{c: a.get(c) for c in COLUMNS} for a in batch])
        df.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
        total_rows += len(batch)
        print(f"  {assay_type}: +{len(batch)} (total {total_rows}, offset {offset})", flush=True)
        offset += len(batch)
        if not js.get("page_meta", {}).get("next"):
            break
        time.sleep(0.3)
    return total_rows, skipped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    total, all_skipped = 0, []
    for at in ["B", "F"]:
        print(f"Fetching assay_type={at} ...", flush=True)
        total, sk = fetch_assay_type(at, total)
        all_skipped += sk
        if total >= TARGET_ROWS:
            break
    print(f"DONE: {total} raw rows -> {OUT_CSV}; skipped windows: {all_skipped}", flush=True)


if __name__ == "__main__":
    main()
