"""Retry ChEMBL API download with backoff (API was returning 500s)."""
import os, sys, time
import pandas as pd
import requests

TARGET_ID = "CHEMBL203"
API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_CSV = os.path.join(OUT_DIR, "egfr_chembl_activities.csv")

COLUMNS = [
    "molecule_chembl_id", "canonical_smiles", "standard_type",
    "standard_value", "standard_units", "pchembl_value", "assay_type",
    "assay_description", "target_pref_name", "bao_label", "document_chembl_id",
]


def fetch_all():
    rows, page, offset, limit = [], 0, 0, 500
    while True:
        params = {"target_chembl_id": TARGET_ID, "standard_type": "IC50",
                  "standard_units": "nM", "limit": limit, "offset": offset}
        r = requests.get(API, params=params, timeout=90)
        r.raise_for_status()
        js = r.json()
        batch = js.get("activities", [])
        print(f"  page {page}: {len(batch)} rows (cumulative {offset + len(batch)})", flush=True)
        rows += [{c: a.get(c) for c in COLUMNS} for a in batch]
        offset += len(batch)
        if not js.get("page_meta", {}).get("next") or not batch:
            break
        page += 1
        time.sleep(0.3)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for attempt in range(1, 26):
        try:
            print(f"Attempt {attempt}/25 ...", flush=True)
            rows = fetch_all()
            df = pd.DataFrame(rows)
            df = df[df["assay_type"].isin(["B", "F"])].copy()
            df.to_csv(OUT_CSV, index=False)
            print(f"SUCCESS: {len(df)} rows -> {OUT_CSV}", flush=True)
            return
        except Exception as e:
            print(f"  failed: {e}", flush=True)
            time.sleep(120)
    print("API still down after all retries.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
