"""Step 1 (robust): download EGFR (CHEMBL203) IC50 data with PER-PAGE retries.

The ChEMBL API was returning intermittent 500s: some backend nodes healthy,
some not. Strategy: paginate 500 rows/page; on failure retry the SAME page
up to 10 times with backoff, then keep going. Progress is saved after every
page so nothing is lost.
"""
import os, time
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


def get_page(offset, limit=500, max_tries=10):
    params = {"target_chembl_id": TARGET_ID, "standard_type": "IC50",
              "standard_units": "nM", "limit": limit, "offset": offset}
    for t in range(1, max_tries + 1):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            print(f"  offset={offset}: HTTP {r.status_code} (try {t}/{max_tries})", flush=True)
        except Exception as e:
            print(f"  offset={offset}: {e} (try {t}/{max_tries})", flush=True)
        time.sleep(min(10 * t, 60))
    raise RuntimeError(f"page at offset={offset} failed after {max_tries} tries")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, offset, limit = [], 0, 500
    while True:
        js = get_page(offset, limit)
        batch = js.get("activities", [])
        print(f"  page ok: +{len(batch)} rows (cumulative {offset + len(batch)})", flush=True)
        rows += [{c: a.get(c) for c in COLUMNS} for a in batch]
        offset += len(batch)
        if not js.get("page_meta", {}).get("next") or not batch:
            break
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    df = df[df["assay_type"].isin(["B", "F"])].copy()
    df.to_csv(OUT_CSV, index=False)
    print(f"SUCCESS: {len(df)} rows -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
