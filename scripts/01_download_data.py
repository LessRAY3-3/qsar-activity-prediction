"""Step 1: Download bioactivity data for EGFR (CHEMBL203) from ChEMBL via API.

Why EGFR / CHEMBL203:
  EGFR is one of the most studied drug targets, with tens of thousands of
  curated IC50 measurements in ChEMBL -> plenty of training data.

Server-side filters:
  - target_chembl_id = CHEMBL203 (human EGFR)
  - standard_type     = IC50    (single, comparable potency readout)
  - standard_units    = nM      (keep one unit so values are comparable)
assay_type (B = binding, F = functional) is filtered client-side afterwards.

Note: we call the REST API with plain `requests` (verified working) instead
of the chembl_webresource_client wrapper, which returned server error pages.
"""
import os
import time
import pandas as pd
import requests

TARGET_ID = "CHEMBL203"  # Epidermal growth factor receptor (human)
API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_CSV = os.path.join(OUT_DIR, "egfr_chembl_activities.csv")

COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_type",
    "assay_description",
    "target_pref_name",
    "bao_label",
    "document_chembl_id",
]


def fetch_all():
    rows, page, offset, limit = [], 0, 0, 1000
    while True:
        params = {
            "target_chembl_id": TARGET_ID,
            "standard_type": "IC50",
            "standard_units": "nM",
            "limit": limit,
            "offset": offset,
        }
        for attempt in range(3):
            try:
                r = requests.get(API, params=params, timeout=60)
                r.raise_for_status()
                js = r.json()
                break
            except Exception as e:
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(5)
        else:
            raise RuntimeError("API failed after 3 retries")

        batch = js.get("activities", [])
        print(f"  page {page}: {len(batch)} rows (total so far {offset + len(batch)})")
        for a in batch:
            rows.append({c: a.get(c) for c in COLUMNS})

        offset += len(batch)
        if not js.get("page_meta", {}).get("next") or not batch:
            break
        page += 1
        time.sleep(0.3)  # be polite to the public API
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Querying ChEMBL API for {TARGET_ID} (EGFR), IC50, nM ...")
    rows = fetch_all()
    df = pd.DataFrame(rows)
    print(f"Downloaded {len(df)} activity rows.")

    df = df[df["assay_type"].isin(["B", "F"])].copy()
    print(f"After keeping assay_type in [B, F]: {len(df)} rows.")

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved raw data -> {os.path.abspath(OUT_CSV)}")


if __name__ == "__main__":
    main()
