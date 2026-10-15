"""Legacy EGFR entry point: fresh capped download, not resume.

Uses the same validated, atomic downloader as the generic entry point.
The 8,000-row cap is intentional; failed pages abort without replacing old data.
"""
import importlib
import os
from argparse import Namespace

TARGET_ID = "CHEMBL203"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_CSV = os.path.join(OUT_DIR, "egfr_activities.csv")
TARGET_ROWS = 8000
PAGE = 500
COLUMNS = importlib.import_module("01e_download_chembl").COLUMNS


def main():
    downloader = importlib.import_module("01e_download_chembl")
    args = Namespace(target_chembl_id=TARGET_ID, tag="egfr", standard_type="IC50",
                     units="nM", max_rows=TARGET_ROWS, page=PAGE)
    downloader.run_download(args, OUT_CSV)


if __name__ == "__main__":
    main()
