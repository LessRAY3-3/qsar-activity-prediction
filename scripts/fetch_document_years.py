"""Fetch the publication year of every ChEMBL document behind the activity sets.

P8 (time-based split) needs each document's year, but the raw activity CSVs
only carry ``document_chembl_id``.  One
``GET activity.json?document_chembl_id=...&limit=1`` probe returns
``document_year`` inside the first activity record, so the dedicated document
endpoint is unnecessary.

Design tradeoffs (retry policy inherited from 01e_download_chembl.py):
  * The API is intermittently 500 (bad backend nodes) -> up to 8 tries per
    doc, backoff ``sleep(min(8 * t, 45))``, 90 s timeout.  A doc that exhausts
    its retries (or whose year cannot be parsed) is written with an empty year
    and listed in the end-of-run summary instead of aborting the run.
  * ~4-5.5k docs total -> 4 workers by default, one thread-local
    ``requests.Session`` per worker thread; rows are appended and flushed as
    results arrive so an interrupted run loses nothing.
  * Re-entrant: docs already recorded with a year are skipped on start-up,
    and empty-year rows left by a previous run are compacted away first, so a
    re-run retries exactly the failures.  Each doc keeps at most one row.
  * A doc with no activity records at all counts as missing (should not
    happen: the ids come from the activity table itself).

Output: data/raw/document_years.csv
        (columns document_chembl_id,document_year; year is an int)
"""
import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_TAGS = ["a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2"]
MAX_TRIES = 8
TIMEOUT = 90
LOG_TRIES = 3         # only the first few retries are printed per doc
PROGRESS_EVERY = 50
HEADER = ["document_chembl_id", "document_year"]

_thread_local = threading.local()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", default=",".join(DEFAULT_TAGS),
                   help="comma-separated dataset tags (default: all eight)")
    p.add_argument("--limit", type=int, default=0,
                   help="only query the first N not-yet-recorded docs; 0 = no limit")
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent API queries (keep at 4 to stay polite to EBI)")
    p.add_argument("--out", default=os.path.join(RAW_DIR, "document_years.csv"),
                   help="output CSV; rows already carrying a year are skipped (re-entrant)")
    return p.parse_args()


def get_session():
    """One requests.Session per worker thread (connection reuse, no sharing)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def collect_doc_ids(raw_dir, tags):
    """Unique document ids across the tags' raw CSVs; first occurrence wins."""
    doc_ids, seen = [], set()
    for tag in tags:
        path = os.path.join(raw_dir, f"{tag}_activities.csv")
        if not os.path.exists(path):
            raise SystemExit(f"raw activity file not found for tag '{tag}': {path}")
        series = pd.read_csv(path, usecols=["document_chembl_id"])["document_chembl_id"]
        for doc_id in series.dropna().astype(str):
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
    return doc_ids


def load_done_ids(out_csv):
    """Ids already recorded with a non-empty year; compact the file first.

    Rows with an empty year are leftovers from failed fetches: they are
    dropped (last row wins for duplicates) so a re-run retries exactly those.
    """
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return set()
    df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    if list(df.columns) != HEADER:
        raise SystemExit(f"{out_csv}: expected columns {HEADER}, got {list(df.columns)}")
    done = df[df["document_year"].str.strip() != ""]
    done = done.drop_duplicates(subset=["document_chembl_id"], keep="last")
    if len(done) != len(df):
        tmp = out_csv + ".tmp"
        done.to_csv(tmp, index=False, header=HEADER)
        os.replace(tmp, out_csv)
    return set(done["document_chembl_id"])


def fetch_document_year(doc_id, session=None, max_tries=MAX_TRIES):
    """-> ("ok", int_year) | ("missing", None) | ("failed", None)."""
    if session is None:
        session = get_session()
    params = {"document_chembl_id": doc_id, "limit": 1}
    for t in range(1, max_tries + 1):
        try:
            r = session.get(API, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict) and "activities" in payload:
                    acts = payload["activities"]
                    if not acts:
                        return "missing", None
                    raw = acts[0].get("document_year")
                    try:
                        return "ok", int(float(raw))
                    except (TypeError, ValueError):
                        print(f"  {doc_id}: unparsable document_year {raw!r}", flush=True)
                        return "failed", None
                elif t <= LOG_TRIES:
                    print(f"  {doc_id}: unexpected JSON body (try {t})", flush=True)
            elif t <= LOG_TRIES:
                print(f"  {doc_id}: HTTP {r.status_code} (try {t})", flush=True)
        except Exception as e:
            if t <= LOG_TRIES:
                print(f"  {doc_id}: {e} (try {t})", flush=True)
        if t < max_tries:
            time.sleep(min(8 * t, 45))
    return "failed", None


def run(args):
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        raise SystemExit("--tags produced no tags")

    doc_ids = collect_doc_ids(RAW_DIR, tags)
    done = load_done_ids(args.out)
    pending = [d for d in doc_ids if d not in done]
    skipped = len(doc_ids) - len(pending)
    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"{len(doc_ids)} unique docs across {len(tags)} tags | "
          f"{skipped} already recorded | {len(pending)} to query "
          f"| workers={args.workers} | out={args.out}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    need_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    stats = {"attempted": len(pending), "skipped": skipped,
             "got": 0, "missing": 0, "failed": 0}
    failed_docs = []

    with open(args.out, "a", newline="") as fh:
        writer = csv.writer(fh)
        if need_header:
            writer.writerow(HEADER)
            fh.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_document_year, d): d for d in pending}
            for n, fut in enumerate(as_completed(futures), 1):
                doc_id = futures[fut]
                try:
                    status, year = fut.result()
                except Exception as e:  # a worker should never raise; be safe
                    status, year = "failed", None
                    print(f"  {doc_id}: unexpected error {e}", flush=True)
                writer.writerow([doc_id, "" if year is None else year])
                fh.flush()
                stats[{"ok": "got", "missing": "missing", "failed": "failed"}[status]] += 1
                if status == "failed":
                    failed_docs.append(doc_id)
                if status != "ok":
                    print(f"  {doc_id}: {status}", flush=True)
                if n % PROGRESS_EVERY == 0:
                    print(f"  {n}/{len(pending)} | got={stats['got']} "
                          f"missing={stats['missing']} failed={stats['failed']}", flush=True)

    print(f"\nDONE: got={stats['got']} missing={stats['missing']} failed={stats['failed']} "
          f"skipped={skipped} attempted={stats['attempted']} -> {args.out}", flush=True)
    if failed_docs:
        print("failed docs (empty year; re-run to retry):", flush=True)
        for doc_id in sorted(failed_docs):
            print(f"  {doc_id}", flush=True)
    return stats


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
