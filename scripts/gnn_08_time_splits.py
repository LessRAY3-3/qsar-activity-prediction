"""GNN step 8: publication-year (time) train/valid/test split.

What / why:
  random and scaffold splits let the model train on molecules published
  AFTER the molecules it is tested on - the score is optimistic for the
  real use case (predicting activity of genuinely new compounds).  A time
  split cuts the dataset at publication-year cutoffs: everything measured
  early trains the model, the newest documents become validation/test.

Year assignment
  One molecule can appear in many documents.  Its year is the MINIMUM
  document_year over all of its activity rows: the earliest publication
  in which it was measured ("known since").  Rows whose document has no
  year are ignored; a molecule with no joinable year stays out of the
  year-ranked pools (it ends up in train).  If more than 1% of the
  fingerprints' SMILES cannot be joined to a year at all, the split is
  refused (SystemExit) - silent shrinkage of the dataset would make
  downstream metrics incomparable.

  The year join runs on 02's standardized parent SMILES (largest
  fragment + uncharge), not on raw strings: the fingerprint npz exports
  parent_smiles (03) while raw activities still carry salt/charge forms,
  so a string-level reindex would miss every multi-fragment compound,
  drop year coverage below the thresholds above and pin/refuse the split.

Split (n = molecule count)
  test  : years counted from the newest down until the cumulative count
          reaches >= 0.2n; cutoff_t = the year where accumulation stops;
          test = all molecules with year >= cutoff_t.  If a single year
          alone exceeds 0.2n it becomes cutoff_t and the test fraction is
          slightly above 0.2 (a note is printed).
  valid : same procedure inside the remaining pool (year < cutoff_t),
          target = 0.1 * pool size; valid = year >= cutoff_v.
  train : everything else, verified disjoint+exhaustive via
          gnn_01_make_splits.check_disjoint.

Guards (write {TAG}_time_NA.json instead of an npz, exit 0)
  * year span max(year)-min(year) < 8  -> too few time points to cut on
  * fewer than 100 molecules in the BUSIEST year (the year holding the
    most molecules) -> the yearly counts are too thin for stable cutoffs.
    NB: this is deliberately NOT "molecules in the newest year": on real
    ChEMBL data the newest year is often a 1-3 molecule straggler, which
    would NA nearly every tag; the busiest year is the density measure
    that matters here.

Outputs
  data/processed/splits/{TAG}_time.npz   train_idx/valid_idx/test_idx
      (int, row order of {TAG}_fingerprints.npz == {TAG}_pic50_clean.csv),
      scaffold_smiles, smiles, year (float32, per molecule)
  data/processed/splits/{TAG}_time_NA.json   guard marker (tag, reason,
      year_min, year_max, n_in_latest_year, n_in_busiest_year)
  figures/time_split/{TAG}_timeline.png  year histogram with both cutoffs

Consumers: scripts/04_train_and_evaluate.py --split time and the GNN
trainers read the npz (or skip on the NA marker).

CLI: --tag (default QSAR_TAG env, then "egfr"), style of 02_clean_data.py.
"""
import argparse
import importlib
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn_01_make_splits import check_disjoint
from qsar_common import murcko_scaffold_list

# 02's parent standardization (LargestFragmentChooser + Uncharger), reused
# verbatim so the year join keys on exactly the same unit 02/03 export.
_parent_smiles = importlib.import_module("02_clean_data")._parent_smiles

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace | ...
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
IN_ACTIVITIES = os.path.join(BASE, "data", "raw", f"{TAG}_activities.csv")
IN_YEARS = os.path.join(BASE, "data", "raw", "document_years.csv")
SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")
FIG_DIR = os.path.join(BASE, "figures", "time_split")

TEST_FRAC = 0.2        # target size of the test set (fraction of all molecules)
VALID_FRAC = 0.1       # target size of the valid set (fraction of the pool)
MIN_YEAR_COVERAGE = 0.99   # above this fraction of dated SMILES, split as-is
FILL_YEAR_COVERAGE = 0.95  # between this and MIN: undated SMILES are pinned to
                           # year_min (train side); below: refuse to split
MIN_YEAR_SPAN = 8          # max(year) - min(year) below this -> NA marker
MIN_N_BUSIEST_YEAR = 100   # busiest year must hold at least this many molecules


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"),
                   help="dataset tag (default: QSAR_TAG env var, else egfr)")
    return p.parse_args(argv)


def _require(path, what, hint=None):
    if not os.path.exists(path):
        msg = f"missing {what}: {path}"
        if hint:
            msg += f" - {hint}"
        raise SystemExit(msg)


def read_inputs(in_npz, in_activities, in_years):
    """-> (smiles ndarray, activities df, document-years df) with checks."""
    _require(in_npz, "fingerprint cache (run scripts/03_featurize.py)")
    _require(in_activities, "raw activities CSV (run scripts/01e_download_chembl.py)")
    _require(in_years, "document years table", "run scripts/fetch_document_years.py first")

    d = np.load(in_npz)
    if "smiles" not in d.files:
        raise SystemExit(f"{in_npz}: missing required key 'smiles' (have {d.files})")
    smiles = d["smiles"]
    d.close()

    activities = pd.read_csv(in_activities)
    need = {"canonical_smiles", "document_chembl_id"}
    if not need.issubset(activities.columns):
        raise SystemExit(f"{in_activities}: missing columns {sorted(need - set(activities.columns))}")

    years = pd.read_csv(in_years)
    need = {"document_chembl_id", "document_year"}
    if not need.issubset(years.columns):
        raise SystemExit(f"{in_years}: missing columns {sorted(need - set(years.columns))}")
    return smiles, activities, years


def _parent_keys(values):
    """02's parent standardization for every entry (memoized), for joining.

    Both join sides go through this so the key space is identical whether
    the npz already holds parent_smiles (03) or a raw canonical string.
    """
    cache = {}
    keys = []
    for v in values:
        s = str(v)
        if s not in cache:
            cache[s] = _parent_smiles(s)[0]
        keys.append(cache[s])
    return keys


def assign_years(smiles, activities, years):
    """Per-SMILES earliest publication year (float array, NaN = no year).

    Joins activities -> document_years on document_chembl_id, then takes
    the minimum year over all rows of each standardized parent SMILES.
    The reindex key is 02's parent form on BOTH sides: raw activities carry
    salt/charge variants while the fingerprint npz exports parent_smiles,
    so plain-string keys would leave those molecules undated.
    """
    acts = activities.dropna(subset=["canonical_smiles", "document_chembl_id"]).copy()
    acts["canonical_smiles"] = acts["canonical_smiles"].astype(str)
    acts["document_chembl_id"] = acts["document_chembl_id"].astype(str)

    docs = years.dropna(subset=["document_chembl_id", "document_year"]).copy()
    docs["document_year"] = pd.to_numeric(docs["document_year"], errors="coerce")
    docs = docs.dropna(subset=["document_year"])
    docs["document_chembl_id"] = docs["document_chembl_id"].astype(str)

    merged = acts.merge(docs, on="document_chembl_id", how="left")
    merged["join_smiles"] = _parent_keys(merged["canonical_smiles"])
    per_smiles = merged.groupby("join_smiles")["document_year"].min()
    keys = _parent_keys(np.asarray(smiles, dtype=str))
    out = per_smiles.reindex(keys).to_numpy(dtype=float)
    return np.array(out, dtype=float, copy=True)  # writable copy: ensure_coverage fills NaNs in place


def ensure_coverage(year, smiles):
    """Check the dated fraction; pin sparse undated SMILES to year_min (train side).

    Coverage > MIN_YEAR_COVERAGE -> split as-is. Between FILL_YEAR_COVERAGE and
    MIN_YEAR_COVERAGE -> undated SMILES (docs with no recorded year in ChEMBL)
    are assigned the dataset minimum year, i.e. they always land in train.
    Below FILL_YEAR_COVERAGE -> refuse: the timeline would be mostly invented.
    Returns the number of SMILES with a real (non-filled) year.
    """
    n = len(year)
    known = ~np.isnan(year)
    k = int(known.sum())
    if k > MIN_YEAR_COVERAGE * n:
        return k
    if k >= FILL_YEAR_COVERAGE * n:
        year_min = int(np.nanmin(year))
        n_fill = int((~known).sum())
        year[~known] = year_min
        print(f"  {n_fill} of {n} smiles have no publication year "
              f"(coverage {k / n:.1%}); pinning them to year_min={year_min} "
              "(they fall on the train side)")
        return k
    missing = np.asarray(smiles, dtype=str)[~known]
    print(f"  {n - k} of {n} smiles have no publication year "
          f"(coverage {k / n:.1%}, need > {MIN_YEAR_COVERAGE:.0%}); e.g. "
          + ", ".join(missing[:5]))
    raise SystemExit(f"publication-year coverage {k}/{n} ({k / n:.1%}) is below "
                     f"the required {FILL_YEAR_COVERAGE:.0%} - not splitting")


def year_stats(year):
    """-> (year_min, year_max, n_in_latest_year, busiest_year, n_in_busiest_year)."""
    known = year[~np.isnan(year)].astype(int)
    uniq, counts = np.unique(known, return_counts=True)
    busiest = int(counts.argmax())
    return (int(uniq[0]), int(uniq[-1]), int(counts[-1]),
            int(uniq[busiest]), int(counts[busiest]))


def cutoff_from_latest(year_values, target):
    """Newest year Y such that count(year >= Y) first reaches >= target.

    Walks years from largest to smallest accumulating molecule counts;
    returns the year at which the cumulative count hits ``target``.
    """
    vals = np.asarray(year_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    uniq, counts = np.unique(vals, return_counts=True)      # ascending
    cum_from_newest = np.cumsum(counts[::-1])
    hit = int(np.searchsorted(cum_from_newest, target))     # first index (newest side)
    if hit >= len(uniq):                                    # target > total (defensive)
        hit = len(uniq) - 1
    return int(uniq[::-1][hit])


def time_split(year, test_frac=TEST_FRAC, valid_frac=VALID_FRAC):
    """-> (train_idx, valid_idx, test_idx, cutoff_v, cutoff_t).

    Molecules without a year can never satisfy year >= cutoff, so they
    fall into train (check_disjoint still sees a full partition).
    """
    n = len(year)
    cutoff_t = cutoff_from_latest(year, test_frac * n)
    if cutoff_t is None:
        raise SystemExit("no publication years at all - cannot time-split")
    in_test = year >= cutoff_t
    n_top = int(np.count_nonzero(year == cutoff_t))
    if n_top > test_frac * n:
        print(f"  note: year {cutoff_t} alone holds {n_top} molecules (> {test_frac:.0%} of n) "
              f"- test fraction will be slightly above {test_frac:.0%}")

    pool = ~in_test
    cutoff_v = cutoff_from_latest(year[pool], valid_frac * int(pool.sum()))
    if cutoff_v is None:
        raise SystemExit("no dated molecules left outside the test set")
    in_valid = pool & (year >= cutoff_v)
    in_train = ~in_test & ~in_valid

    return (np.flatnonzero(in_train), np.flatnonzero(in_valid), np.flatnonzero(in_test),
            int(cutoff_v), int(cutoff_t))


def write_na_marker(split_dir, tag, reason, year_min, year_max, n_latest, n_busiest):
    path = os.path.join(split_dir, f"{tag}_time_NA.json")
    payload = {
        "tag": tag,
        "reason": reason,
        "year_min": year_min,
        "year_max": year_max,
        "n_in_latest_year": n_latest,
        "n_in_busiest_year": n_busiest,
    }
    os.makedirs(split_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"  wrote {os.path.relpath(path, BASE)}")
    return path


def plot_timeline(year, cutoff_v, cutoff_t, n_train, n_valid, n_test, tag, out_png):
    known = year[~np.isnan(year)].astype(int)
    bins = np.arange(known.min(), known.max() + 2)
    counts, edges = np.histogram(known, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(edges[:-1], counts, width=0.9, color="steelblue", edgecolor="none")
    ax.axvline(cutoff_v, color="tab:orange", ls="--", lw=1.5,
               label=f"valid cutoff {cutoff_v}")
    ax.axvline(cutoff_t, color="tab:red", ls="--", lw=1.5,
               label=f"test cutoff {cutoff_t}")
    ax.set_xlabel("publication year")
    ax.set_ylabel("molecules")
    ax.set_title(f"{tag} time split: train={n_train} valid={n_valid} test={n_test}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run(tag, in_npz, in_activities, in_years, split_dir, fig_dir):
    smiles, activities, years_df = read_inputs(in_npz, in_activities, in_years)
    n = len(smiles)
    print(f"[{tag}] {n} molecules")

    year = assign_years(smiles, activities, years_df)
    n_with_year = ensure_coverage(year, smiles)

    year_min, year_max, n_latest, busiest_year, n_busiest = year_stats(year)
    print(f"  year coverage: {n_with_year}/{n} ({n_with_year / n:.1%}) | "
          f"{year_min}-{year_max} | busiest year {busiest_year}: {n_busiest} molecules")

    reasons = []
    if year_max - year_min < MIN_YEAR_SPAN:
        reasons.append(f"year span {year_max - year_min} < {MIN_YEAR_SPAN} ({year_min}-{year_max})")
    if n_busiest < MIN_N_BUSIEST_YEAR:
        reasons.append(f"largest year bucket ({busiest_year}) holds only {n_busiest} "
                       f"molecules (< {MIN_N_BUSIEST_YEAR})")
    if reasons:
        reason = "; ".join(reasons)
        print(f"  time split not feasible: {reason}")
        write_na_marker(split_dir, tag, reason, year_min, year_max, n_latest, n_busiest)
        return None

    train_idx, valid_idx, test_idx, cutoff_v, cutoff_t = time_split(year)
    check_disjoint(n, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

    os.makedirs(split_dir, exist_ok=True)
    out_npz = os.path.join(split_dir, f"{tag}_time.npz")
    np.savez(
        out_npz,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        scaffold_smiles=murcko_scaffold_list(smiles),
        smiles=smiles,
        year=year.astype(np.float32),
    )
    print(f"  saved {os.path.relpath(out_npz, BASE)}")

    out_png = os.path.join(fig_dir, f"{tag}_timeline.png")
    plot_timeline(year, cutoff_v, cutoff_t, len(train_idx), len(valid_idx), len(test_idx),
                  tag, out_png)
    print(f"  saved {os.path.relpath(out_png, BASE)}")

    print(f"[{tag}] n={n} years={year_min}-{year_max} cutoff_v={cutoff_v} cutoff_t={cutoff_t} "
          f"train={len(train_idx)} ({len(train_idx) / n:.1%}) "
          f"valid={len(valid_idx)} ({len(valid_idx) / n:.1%}) "
          f"test={len(test_idx)} ({len(test_idx) / n:.1%})")
    return None


def main(argv=None):
    args = parse_args(argv)
    in_npz, in_activities = IN_NPZ, IN_ACTIVITIES
    if args.tag != TAG:  # --tag overrides QSAR_TAG: rebuild the tag-specific paths
        in_npz = os.path.join(BASE, "data", "processed", f"{args.tag}_fingerprints.npz")
        in_activities = os.path.join(BASE, "data", "raw", f"{args.tag}_activities.csv")
    return run(args.tag, in_npz, in_activities, IN_YEARS, SPLIT_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
