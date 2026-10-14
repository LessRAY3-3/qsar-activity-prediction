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
  year are ignored. At least 95% year coverage is required. Sparse
  undated molecules may provisionally occupy the train side during cutoff
  selection, but are excluded from every model pool if they lack a dated
  exact measurement; this exclusion is explicitly recorded.

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
  train : everything else before label filtering. Each pool then receives
          median exact nM labels from eligible dated records: train strictly
          before cutoff_v; valid strictly before cutoff_t; test at/after
          cutoff_t. Missing eligible labels are excluded. The three pools
          plus excluded_idx partition the original dataset.

First-generation guards (write {TAG}_time_NA.json instead of an npz, exit 0;
  an unavailable regeneration retains an existing split and exits nonzero)
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
      excluded_idx, y (as-of; NaN for exclusions), original_y,
      scaffold_smiles, smiles, year (float32, per molecule), cutoffs,
      label_policy, source/data signatures and label_id; new splits also carry
      integrity_schema and integrity_sha256 covering every payload field
  data/processed/splits/{TAG}_time_NA.json   guard marker (tag, reason,
      year_min, year_max, n_in_latest_year, n_in_busiest_year)
  figures/time_split/{TAG}_timeline.png  year histogram with both cutoffs

Consumers: scripts/04_train_and_evaluate.py --split time and the GNN
trainers read the npz (or skip on the NA marker).

CLI: --tag (default QSAR_TAG env, then "egfr"), style of 02_clean_data.py.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnn_01_make_splits import check_disjoint
from qsar_common import exact_relation_mask, murcko_scaffold_list
from time_labels import (INTEGRITY_SCHEMA, LABEL_POLICY, dataset_signature,
                         label_signature, split_integrity_digest)

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
                           # year_min for cutoff selection; no dated label -> excluded
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
    """Standardize raw activity SMILES once, exactly as the cache producer.

    Cached SMILES already contain 02's parent output and must not pass
    through this function again: the RDKit round trip is not idempotent
    for every molecule (e.g. CHEMBL2035183).
    """
    cache = {}
    keys = []
    for v in values:
        s = str(v)
        if s not in cache:
            cache[s] = _parent_smiles(s)[0]
        keys.append(cache[s])
    return keys



def _activities_with_parent(smiles, activities):
    """Validate cache keys against ALL raw structures before eligibility filtering.

    A missing document/year is allowed; an absent structural key indicates a
    cache/raw mismatch, not missing publication metadata. Never normalize the
    cached keys again, because parent standardization need not be idempotent.
    """
    acts = activities.copy()
    acts["parent"] = _parent_keys(acts["canonical_smiles"])
    raw_keys = {key for key in acts["parent"] if key is not None}
    missing = [(row, key) for row, key in enumerate(np.asarray(smiles, dtype=str))
               if key not in raw_keys]
    if missing:
        details = "; ".join(f"row={row} key={key!r}" for row, key in missing)
        raise ValueError(
            f"{len(missing)} cached parent key(s) absent from standardized raw activities "
            f"(zero-based cache rows): {details}")
    return acts


def assign_years(smiles, activities, years):
    """Per-SMILES earliest publication year (float array, NaN = no year).

    Joins activities -> document_years on document_chembl_id, then takes
    the minimum year over all rows of each standardized parent SMILES.
    Raw salt/charge forms are standardized once, as in 02. The fingerprint
    cache already exports those parent keys; use them unchanged rather than
    standardizing a second time and potentially changing the join key.
    """
    acts = _activities_with_parent(smiles, activities).dropna(
        subset=["canonical_smiles", "document_chembl_id"]).copy()
    acts["canonical_smiles"] = acts["canonical_smiles"].astype(str)
    acts["document_chembl_id"] = acts["document_chembl_id"].astype(str)

    docs = years.dropna(subset=["document_chembl_id", "document_year"]).copy()
    docs["document_year"] = pd.to_numeric(docs["document_year"], errors="coerce")
    docs = docs.dropna(subset=["document_year"])
    docs["document_chembl_id"] = docs["document_chembl_id"].astype(str)

    merged = acts.merge(docs, on="document_chembl_id", how="left")
    per_smiles = merged.groupby("parent")["document_year"].min()
    keys = np.asarray(smiles, dtype=str)
    out = per_smiles.reindex(keys).to_numpy(dtype=float)
    return np.array(out, dtype=float, copy=True)  # writable copy: ensure_coverage fills NaNs in place


def asof_labels(smiles, activities, years, pools, cutoff_v, cutoff_t):
    """Median of dated exact nM measurements available to each pool.

    Train uses years < cutoff_v; validation uses years < cutoff_t; test
    uses dated measurements >= cutoff_t. Undated records never become
    historical labels. Molecules without an eligible label are excluded,
    never assigned a later measurement or silently moved to another pool.
    """
    required = {"standard_value", "standard_units", "standard_relation",
                "canonical_smiles", "document_chembl_id"}
    if not required.issubset(activities.columns):
        raise ValueError(f"as-of labels require columns: {sorted(required - set(activities.columns))}")
    acts = _activities_with_parent(smiles, activities)
    acts["standard_value"] = pd.to_numeric(acts["standard_value"], errors="coerce")
    keep = (exact_relation_mask(acts["standard_relation"])
            & acts["standard_units"].eq("nM")
            & np.isfinite(acts["standard_value"]) & acts["standard_value"].gt(0)
            & acts["canonical_smiles"].fillna("").astype(str).str.strip().ne(""))
    acts = acts.loc[keep].dropna(subset=["document_chembl_id"]).copy()
    docs = years.dropna(subset=["document_chembl_id"]).copy()
    docs["document_year"] = pd.to_numeric(docs["document_year"], errors="coerce")
    docs = docs[np.isfinite(docs["document_year"])].drop_duplicates()
    if docs["document_chembl_id"].duplicated().any():
        raise ValueError("conflicting publication years for the same document")
    acts["document_chembl_id"] = acts["document_chembl_id"].astype(str)
    docs["document_chembl_id"] = docs["document_chembl_id"].astype(str)
    acts = acts.merge(docs, on="document_chembl_id", how="inner", validate="many_to_one")
    acts["pic50"] = -np.log10(acts["standard_value"] * 1e-9)
    keys = np.asarray(smiles, dtype=str)
    y = np.full(len(smiles), np.nan, dtype=np.float32)
    retained = []
    conditions = (acts["document_year"] < cutoff_v,
                  acts["document_year"] < cutoff_t,
                  acts["document_year"] >= cutoff_t)
    for idx, condition in zip(pools, conditions):
        medians = acts.loc[condition].groupby("parent")["pic50"].median()
        values = medians.reindex(np.asarray(keys)[idx]).to_numpy(dtype=float)
        usable = np.isfinite(values)
        y[idx[usable]] = values[usable]
        retained.append(idx[usable])
    excluded = np.flatnonzero(~np.isfinite(y))
    return y, retained, excluded


def ensure_coverage(year, smiles):
    """Check the dated fraction; pin sparse undated SMILES to year_min (train side).

    Coverage > MIN_YEAR_COVERAGE -> split as-is. Between FILL_YEAR_COVERAGE and
    MIN_YEAR_COVERAGE -> undated SMILES (docs with no recorded year in ChEMBL)
    are assigned the dataset minimum year for cutoff selection only.
    asof_labels subsequently excludes them for lack of a dated label.
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
              "(cutoff selection only; no dated label means exclusion)")
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
    provisionally fall into train; asof_labels later excludes undated rows.
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
    old_split = os.path.join(split_dir, f"{tag}_time.npz")
    if os.path.exists(old_split):
        os.unlink(old_split)
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


def _stage_run(tag, in_npz, in_activities, in_years, split_dir, fig_dir):
    # All writes here are confined to the unpublished staging directory.
    write_na_marker(split_dir, tag, "time split regeneration incomplete", None, None, None, None)
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

    y, pools, excluded = asof_labels(
        smiles, activities, years_df, (train_idx, valid_idx, test_idx), cutoff_v, cutoff_t)
    train_idx, valid_idx, test_idx = pools
    if any(len(idx) < 2 for idx in pools):
        write_na_marker(split_dir, tag, "fewer than two dated exact labels in a time pool",
                        year_min, year_max, n_latest, n_busiest)
        return None
    source_hash = hashlib.sha256()
    for path in (in_activities, in_years):
        with open(path, "rb") as fh:
            source_hash.update(fh.read())
    with np.load(in_npz) as fp:
        original_y = fp["y"]
        signature = dataset_signature(smiles, original_y)
    payload = dict(train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx,
                   excluded_idx=excluded, y=y, original_y=original_y, label_policy=LABEL_POLICY,
                   dataset_signature=signature, source_sha256=source_hash.hexdigest(),
                   cutoff_valid=cutoff_v, cutoff_test=cutoff_t,
                   scaffold_smiles=murcko_scaffold_list(smiles), smiles=smiles,
                   year=year.astype(np.float32))
    payload["label_id"] = label_signature(payload)
    payload["integrity_schema"] = INTEGRITY_SCHEMA
    payload["integrity_sha256"] = split_integrity_digest(payload)

    os.makedirs(split_dir, exist_ok=True)
    out_npz = os.path.join(split_dir, f"{tag}_time.npz")
    np.savez(out_npz, **payload)
    na_path = os.path.join(split_dir, f"{tag}_time_NA.json")
    if os.path.exists(na_path):
        os.unlink(na_path)
    print(f"  excluded {len(excluded)} molecules without an eligible dated exact label")
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



def _atomic_replace(source, destination):
    """Copy into the destination filesystem before one atomic replacement."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError(f"refusing to replace symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".time-regenerate-", suffix=".tmp",
                                         dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with Path(source).open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



def _file_hash(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"symlink publication path forbidden: {path}")
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path):
    if not Path(path).exists():
        return None
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino]


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_publication_record(path, payload):
    """Write journal metadata durably before changing a canonical artifact."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _regeneration_lock(split_dir, tag):
    """A crash releases this OS lock; journal recovery never races a live run."""
    import fcntl  # supported campaign hosts are Linux and macOS

    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f".{tag}_time_regeneration.lock"
    if path.is_symlink():
        raise ValueError(f"symlink lock forbidden: {path}")
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{tag}: another time split regeneration is active") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _recover_publication(split_dir, fig_dir, tag):
    """Finish a committed pair or roll back its figure; never guess on conflicts.

    NPZ replacement is the commit point. Its inode distinguishes a completed
    replacement even when the new NPZ bytes equal the old bytes. There can be a
    short mixed-version window before this point; recovery is not multi-file
    instantaneous atomicity. Readers needing a pair should inspect the journal.
    """
    journal = split_dir / f".{tag}_time_publication"
    if not journal.exists():
        return
    if journal.is_symlink():
        raise ValueError(f"symlink journal forbidden: {journal}")
    record = journal / "pending.json"
    if not record.exists():
        # Preparation did not reach its durable write-ahead record. Neither
        # canonical artifact can have been changed by this transaction yet.
        shutil.rmtree(journal)
        return
    state = json.loads(record.read_text())
    target, figure = split_dir / f"{tag}_time.npz", fig_dir / f"{tag}_timeline.png"
    if (state.get("schema") != "qsar-time-publication-v1" or
            state.get("split_path") != str(target.resolve()) or
            state.get("figure_path") != str(figure.resolve())):
        raise ValueError(f"{tag}: pending publication paths/schema differ; inspect {record}")
    actual, identity = _file_hash(target), _file_identity(target)
    committed = actual == state["new_split_sha256"] and identity != state["old_split_identity"]
    if committed:
        if _file_hash(figure) != state["new_figure_sha256"]:
            raise ValueError(f"{tag}: committed figure changed externally; inspect {record}")
        (split_dir / f"{tag}_time_NA.json").unlink(missing_ok=True)
        outcome = "committed"
    else:
        if actual != state["old_split_sha256"] or identity != state["old_split_identity"]:
            raise ValueError(f"{tag}: split changed externally during publication; inspect {record}")
        current_figure = _file_hash(figure)
        if current_figure not in (state["old_figure_sha256"], state["new_figure_sha256"]):
            raise ValueError(f"{tag}: figure changed externally during publication; inspect {record}")
        if current_figure != state["old_figure_sha256"]:
            if state["old_figure_sha256"] is None:
                figure.unlink(missing_ok=True)
            else:
                backup = journal / "previous_timeline.png"
                if _file_hash(backup) != state["old_figure_sha256"]:
                    raise ValueError(f"{tag}: timeline backup is corrupt; inspect {record}")
                _atomic_replace(backup, figure)
        outcome = "rolled_back"
    _sync_directory(split_dir)
    if figure.parent.exists():
        _sync_directory(figure.parent)
    _write_publication_record(split_dir / f".{tag}_time_last_publication.json",
                              {**state, "outcome": outcome})
    shutil.rmtree(journal)
    _sync_directory(split_dir)
    print(f"  {tag}: publication {outcome}")


def _publish_pair(staged_split, staged_figure, split_dir, fig_dir, tag):
    target, figure = split_dir / f"{tag}_time.npz", fig_dir / f"{tag}_timeline.png"
    journal = split_dir / f".{tag}_time_publication"
    state = {"schema": "qsar-time-publication-v1", "split_path": str(target.resolve()),
             "figure_path": str(figure.resolve()), "old_split_sha256": _file_hash(target),
             "old_split_identity": _file_identity(target), "new_split_sha256": _file_hash(staged_split),
             "old_figure_sha256": _file_hash(figure), "new_figure_sha256": _file_hash(staged_figure)}
    journal.mkdir()
    try:
        if state["old_figure_sha256"] is not None:
            _atomic_replace(figure, journal / "previous_timeline.png")
        _write_publication_record(journal / "pending.json", state)
        _sync_directory(split_dir)
        _atomic_replace(staged_figure, figure)
        _sync_directory(figure.parent)
        _atomic_replace(staged_split, target)  # commit point
        _sync_directory(split_dir)
    except BaseException:
        # SIGKILL cannot execute this block; the durable journal is recovered
        # before the next run reads raw inputs. Failed rollback leaves evidence.
        _recover_publication(split_dir, fig_dir, tag)
        raise
    _recover_publication(split_dir, fig_dir, tag)


def run(tag, in_npz, in_activities, in_years, split_dir, fig_dir):
    """Compute first; preserve old outputs on failure and recover interrupted publication.

    A legitimately unavailable first split may publish an NA marker. An
    unavailable replacement fails explicitly and retains the previous split.
    Each file replacement is atomic; the NPZ/timeline pair uses a journal and
    rollback, not a claim of instantaneous multi-file atomicity.
    """
    split_dir, fig_dir = Path(split_dir), Path(fig_dir)
    with _regeneration_lock(split_dir, tag):
        _recover_publication(split_dir, fig_dir, tag)
        with tempfile.TemporaryDirectory(prefix=".time-regenerate-", dir=split_dir.parent) as temporary:
            stage = Path(temporary)
            staged_splits, staged_figures = stage / "splits", stage / "figures"
            _stage_run(tag, in_npz, in_activities, in_years, staged_splits, staged_figures)
            name, marker = f"{tag}_time.npz", f"{tag}_time_NA.json"
            if (staged_splits / marker).exists():
                if (split_dir / name).exists() or (split_dir / name).is_symlink():
                    reason = json.loads((staged_splits / marker).read_text())["reason"]
                    raise ValueError(f"{tag}: replacement time split unavailable ({reason}); existing split retained")
                _atomic_replace(staged_splits / marker, split_dir / marker)
                return None
            _publish_pair(staged_splits / name, staged_figures / f"{tag}_timeline.png",
                          split_dir, fig_dir, tag)
            print(f"  published {split_dir / name}")
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
