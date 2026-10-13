"""Step 2: Clean the raw ChEMBL activity data.

What / why:
  1. Keep only rows whose IC50 unit is exactly 'nM'  -> comparable numbers.
  2. Drop rows with missing standard_value (IC50) or empty canonical_smiles.
  3. Drop non-positive IC50 values (a 0 breaks the log transform).
  4. Convert IC50 (nM) -> pIC50 = -log10(IC50 * 1e-9).
     Why: IC50 spans many orders of magnitude (1 nM ... 100 uM). Raw values
     are heavy-tailed and would dominate model training; pIC50 is roughly
     normal and is THE standard target transform in QSAR.
  5. Censoring filter: keep only rows with standard_relation == "=".
     Why (review finding H1): rows like "IC50 > 10000 nM" are bounds, not
     exact measurements; treating them as exact labels corrupts the
     regression target (e.g. 20.17% of egfr_full compounds had only censored
     records). Policy: relation == "=" is THE inclusion rule. Using
     pchembl_value.notna() instead would be wrong: exact rows can have an
     empty pChEMBL ("Outside typical range"), so an empty pChEMBL does not
     imply censoring (docs/methodology.md: the reverse case -- bounded rows
     carrying a pChEMBL value -- occurs 0 times in the current raw pulls, so
     only that one direction exists). If the input CSV has no
     standard_relation column, the filter is skipped (with a printed note)
     ONLY for genuine non-ChEMBL inputs: TAG == 'bace' or molecule IDs that
     are not ChEMBL-style after stripping whitespace -- no CHEMBL prefix and
     not a bare numeric ChEMBL number (e.g. bace download format, BACE_1
     ids). A ChEMBL-style file without the column predates the 2026-10-11 data
     refresh (e.g. raw files from the artifacts-v1 release); censoring
     cannot be verified on it, so the script aborts with a non-zero exit
     instead of silently reintroducing the H1 censored-label bug. Relation
     values are compared after strip(), so a padded "=" counts as exact
     (a raw-column comparison would drop every padded row and write a
     header-only CSV with exit 0). Degenerate ID columns also fail
     closed: a pandas float-coerced number column ("4069737.0") is still
     ChEMBL-style, and an all-empty ID column (or blank cells mixed with
     numeric IDs) cannot certify a non-ChEMBL input.
  6. Sanity cross-check against ChEMBL's own pchembl_value (should match),
     over ALL retained rows, and explicitly report the retained rows whose
     pchembl_value is empty (exact measurements ChEMBL could not convert,
     e.g. "Outside typical range"). atol=0.02.
  7. Parent standardization (review finding M1) before deduplication:
     strip salts/counterions with rdMolStandardize.LargestFragmentChooser
     and neutralize charges with rdMolStandardize.Uncharger, giving
     parent_smiles. Scope limit: this deliberately does NOT resolve
     tautomer or stereochemical equivalence (consistent with the audit
     definition in docs/reviews/codex-verification-2026-10-10.md), it only
     merges multi-fragment/charge variants of the same structure.
     Why: the same parent compound measured as free base and as a salt was
     counted as two compounds, leaking parent structures across splits
     (<= 2.04% of RF test molecules). Group by parent_smiles and take the
     MEDIAN pIC50 (robust to outliers), keeping canonical_smiles (first, for
     provenance), n_measurements (records per parent) and
     n_structural_variants (distinct canonical_smiles per parent).
     Unparseable SMILES fall back to the raw string and are counted.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from qsar_common import exact_relation_mask

RDLogger.DisableLog("rdApp.*")  # per-molecule parse issues are counted and printed instead

BASE = os.path.join(os.path.dirname(__file__), "..")
TAG = os.environ.get("QSAR_TAG", "egfr")  # which dataset: egfr | bace
IN_CSV = os.path.join(BASE, "data", "raw", f"{TAG}_activities.csv")
OUT_CSV = os.path.join(BASE, "data", "processed", f"{TAG}_pic50_clean.csv")

_FRAG_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def _parent_smiles(smiles):
    """Parent SMILES of one canonical SMILES via largest fragment + uncharge.

    Returns (parent, n_fragments, parsed): unparseable input falls back to
    the raw string with parsed=False; n_fragments is 0 in that case.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, 0, False
    parent = _UNCHARGER.uncharge(_FRAG_CHOOSER.choose(mol))
    return Chem.MolToSmiles(parent), len(Chem.GetMolFrags(mol)), True


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} raw rows.")

    # 1. unit must be nM
    df = df[df["standard_units"] == "nM"].copy()
    print(f"After unit == nM: {len(df)}")

    # 2. drop missing IC50 / SMILES
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    print(f"After dropping missing IC50/SMILES: {len(df)}")

    # 3. non-positive IC50 is physically meaningless for the log transform
    df = df[df["standard_value"] > 0].copy()
    print(f"After dropping IC50 <= 0: {len(df)}")

    # 4. IC50 (nM) -> pIC50
    df["pic50"] = -np.log10(df["standard_value"] * 1e-9)

    # 5. censoring filter: only exact measurements ("=") become labels
    if "standard_relation" in df.columns:
        # Fail closed (Low, PR #5): a relation column that is present but
        # entirely empty would drop every row and write a header-only clean
        # CSV with exit 0 -- silently "cleaning" the dataset to nothing.
        rel_text = df["standard_relation"].fillna("").astype(str).str.strip()
        if len(df) and not (rel_text != "").any():
            raise SystemExit(
                f"ERROR: {IN_CSV} has a standard_relation column, but every "
                "value is empty/NaN, so no row can pass the censoring filter "
                "(H1: keep only standard_relation == '=') and the clean "
                "output would be header-only. Re-fetch a fresh raw download "
                "with a populated standard_relation column "
                "(scripts/01e_download_chembl.py already writes that column)."
            )
        n_in = len(df)
        # Compare the STRIPPED value: a padded "=" (" = ", "= ") is an
        # exact measurement, and comparing the raw column would drop every
        # padded row -- another header-only-with-exit-0 fail-open (Low,
        # PR #6).
        keep = exact_relation_mask(df["standard_relation"])
        removed_rel = df.loc[~keep, "standard_relation"]
        df = df[keep].copy()
        n_removed = n_in - len(df)
        print(f"Censoring filter: {n_in} rows in, removed {n_removed} non-exact rows, {len(df)} remain.")
        print("  removed rows by standard_relation:")
        for rel, n in removed_rel.value_counts(dropna=False).items():
            print(f"    {rel!r}: {n}" if pd.notna(rel) else f"    (empty): {n}")
    else:
        # Fail closed (review finding N5): skipping the filter is only safe for
        # genuine non-ChEMBL inputs. A ChEMBL-style file without the column
        # predates the 2026-10-11 refresh (artifacts-v1 release raw files), and
        # running the filter-less would keep censored bounds as exact labels.
        # IDs are checked after strip()+upper() so padded (" CHEMBL4069737 ")
        # and bare numeric ChEMBL numbers cannot slip through as "non-ChEMBL".
        # Degenerate ID columns fail closed the same way (Low, PR #6): a
        # pandas float-coerced number column ("4069737.0") is still a
        # ChEMBL-style ID, and an all-empty column (or blank cells mixed
        # with numeric IDs) cannot certify a non-ChEMBL input either --
        # only uniformly non-ChEMBL IDs (e.g. BACE_1) may skip the filter.
        if "molecule_chembl_id" in df.columns:
            ids = (df["molecule_chembl_id"].fillna("").astype(str)
                   .str.strip().str.upper())
            is_chembl = (ids.str.startswith("CHEMBL")
                         | ids.str.fullmatch(r"\d+")
                         | ids.str.fullmatch(r"\d+\.0+"))
            n_chembl = int(is_chembl.sum())
            n_blank = int((ids == "").sum())
        else:
            n_chembl = len(df)  # cannot verify IDs -> treat as ChEMBL-style
            n_blank = 0
        if TAG != "bace" and (n_chembl > 0 or n_blank > 0):
            raise SystemExit(
                f"ERROR: {IN_CSV} has no standard_relation column, but this is "
                f"not a non-ChEMBL input (TAG={TAG!r}, {n_chembl} ChEMBL-style "
                f"molecule IDs of {len(df)}"
                + (f", {n_blank} blank ID cell(s)" if n_blank else "")
                + "). This raw file predates the "
                "2026-10-11 data refresh (e.g. a download from the artifacts-v1 "
                "release), so the censoring filter (H1: keep only "
                "standard_relation == '=') cannot be verified and censored rows "
                "such as 'IC50 > 10000 nM' would be kept as exact pIC50 labels. "
                "Re-fetch a raw download that carries the standard_relation "
                "column (scripts/01e_download_chembl.py already writes it)."
            )
        print("No standard_relation column in input (bace-style download format); skipping censoring filter.")

    # 6. sanity cross-check against ChEMBL's own pchembl_value (should match),
    #    over all retained rows, not just the subset with a pChEMBL value
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    has_pchembl = pchembl.notna()
    if has_pchembl.any():
        agree = np.isclose(df.loc[has_pchembl, "pic50"], pchembl[has_pchembl], atol=0.02).mean()
        print(f"Sanity check: {int(has_pchembl.sum())} retained rows have pchembl_value; "
              f"pIC50 matches for {agree:.1%} of them (atol=0.02).")
    else:
        print("Sanity check: no retained rows have pchembl_value to compare against.")
    print(f"Retained rows with empty pchembl_value: {int((~has_pchembl).sum())} "
          "(exact measurements ChEMBL could not convert, e.g. 'Outside typical range').")

    # 7. parent standardization, then deduplicate by parent_smiles -> median pIC50
    print(f"Standardizing {df['canonical_smiles'].nunique()} unique SMILES to parent structures ...")
    parent_map = {}
    n_multi = n_changed = n_fallback = 0
    for smi in df["canonical_smiles"].unique():
        parent, n_frag, parsed = _parent_smiles(smi)
        parent_map[smi] = parent
        if not parsed:
            n_fallback += 1
        else:
            n_multi += n_frag > 1
            n_changed += parent != smi
    df["parent_smiles"] = df["canonical_smiles"].map(parent_map)
    print(f"  multi-fragment inputs: {n_multi}; parent differs from input: {n_changed}; "
          f"standardization failed (kept raw SMILES): {n_fallback}")

    n_rows = len(df)
    target_name = df["target_pref_name"].dropna().iloc[0] if df["target_pref_name"].notna().any() else TAG
    grp = df.groupby("parent_smiles")
    sizes = grp.size()
    n_structural = grp["canonical_smiles"].nunique()
    clean = (
        grp.agg(
            pic50=("pic50", "median"),
            ic50_nm_median=("standard_value", "median"),
            n_measurements=("pic50", "size"),
            n_structural_variants=("canonical_smiles", "nunique"),
            canonical_smiles=("canonical_smiles", "first"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
        )
        .reset_index()
    )
    clean["target"] = target_name
    clean = clean.sort_values("parent_smiles", kind="stable").reset_index(drop=True)
    clean = clean[
        [
            "parent_smiles",
            "canonical_smiles",
            "pic50",
            "ic50_nm_median",
            "n_measurements",
            "n_structural_variants",
            "molecule_chembl_id",
            "target",
        ]
    ]
    print(f"Target: {target_name}")
    print(f"Removed {n_rows - len(clean)} duplicate rows -> {len(clean)} unique parent compounds.")
    n_merged = int((sizes > 1).sum())
    n_merged_struct = int((n_structural > 1).sum())
    print(f"Parent groups merged away: {n_merged} (of which {n_merged_struct} held >1 distinct "
          "canonical SMILES, i.e. salt/form duplicates).")

    print("\npIC50 distribution:")
    print(clean["pic50"].describe())

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    clean.to_csv(OUT_CSV, index=False)
    print(f"\nSaved cleaned data -> {os.path.abspath(OUT_CSV)}")


if __name__ == "__main__":
    main()
