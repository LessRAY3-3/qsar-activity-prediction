"""scripts/gnn_08_time_splits.py - synthetic publication-year split, tmp_path only.

Builds 200 unique RDKit-parseable SMILES (base fragments cycled with an
index-dependent carbon tail), one document per molecule with a synthetic
1990-2020 year, plus a synthetic document_years.csv - then asserts:

  * happy path: train/valid/test disjoint + exhaustive, strictly
    time-ordered (max train year < min valid year < min test year),
    test fraction in [0.20, 0.36], valid fraction in [0.08, 0.22],
    npz schema (int indices, per-molecule float32 year), timeline PNG,
    one-line summary;
  * guards: 3-year span -> {TAG}_time_NA.json marker, no npz, exit 0;
    busiest year < 100 molecules -> same marker; 2 of 200 SMILES
    undated (99.0% coverage) -> pinned for cutoff selection, then
    excluded from training for lack of dated labels; 12 of 200 undated (94%, below
    FILL_YEAR_COVERAGE) -> SystemExit; missing
    document_years.csv -> SystemExit naming fetch_document_years.py;
  * M1/03 regime: activities rows written as salt forms while the npz
    holds parents -> years join through 02's parent standardization
    (200/200 coverage, exact per-molecule years), no coverage collapse.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from qsar_common import murcko_scaffold_list

N = 200
YEAR_LO, YEAR_HI = 1990, 2020
BUCKET_YEAR, BUCKET_N = 2000, 100  # busiest year holds >= MIN_N_BUSIEST_YEAR molecules
SHORT_LO, SHORT_HI = 2018, 2020    # span 2 -> NA marker

BASES = ["CC", "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CC(C)C", "c1ccncc1",
         "O=C(O)C", "C1CCCCC1", "c1ccoc1", "C1CCNCC1", "c1cc2ccc2cc1"]


def _smiles(n=N):
    # every base cycled with a per-index carbon tail: unique + parseable
    return [BASES[i % len(BASES)] + "C" * (i // len(BASES) + 1) for i in range(n)]


def _years():
    """100 molecules in BUCKET_YEAR, the rest spread over the full range."""
    y = np.empty(N, dtype=int)
    y[:BUCKET_N] = BUCKET_YEAR
    y[BUCKET_N:] = YEAR_LO + np.arange(N - BUCKET_N) * (YEAR_HI - YEAR_LO + 1) // (N - BUCKET_N)
    return y


def _years_uniform():
    """Even spread over the full range: span OK, busiest year far below 100."""
    return YEAR_LO + np.arange(N) * (YEAR_HI - YEAR_LO + 1) // N


def _years_short():
    return SHORT_LO + np.arange(N) % (SHORT_HI - SHORT_LO + 1)


def _write_inputs(tmp_path, years, broken=(), salt=()):
    """Synthetic fingerprints.npz + activities.csv + document_years.csv.

    Each molecule gets one document carrying its year; every 7th also gets
    a NEWER second document so the per-SMILES min() is exercised (the
    effective year stays the primary document's).  Indices in ``broken``
    never get a document-year row -> those SMILES cannot join a year.
    Indices in ``salt`` are written to activities.csv as a multi-fragment
    (salt) form while the npz keeps the plain SMILES - the M1/03 regime.
    """
    # Match the real cache contract: raw strings are standardized by 02
    # once before 03 exports their parent keys.
    import importlib
    parent = importlib.import_module("02_clean_data")._parent_smiles
    raw_smiles = np.array(_smiles())
    smiles = np.array([parent(s)[0] for s in raw_smiles])
    years = np.asarray(years)
    assert len(years) == len(smiles)

    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez(
        proc / "egfr_fingerprints.npz",
        X=rng.integers(0, 2, size=(len(smiles), 2048)).astype(np.uint8),
        y=rng.normal(6.0, 1.0, len(smiles)).astype(np.float32),
        smiles=smiles,
    )

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    broken = set(broken)
    salt = set(salt)
    act_rows, year_rows = [], []
    for i, smi in enumerate(raw_smiles):
        act_smi = f"{smi}.Cl" if i in salt else smi
        act_rows.append((act_smi, f"DOC{i:03d}"))
        if i % 7 == 0:
            act_rows.append((act_smi, f"DOCX{i:03d}"))
        if i in broken:
            continue
        year_rows.append((f"DOC{i:03d}", int(years[i])))
        if i % 7 == 0:
            year_rows.append((f"DOCX{i:03d}", int(years[i]) + 5))  # newer -> min() unchanged
    acts = pd.DataFrame(act_rows, columns=["canonical_smiles", "document_chembl_id"])
    acts["standard_value"] = 100.0
    acts["standard_units"] = "nM"
    acts["standard_relation"] = "="
    acts.to_csv(raw / "egfr_activities.csv", index=False)
    pd.DataFrame(year_rows, columns=["document_chembl_id", "document_year"]).to_csv(
        raw / "document_years.csv", index=False)

    split_dir = proc / "splits"
    split_dir.mkdir()
    return {
        "npz": proc / "egfr_fingerprints.npz",
        "acts": raw / "egfr_activities.csv",
        "years": raw / "document_years.csv",
        "splits": split_dir,
        "figs": tmp_path / "figures" / "time_split",
    }


def _setup(tmp_path, monkeypatch, mod, years=None, broken=(), salt=()):
    paths = _write_inputs(tmp_path, _years() if years is None else years,
                          broken=broken, salt=salt)
    monkeypatch.setattr(mod, "IN_NPZ", str(paths["npz"]))
    monkeypatch.setattr(mod, "IN_ACTIVITIES", str(paths["acts"]))
    monkeypatch.setattr(mod, "IN_YEARS", str(paths["years"]))
    monkeypatch.setattr(mod, "SPLIT_DIR", str(paths["splits"]))
    monkeypatch.setattr(mod, "FIG_DIR", str(paths["figs"]))
    return paths


@pytest.fixture
def mod(load_script):
    return load_script("gnn_08_time_splits")


def test_fixture_smiles_unique_parseable_and_diverse():
    smiles = _smiles()
    assert len(smiles) == N == len(set(smiles))
    assert all(Chem.MolFromSmiles(s) is not None for s in smiles)
    assert len(set(murcko_scaffold_list(smiles))) >= 3  # scaffolds must vary


def test_happy_path_disjoint_exhaustive_and_artifacts(tmp_path, monkeypatch, mod, capsys):
    paths = _setup(tmp_path, monkeypatch, mod)
    assert mod.main([]) is None  # exit 0

    out = np.load(paths["splits"] / f"{mod.TAG}_time.npz")
    assert set(out.files) >= {"train_idx", "valid_idx", "test_idx",
                              "scaffold_smiles", "smiles", "year"}
    tr, va, te = out["train_idx"], out["valid_idx"], out["test_idx"]
    assert all(idx.dtype.kind == "i" for idx in (tr, va, te))
    assert len(tr) + len(va) + len(te) == N
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert set(tr) | set(va) | set(te) == set(range(N))

    with np.load(paths["npz"]) as cache:
        smiles = cache["smiles"]
    np.testing.assert_array_equal(out["smiles"], smiles)
    np.testing.assert_array_equal(out["scaffold_smiles"], murcko_scaffold_list(smiles))

    assert (paths["figs"] / f"{mod.TAG}_timeline.png").exists()
    printed = capsys.readouterr().out
    assert f"[{mod.TAG}] n={N}" in printed and "cutoff_t=" in printed


def test_time_ordering_and_year_column(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    assert mod.main([]) is None

    out = np.load(paths["splits"] / f"{mod.TAG}_time.npz")
    year = out["year"]
    assert len(year) == N and year.dtype == np.float32
    tr, va, te = out["train_idx"], out["valid_idx"], out["test_idx"]
    # strict chronological separation of the three segments
    assert year[tr].max() < year[va].min()
    assert year[va].max() < year[te].min()
    assert year[va].min() < year[te].min()


def test_split_fractions_within_spec(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    assert mod.main([]) is None

    out = np.load(paths["splits"] / f"{mod.TAG}_time.npz")
    frac_test = len(out["test_idx"]) / N
    frac_valid = len(out["valid_idx"]) / N
    assert 0.20 <= frac_test <= 0.36, f"test fraction {frac_test:.3f}"
    assert 0.08 <= frac_valid <= 0.22, f"valid fraction {frac_valid:.3f}"


def test_salt_activities_join_years_on_parent_keys(tmp_path, monkeypatch, mod, capsys):
    """activities hold salt forms, the npz holds parents -> years still join.

    03 exports parent_smiles in the fingerprint npz while raw activities
    keep multi-fragment canonical variants, so the year join must key on
    02's parent standardization; a string-level reindex would leave those
    molecules undated and collapse the coverage guard.
    """
    salted = set(range(0, N, 3))  # 1/3 salt-form activity rows, 2/3 plain
    paths = _setup(tmp_path, monkeypatch, mod, salt=salted)
    acts = pd.read_csv(paths["acts"])
    assert acts["canonical_smiles"].str.endswith(".Cl").sum() > 0  # fixture really salts rows

    assert mod.main([]) is None  # exit 0: coverage not refused

    printed = capsys.readouterr().out
    assert f"year coverage: {N}/{N}" in printed
    assert "have no publication year" not in printed

    d = np.load(paths["splits"] / f"{mod.TAG}_time.npz")
    np.testing.assert_array_equal(d["year"].astype(int), _years())  # per-molecule years exact
    with np.load(paths["npz"]) as cache:
        np.testing.assert_array_equal(d["smiles"], cache["smiles"])


def test_cached_parent_is_not_standardized_twice(mod):
    """CHEMBL2035183 has a non-idempotent parent round trip in pinned RDKit.

    Its exact 2012 record belongs in the pre-2016 training pool, even when
    reparsing the parent cache would produce a different SMILES string.
    """
    raw = "C1=C/COCc2cc(ccc2OCCN2CCCC2)Nc2nccc(n2)-c2ccc(s2)COC/1"
    cached_parent = mod._parent_smiles(raw)[0]
    assert mod._parent_smiles(cached_parent)[0] != cached_parent, (
        "Pinned RDKit no longer reproduces CHEMBL2035183's non-idempotent round trip; "
        "refresh this real regression case explicitly (do not silently pass).")
    smiles = np.array([cached_parent])
    acts = pd.DataFrame({"canonical_smiles": [raw],
                         "document_chembl_id": ["CHEMBL2034857"],
                         "standard_value": [7200.], "standard_units": ["nM"],
                         "standard_relation": ["="]})
    years = pd.DataFrame({"document_chembl_id": ["CHEMBL2034857"],
                          "document_year": [2012]})
    np.testing.assert_array_equal(mod.assign_years(smiles, acts, years), [2012.])
    pools = [np.array([0]), np.array([], dtype=int), np.array([], dtype=int)]
    y, retained, excluded = mod.asof_labels(smiles, acts, years, pools, 2016, 2018)
    assert float(y[0]) == pytest.approx(9 - np.log10(7200), abs=1e-6)
    assert retained[0].tolist() == [0]
    assert len(excluded) == 0


def test_short_year_span_writes_na_marker(tmp_path, monkeypatch, mod, capsys):
    paths = _setup(tmp_path, monkeypatch, mod, years=_years_short())
    assert mod.main([]) is None  # A legitimately unavailable first split may be marked NA

    payload = json.loads((paths["splits"] / f"{mod.TAG}_time_NA.json").read_text())
    for key in ("tag", "reason", "year_min", "year_max", "n_in_latest_year"):
        assert key in payload
    assert payload["tag"] == mod.TAG
    assert payload["year_max"] - payload["year_min"] == 2
    assert not (paths["splits"] / f"{mod.TAG}_time.npz").exists()
    assert not (paths["figs"] / f"{mod.TAG}_timeline.png").exists()
    assert "not feasible" in capsys.readouterr().out


def test_sparse_busiest_year_writes_na_marker(tmp_path, monkeypatch, mod):
    """Span is fine (30 y) but no year holds 100 molecules -> NA marker."""
    paths = _setup(tmp_path, monkeypatch, mod, years=_years_uniform())
    assert mod.main([]) is None

    payload = json.loads((paths["splits"] / f"{mod.TAG}_time_NA.json").read_text())
    assert payload["year_max"] - payload["year_min"] == 30
    assert payload["n_in_busiest_year"] < mod.MIN_N_BUSIEST_YEAR
    assert not (paths["splits"] / f"{mod.TAG}_time.npz").exists()


def test_two_raw_structures_without_document_years_are_excluded(tmp_path, monkeypatch, mod, capsys):
    paths = _setup(tmp_path, monkeypatch, mod, broken=(30, 77))
    mod.main([])
    printed = capsys.readouterr().out
    assert "2 of 200 smiles have no publication year" in printed
    assert "pinning them to year_min" in printed
    d = np.load(paths["splits"] / f"{mod.TAG}_time.npz")
    assert not np.isnan(d["year"]).any()
    filled = set(d["smiles"][[30, 77]].tolist())
    train_smiles = set(np.asarray(d["smiles"], dtype=str)[d["train_idx"]].tolist())
    assert not (filled & train_smiles)
    assert set(d["excluded_idx"]) == {30, 77}


def test_massive_undated_share_aborts(tmp_path, monkeypatch, mod, capsys):
    broken = tuple(range(12))  # 12/200 = 6% undated, below FILL_YEAR_COVERAGE
    paths = _setup(tmp_path, monkeypatch, mod, broken=broken)
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    printed = capsys.readouterr().out
    assert "12 of 200 smiles have no publication year" in printed
    assert "188/200" in str(exc.value)
    assert not (paths["splits"] / f"{mod.TAG}_time.npz").exists()


def test_missing_years_file_points_to_fetch_script(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    paths["years"].unlink()
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert "fetch_document_years" in str(exc.value)


def test_successful_regeneration_clears_na(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    marker = paths["splits"] / f"{mod.TAG}_time_NA.json"
    marker.write_text('{"reason": "previous failure"}')
    mod.main([])
    assert not marker.exists()


def test_versioned_split_self_validates_without_external_cache(tmp_path, monkeypatch, mod):
    from time_labels import label_signature, load_time_split
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    path = paths["splits"] / f"{mod.TAG}_time.npz"
    good = load_time_split(paths["splits"], mod.TAG)
    for key, changed, message in (
        ("year", good["year"][:-1], "label count"),
        ("smiles", good["smiles"][:-1], "label count"),
        ("original_y", good["original_y"] + 1, "source dataset signature"),
    ):
        bad = {**good, key: changed}
        bad["label_id"] = label_signature(bad)
        np.savez(path, **bad)
        with pytest.raises(ValueError, match=message):
            load_time_split(paths["splits"], mod.TAG)


def test_prediction_version_checks_rows_and_actual_labels(tmp_path, monkeypatch, mod):
    from time_labels import load_time_split, require_prediction_version
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    sp = load_time_split(paths["splits"], mod.TAG)
    good = dict(label_id=sp["label_id"], test_idx=sp["test_idx"],
                y_true=sp["y"][sp["test_idx"]])
    require_prediction_version(good, sp)
    # Keep the actual labels and row order identical so only the version gate
    # can reject an older campaign (the test must kill removal of that gate).
    for stale_id in ("0" * 64, ""):
        with pytest.raises(ValueError, match="label version differs"):
            require_prediction_version({**good, "label_id": stale_id}, sp)
    with pytest.raises(ValueError, match="label version differs"):
        require_prediction_version({k: v for k, v in good.items() if k != "label_id"}, sp)
    with pytest.raises(ValueError, match="test indices"):
        require_prediction_version({**good, "test_idx": good["test_idx"][::-1]}, sp)
    with pytest.raises(ValueError, match="labels differ"):
        require_prediction_version({**good, "y_true": good["y_true"] + 0.1}, sp)


def test_failed_regeneration_preserves_readable_old_split(tmp_path, monkeypatch, mod):
    from time_labels import load_time_split
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    existing = paths["splits"] / f"{mod.TAG}_time.npz"
    original = existing.read_bytes()
    paths["years"].unlink()
    with pytest.raises(SystemExit):
        mod.main([])
    assert existing.read_bytes() == original
    assert not (paths["splits"] / f"{mod.TAG}_time_NA.json").exists()
    assert load_time_split(paths["splits"], mod.TAG) is not None


@pytest.mark.parametrize("cached", ["C1=CC=CC=C1", "c1ccccc1.Cl"])
def test_cached_key_missing_from_raw_fails_with_row_and_key(mod, cached):
    acts = pd.DataFrame({"canonical_smiles": ["CC", "c1ccccc1"],
                         "document_chembl_id": ["D0", "D1"],
                         "standard_value": [100., 100.], "standard_units": ["nM", "nM"],
                         "standard_relation": ["=", "="]})
    years = pd.DataFrame({"document_chembl_id": ["D0", "D1"], "document_year": [2000, 2001]})
    keys = np.array(["CC", cached])
    pools = [np.array([0, 1]), np.array([], dtype=int), np.array([], dtype=int)]
    for call in (lambda: mod.assign_years(keys, acts, years),
                 lambda: mod.asof_labels(keys, acts, years, pools, 2010, 2020)):
        with pytest.raises(ValueError, match="cached parent key") as error:
            call()
        assert "row=1" in str(error.value) and repr(cached) in str(error.value)


@pytest.mark.parametrize("missing", ["document", "year", "exact_measurement"])
def test_raw_structure_exists_but_metadata_or_eligible_label_missing_is_allowed(mod, missing):
    acts = pd.DataFrame({"canonical_smiles": ["CC"], "document_chembl_id": ["D"],
                         "standard_value": [100.], "standard_units": ["nM"],
                         "standard_relation": ["="]})
    years = pd.DataFrame({"document_chembl_id": ["D"], "document_year": [2000.]})
    if missing == "document":
        acts.loc[0, "document_chembl_id"] = None
    elif missing == "year":
        years.loc[0, "document_year"] = np.nan
    else:
        acts.loc[0, "standard_relation"] = ">"
    keys = np.array(["CC"])
    pools = [np.array([0]), np.array([], dtype=int), np.array([], dtype=int)]
    assigned = mod.assign_years(keys, acts, years)
    assert np.isnan(assigned[0]) if missing != "exact_measurement" else assigned[0] == 2000
    y, retained, excluded = mod.asof_labels(keys, acts, years, pools, 2010, 2020)
    assert np.isnan(y[0]) and excluded.tolist() == [0]
    assert all(len(idx) == 0 for idx in retained)


def test_asof_exact_record_boundaries_median_and_units(mod):
    # Explicit pIC50 answers: train median(6,7,4)=6; validation median(6,7)=6.5;
    # test median(5,7)=6. Boundary/future records carry very different values.
    records = [
        ("CC", 2009, 1000., "nM", "="),
        ("CC", 2009, 100., "nM", "="),
        ("CC", 2009, 100000., "nM", "="),
        ("CC", 2010, 1., "nM", "="),
        ("CC", 2019, .1, "nM", "="),
        ("CC", 2020, .01, "nM", "="),
        ("CCC", 2010, 1000., "nM", "="),
        ("CCC", 2019, 100., "nM", "="),
        ("CCC", 2020, 1., "nM", "="),
        ("CCCC", 2019, 1., "nM", "="),
        ("CCCC", 2020, 10000., "nM", "="),
        ("CCCC", 2021, 100., "nM", "="),
        ("CCO", 2009, 1., "uM", "="),
        ("CCO", 2009, 1., "nM", ">"),
    ]
    acts = pd.DataFrame(records, columns=["canonical_smiles", "year", "standard_value",
                                         "standard_units", "standard_relation"])
    acts["document_chembl_id"] = [f"D{i}" for i in range(len(acts))]
    years = acts[["document_chembl_id", "year"]].rename(columns={"year": "document_year"})
    keys = np.array(["CC", "CCC", "CCCC", "CCO"])
    pools = [np.array([0, 3]), np.array([1]), np.array([2])]
    y, retained, excluded = mod.asof_labels(keys, acts, years, pools, 2010, 2020)
    np.testing.assert_allclose(y[:3], [6., 6.5, 6.], atol=1e-6, rtol=0)
    assert np.isnan(y[3])
    assert [idx.tolist() for idx in retained] == [[0], [1], [2]]
    assert excluded.tolist() == [3]


def test_single_structural_mismatch_aborts_instead_of_becoming_an_exclusion(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    with np.load(paths["npz"]) as cache:
        data = dict(cache)
    # Only 0.5% affected: year-coverage thresholds must not hide this mismatch.
    keys = data["smiles"].astype(object)
    keys[30] = str(keys[30]) + ".Cl"
    data["smiles"] = keys.astype(str)
    np.savez(paths["npz"], **data)
    with pytest.raises(ValueError, match="row=30"):
        mod.main([])
    assert not (paths["splits"] / f"{mod.TAG}_time.npz").exists()
    assert not (paths["splits"] / f"{mod.TAG}_time_NA.json").exists()



def test_cached_parent_is_not_renormalized_independently_of_rdkit_version(monkeypatch, mod):
    raw, parent = "raw structure key", "cached parent key"
    seen = []
    def non_idempotent(value):
        seen.append(value)
        if value == raw:
            return parent, None
        raise AssertionError("Cached parent was normalized a second time")
    monkeypatch.setattr(mod, "_parent_smiles", non_idempotent)
    acts = pd.DataFrame({"canonical_smiles": [raw], "document_chembl_id": ["D"],
                         "standard_value": [7200.], "standard_units": ["nM"], "standard_relation": ["="]})
    years = pd.DataFrame({"document_chembl_id": ["D"], "document_year": [2012]})
    keys = np.array([parent])
    np.testing.assert_array_equal(mod.assign_years(keys, acts, years), [2012.])
    y, _, excluded = mod.asof_labels(keys, acts, years,
                                     [np.array([0]), np.array([], dtype=int), np.array([], dtype=int)],
                                     2016, 2018)
    assert y[0] == pytest.approx(9 - np.log10(7200), abs=1e-6)
    assert not len(excluded) and seen == [raw, raw]


def test_unavailable_regeneration_preserves_old_split_and_returns_error(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    existing = paths["splits"] / f"{mod.TAG}_time.npz"
    original = existing.read_bytes()
    years = pd.read_csv(paths["years"])
    years["document_year"] = 2020
    years.to_csv(paths["years"], index=False)
    with pytest.raises(ValueError, match="existing split retained"):
        mod.main([])
    assert existing.read_bytes() == original
    assert not (paths["splits"] / f"{mod.TAG}_time_NA.json").exists()


def test_plot_failure_does_not_publish_new_split_or_invalidate_old_one(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    existing = paths["splits"] / f"{mod.TAG}_time.npz"
    figure = paths["figs"] / f"{mod.TAG}_timeline.png"
    original, original_figure = existing.read_bytes(), figure.read_bytes()
    def fail_plot(*args, **kwargs):
        raise OSError("injected plot failure")
    monkeypatch.setattr(mod, "plot_timeline", fail_plot)
    with pytest.raises(OSError, match="plot failure"):
        mod.main([])
    assert existing.read_bytes() == original and figure.read_bytes() == original_figure
    assert not (paths["splits"] / f"{mod.TAG}_time_NA.json").exists()


def test_full_integrity_digest_covers_scaffolds_without_changing_legacy_label_id(tmp_path, monkeypatch, mod):
    from time_labels import label_signature, load_time_split
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    file = paths["splits"] / f"{mod.TAG}_time.npz"
    good = load_time_split(paths["splits"], mod.TAG)
    original_id = str(good["label_id"])
    bad = dict(good, scaffold_smiles=good["scaffold_smiles"].copy())
    bad["scaffold_smiles"][0] = "different"
    assert label_signature(bad) == original_id  # legacy ID algorithm remains compatible
    np.savez(file, **bad)
    with pytest.raises(ValueError, match="full-payload integrity mismatch"):
        load_time_split(paths["splits"], mod.TAG)
    for missing in ("integrity_schema", "integrity_sha256"):
        incomplete = {key: value for key, value in good.items() if key != missing}
        np.savez(file, **incomplete)
        with pytest.raises(ValueError, match="integrity schema"):
            load_time_split(paths["splits"], mod.TAG)
    historical = {key: value for key, value in good.items() if not key.startswith("integrity_")}
    np.savez(file, **historical)
    restored = load_time_split(paths["splits"], mod.TAG)
    assert str(restored["label_id"]) == original_id
    np.testing.assert_array_equal(restored["scaffold_smiles"], good["scaffold_smiles"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal interruption test")
def test_sigkill_during_regeneration_keeps_existing_split(tmp_path, monkeypatch, mod):
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    existing = paths["splits"] / f"{mod.TAG}_time.npz"
    original = existing.read_bytes()
    ready = tmp_path / "ready"
    code = """
import importlib, pathlib, signal, sys
sys.path.insert(0, sys.argv[1])
module = importlib.import_module('gnn_08_time_splits')
def interrupted_plot(*args, **kwargs):
    pathlib.Path(sys.argv[-1]).touch()
    signal.pause()
module.plot_timeline = interrupted_plot
module.run('egfr', *sys.argv[2:-1])
"""
    args = [str(paths[key]) for key in ("npz", "acts", "years", "splits", "figs")]
    process = subprocess.Popen([sys.executable, "-B", "-c", code,
                                str(Path(mod.__file__).parent), *args, str(ready)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None, "regeneration exited before interruption point"
            time.sleep(0.02)
        assert ready.exists(), "regeneration did not reach interruption point"
        process.kill()
        process.wait(timeout=10)
        assert existing.read_bytes() == original
        assert not (paths["splits"] / "egfr_time_NA.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()


@pytest.mark.parametrize("same_split_bytes", [False, True])
def test_npz_publication_failure_rolls_back_already_replaced_timeline(
        tmp_path, monkeypatch, mod, same_split_bytes):
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    split = paths["splits"] / "egfr_time.npz"
    figure = paths["figs"] / "egfr_timeline.png"
    old_split, old_figure = split.read_bytes(), figure.read_bytes()
    staged_split, staged_figure = tmp_path / "next.npz", tmp_path / "next.png"
    staged_split.write_bytes(old_split if same_split_bytes else b"next split payload")
    staged_figure.write_bytes(b"distinct next figure payload")
    original_replace = mod._atomic_replace
    def fail_npz(source, destination):
        if Path(destination) == split:
            assert figure.read_bytes() == staged_figure.read_bytes()
            raise OSError("injected second canonical replacement failure")
        return original_replace(source, destination)
    monkeypatch.setattr(mod, "_atomic_replace", fail_npz)
    with pytest.raises(OSError, match="second canonical"):
        mod._publish_pair(staged_split, staged_figure, paths["splits"], paths["figs"], "egfr")
    assert split.read_bytes() == old_split
    assert figure.read_bytes() == old_figure
    assert not (paths["splits"] / ".egfr_time_publication").exists()
    record = json.loads((paths["splits"] / ".egfr_time_last_publication.json").read_text())
    assert record["outcome"] == "rolled_back"


def test_first_publication_failure_removes_new_figure_and_leaves_no_split(tmp_path, monkeypatch, mod):
    split_dir, fig_dir = tmp_path / "splits", tmp_path / "figures"
    split_dir.mkdir()
    candidate, figure = tmp_path / "candidate.npz", tmp_path / "candidate.png"
    candidate.write_bytes(b"complete candidate")
    figure.write_bytes(b"complete figure")
    original_replace = mod._atomic_replace
    def fail_npz(source, destination):
        if Path(destination) == split_dir / "egfr_time.npz":
            raise OSError("injected first publication failure")
        return original_replace(source, destination)
    monkeypatch.setattr(mod, "_atomic_replace", fail_npz)
    with pytest.raises(OSError, match="first publication"):
        mod._publish_pair(candidate, figure, split_dir, fig_dir, "egfr")
    assert not (split_dir / "egfr_time.npz").exists()
    assert not (fig_dir / "egfr_timeline.png").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal interruption test")
@pytest.mark.parametrize("after_commit", [False, True])
def test_sigkill_between_publication_steps_is_recovered_before_next_input_read(
        tmp_path, monkeypatch, mod, after_commit):
    paths = _setup(tmp_path, monkeypatch, mod)
    mod.main([])
    split, figure = paths["splits"] / "egfr_time.npz", paths["figs"] / "egfr_timeline.png"
    old_split, old_figure = split.read_bytes(), figure.read_bytes()
    # Real changed year input produces a genuinely different NPZ and timeline.
    years = pd.read_csv(paths["years"])
    years["document_year"] += 1
    years.to_csv(paths["years"], index=False)
    ready = tmp_path / "publication_ready"
    code = """
import importlib, pathlib, signal, sys
sys.path.insert(0, sys.argv[1])
module = importlib.import_module('gnn_08_time_splits')
original = module._atomic_replace
split = pathlib.Path(sys.argv[5]) / 'egfr_time.npz'
after_commit = sys.argv[-1] == 'True'
def interrupted_replace(source, destination):
    if pathlib.Path(destination) == split:
        if after_commit:
            original(source, destination)
        pathlib.Path(sys.argv[-2]).touch()
        signal.pause()
    return original(source, destination)
module._atomic_replace = interrupted_replace
module.run('egfr', *sys.argv[2:7])
"""
    args = [str(paths[key]) for key in ("npz", "acts", "years", "splits", "figs")]
    process = subprocess.Popen([sys.executable, "-B", "-c", code,
                                str(Path(mod.__file__).parent), *args, str(ready), str(after_commit)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None, "publication exited before interruption point"
            time.sleep(0.02)
        assert ready.exists(), "publication did not reach interruption point"
        assert figure.read_bytes() != old_figure
        assert (split.read_bytes() != old_split) == after_commit
        journal = paths["splits"] / ".egfr_time_publication"
        assert (journal / "pending.json").exists()
        # A live process must not be mistaken for an interrupted transaction.
        with pytest.raises(RuntimeError, match="another time split regeneration"):
            mod.main([])
        assert (journal / "pending.json").exists()
        process.kill()
        process.wait(timeout=10)
        published_split, published_figure = split.read_bytes(), figure.read_bytes()
        paths["years"].unlink()
        with pytest.raises(SystemExit, match="document years"):
            mod.main([])
        assert not journal.exists()
        record = json.loads((paths["splits"] / ".egfr_time_last_publication.json").read_text())
        if after_commit:
            assert record["outcome"] == "committed"
            assert split.read_bytes() == published_split and figure.read_bytes() == published_figure
        else:
            assert record["outcome"] == "rolled_back"
            assert split.read_bytes() == old_split and figure.read_bytes() == old_figure
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()


def test_pending_publication_refuses_external_figure_edits(tmp_path, monkeypatch, mod):
    split_dir, fig_dir = tmp_path / "splits", tmp_path / "figures"
    split_dir.mkdir()
    fig_dir.mkdir()
    target, figure = split_dir / "egfr_time.npz", fig_dir / "egfr_timeline.png"
    target.write_bytes(b"old split")
    figure.write_bytes(b"old figure")
    candidate, candidate_figure = tmp_path / "next.npz", tmp_path / "next.png"
    candidate.write_bytes(b"new split")
    candidate_figure.write_bytes(b"new figure")
    original_replace, original_recover = mod._atomic_replace, mod._recover_publication
    def fail_npz(source, destination):
        if Path(destination) == target:
            raise OSError("injected split publish failure")
        return original_replace(source, destination)
    monkeypatch.setattr(mod, "_atomic_replace", fail_npz)
    monkeypatch.setattr(mod, "_recover_publication", lambda *args: None)
    with pytest.raises(OSError):
        mod._publish_pair(candidate, candidate_figure, split_dir, fig_dir, "egfr")
    figure.write_bytes(b"external user edit")
    with pytest.raises(ValueError, match="figure changed externally"):
        original_recover(split_dir, fig_dir, "egfr")
    assert figure.read_bytes() == b"external user edit" and target.read_bytes() == b"old split"
    assert (split_dir / ".egfr_time_publication/pending.json").exists()
    # A previously completed rollback may have been killed during journal
    # cleanup, after deleting the backup. Recovery must remain idempotent.
    figure.write_bytes(b"old figure")
    (split_dir / ".egfr_time_publication/previous_timeline.png").unlink()
    original_recover(split_dir, fig_dir, "egfr")
    assert not (split_dir / ".egfr_time_publication").exists()
    assert figure.read_bytes() == b"old figure"
