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
    unjoinable (99.0% coverage) -> pinned to year_min on the train side
    and the split proceeds; 12 of 200 unjoinable (94%, below
    FILL_YEAR_COVERAGE) -> SystemExit; missing
    document_years.csv -> SystemExit naming fetch_document_years.py.
"""
import json

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


def _write_inputs(tmp_path, years, broken=()):
    """Synthetic fingerprints.npz + activities.csv + document_years.csv.

    Each molecule gets one document carrying its year; every 7th also gets
    a NEWER second document so the per-SMILES min() is exercised (the
    effective year stays the primary document's).  Indices in ``broken``
    never get a document-year row -> those SMILES cannot join a year.
    """
    smiles = np.array(_smiles())
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
    act_rows, year_rows = [], []
    for i, smi in enumerate(smiles):
        act_rows.append((smi, f"DOC{i:03d}"))
        if i % 7 == 0:
            act_rows.append((smi, f"DOCX{i:03d}"))
        if i in broken:
            continue
        year_rows.append((f"DOC{i:03d}", int(years[i])))
        if i % 7 == 0:
            year_rows.append((f"DOCX{i:03d}", int(years[i]) + 5))  # newer -> min() unchanged
    pd.DataFrame(act_rows, columns=["canonical_smiles", "document_chembl_id"]).to_csv(
        raw / "egfr_activities.csv", index=False)
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


def _setup(tmp_path, monkeypatch, mod, years=None, broken=()):
    paths = _write_inputs(tmp_path, _years() if years is None else years, broken=broken)
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
    assert set(out.files) == {"train_idx", "valid_idx", "test_idx",
                              "scaffold_smiles", "smiles", "year"}
    tr, va, te = out["train_idx"], out["valid_idx"], out["test_idx"]
    assert all(idx.dtype.kind == "i" for idx in (tr, va, te))
    assert len(tr) + len(va) + len(te) == N
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert set(tr) | set(va) | set(te) == set(range(N))

    smiles = np.array(_smiles())
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


def test_short_year_span_writes_na_marker(tmp_path, monkeypatch, mod, capsys):
    paths = _setup(tmp_path, monkeypatch, mod, years=_years_short())
    assert mod.main([]) is None  # NA is a normal, successful exit

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


def test_two_unjoinable_smiles_filled_to_train(tmp_path, monkeypatch, mod, capsys):
    paths = _setup(tmp_path, monkeypatch, mod, broken=(30, 77))
    mod.main([])
    printed = capsys.readouterr().out
    assert "2 of 200 smiles have no publication year" in printed
    assert "pinning them to year_min" in printed
    d = np.load(paths["splits"] / f"{mod.TAG}_time.npz")
    assert not np.isnan(d["year"]).any()
    filled = {_smiles()[30], _smiles()[77]}
    train_smiles = set(np.asarray(d["smiles"], dtype=str)[d["train_idx"]].tolist())
    assert filled <= train_smiles  # pinned to year_min -> always train side


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
