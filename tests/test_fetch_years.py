"""Unit tests for scripts/fetch_document_years.py — every HTTP call is faked.

Covers: year parsing/int conversion, retry-then-success, retry exhaustion
(failed list, run survives, other docs continue), resume skip of recorded
ids (incl. empty-year retry), and --limit.
"""
import sys

import pandas as pd
import pytest


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _activities(year):
    return {"activities": [{"document_chembl_id": "CHEMBL_FAKE", "document_year": year}]}


class FakeSession:
    """Routes each GET to handler(doc_id) -> FakeResponse, recording calls."""

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def get(self, url, params=None, timeout=None):
        doc_id = params["document_chembl_id"]
        self.calls.append(doc_id)
        return self._handler(doc_id)


@pytest.fixture
def mod(load_script, monkeypatch):
    m = load_script("fetch_document_years")
    monkeypatch.setattr(m.time, "sleep", lambda seconds: None)  # retries must not wait
    return m


def _write_raw(tmp_path, tag, doc_ids):
    path = tmp_path / f"{tag}_activities.csv"
    pd.DataFrame({"document_chembl_id": doc_ids}).to_csv(path, index=False)
    return path


def _parse(mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fetch_document_years.py", *argv])
    return mod.parse_args()


def test_year_parsed_as_int(mod):
    session = FakeSession(lambda doc: FakeResponse(200, _activities("2019")))
    status, year = mod.fetch_document_year("CHEMBL1", session)
    assert status == "ok" and year == 2019 and isinstance(year, int)

    session = FakeSession(lambda doc: FakeResponse(200, _activities(2005)))
    status, year = mod.fetch_document_year("CHEMBL1", session)
    assert status == "ok" and year == 2005 and isinstance(year, int)


def test_unparsable_year_counts_as_failed(mod):
    session = FakeSession(lambda doc: FakeResponse(200, _activities("not-a-year")))
    status, year = mod.fetch_document_year("CHEMBL1", session)
    assert status == "failed" and year is None


def test_retry_then_success(mod):
    attempts = {"n": 0}

    def handler(doc):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return FakeResponse(500)
        return FakeResponse(200, _activities(2017))

    session = FakeSession(handler)
    status, year = mod.fetch_document_year("CHEMBL1", session)
    assert (status, year) == ("ok", 2017)
    assert len(session.calls) == 3


def test_retry_exhaustion_recorded_and_run_continues(mod, tmp_path, monkeypatch, capsys):
    _write_raw(tmp_path, "t1", ["CHEMBL_BAD", "CHEMBL_OK", "CHEMBL_EMPTY"])

    def handler(doc):
        if doc == "CHEMBL_BAD":
            return FakeResponse(500)
        if doc == "CHEMBL_EMPTY":
            return FakeResponse(200, {"activities": []})
        return FakeResponse(200, _activities(1998))

    fake = FakeSession(handler)
    monkeypatch.setattr(mod, "get_session", lambda: fake)
    monkeypatch.setattr(mod, "RAW_DIR", str(tmp_path))
    args = _parse(mod, monkeypatch, "--tags", "t1",
                  "--out", str(tmp_path / "years.csv"))

    stats = mod.run(args)  # must not raise despite the permanent 500

    assert stats["got"] == 1 and stats["missing"] == 1 and stats["failed"] == 1
    assert stats["attempted"] == 3 and stats["skipped"] == 0
    assert fake.calls.count("CHEMBL_BAD") == mod.MAX_TRIES  # exhausted every retry

    out = pd.read_csv(tmp_path / "years.csv", dtype=str, keep_default_na=False)
    years = dict(zip(out["document_chembl_id"], out["document_year"]))
    assert years == {"CHEMBL_BAD": "", "CHEMBL_EMPTY": "", "CHEMBL_OK": "1998"}

    printed = capsys.readouterr().out
    assert "failed=1" in printed and "CHEMBL_BAD" in printed  # failure summary


def test_resume_skips_recorded_ids(mod, tmp_path, monkeypatch):
    _write_raw(tmp_path, "t1", ["CHEMBL_DONE", "CHEMBL_EMPTY", "CHEMBL_NEW"])
    out = tmp_path / "years.csv"
    pd.DataFrame({"document_chembl_id": ["CHEMBL_DONE", "CHEMBL_EMPTY"],
                  "document_year": ["2001", ""]}).to_csv(out, index=False)

    fake = FakeSession(lambda doc: FakeResponse(200, _activities(2015)))
    monkeypatch.setattr(mod, "get_session", lambda: fake)
    monkeypatch.setattr(mod, "RAW_DIR", str(tmp_path))
    args = _parse(mod, monkeypatch, "--tags", "t1", "--out", str(out))

    stats = mod.run(args)

    assert stats["skipped"] == 1 and stats["attempted"] == 2 and stats["got"] == 2
    assert "CHEMBL_DONE" not in fake.calls  # already recorded -> never re-queried
    assert set(fake.calls) == {"CHEMBL_EMPTY", "CHEMBL_NEW"}

    # failure row (empty year) was retried; compaction leaves one row per doc
    result = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(result["document_chembl_id"]) == ["CHEMBL_DONE", "CHEMBL_EMPTY", "CHEMBL_NEW"]
    assert dict(zip(result["document_chembl_id"], result["document_year"])) == {
        "CHEMBL_DONE": "2001", "CHEMBL_EMPTY": "2015", "CHEMBL_NEW": "2015",
    }


def test_limit_caps_number_of_queries(mod, tmp_path, monkeypatch):
    _write_raw(tmp_path, "t1", [f"CHEMBL{i}" for i in range(5)])
    fake = FakeSession(lambda doc: FakeResponse(200, _activities(2010)))
    monkeypatch.setattr(mod, "get_session", lambda: fake)
    monkeypatch.setattr(mod, "RAW_DIR", str(tmp_path))
    args = _parse(mod, monkeypatch, "--tags", "t1", "--limit", "2",
                  "--out", str(tmp_path / "years.csv"))
    assert args.limit == 2

    stats = mod.run(args)

    assert stats["attempted"] == 2
    assert len(fake.calls) == 2
    out = pd.read_csv(tmp_path / "years.csv", dtype=str, keep_default_na=False)
    assert len(out) == 2
    assert set(out["document_chembl_id"]) == {"CHEMBL0", "CHEMBL1"}
