"""Unit tests for scripts/fetch_artifacts.py (review findings N5, P3, P4).

The release manifest only proves the archives downloaded intact; after
extraction the script must also verify every raw file against the repo's
data/raw/sha256sums.txt and abort loudly when they disagree — e.g. for the
artifacts-v1 release, whose raw CSVs predate the 2026-10-11 refresh.

P3 hardening: the repo manifest is the trust anchor — it is read before any
extraction, an archive that ships its own data/raw/sha256sums.txt member is
refused, verification happens inside a temp directory, and document_years.csv
is covered by the manifest. P4: the fingerprint cache asset extracts through
the same temp flow; the download cache is bucketed by release tag.

Everything runs offline in a tmpdir: `run()` accepts injectable
`fetch_assets` / `downloader` callables, so no network is touched.
"""
import hashlib
import io
import shutil
import tarfile

import pytest

FRESH = b"molecule_chembl_id,standard_relation\nCHEMBL1,=\n"
STALE = b"molecule_chembl_id\nCHEMBL1\n"  # artifacts-v1 shape: no standard_relation
TRUST_ANCHOR = "data/raw/sha256sums.txt"


def _api_url(tag):
    return (
        f"https://api.github.com/repos/LessRAY3-3/qsar-activity-prediction"
        f"/releases/tags/{tag}"
    )


def _build_archive(path, members):
    """Create a tar.gz with the given {member name: bytes} entries."""
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _setup(tmp_path, extracted_payload, repo_sums_payload=None, extra_members=None,
           on_download=None, tag="artifacts-v2"):
    """Return (repo_root, fetch_assets, downloader) for an offline run().

    - repo_root/data/raw/sha256sums.txt pins `repo_sums_payload` (default:
      the same bytes the archive extracts, i.e. a matching repo).
    - repo_root/.artifacts_cache/<tag>/sha256sums.txt is the release
      manifest for the built archive (the cache is bucketed by release tag),
      so the release-manifest check passes in match and mismatch scenarios.
    - `extra_members` adds archive members (e.g. a smuggled manifest);
      `on_download(repo)` runs inside the downloader, i.e. while `run()` is
      in flight, to simulate writes that would otherwise happen at
      extraction time.
    """
    if repo_sums_payload is None:
        repo_sums_payload = extracted_payload
    repo = tmp_path / "repo"
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True)
    repo_digest = hashlib.sha256(repo_sums_payload).hexdigest()
    (raw_dir / "sha256sums.txt").write_text(f"# comment line\n{repo_digest}  2  egfr_activities.csv\n")

    archive = tmp_path / "raw_data.tar.gz"
    members = {"data/raw/egfr_activities.csv": extracted_payload}
    members.update(extra_members or {})
    _build_archive(archive, members)
    cache = repo / ".artifacts_cache" / tag
    cache.mkdir(parents=True)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (cache / "sha256sums.txt").write_text(f"{archive_digest}  raw_data.tar.gz\n")

    urls = {"raw_data.tar.gz": "https://example.invalid/raw_data.tar.gz"}

    def downloader(url, dest, label=""):
        if on_download is not None:
            on_download(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(archive, dest)

    return repo, (lambda api_url=...: urls), downloader


def test_raw_matching_repo_checksums_extracts(tmp_path, load_script, capsys):
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)

    mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    out = capsys.readouterr().out
    extracted = repo / "data" / "raw" / "egfr_activities.csv"
    assert extracted.read_bytes() == FRESH
    assert "repo checksums ok: 1 files match" in out
    assert "done." in out


def test_raw_stale_vs_repo_checksums_aborts(tmp_path, load_script):
    # The failure scenario from N5: release manifest passes (archive intact),
    # but the extracted raw file predates the 2026-10-11 refresh and does not
    # match the repo's data/raw/sha256sums.txt -> non-zero exit, loudly.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(
        tmp_path, extracted_payload=STALE, repo_sums_payload=FRESH
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "data/raw/sha256sums.txt" in msg
    assert "predates the 2026-10-11 data refresh" in msg
    assert "egfr_activities.csv" in msg


def test_raw_missing_from_repo_checksums_aborts(tmp_path, load_script):
    # A file listed in the repo's sums but absent after extraction must abort.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)
    sums_path = repo / "data" / "raw" / "sha256sums.txt"
    sums_path.write_text(
        sums_path.read_text()
        + f"{hashlib.sha256(FRESH).hexdigest()}  3  herg_activities.csv\n"
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    assert "missing herg_activities.csv" in str(excinfo.value)


def test_parse_sha256sums_handles_repo_format(tmp_path, load_script):
    mod = load_script("fetch_artifacts")
    text = (
        "# sha256  rows  filename  -- generated\n"
        f"{hashlib.sha256(FRESH).hexdigest()}  2  egfr_activities.csv\n"
        "abc123  other.csv\n"
    )
    sums = mod.parse_sha256sums(text)
    assert sums == {
        "egfr_activities.csv": hashlib.sha256(FRESH).hexdigest(),
        "other.csv": "abc123",
    }


def test_missing_trust_anchor_aborts_before_download(tmp_path, load_script):
    # P3(a): the repo manifest is read before anything else; without it,
    # nothing is downloaded and nothing is extracted.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)
    (repo / "data" / "raw" / "sha256sums.txt").unlink()

    calls = []

    def spy(url, dest, label=""):
        calls.append(dest)
        downloader(url, dest, label)

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=spy)

    assert "sha256sums.txt" in str(excinfo.value)
    assert calls == []  # no download happened
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_archive_with_smuggled_matching_manifest_rejected(tmp_path, load_script):
    # P3 attack: a stale CSV plus a forged manifest that matches it. The
    # archive must be refused before anything reaches data/raw/, leaving the
    # tracked trust anchor and the working tree untouched.
    mod = load_script("fetch_artifacts")
    forged = f"{hashlib.sha256(STALE).hexdigest()}  2  egfr_activities.csv\n"
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=STALE,
        repo_sums_payload=FRESH,
        extra_members={TRUST_ANCHOR: forged.encode()},
    )
    manifest = repo / "data" / "raw" / "sha256sums.txt"
    original = manifest.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert TRUST_ANCHOR in msg
    assert "trust anchor" in msg
    # the tracked trust anchor was NOT overwritten by the archive
    assert manifest.read_bytes() == original
    # and no half-installed files were left behind
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()
    assert not (repo / ".artifacts_cache" / "artifacts-v2" / "_extract_tmp").exists()


def test_archive_containing_manifest_member_refused(tmp_path, load_script):
    # Even a member byte-identical to the repo's manifest is refused: the
    # trust anchor must never come from an archive at all. (artifacts-v2's
    # current raw_data.tar.gz ships one, so the release asset has to be
    # repackaged without it.)
    mod = load_script("fetch_artifacts")
    matching = f"# comment line\n{hashlib.sha256(FRESH).hexdigest()}  2  egfr_activities.csv\n"
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        repo_sums_payload=FRESH,
        extra_members={TRUST_ANCHOR: matching.encode()},
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert TRUST_ANCHOR in msg
    assert "trust anchor" in msg
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_trust_anchor_read_before_extraction(tmp_path, load_script):
    # P3: the manifest must be read into memory before extraction could
    # clobber it. The downloader rewrites the on-disk manifest mid-flight
    # with a forged pin for the stale payload — verification must still fail
    # against the original pin read at the start of run().
    mod = load_script("fetch_artifacts")

    def clobber(repo):
        (repo / "data" / "raw" / "sha256sums.txt").write_text(
            f"# forged\n{hashlib.sha256(STALE).hexdigest()}  2  egfr_activities.csv\n"
        )

    repo, fetch_assets, downloader = _setup(
        tmp_path, extracted_payload=STALE, repo_sums_payload=FRESH, on_download=clobber
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "predates the 2026-10-11 data refresh" in msg
    # checked against the ORIGINAL (in-memory) pin, not the on-disk forged one
    assert f"expected {hashlib.sha256(FRESH).hexdigest()}" in msg
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_document_years_wrong_copy_aborts(tmp_path, load_script):
    # document_years.csv is pinned by the repo manifest (added for P3), so a
    # substituted copy aborts even though the activities CSV is correct.
    mod = load_script("fetch_artifacts")
    years_ok = b"document_chembl_id,document_year\nCHEMBL1,2004\n"
    years_bad = b"document_chembl_id,document_year\nCHEMBL1,1999\n"
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={"data/raw/document_years.csv": years_bad},
    )
    sums_path = repo / "data" / "raw" / "sha256sums.txt"
    sums_path.write_text(
        sums_path.read_text()
        + f"{hashlib.sha256(years_ok).hexdigest()}  2  document_years.csv\n"
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "document_years.csv" in msg
    assert "data/raw/sha256sums.txt" in msg
    # nothing installed — not even the correct egfr CSV
    assert not (repo / "data" / "raw" / "document_years.csv").exists()
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_document_years_matching_copy_extracts(tmp_path, load_script, capsys):
    # The positive counterpart: with the manifest entry present and matching,
    # document_years.csv installs together with the activities CSV.
    mod = load_script("fetch_artifacts")
    years = b"document_chembl_id,document_year\nCHEMBL1,2004\n"
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={"data/raw/document_years.csv": years},
    )
    sums_path = repo / "data" / "raw" / "sha256sums.txt"
    sums_path.write_text(
        sums_path.read_text()
        + f"{hashlib.sha256(years).hexdigest()}  2  document_years.csv\n"
    )

    mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    out = capsys.readouterr().out
    assert (repo / "data" / "raw" / "document_years.csv").read_bytes() == years
    assert (repo / "data" / "raw" / "egfr_activities.csv").read_bytes() == FRESH
    assert "repo checksums ok: 2 files match" in out
    assert "done." in out


def test_cache_bucketed_by_release_tag(tmp_path, load_script, capsys):
    # P3(g): downloads live in .artifacts_cache/<tag>/; a stale archive from
    # the old flat layout (or another release) is never reused.
    mod = load_script("fetch_artifacts")
    tag = "artifacts-v3"
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH, tag=tag)
    flat_stale = repo / ".artifacts_cache" / "raw_data.tar.gz"
    flat_stale.write_bytes(b"stale flat-layout archive")

    mod.run(
        ["raw"],
        repo_root=repo,
        api_url=_api_url(tag),
        fetch_assets=fetch_assets,
        downloader=downloader,
    )

    out = capsys.readouterr().out
    assert "downloading raw_data.tar.gz" in out  # bucket miss, flat file ignored
    bucket = repo / ".artifacts_cache" / tag
    assert (bucket / "raw_data.tar.gz").exists()
    assert (bucket / "sha256sums.txt").exists()
    assert flat_stale.read_bytes() == b"stale flat-layout archive"  # untouched
    assert not (repo / ".artifacts_cache" / "artifacts-v2").exists()
    assert (repo / "data" / "raw" / "egfr_activities.csv").read_bytes() == FRESH


def test_fingerprint_cache_extracts_via_temp(tmp_path, load_script, capsys):
    # P4: processed_cache.tar.gz ships data/processed/*_fingerprints.npz
    # through the same temp-extraction flow; caches have no repo manifest,
    # so key-file existence is the check.
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sha256sums.txt").write_text(
        f"# comment line\n{hashlib.sha256(FRESH).hexdigest()}  2  egfr_activities.csv\n"
    )
    npz = b"PK\x03\x04 fake npz payload"
    archive = tmp_path / "processed_cache.tar.gz"
    _build_archive(archive, {"data/processed/egfr_fingerprints.npz": npz})
    bucket = repo / ".artifacts_cache" / "artifacts-v2"
    bucket.mkdir(parents=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (bucket / "sha256sums.txt").write_text(f"{digest}  processed_cache.tar.gz\n")
    urls = {"processed_cache.tar.gz": "https://example.invalid/processed_cache.tar.gz"}

    def downloader(url, dest, label=""):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(archive, dest)

    mod.run(
        ["cache"],
        repo_root=repo,
        fetch_assets=lambda api_url=...: urls,
        downloader=downloader,
    )

    out = capsys.readouterr().out
    extracted = repo / "data" / "processed" / "egfr_fingerprints.npz"
    assert extracted.read_bytes() == npz
    assert not (bucket / "_extract_tmp").exists()  # temp dir cleaned up
    assert "done." in out
