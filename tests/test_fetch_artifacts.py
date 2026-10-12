"""Unit tests for scripts/fetch_artifacts.py (review finding N5).

The release manifest only proves the archives downloaded intact; after
extraction the script must also verify every raw file against the repo's
data/raw/sha256sums.txt and abort loudly when they disagree — e.g. for the
artifacts-v1 release, whose raw CSVs predate the 2026-10-11 refresh.

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


def _build_archive(path, payload):
    """Create raw_data.tar.gz containing data/raw/egfr_activities.csv."""
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name="data/raw/egfr_activities.csv")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def _setup(tmp_path, extracted_payload, repo_sums_payload=None):
    """Return (repo_root, fetch_assets, downloader) for an offline run().

    - repo_root/data/raw/sha256sums.txt pins `repo_sums_payload` (default:
      the same bytes the archive extracts, i.e. a matching repo).
    - repo_root/.artifacts_cache/sha256sums.txt is the release manifest for
      the built archive, so the release-manifest check passes in both the
      match and mismatch scenarios.
    """
    if repo_sums_payload is None:
        repo_sums_payload = extracted_payload
    repo = tmp_path / "repo"
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True)
    repo_digest = hashlib.sha256(repo_sums_payload).hexdigest()
    (raw_dir / "sha256sums.txt").write_text(f"# comment line\n{repo_digest}  2  egfr_activities.csv\n")

    archive = tmp_path / "raw_data.tar.gz"
    _build_archive(archive, extracted_payload)
    cache = repo / ".artifacts_cache"
    cache.mkdir()
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (cache / "sha256sums.txt").write_text(f"{archive_digest}  raw_data.tar.gz\n")

    urls = {"raw_data.tar.gz": "https://example.invalid/raw_data.tar.gz"}

    def downloader(url, dest, label=""):
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
