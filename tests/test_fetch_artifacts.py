"""Unit tests for scripts/fetch_artifacts.py (review findings N5, P3, P4, R3, R4).

The release manifest only proves the archives downloaded intact; after
extraction the script must also verify every raw file against the repo's
data/raw/sha256sums.txt and abort loudly when they disagree — e.g. for the
artifacts-v1 release, whose raw CSVs predate the 2026-10-11 refresh.

P3/R3 hardening: the repo manifest is the trust anchor — it is read before
any extraction, an archive that ships any member matching the anchor
(case-insensitively, for macOS) is refused, symlink/hardlink members are
refused, every member must sit under a per-asset prefix allowlist, the
verification manifest always comes from memory (never from disk), and after
the move the on-disk anchor must still be a regular file whose bytes equal
the copy read at start. R4: an asset counts as installed only when ALL its
key files exist; a stale local raw copy is repaired by download+verify+atomic
replace (with rollback on failure) instead of a fatal lock-out; the release
manifest is re-downloaded every run. P4: the fingerprint cache asset extracts
through the same temp flow; the download cache is bucketed by release tag.

Everything runs offline in a tmpdir: `run()` accepts injectable
`fetch_assets` / `downloader` callables, so no network is touched.
"""
import hashlib
import io
import os
from pathlib import Path
import shutil
import tarfile

import pytest

FRESH = b"molecule_chembl_id,standard_relation\nCHEMBL1,=\n"
STALE = b"molecule_chembl_id\nCHEMBL1\n"  # artifacts-v1 shape: no standard_relation
TRUST_ANCHOR = "data/raw/sha256sums.txt"
TAGS = ["a2a", "abl1", "bace", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2"]
MODEL_FILES = (
    [f"models/rf_random_split_{tag}.joblib" for tag in TAGS]
    + [f"models/rf_scaffold_split_{tag}.joblib" for tag in TAGS]
)
LEGACY_TIME_FILES = [f"models/rf_time_split_{tag}.joblib" for tag in TAGS if tag != "bace"]
CACHE_FILES = [f"data/processed/{tag}_fingerprints.npz" for tag in TAGS]


def _api_url(tag):
    return (
        f"https://api.github.com/repos/LessRAY3-3/qsar-activity-prediction"
        f"/releases/tags/{tag}"
    )


def _build_archive(path, members):
    """Create a tar.gz with the given entries.

    Values are bytes (regular file), ("symlink", target), ("hardlink",
    target), or ("dir",) for a directory member.
    """
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            if isinstance(payload, bytes):
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            elif payload[0] == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload[1]
                tar.addfile(info)
            elif payload[0] == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = payload[1]
                tar.addfile(info)
            elif payload[0] == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                raise ValueError(f"unknown member kind: {payload!r}")


def _write_anchor(repo, payload=FRESH):
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    (raw_dir / "sha256sums.txt").write_text(f"# comment line\n{digest}  2  egfr_activities.csv\n")


def _setup(tmp_path, extracted_payload, repo_sums_payload=None, extra_members=None,
           on_download=None, tag="artifacts-v2"):
    """Return (repo_root, fetch_assets, downloader) for an offline run().

    - repo_root/data/raw/sha256sums.txt pins `repo_sums_payload` (default:
      the same bytes the archive extracts, i.e. a matching repo).
    - The fake downloader serves BOTH release assets the real one would: the
      manifest (sha256sums.txt, re-downloaded every run) and the archive.
    - `extra_members` adds archive members (e.g. a smuggled manifest);
      `on_download(repo)` runs inside the downloader, i.e. while `run()` is
      in flight, to simulate writes that would otherwise happen at
      extraction time.
    """
    if repo_sums_payload is None:
        repo_sums_payload = extracted_payload
    repo = tmp_path / "repo"
    _write_anchor(repo, repo_sums_payload)

    archive = tmp_path / "raw_data.tar.gz"
    members = {"data/raw/egfr_activities.csv": extracted_payload}
    members.update(extra_members or {})
    _build_archive(archive, members)
    cache = repo / ".artifacts_cache" / tag
    cache.mkdir(parents=True)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sums_text = f"{archive_digest}  raw_data.tar.gz\n"

    urls = {
        "sha256sums.txt": "https://example.invalid/sha256sums.txt",
        "raw_data.tar.gz": "https://example.invalid/raw_data.tar.gz",
    }

    def downloader(url, dest, label=""):
        if on_download is not None:
            on_download(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("sha256sums.txt"):
            dest.write_text(sums_text)
        else:
            shutil.copy(archive, dest)

    return repo, (lambda api_url=...: urls), downloader


def _offline_asset(tmp_path, repo, archive_name, members, tag="artifacts-v2"):
    """Build a release archive + fake downloader for a non-raw asset.

    Returns (fetch_assets, downloader) serving `archive_name` and the
    manifest that pins it.
    """
    archive = tmp_path / archive_name
    _build_archive(archive, members)
    cache = repo / ".artifacts_cache" / tag
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sums_text = f"{digest}  {archive_name}\n"
    urls = {
        "sha256sums.txt": "https://example.invalid/sha256sums.txt",
        archive_name: f"https://example.invalid/{archive_name}",
    }

    def downloader(url, dest, label=""):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("sha256sums.txt"):
            dest.write_text(sums_text)
        else:
            shutil.copy(archive, dest)

    return (lambda api_url=...: urls), downloader


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
    # document_years.csv installs together with the activities CSV. Ancestor
    # directory members ("data") are allowed, but a non-manifest file under
    # data/raw/ is never installed (only verified manifest files move).
    mod = load_script("fetch_artifacts")
    years = b"document_chembl_id,document_year\nCHEMBL1,2004\n"
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={
            "data": ("dir",),
            "data/raw/document_years.csv": years,
            "data/raw/unpinned.txt": b"not in the manifest\n",
        },
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
    assert not (repo / "data" / "raw" / "unpinned.txt").exists()  # not verified -> not installed
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
    # so completeness of all 9 npz key files is the check.
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    npz_payloads = {name: b"PK\x03\x04 fake npz payload" for name in CACHE_FILES}
    fetch_assets, downloader = _offline_asset(
        tmp_path, repo, "processed_cache.tar.gz", npz_payloads
    )

    mod.run(["cache"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    out = capsys.readouterr().out
    for name in CACHE_FILES:
        assert (repo / name).read_bytes() == npz_payloads[name]
    assert not (repo / ".artifacts_cache" / "artifacts-v2" / "_extract_tmp").exists()
    assert "done." in out


# --- R3: trust-anchor hardening — every attack vector refused ---


def test_symlink_members_refused(tmp_path, load_script):
    # R3: a directory symlink inside the allowlist (the shape of the PR #6
    # attack: data/raw -> payload) and a file symlink are both refused
    # member-by-member, before extraction — the tar data filter would follow
    # in-tree links, which is how a forged manifest used to slip past a
    # filename-only check.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={
            "data/raw/payload/egfr_activities.csv": FRESH,
            "data/raw/evil_dir": ("symlink", "payload"),
            "data/raw/sneaky.csv": ("symlink", "egfr_activities.csv"),
        },
    )
    manifest = repo / "data" / "raw" / "sha256sums.txt"
    original = manifest.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "symbolic link" in msg
    assert "data/raw/evil_dir" in msg
    assert "data/raw/sneaky.csv" in msg
    assert manifest.read_bytes() == original
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()
    assert not (repo / "data" / "raw" / "evil_dir").exists()
    assert not (repo / "data" / "raw" / "sneaky.csv").exists()


def test_hardlink_member_refused(tmp_path, load_script):
    # R3: hardlink members are refused the same way (a hardlink to a
    # manifest file could alias the trust anchor's inode on the same fs).
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={
            "data/raw/hard.csv": ("hardlink", "data/raw/egfr_activities.csv"),
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "hard link" in msg
    assert "data/raw/hard.csv" in msg
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()
    assert not (repo / "data" / "raw" / "hard.csv").exists()


def test_case_variant_anchor_member_refused(tmp_path, load_script):
    # R3: on macOS, data/raw/SHA256SUMS.txt IS data/raw/sha256sums.txt. The
    # anchor comparison is case-insensitive, so both variants are refused by
    # name before anything is extracted.
    mod = load_script("fetch_artifacts")
    forged = b"# forged\n0000  0  nothing\n"
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={
            "data/raw/SHA256SUMS.txt": forged,
            "Data/Raw/SHA256SUMS.txt": forged,
        },
    )
    manifest = repo / "data" / "raw" / "sha256sums.txt"
    original = manifest.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "trust anchor member" in msg
    assert "data/raw/SHA256SUMS.txt" in msg
    assert "Data/Raw/SHA256SUMS.txt" in msg
    assert manifest.read_bytes() == original
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_members_outside_allowlist_refused(tmp_path, load_script):
    # R3: a release archive can never install unverified content elsewhere
    # in the repo — results/metrics_*.json (verified by gnn_01), scripts/,
    # and .git/hooks/ were the PR #6 examples.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={
            "results/metrics_egfr.json": b'{"r2": 0.99}\n',
            ".git/hooks/pre-commit": b"#!/bin/sh\necho pwned\n",
            "scripts/02_clean_data.py": b"# replaced pipeline script\n",
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "outside the allowed prefix data/raw" in msg
    assert "results/metrics_egfr.json" in msg
    assert ".git/hooks/pre-commit" in msg
    assert "scripts/02_clean_data.py" in msg
    assert not (repo / "results" / "metrics_egfr.json").exists()
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()
    assert not (repo / "scripts" / "02_clean_data.py").exists()
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_models_archive_member_outside_allowlist_refused(tmp_path, load_script):
    # The allowlist is per-asset (models/ for models.tar.xz), not hardcoded
    # to the raw prefix.
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    fetch_assets, downloader = _offline_asset(
        tmp_path,
        repo,
        "models.tar.xz",
        {
            "models/rf_random_split_egfr.joblib": b"model",
            "results/metrics_egfr.json": b'{"r2": 0.99}\n',
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "outside the allowed prefix models" in msg
    assert "results/metrics_egfr.json" in msg
    assert not (repo / "results" / "metrics_egfr.json").exists()
    assert not (repo / "models" / "rf_random_split_egfr.joblib").exists()


def test_unsafe_paths_refused(tmp_path, load_script):
    # R3: `..` and absolute member paths are refused before extraction.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(
        tmp_path,
        extracted_payload=FRESH,
        extra_members={"../escape.txt": b"escaped", "/abs.txt": b"absolute"},
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "unsafe path" in msg
    assert "../escape.txt" in msg
    assert "/abs.txt" in msg
    assert not (repo.parent / "escape.txt").exists()
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


def test_verify_uses_memory_manifest_not_disk_copy(tmp_path, load_script):
    # R3: verification must use the manifest passed in (the copy read at the
    # start of run()), never one found on disk under the verified base —
    # that is how a smuggled or clobbered file would pass itself.
    mod = load_script("fetch_artifacts")
    base = tmp_path / "base"
    (base / "data" / "raw").mkdir(parents=True)
    (base / "data" / "raw" / "egfr_activities.csv").write_bytes(STALE)
    forged = f"# forged\n{hashlib.sha256(STALE).hexdigest()}  2  egfr_activities.csv\n"
    (base / "data" / "raw" / "sha256sums.txt").write_text(forged)
    expected = {"egfr_activities.csv": hashlib.sha256(FRESH).hexdigest()}

    with pytest.raises(SystemExit) as excinfo:
        mod.verify_raw_against_manifest(base, expected)

    assert "egfr_activities.csv" in str(excinfo.value)


def test_anchor_bytes_changed_during_run_detected(tmp_path, load_script):
    # R3: after the move, the on-disk anchor is re-read and must equal the
    # bytes captured at the start of run() — a mid-run rewrite (even one that
    # keeps the same parsed pins) is refused.
    mod = load_script("fetch_artifacts")

    def clobber(repo):
        (repo / "data" / "raw" / "sha256sums.txt").write_text(
            f"# forged\n{hashlib.sha256(FRESH).hexdigest()}  2  egfr_activities.csv\n"
        )

    repo, fetch_assets, downloader = _setup(
        tmp_path, extracted_payload=FRESH, on_download=clobber
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "trust anchor check failed" in msg
    assert TRUST_ANCHOR in msg


def test_anchor_swapped_for_symlink_detected(tmp_path, load_script):
    # R3: a symlink swapped in for the anchor mid-run is refused even when
    # its target has the right bytes.
    mod = load_script("fetch_artifacts")

    def clobber(repo):
        anchor = repo / "data" / "raw" / "sha256sums.txt"
        target = repo / "data" / "raw" / "anchor_copy.txt"
        target.write_bytes(anchor.read_bytes())
        anchor.unlink()
        anchor.symlink_to("anchor_copy.txt")

    repo, fetch_assets, downloader = _setup(
        tmp_path, extracted_payload=FRESH, on_download=clobber
    )

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    msg = str(excinfo.value)
    assert "trust anchor check failed" in msg
    assert "symlink" in msg


# --- R4: self-repair instead of lock-out; full key-file completeness ---


def test_stale_local_raw_repaired_not_locked_out(tmp_path, load_script, capsys):
    # R4: the artifacts-v1 user (old CSV on disk, no cache) used to hit a
    # fatal "already extracted" abort whose advice did not work. Now the
    # stale file is named, the archive is re-downloaded and the verified copy
    # atomically replaces it.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)
    local = repo / "data" / "raw" / "egfr_activities.csv"
    local.write_bytes(STALE)
    calls = []

    def spy(url, dest, label=""):
        calls.append(url)
        downloader(url, dest, label)

    mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=spy)

    out = capsys.readouterr().out
    assert local.read_bytes() == FRESH  # repaired
    assert "already extracted" not in out
    assert "re-downloading raw_data.tar.gz" in out
    assert "egfr_activities.csv: sha256" in out  # the stale file is named
    assert any(url.endswith("raw_data.tar.gz") for url in calls)
    assert "replaced: data/raw/egfr_activities.csv" in out
    assert "done." in out


def test_partial_raw_key_files_refilled(tmp_path, load_script, capsys):
    # R4: the raw asset counts as installed only when EVERY manifest file is
    # present; egfr_activities.csv missing locally triggers the fetch, and the
    # already-matching document_years.csv is kept, not rewritten.
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
    local = repo / "data" / "raw" / "document_years.csv"
    local.write_bytes(years)  # first sorted key file present, second missing

    mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    out = capsys.readouterr().out
    assert (repo / "data" / "raw" / "egfr_activities.csv").read_bytes() == FRESH
    assert local.read_bytes() == years
    assert "already extracted" not in out
    assert "replaced: data/raw/egfr_activities.csv" in out
    assert "kept up to date: data/raw/document_years.csv" in out


def test_single_model_file_does_not_mark_asset_installed(tmp_path, load_script, capsys):
    # R4: one stray local file used to make the whole models asset count as
    # installed. Now all 26 joblibs must be present as regular files; a
    # symlink standing in for one key file triggers the fetch, which replaces
    # it with the verified regular file. A second run then skips cleanly.
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    # all 18 supported models present locally, but the first key file is a SYMLINK —
    # the old code treated one probe file as the whole asset being installed,
    # and a symlink probe must not count as installed either
    blob = repo / "models" / "old_model_blob.bin"
    blob.parent.mkdir()
    blob.write_bytes(b"old model")
    for name in MODEL_FILES:
        (repo / name).write_bytes(b"old model")
    probe = repo / "models" / "rf_random_split_a2a.joblib"
    probe.unlink()
    probe.symlink_to("old_model_blob.bin")
    members = {name: f"model:{name}".encode() for name in MODEL_FILES}
    fetch_assets, downloader = _offline_asset(tmp_path, repo, "models.tar.xz", members)
    calls = []

    def spy(url, dest, label=""):
        calls.append(url)
        downloader(url, dest, label)

    mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=spy)

    out = capsys.readouterr().out
    assert "already extracted" not in out
    assert any(url.endswith("models.tar.xz") for url in calls)
    for name in MODEL_FILES:
        assert (repo / name).read_bytes() == f"model:{name}".encode()
    assert not probe.is_symlink()  # the symlink was replaced by a real file

    # idempotent second run: all 26 present -> skip, no re-download
    downloads_before = sum(url.endswith("models.tar.xz") for url in calls)
    mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=spy)
    out2 = capsys.readouterr().out
    downloads_after = sum(url.endswith("models.tar.xz") for url in calls)
    assert "already extracted, skipping: models.tar.xz" in out2
    assert downloads_after == downloads_before


def test_partial_cache_refilled(tmp_path, load_script, capsys):
    # R4: a single local npz no longer marks the cache asset installed; all
    # 9 fingerprint caches are fetched.
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    probe = repo / "data" / "processed" / "a2a_fingerprints.npz"
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"old npz")
    npz_payloads = {name: b"PK\x03\x04 fresh" for name in CACHE_FILES}
    fetch_assets, downloader = _offline_asset(
        tmp_path, repo, "processed_cache.tar.gz", npz_payloads
    )

    mod.run(["cache"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    out = capsys.readouterr().out
    assert "already extracted" not in out
    for name in CACHE_FILES:
        assert (repo / name).read_bytes() == b"PK\x03\x04 fresh"


def test_failed_install_rolls_back(tmp_path, load_script, monkeypatch):
    # R4: installation is atomic in spirit — if any per-file replace fails,
    # already-replaced files are rolled back from their backups and nothing
    # half-installed (and no leftover backup) is left behind.
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
    stale = repo / "data" / "raw" / "egfr_activities.csv"
    stale.write_bytes(STALE)
    local_years = b"old document years\n"
    (repo / "data" / "raw" / "document_years.csv").write_bytes(local_years)

    real_replace = os.replace

    def flaky(src, dst):
        if str(dst).endswith("egfr_activities.csv"):
            raise OSError("simulated disk failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    assert "rolled back" in str(excinfo.value)
    assert stale.read_bytes() == STALE  # old file kept
    # the already-replaced document_years.csv was restored from its backup
    assert (repo / "data" / "raw" / "document_years.csv").read_bytes() == local_years
    assert not list((repo / "data" / "raw").glob("*.fetch_backup"))


def test_install_refuses_directory_destination(tmp_path, load_script):
    # R4: a directory squatting on a file path aborts BEFORE any file is
    # touched, with a clear SystemExit (no traceback, no partial install).
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)
    squat = repo / "data" / "raw" / "egfr_activities.csv"
    squat.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    assert "directory" in str(excinfo.value)
    assert squat.is_dir()  # untouched


# --- release-manifest freshness and archive hash checks ---


def test_release_manifest_refetched_every_run(tmp_path, load_script, capsys):
    # Low: release assets may be re-uploaded in place (artifacts-v2 was), so
    # the cached sha256sums.txt is re-downloaded and used, not trusted.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)
    bucket = repo / ".artifacts_cache" / "artifacts-v2"
    zeros = "0" * 64
    (bucket / "sha256sums.txt").write_text(f"{zeros}  raw_data.tar.gz\n")
    fetched = []

    def spy(url, dest, label=""):
        fetched.append(url)
        downloader(url, dest, label)

    mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=spy)

    out = capsys.readouterr().out
    assert "downloading sha256sums.txt" in out
    assert any(url.endswith("sha256sums.txt") for url in fetched)
    assert zeros not in (bucket / "sha256sums.txt").read_text()  # cache replaced
    assert (repo / "data" / "raw" / "egfr_activities.csv").read_bytes() == FRESH


def test_cached_archive_hash_mismatch_redownloaded(tmp_path, load_script, capsys):
    # A tampered cached archive never satisfies the freshly downloaded
    # release manifest: it is discarded and re-downloaded.
    mod = load_script("fetch_artifacts")
    repo, fetch_assets, downloader = _setup(tmp_path, extracted_payload=FRESH)
    bucket = repo / ".artifacts_cache" / "artifacts-v2"
    (bucket / "raw_data.tar.gz").write_bytes(b"tampered cache bytes")

    mod.run(["raw"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)

    out = capsys.readouterr().out
    assert "cache stale for raw_data.tar.gz" in out
    assert "downloading raw_data.tar.gz" in out
    assert (repo / "data" / "raw" / "egfr_activities.csv").read_bytes() == FRESH


def test_archive_bytes_must_match_release_manifest(tmp_path, load_script):
    # A downloaded archive whose sha256 disagrees with the release manifest
    # is refused: the cache is cleared and nothing is installed.
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    archive_a = tmp_path / "a.tar.gz"
    _build_archive(archive_a, {"data/raw/egfr_activities.csv": FRESH})
    archive_b = tmp_path / "b.tar.gz"
    _build_archive(archive_b, {"data/raw/egfr_activities.csv": STALE})
    sums_text = f"{hashlib.sha256(archive_a.read_bytes()).hexdigest()}  raw_data.tar.gz\n"
    urls = {
        "sha256sums.txt": "https://example.invalid/sha256sums.txt",
        "raw_data.tar.gz": "https://example.invalid/raw_data.tar.gz",
    }

    def downloader(url, dest, label=""):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("sha256sums.txt"):
            dest.write_text(sums_text)
        else:
            shutil.copy(archive_b, dest)

    with pytest.raises(SystemExit) as excinfo:
        mod.run(
            ["raw"],
            repo_root=repo,
            fetch_assets=lambda api_url=...: urls,
            downloader=downloader,
        )

    msg = str(excinfo.value)
    assert "verification FAILED" in msg
    assert "raw_data.tar.gz" in msg
    assert not (repo / "data" / "raw" / "egfr_activities.csv").exists()


@pytest.mark.parametrize("missing_tag", TAGS)
def test_models_repair_preserves_all_accepted_time_checkpoints(tmp_path, load_script, capsys, missing_tag):
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    (repo / "models").mkdir()
    accepted = {name: f"accepted-asof:{name}".encode() for name in LEGACY_TIME_FILES}
    for name, payload in accepted.items():
        (repo / name).write_bytes(payload)
    missing = f"models/rf_random_split_{missing_tag}.joblib"
    for name in MODEL_FILES:
        if name != missing:
            (repo / name).write_bytes(f"model:{name}".encode())
    members = {name: f"model:{name}".encode() for name in MODEL_FILES}
    members.update({name: f"legacy:{name}".encode() for name in LEGACY_TIME_FILES})
    fetch_assets, downloader = _offline_asset(tmp_path, repo, "models.tar.xz", members)
    mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)
    assert (repo / missing).read_bytes() == members[missing]
    assert {name: (repo / name).read_bytes() for name in accepted} == accepted
    assert "skipping legacy rf_time_split_*" in capsys.readouterr().out
    assert not list((repo / "models").glob("*.fetch_backup"))


def test_fresh_models_install_excludes_legacy_time_and_other_extra_files(tmp_path, load_script):
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    members = {name: b"supported model" for name in MODEL_FILES}
    members.update({name: b"legacy time model" for name in LEGACY_TIME_FILES})
    members["models/extra.joblib"] = b"unrequested model"
    fetch_assets, downloader = _offline_asset(tmp_path, repo, "models.tar.xz", members)
    mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)
    assert all((repo / name).read_bytes() == b"supported model" for name in MODEL_FILES)
    assert all(not (repo / name).exists() for name in LEGACY_TIME_FILES)
    assert not (repo / "models/extra.joblib").exists()
    # Missing temporal models never cause another download of legacy assets.
    def manifest_only(url, dest, label=""):
        assert url.endswith("sha256sums.txt"), "Valid receipt must skip the archive download"
        downloader(url, dest, label=label)
    mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=manifest_only)


@pytest.mark.parametrize("member,payload", [
    ("models/rf_time_split_herg.joblib", ("symlink", "rf_random_split_herg.joblib")),
    ("models/rf_time_split_herg.joblib", ("hardlink", "models/rf_random_split_herg.joblib")),
    ("models/../rf_time_split_herg.joblib", b"unsafe traversal"),
    ("results/rf_time_split_herg.joblib", b"outside models"),
])
def test_skipped_model_members_still_require_archive_safety(tmp_path, load_script, member, payload):
    mod = load_script("fetch_artifacts")
    repo = tmp_path / "repo"
    _write_anchor(repo)
    members = {name: b"model" for name in MODEL_FILES}
    members[member] = payload
    fetch_assets, downloader = _offline_asset(tmp_path, repo, "models.tar.xz", members)
    with pytest.raises(SystemExit, match="unsafe archive members"):
        mod.run(["models"], repo_root=repo, fetch_assets=fetch_assets, downloader=downloader)
    assert not (repo / "models").exists()


def test_cache_installs_only_declared_keys_and_ignores_tar_metadata(tmp_path, load_script):
    mod = load_script('fetch_artifacts')
    repo = tmp_path / 'repo'
    _write_anchor(repo)
    protected = repo / 'data/processed/splits/a2a_time.npz'
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b'accepted split')
    members = {p: b'fingerprint' for p in CACHE_FILES}
    members.update({'.': ('dir',), '._data': b'metadata',
                    'data/processed/._a2a_fingerprints.npz': b'metadata',
                    'data/processed/splits/a2a_time.npz': b'unverified overwrite'})
    fetch, dl = _offline_asset(tmp_path, repo, 'processed_cache.tar.gz', members)
    mod.run(['cache'], repo_root=repo, fetch_assets=fetch, downloader=dl)
    assert protected.read_bytes() == b'accepted split'
    assert not (repo / 'data/processed/._a2a_fingerprints.npz').exists()


@pytest.mark.parametrize('kind,archive,paths', [
    ('models', 'models.tar.xz', MODEL_FILES),
    ('cache', 'processed_cache.tar.gz', CACHE_FILES),
])
def test_complete_but_corrupt_install_is_repaired(tmp_path, load_script, kind, archive, paths):
    mod = load_script('fetch_artifacts')
    repo = tmp_path / 'repo'
    _write_anchor(repo)
    fetch, dl = _offline_asset(tmp_path, repo, archive, {p: b'current' for p in paths})
    mod.run([kind], repo_root=repo, fetch_assets=fetch, downloader=dl)
    stale = repo / paths[0]
    stale.write_bytes(b'changed locally')
    mod.run([kind], repo_root=repo, fetch_assets=fetch, downloader=dl)
    assert stale.read_bytes() == b'current'
    backups = list((repo / '.artifacts_cache/backups').rglob(stale.name))
    assert any(p.read_bytes() == b'changed locally' for p in backups)
    # Complete legacy installations without a receipt must also be verified.
    (repo / '.artifacts_cache/artifacts-v2' / f'{kind}_installed.json').unlink()
    stale.write_bytes(b'legacy')
    mod.run([kind], repo_root=repo, fetch_assets=fetch, downloader=dl)
    assert stale.read_bytes() == b'current'


def test_install_keyboard_interrupt_restores_old_files(tmp_path, load_script, monkeypatch):
    mod = load_script('fetch_artifacts')
    repo, source = tmp_path / 'repo', tmp_path / 'source'
    repo.mkdir()
    source.mkdir()
    for name in ['a', 'b']:
        (repo / name).write_bytes(b'old')
        (source / name).write_bytes(b'new')
    real = os.replace
    def interrupt(src, dst):
        if Path(src) == source / 'b':
            raise KeyboardInterrupt()
        return real(src, dst)
    monkeypatch.setattr(mod.os, 'replace', interrupt)
    with pytest.raises(SystemExit, match='rolled back'):
        mod.install_verified_files(repo, source, ['a', 'b'])
    assert all((repo / p).read_bytes() == b'old' for p in ['a', 'b'])


def test_failed_rollback_preserves_backup_and_reports_incomplete(tmp_path, load_script, monkeypatch):
    mod = load_script('fetch_artifacts')
    repo, source = tmp_path / 'repo', tmp_path / 'source'
    repo.mkdir()
    source.mkdir()
    for name in ['a', 'b']:
        (repo / name).write_bytes(b'old')
        (source / name).write_bytes(b'new')
    real = os.replace
    def fail(src, dst):
        if Path(src) == source / 'b' or str(src).endswith('a.fetch_backup'):
            raise OSError('injected failure')
        return real(src, dst)
    monkeypatch.setattr(mod.os, 'replace', fail)
    with pytest.raises(SystemExit, match='rollback incomplete'):
        mod.install_verified_files(repo, source, ['a', 'b'])
    assert (repo / 'a.fetch_backup').read_bytes() == b'old'
    assert (repo / 'a').read_bytes() == b'new'
    assert (repo / 'b').read_bytes() == b'old'
    with pytest.raises(SystemExit, match='existing backup'):
        mod.install_verified_files(repo, source, ['a'])
    assert (repo / 'a.fetch_backup').read_bytes() == b'old'


def test_models_prefix_boundary_rejected(tmp_path, load_script):
    mod = load_script('fetch_artifacts')
    archive = tmp_path / 'bad.tar.gz'
    _build_archive(archive, {'models_evil/payload.py': b'wrong prefix'})
    with pytest.raises(SystemExit, match='outside the allowed prefix'):
        mod.extract_archive(archive, tmp_path / 'out', 'models/')


def test_release_reupload_invalidates_complete_install_receipt(tmp_path, load_script):
    mod = load_script('fetch_artifacts')
    repo = tmp_path / 'repo'
    _write_anchor(repo)
    fetch, dl = _offline_asset(tmp_path, repo, 'models.tar.xz',
                               {p: b'first release' for p in MODEL_FILES})
    mod.run(['models'], repo_root=repo, fetch_assets=fetch, downloader=dl)
    fetch, dl = _offline_asset(tmp_path, repo, 'models.tar.xz',
                               {p: b'replaced release' for p in MODEL_FILES})
    mod.run(['models'], repo_root=repo, fetch_assets=fetch, downloader=dl)
    assert all((repo / p).read_bytes() == b'replaced release' for p in MODEL_FILES)
    assert not any((repo / p).exists() for p in LEGACY_TIME_FILES)


def test_appledouble_link_is_rejected_not_hidden(tmp_path, load_script):
    mod = load_script('fetch_artifacts')
    archive = tmp_path / 'link.tar.gz'
    _build_archive(archive, {'models/._payload': ('symlink', 'elsewhere')})
    with pytest.raises(SystemExit, match='symbolic link'):
        mod.extract_archive(archive, tmp_path / 'out', 'models/')


def test_receipt_cannot_skip_symlink_parent(tmp_path, load_script):
    mod = load_script('fetch_artifacts')
    repo = tmp_path / 'repo'
    _write_anchor(repo)
    fetch, dl = _offline_asset(tmp_path, repo, 'models.tar.xz',
                               {p: b'model bytes' for p in MODEL_FILES})
    mod.run(['models'], repo_root=repo, fetch_assets=fetch, downloader=dl)
    elsewhere = tmp_path / 'elsewhere'
    (repo / 'models').rename(elsewhere)
    (repo / 'models').symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(SystemExit, match='parent is a symlink'):
        mod.run(['models'], repo_root=repo, fetch_assets=fetch, downloader=dl)
    assert all(p.read_bytes() == b'model bytes' for p in elsewhere.glob('*.joblib'))


def test_backup_archival_failure_reports_committed_state(tmp_path, load_script, monkeypatch):
    mod = load_script('fetch_artifacts')
    repo, source = tmp_path / 'repo', tmp_path / 'source'
    repo.mkdir()
    source.mkdir()
    (repo / 'model').write_bytes(b'old')
    (source / 'model').write_bytes(b'new')
    real = os.replace
    def fail(src, dst):
        if 'backups' in Path(dst).parts:
            raise OSError('archive failure')
        return real(src, dst)
    monkeypatch.setattr(mod.os, 'replace', fail)
    with pytest.raises(SystemExit, match='verified files installed, but backup archival incomplete'):
        mod.install_verified_files(repo, source, ['model'])
    assert (repo / 'model').read_bytes() == b'new'
    assert (repo / 'model.fetch_backup').read_bytes() == b'old'
