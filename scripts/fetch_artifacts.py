#!/usr/bin/env python3
"""Download models, raw data and fingerprint caches from the `artifacts-v2` GitHub Release.

Assets are cached in <repo>/.artifacts_cache/<release-tag>/ (bucketed per
release, so an archive left over from another release — or from the old flat
layout — is never reused), checked against the release's own sha256sums.txt,
and unpacked into a temporary directory inside that cache, never straight
into the working tree.

Trust model for raw data (review finding P3): the repo's tracked
data/raw/sha256sums.txt is the trust anchor and is read into memory before
any archive is downloaded or extracted. An archive carrying its own
data/raw/sha256sums.txt member is refused — it could only be an attempt to
overwrite the trust anchor. Every file named by the in-memory manifest is
then sha256-verified inside the temporary directory (a missing file counts
as a failure), and only after the whole manifest passes are the files moved
into place. A failed check therefore never leaves half-installed data in
data/raw/, and the tracked manifest is never replaced by release content.

processed_cache.tar.gz (review finding P4) ships the gitignored
data/processed/*_fingerprints.npz caches so the fast path
(`gnn_01_make_splits.py` onward) runs without re-featurizing first. It has
no repo-side manifest; extraction is validated by key-file existence only.

Known limitation: models/ has no repo-side manifest either — a model file
already present in the working tree is reused as-is (only cached archives
are hash-checked against the release manifest).

Usage:
    python scripts/fetch_artifacts.py [--models | --raw | --cache | --all]

Standard library only; no third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR_NAME = ".artifacts_cache"
EXTRACT_TMP_NAME = "_extract_tmp"
SUMS_NAME = "sha256sums.txt"
TRUST_ANCHOR = "data/raw/sha256sums.txt"
RELEASE_TAG = "artifacts-v2"
API_URL = (
    "https://api.github.com/repos/LessRAY3-3/qsar-activity-prediction"
    f"/releases/tags/{RELEASE_TAG}"
)
USER_AGENT = "qsar-activity-prediction-fetch-artifacts"

# asset key -> (archive filename, key files that must exist after extraction)
ASSETS = {
    "models": ("models.tar.xz", ["models/rf_random_split_egfr.joblib"]),
    "raw": ("raw_data.tar.gz", ["data/raw/egfr_activities.csv"]),
    "cache": ("processed_cache.tar.gz", ["data/processed/egfr_fingerprints.npz"]),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_release_assets(api_url: str = API_URL) -> dict:
    """Return {asset_name: browser_download_url} from the GitHub release API."""
    request = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return {asset["name"]: asset["browser_download_url"] for asset in payload["assets"]}


def download(url: str, dest: Path, label: str = "") -> None:
    """Stream `url` to `dest` (via a .part temp file), with coarse progress."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_pct = 10
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= next_pct:
                    print(f"  {label or dest.name}: {pct}% ({done}/{total} bytes)", flush=True)
                    next_pct += 10
    tmp.replace(dest)


def parse_sha256sums(text: str) -> dict:
    """Parse checksum output: '<hex>  <filename>' / '<hex> *<filename>'.

    The repo's data/raw/sha256sums.txt is '<hex>  <rows>  <filename>' with a
    leading '#' comment line; both shapes are handled (filename = last token).
    """
    sums = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        sums[name] = digest.lower()
    return sums


def cache_bucket(api_url: str) -> str:
    """Return the release tag named in `api_url`; the cache is bucketed by it.

    Keeps a cached archive from a different release (or from the pre-bucketing
    flat `.artifacts_cache/` layout) from ever being reused.
    """
    if "/tags/" in api_url:
        return api_url.rsplit("/tags/", 1)[1].strip("/") or "untagged"
    return "untagged"


def read_trust_anchor(repo_root: Path) -> dict:
    """Read the repo's tracked data/raw/sha256sums.txt into memory.

    `run()` calls this before any download or extraction: the in-memory copy
    is the only manifest raw data is ever verified against, so neither an
    archive nor any other write can replace the trust anchor first (P3).
    """
    path = Path(repo_root) / "data" / "raw" / SUMS_NAME
    if not path.exists():
        raise SystemExit(
            f"cannot verify raw data: {path} is missing "
            "(it ships with the repo and pins the refreshed raw data; "
            "restore it and re-run — never delete data/raw/sha256sums.txt)"
        )
    expected = parse_sha256sums(path.read_text())
    if not expected:
        raise SystemExit(f"{path} contains no checksums; cannot verify raw data")
    return expected


def ensure_sums(cache_dir: Path, get_urls, downloader) -> dict:
    """Return {filename: sha256} from sha256sums.txt, downloading it if needed."""
    sums_path = cache_dir / SUMS_NAME

    def load() -> dict:
        return parse_sha256sums(sums_path.read_text())

    if sums_path.exists():
        sums = load()
        if sums:
            return sums
        sums_path.unlink()

    urls = get_urls()
    if SUMS_NAME not in urls:
        raise SystemExit(f"release is missing asset {SUMS_NAME}")
    print(f"downloading {SUMS_NAME}")
    downloader(urls[SUMS_NAME], sums_path)
    return load()


def ensure_archive(
    asset_key: str, cache_dir: Path, get_urls, sums: dict, downloader
) -> Path:
    """Return a verified archive path in the cache, downloading it if needed.

    Skips the download when the cached file's sha256 already matches.
    On verification failure: delete the cache bucket and exit non-zero.
    """
    archive_name, _ = ASSETS[asset_key]
    if archive_name not in sums:
        raise SystemExit(f"{SUMS_NAME} has no entry for {archive_name}")
    expected = sums[archive_name]
    dest = cache_dir / archive_name

    if dest.exists():
        actual = sha256_file(dest)
        if actual == expected:
            print(f"cache hit: {archive_name} (sha256 ok), skipping download")
            return dest
        print(f"cache stale for {archive_name} (hash mismatch), re-downloading")
        dest.unlink()

    urls = get_urls()
    if archive_name not in urls:
        raise SystemExit(f"release is missing asset {archive_name}")
    print(f"downloading {archive_name}")
    downloader(urls[archive_name], dest, label=archive_name)

    actual = sha256_file(dest)
    if actual != expected:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise SystemExit(
            f"SHA256 verification FAILED for {archive_name}: "
            f"expected {expected}, got {actual}; cache removed, please re-run"
        )
    print(f"sha256 ok: {archive_name}")
    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract `archive_path` into `dest_dir` (a temp dir, never the repo root).

    Refuses archives containing a data/raw/sha256sums.txt member outright:
    the repo's tracked trust anchor must never come from a release archive,
    because extraction would replace it before anything could verify against
    it (P3).
    """
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise SystemExit(f"unsafe path in archive {archive_path.name}: {name}")
            if "/".join(Path(name).parts) == TRUST_ANCHOR:
                raise SystemExit(
                    f"refusing {archive_path.name}: it contains {TRUST_ANCHOR}. "
                    "The repo manifest is the trust anchor for raw data and "
                    "cannot be overwritten by a release archive; repackage the "
                    "release without that member and re-run."
                )
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest_dir, filter="data")
        else:
            tar.extractall(dest_dir)
    print(f"extracted {archive_path.name}")


def missing_key_files(base: Path, key_files) -> list:
    return [k for k in key_files if not (base / k).exists()]


def verify_raw_against_manifest(base: Path, expected: dict) -> None:
    """Fail closed: check every file of the in-memory manifest under `base`.

    `base` is the repo root for already-extracted assets, or the temporary
    extraction directory for freshly unpacked archives — verification never
    runs against a tree an archive could have overwritten. The manifest
    itself always comes from the repo copy read at the start of `run()`.
    The release's own manifest only proves the archive downloaded intact;
    the repo's sums pin the CURRENT refreshed raw data, so a release that
    predates the 2026-10-11 refresh (or ships substituted files) aborts
    loudly instead of silently installing stale data.
    """
    problems = []
    for name, digest in sorted(expected.items()):
        path = base / "data" / "raw" / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            problems.append(f"{name}: sha256 {actual} != expected {digest}")
    if problems:
        raise SystemExit(
            f"raw data does not match {TRUST_ANCHOR} "
            "(the release likely predates the 2026-10-11 data refresh):\n  "
            + "\n  ".join(problems)
            + f"\nNothing was installed; {TRUST_ANCHOR} remains the tracked "
            "trust anchor — do not delete data/raw/. Wait for a release whose "
            f"files match the repo manifest, or clear {CACHE_DIR_NAME}/ and re-run."
        )
    print(f"repo checksums ok: {len(expected)} files match data/raw/{SUMS_NAME}")


def move_tree(src: Path, dst: Path) -> None:
    """Move everything under `src` into `dst`, merging existing directories."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        if child.is_dir():
            move_tree(child, dst / child.name)
        else:
            shutil.move(str(child), str(dst / child.name))


def run(asset_keys, repo_root: Path = REPO_ROOT, api_url: str = API_URL,
        fetch_assets=None, downloader=None) -> None:
    """Download/verify/extract the selected assets into `repo_root`.

    `fetch_assets` and `downloader` are injectable for offline testing.
    Idempotent: already-extracted assets are skipped (raw still re-verifies
    against the in-memory trust anchor); cached archives whose sha256
    matches skip the download.

    Order matters (P3): the repo's data/raw/sha256sums.txt is read into
    memory before anything is downloaded or extracted; archives unpack into
    `<cache>/<tag>/_extract_tmp/`; raw files are verified there against the
    in-memory manifest (key files for models/cache); only a fully verified
    tree is moved into the working tree, so a failed check never leaves a
    half-installed set of files behind.
    """
    repo_root = Path(repo_root)
    # (a) trust anchor into memory before any archive is touched
    expected_raw = read_trust_anchor(repo_root)
    # (g) cache bucketed by release tag so cross-release reuse is impossible
    cache_dir = repo_root / CACHE_DIR_NAME / cache_bucket(api_url)
    fetch = fetch_assets or fetch_release_assets
    dl = downloader or download

    urls_cache = {}

    def get_urls() -> dict:
        if "urls" not in urls_cache:
            urls_cache["urls"] = fetch(api_url)
        return urls_cache["urls"]

    sums = None
    for key in asset_keys:
        archive_name, key_files = ASSETS[key]
        if not missing_key_files(repo_root, key_files):
            print(f"already extracted, skipping: {archive_name}")
            if key == "raw":
                verify_raw_against_manifest(repo_root, expected_raw)
            continue
        if sums is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            sums = ensure_sums(cache_dir, get_urls, dl)
        archive = ensure_archive(key, cache_dir, get_urls, sums, dl)
        # (b) unpack inside the cache, never into the working tree
        tmp_dir = cache_dir / EXTRACT_TMP_NAME
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        try:
            # (c) refuses an archive-bundled data/raw/sha256sums.txt member
            extract_archive(archive, tmp_dir)
            still_missing = missing_key_files(tmp_dir, key_files)
            if still_missing:
                raise SystemExit(
                    f"extraction of {archive_name} did not produce: {', '.join(still_missing)}"
                )
            if key == "raw":
                # (d) verify against the in-memory manifest inside the temp dir;
                # only a fully verified tree is moved into place
                verify_raw_against_manifest(tmp_dir, expected_raw)
            move_tree(tmp_dir, repo_root)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"verified and installed {archive_name}")
    print("done.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch models / raw data / fingerprint caches from the artifacts-v2 release."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", action="store_true", help="fetch models.tar.xz only")
    group.add_argument("--raw", action="store_true", help="fetch raw_data.tar.gz only")
    group.add_argument(
        "--cache", action="store_true", help="fetch processed_cache.tar.gz only"
    )
    group.add_argument("--all", action="store_true", help="fetch everything (default)")
    args = parser.parse_args(argv)

    if args.models:
        keys = ["models"]
    elif args.raw:
        keys = ["raw"]
    elif args.cache:
        keys = ["cache"]
    else:
        keys = ["models", "raw", "cache"]

    try:
        run(keys)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
