#!/usr/bin/env python3
"""Download trained models and raw data from the `artifacts-v1` GitHub Release.

Assets are cached in <repo>/.artifacts_cache/, verified against the
release's own sha256sums.txt, then unpacked into the repo root (the archives
already contain the models/ and data/raw/ prefixes). Extracted raw files are
additionally verified against the repo's data/raw/sha256sums.txt (the
release manifest only proves the download is intact; the repo sums prove the
files are the current, refreshed data) and the script aborts loudly on any
mismatch — e.g. a release that predates the 2026-10-11 data refresh.

Usage:
    python scripts/fetch_artifacts.py [--models | --raw | --all]

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
SUMS_NAME = "sha256sums.txt"
API_URL = (
    "https://api.github.com/repos/LessRAY3-3/qsar-activity-prediction"
    "/releases/tags/artifacts-v1"
)
USER_AGENT = "qsar-activity-prediction-fetch-artifacts"

# asset key -> (archive filename, key files that must exist after extraction)
ASSETS = {
    "models": ("models.tar.xz", ["models/rf_random_split_egfr.joblib"]),
    "raw": ("raw_data.tar.gz", ["data/raw/egfr_activities.csv"]),
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
    On verification failure: delete the cache and exit non-zero.
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


def extract_archive(archive_path: Path, repo_root: Path) -> None:
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise SystemExit(f"unsafe path in archive {archive_path.name}: {name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(repo_root, filter="data")
        else:
            tar.extractall(repo_root)
    print(f"extracted {archive_path.name} -> {repo_root}")


def missing_key_files(repo_root: Path, key_files) -> list:
    return [k for k in key_files if not (repo_root / k).exists()]


def verify_extracted_raw(repo_root: Path) -> None:
    """Fail closed: check every file in the repo's data/raw/sha256sums.txt.

    The release's own manifest only proves the archives downloaded intact;
    the repo's sums file pins the CURRENT refreshed raw data. Verifying both
    means a release that predates the data refresh (e.g. artifacts-v1, whose
    raw CSVs predate the 2026-10-11 standard_relation refresh) aborts loudly
    instead of silently installing stale files.
    """
    repo_root = Path(repo_root)
    sums_path = repo_root / "data" / "raw" / SUMS_NAME
    if not sums_path.exists():
        raise SystemExit(
            f"cannot verify extracted raw data: {sums_path} is missing "
            "(it ships with the repo and pins the refreshed raw files)"
        )
    expected = parse_sha256sums(sums_path.read_text())
    if not expected:
        raise SystemExit(f"{sums_path} contains no checksums; cannot verify raw data")
    problems = []
    for name, digest in sorted(expected.items()):
        path = repo_root / "data" / "raw" / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            problems.append(f"{name}: sha256 {actual} != expected {digest}")
    if problems:
        raise SystemExit(
            "raw data does not match data/raw/sha256sums.txt after extraction "
            "(the release likely predates the 2026-10-11 data refresh):\n  "
            + "\n  ".join(problems)
            + "\nRemove data/raw/ and .artifacts_cache/ and re-run once a "
            "refreshed release is published."
        )
    print(f"repo checksums ok: {len(expected)} files match data/raw/{SUMS_NAME}")


def run(asset_keys, repo_root: Path = REPO_ROOT, api_url: str = API_URL,
        fetch_assets=None, downloader=None) -> None:
    """Download/verify/extract the selected assets into `repo_root`.

    `fetch_assets` and `downloader` are injectable for offline testing.
    Idempotent: already-extracted assets are skipped entirely; cached
    archives whose sha256 matches skip the download.
    """
    repo_root = Path(repo_root)
    cache_dir = repo_root / CACHE_DIR_NAME
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
                verify_extracted_raw(repo_root)
            continue
        if sums is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            sums = ensure_sums(cache_dir, get_urls, dl)
        archive = ensure_archive(key, cache_dir, get_urls, sums, dl)
        extract_archive(archive, repo_root)
        still_missing = missing_key_files(repo_root, key_files)
        if still_missing:
            raise SystemExit(
                f"extraction of {archive_name} did not produce: {', '.join(still_missing)}"
            )
        print(f"verified extracted files for {archive_name}")
        if key == "raw":
            # release-manifest check (above) + repo checksums: fail closed on
            # a release whose raw files predate the data refresh (N5)
            verify_extracted_raw(repo_root)
    print("done.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch models/raw-data artifacts from the artifacts-v1 GitHub release."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", action="store_true", help="fetch models.tar.xz only")
    group.add_argument("--raw", action="store_true", help="fetch raw_data.tar.gz only")
    group.add_argument("--all", action="store_true", help="fetch everything (default)")
    args = parser.parse_args(argv)

    if args.models:
        keys = ["models"]
    elif args.raw:
        keys = ["raw"]
    else:
        keys = ["models", "raw"]

    try:
        run(keys)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
