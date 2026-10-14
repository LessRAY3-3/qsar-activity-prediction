#!/usr/bin/env python3
"""Download models, raw data and fingerprint caches from the `artifacts-v2` GitHub Release.

Assets are cached in <repo>/.artifacts_cache/<release-tag>/ (bucketed per
release, so an archive left over from another release — or from the old flat
layout — is never reused), checked against the release's own sha256sums.txt —
re-downloaded on every run, because release assets may be re-uploaded in place
(artifacts-v2 was) — and unpacked into a temporary directory inside that cache,
never straight into the working tree.

Trust model for raw data (review findings P3 and R3):

* The repo's tracked data/raw/sha256sums.txt is the trust anchor. It is read
  into memory (parsed pins and raw bytes) before any archive is downloaded or
  extracted; after files are moved into place it is re-read from disk and must
  still be a regular file whose bytes equal that in-memory copy.
* Every archive member is checked before extraction: absolute / `..` paths are
  refused; symlink and hardlink members are refused outright (the data filter
  would otherwise follow in-tree links such as `data/raw -> payload`); members
  must sit under a per-asset prefix allowlist (raw: data/raw/, models:
  models/, cache: data/processed/), so an archive can never install unverified
  files elsewhere in the repo (results/, scripts/, .git/hooks/ ...); and a
  member whose path matches the trust anchor case-insensitively (macOS file
  systems compare names case-insensitively) is refused like an exactly named
  one. All refusals raise SystemExit listing the offending members — no bare
  traceback escapes main().
* Every file named by the in-memory manifest is sha256-verified inside the
  temporary directory (a missing file counts as a failure), and only after the
  whole manifest passes are those verified files moved into place — atomically,
  one os.replace per file with backups, so a failed install rolls back instead
  of leaving half a set behind. The tracked manifest is never replaced by
  release content.

Installation is idempotent but self-repairing (review finding R4): an asset
counts as installed only when ALL of its key files are present as regular
files (models: all 18 random/scaffold joblibs, raw: every manifest file, cache: all 9 npz
caches) — a single stray file no longer marks the asset installed. When the
local raw copy no longer matches the manifest the script no longer aborts with
unusable advice: it names the stale files, re-downloads, verifies in the temp
dir and atomically replaces only what verified; a failed verification keeps
the local files untouched and lists exactly what is stale.

processed_cache.tar.gz (review finding P4) ships the gitignored
data/processed/*_fingerprints.npz caches so the fast path
(`gnn_01_make_splits.py` onward) runs without re-featurizing first. It has
no repo-side manifest; key-file completeness (all 9 npz) is the check.

Known limitation: models/ has no repo-side manifest either — model files are
not hash-pinned, only completeness-checked (all 18 random/scaffold joblibs) before install and
before the "already extracted" skip.

Legacy RF time models in artifacts-v2 predate the accepted as-of labels.
They are validated as archive members but never installed; existing local
time models are preserved. Only the named random/scaffold models are installed.

Usage:
    python scripts/fetch_artifacts.py [--models | --raw | --cache | --all]

Standard library only; no third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

_ASSET_TAGS = ["a2a", "abl1", "bace", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2"]

# asset key -> (archive filename, key files that must all exist after extraction).
# Raw key files are derived from the trust anchor at run time (every manifest
# file); models and cache have no repo-side manifest, so completeness is
# checked against this full list — not a single probe file (R4).
ASSETS = {
    "models": (
        "models.tar.xz",
        [f"models/rf_random_split_{tag}.joblib" for tag in _ASSET_TAGS]
        + [f"models/rf_scaffold_split_{tag}.joblib" for tag in _ASSET_TAGS],
    ),
    "raw": ("raw_data.tar.gz", []),
    "cache": (
        "processed_cache.tar.gz",
        [f"data/processed/{tag}_fingerprints.npz" for tag in _ASSET_TAGS],
    ),
}

# Every archive member must live under its asset's prefix (R3); only these
# directories may ever be written to by a release archive.
ASSET_PREFIXES = {"raw": "data/raw/", "models": "models/", "cache": "data/processed/"}


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


def read_trust_anchor(repo_root: Path) -> tuple[dict, bytes]:
    """Read the repo's tracked data/raw/sha256sums.txt into memory.

    Returns (parsed pins, raw bytes). `run()` calls this before any download
    or extraction: the in-memory copy is the only manifest raw data is ever
    verified against, and the bytes are re-checked against the on-disk file
    after installation (P3, R3).
    """
    path = Path(repo_root) / "data" / "raw" / SUMS_NAME
    if not path.exists():
        raise SystemExit(
            f"cannot verify raw data: {path} is missing "
            "(it ships with the repo and pins the refreshed raw data; "
            "restore it and re-run — never delete data/raw/sha256sums.txt)"
        )
    raw_bytes = path.read_bytes()
    expected = parse_sha256sums(raw_bytes.decode("utf-8", errors="replace"))
    if not expected:
        raise SystemExit(f"{path} contains no checksums; cannot verify raw data")
    return expected, raw_bytes


def ensure_sums(cache_dir: Path, get_urls, downloader) -> dict:
    """Return {filename: sha256} from the release's sha256sums.txt.

    Always re-downloaded on every run that installs something: release assets
    may be re-uploaded in place (the artifacts-v2 raw archive was), so a
    cached copy of the manifest cannot be trusted as current (Low).
    """
    urls = get_urls()
    if SUMS_NAME not in urls:
        raise SystemExit(f"release is missing asset {SUMS_NAME}")
    sums_path = cache_dir / SUMS_NAME
    print(f"downloading {SUMS_NAME}")
    downloader(urls[SUMS_NAME], sums_path, label=SUMS_NAME)
    try:
        text = sums_path.read_text()
    except UnicodeDecodeError:
        sums_path.unlink(missing_ok=True)
        raise SystemExit(f"release {SUMS_NAME} is not a text manifest") from None
    sums = parse_sha256sums(text)
    if not sums:
        sums_path.unlink(missing_ok=True)
        raise SystemExit(f"release {SUMS_NAME} contains no checksums")
    return sums


def ensure_archive(
    asset_key: str, cache_dir: Path, get_urls, sums: dict, downloader
) -> Path:
    """Return a verified archive path in the cache, downloading it if needed.

    Skips the download when the cached file's sha256 already matches the
    freshly downloaded release manifest. On verification failure: delete the
    cache bucket and exit non-zero.
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


def extract_archive(archive_path: Path, dest_dir: Path, allow_prefix: str) -> None:
    """Extract `archive_path` into `dest_dir` (a temp dir, never the repo root).

    Before extraction every member is checked (R3), collecting all violations
    into one SystemExit:
    * no absolute paths and no `..` components;
    * no symlink or hardlink members (the tar data filter follows in-tree
      links, which is how `data/raw -> payload` used to smuggle a forged
      manifest past a filename-only check);
    * no special files (devices, fifos);
    * no member whose path matches data/raw/sha256sums.txt case-insensitively
      (macOS names compare case-insensitively, so data/raw/SHA256SUMS.txt IS
      the trust anchor there) — the anchor must never come from an archive;
    * every other member must sit under `allow_prefix` (per-asset allowlist),
      so an archive can never install unverified content elsewhere in the repo.
    """
    prefix = allow_prefix.rstrip("/")
    anchor_folded = TRUST_ANCHOR.casefold()
    violations = []
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name
            norm = "/".join(Path(name).parts)
            if name.startswith("/") or ".." in Path(name).parts:
                violations.append(f"unsafe path: {name}")
                continue
            if member.issym() or member.islnk():
                kind = "symbolic link" if member.issym() else "hard link"
                violations.append(f"{kind} member: {name}")
                continue
            if not (member.isfile() or member.isdir()):
                violations.append(f"special file (device/fifo): {name}")
                continue
            if norm.casefold() == anchor_folded:
                violations.append(
                    f"trust anchor member (matches {TRUST_ANCHOR} "
                    f"case-insensitively): {name}"
                )
                continue
            in_prefix = norm == prefix or norm.startswith(prefix + "/")
            ancestor_dir = member.isdir() and prefix.startswith(norm + "/")
            if not in_prefix and not ancestor_dir:
                violations.append(f"outside the allowed prefix {prefix}/: {name}")
        if violations:
            raise SystemExit(
                f"refusing {archive_path.name}: unsafe archive members:\n  "
                + "\n  ".join(violations)
                + "\nThe trust anchor can never be overwritten by a release "
                "archive, and every member must be a regular file or directory "
                f"under {prefix}/. Repackage the release and re-run."
            )
        try:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(dest_dir, filter="data")
            else:
                tar.extractall(dest_dir)
        except tarfile.TarError as exc:
            raise SystemExit(f"extraction of {archive_path.name} failed: {exc}") from None
    print(f"extracted {archive_path.name}")


def key_files_for(asset_key: str, expected_raw: dict) -> list:
    """Return the full key-file list for an asset (R4: completeness, not a probe).

    For raw data the key files are every file named by the trust anchor; for
    models and cache they are the static full lists in ASSETS.
    """
    if asset_key == "raw":
        return [f"data/raw/{name}" for name in sorted(expected_raw)]
    return list(ASSETS[asset_key][1])


def missing_key_files(base: Path, key_files) -> list:
    """Key files absent (or not regular files) under `base`."""
    missing = []
    for key in key_files:
        path = base / key
        if path.is_symlink() or not path.is_file():
            missing.append(key)
    return missing


def collect_relative_files(root: Path) -> list:
    """Return sorted repo-relative paths of all regular files under `root`."""
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


def raw_manifest_problems(base: Path, expected: dict) -> list:
    """Return the list of raw-data problems under `base` (empty = fully matching).

    `base` is the repo root for already-extracted assets, or the temporary
    extraction directory for freshly unpacked archives. Checks every file of
    the in-memory manifest; a missing file counts as a problem.
    """
    problems = []
    for name, digest in sorted(expected.items()):
        path = base / "data" / "raw" / name
        if path.is_symlink():
            problems.append(f"{name}: is a symlink, not a regular file")
            continue
        if not path.is_file():
            if path.exists():
                problems.append(f"{name}: not a regular file")
            else:
                problems.append(f"missing {name}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            problems.append(f"{name}: sha256 {actual} != expected {digest}")
    return problems


def verify_raw_against_manifest(base: Path, expected: dict, local_problems=None) -> None:
    """Fail closed: check every file of the in-memory manifest under `base`.

    The manifest always comes from the repo copy read at the start of `run()`,
    never from `base` or from a release archive. The release's own manifest
    only proves the archive downloaded intact; the repo's sums pin the CURRENT
    refreshed raw data, so a release that predates the 2026-10-11 refresh (or
    ships substituted files) aborts loudly instead of silently installing stale
    data. `local_problems`, when given, are appended to the failure message so
    users see exactly which local files are stale and were left untouched (R4).
    """
    problems = raw_manifest_problems(base, expected)
    if problems:
        msg = (
            f"raw data does not match {TRUST_ANCHOR} "
            "(the release likely predates the 2026-10-11 data refresh):\n  "
            + "\n  ".join(problems)
            + f"\nNothing was installed; {TRUST_ANCHOR} remains the tracked "
            "trust anchor — do not delete data/raw/."
        )
        if local_problems:
            msg += (
                "\nYour local files were left unchanged; stale copies that still "
                "need repairing (delete these specific files and re-run to force):\n  "
                + "\n  ".join(local_problems)
            )
        else:
            msg += "\nWait for a release whose files match the repo manifest, or re-run later."
        raise SystemExit(msg)
    print(f"repo checksums ok: {len(expected)} files match data/raw/{SUMS_NAME}")


def install_verified_files(repo_root: Path, tmp_dir: Path, rel_paths) -> tuple[list, list]:
    """Atomically move verified files from `tmp_dir` into `repo_root` (R4).

    Each destination is either left untouched (bytes already identical —
    reported as kept) or replaced with an os.replace of a verified copy.
    Before any write, existing destinations are backed up; if any step fails,
    every already-replaced file is rolled back and a SystemExit reports the
    failure — a failed install never leaves half a set behind. Returns
    (replaced, kept) as repo-relative path strings.
    """
    for rel in rel_paths:
        dst = repo_root / rel
        if dst.is_dir() and not dst.is_symlink():
            raise SystemExit(
                f"cannot install {rel}: destination is a directory; "
                "remove it and re-run"
            )
    to_replace, kept = [], []
    for rel in rel_paths:
        src, dst = tmp_dir / rel, repo_root / rel
        if not src.is_file():
            raise SystemExit(f"cannot install {rel}: verified copy missing from {tmp_dir}")
        if dst.is_file() and not dst.is_symlink() and dst.read_bytes() == src.read_bytes():
            kept.append(rel)
            continue
        to_replace.append(rel)
    backups: dict = {}
    installed = []
    try:
        for rel in to_replace:
            src, dst = tmp_dir / rel, repo_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            backup = dst.parent / (dst.name + ".fetch_backup")
            if backup.exists() or backup.is_symlink():
                backup.unlink()
            if dst.is_symlink():
                os.replace(dst, backup)
                backups[rel] = backup
            elif dst.exists():
                shutil.copy2(dst, backup)
                backups[rel] = backup
            os.replace(src, dst)
            installed.append(rel)
    except OSError as exc:
        for rel in reversed(installed):
            dst = repo_root / rel
            try:
                if rel in backups:
                    os.replace(backups.pop(rel), dst)
                else:
                    dst.unlink()
            except OSError:
                pass
        for rel, backup in list(backups.items()):
            dst = repo_root / rel
            try:
                if not dst.exists() and not dst.is_symlink():
                    os.replace(backup, dst)
                else:
                    backup.unlink()
            except OSError:
                pass
            backups.pop(rel, None)
        raise SystemExit(
            f"failed to install verified files: {exc}; rolled back — "
            "your existing files were left unchanged"
        ) from None
    for backup in backups.values():
        backup.unlink()
    return to_replace, kept


def check_trust_anchor_on_disk(repo_root: Path, anchor_bytes: bytes) -> None:
    """Re-read the on-disk trust anchor after a move and compare (R3).

    The anchor on disk must still be a regular file (not a symlink swapped in
    mid-run) whose bytes equal the copy read at the start of run().
    """
    path = repo_root / "data" / "raw" / SUMS_NAME
    if path.is_symlink():
        raise SystemExit(
            f"trust anchor check failed: {TRUST_ANCHOR} is a symlink after "
            "install; restore the tracked file and re-run"
        )
    if not path.is_file():
        raise SystemExit(
            f"trust anchor check failed: {TRUST_ANCHOR} is missing or not a "
            "regular file after install; restore the tracked file and re-run"
        )
    if path.read_bytes() != anchor_bytes:
        raise SystemExit(
            f"trust anchor check failed: {TRUST_ANCHOR} bytes changed during "
            "this run (expected the copy read at start); restore the tracked "
            "file and re-run"
        )


def run(asset_keys, repo_root: Path = REPO_ROOT, api_url: str = API_URL,
        fetch_assets=None, downloader=None) -> None:
    """Download/verify/extract the selected assets into `repo_root`.

    `fetch_assets` and `downloader` are injectable for offline testing.
    Idempotent: an asset whose full key-file set is present and (for raw)
    matches the trust anchor is skipped; otherwise the archive is downloaded,
    verified in a temp dir and installed atomically — including repairing a
    stale local raw copy instead of aborting (R4). Cached archives are only
    reused after their sha256 matches the freshly re-downloaded release
    manifest.

    Order matters (P3/R3): the repo's data/raw/sha256sums.txt is read into
    memory before anything is downloaded or extracted; archives unpack into
    `<cache>/<tag>/_extract_tmp/` only after every member passes the allowlist
    and link checks; raw files are verified there against the in-memory
    manifest; only fully verified files are moved into the working tree; and
    after the move the on-disk anchor is re-checked against the in-memory
    bytes.
    """
    repo_root = Path(repo_root)
    # (a) trust anchor into memory before any archive is touched
    expected_raw, anchor_bytes = read_trust_anchor(repo_root)
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
        archive_name, _ = ASSETS[key]
        if key == "models":
            print("models: random/scaffold only; skipping legacy rf_time_split_* "
                  "assets (existing local time models are preserved)")
        key_files = key_files_for(key, expected_raw)
        missing = missing_key_files(repo_root, key_files)
        local_problems = raw_manifest_problems(repo_root, expected_raw) if key == "raw" else []
        if not missing and not local_problems:
            print(f"already extracted, skipping: {archive_name}")
            continue
        if not missing and local_problems:
            # R4: a stale local raw copy is repaired, not a fatal lock-out
            print(
                f"local files do not match {TRUST_ANCHOR}; re-downloading "
                f"{archive_name} to replace them (no manual deletion needed). "
                "Files that will be replaced:"
            )
            for problem in local_problems:
                print(f"  {problem}")
        if sums is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            sums = ensure_sums(cache_dir, get_urls, dl)
        archive = ensure_archive(key, cache_dir, get_urls, sums, dl)
        # (b) unpack inside the cache, never into the working tree
        tmp_dir = cache_dir / EXTRACT_TMP_NAME
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        try:
            # (c) member checks: allowlist, no links, case-insensitive anchor
            extract_archive(archive, tmp_dir, ASSET_PREFIXES[key])
            if key == "raw":
                # (d) verify against the in-memory manifest inside the temp dir;
                # only fully verified files are moved into place (atomically)
                verify_raw_against_manifest(tmp_dir, expected_raw, local_problems)
                install_list = [f"data/raw/{name}" for name in sorted(expected_raw)]
            else:
                still_missing = missing_key_files(tmp_dir, key_files)
                if still_missing:
                    raise SystemExit(
                        f"extraction of {archive_name} did not produce: {', '.join(still_missing)}"
                    )
                # Every archive member has already passed the safety checks.
                # Install only the current model allowlist: artifacts-v2 time
                # models must never replace accepted as-of checkpoints.
                install_list = key_files if key == "models" else collect_relative_files(tmp_dir)
            replaced, kept = install_verified_files(repo_root, tmp_dir, install_list)
            if key == "raw":
                # (e) the on-disk anchor must equal the bytes read at start
                check_trust_anchor_on_disk(repo_root, anchor_bytes)
            print(f"verified and installed {archive_name}")
            print(f"  replaced: {', '.join(replaced) or '(none)'}")
            print(f"  kept up to date: {', '.join(kept) or '(none)'}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print("done.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch models / raw data / fingerprint caches from the artifacts-v2 release."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models", action="store_true", help="fetch random/scaffold models only; skip legacy time checkpoints")
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
    except (urllib.error.URLError, OSError, ValueError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
