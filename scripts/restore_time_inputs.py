"""Pack or restore the committed frozen inputs for temporal statistics.

No downloads or model inference. Restoration checks the complete package and
published prediction/metric and 24 GIN checkpoint versions before writing any
split. Checkpoints load on CPU with weights_only=True. Existing
conflicting splits and NA markers are never overwritten or removed.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import numpy as np

from time_campaign_bundle import TAGS, safe_file, sha256
from time_labels import LABEL_POLICY, load_time_split, require_prediction_version

SCHEMA = "qsar-time-analysis-inputs-v1"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("results/time_split/inputs")
MAX_FILE_BYTES = 256 * 1024 * 1024


def validate_publication(root, tag, split):
    """Reject a package from a different result generation, even if self-consistent."""
    import torch

    cells = [(f"rf_preds_{tag}_time.npz", f"metrics_{tag}.json")]
    cells += [(f"gnn_preds_{tag}_time{suf}.npz", f"gnn_metrics_{tag}{suf}.json")
              for suf in ("", "_seed1", "_seed2")]
    for prediction, metric in cells:
        with np.load(safe_file(root, f"results/{prediction}"), allow_pickle=False) as data:
            require_prediction_version(data, split)
        scores = json.loads(safe_file(root, f"results/{metric}").read_text())["time"]
        if scores.get("label_id") != str(split["label_id"]):
            raise ValueError(f"{metric}: published metric label version differs from package")

    for suffix, seed in (("", 42), ("_seed1", 1), ("_seed2", 2)):
        relative = f"results/gnn_models/{tag}_gin_time{suffix}.pt"
        checkpoint = torch.load(safe_file(root, relative), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("label_id") != str(split["label_id"]):
            raise ValueError(f"{relative}: checkpoint label version differs from package")
        expected = {"tag": tag, "model": "gin", "split": "time", "seed": seed, "suffix": suffix}
        args = checkpoint.get("args", {})
        if not isinstance(args, dict) or any(args.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{relative}: checkpoint role/seed differs from published cell")


def pack(root, package):
    """Write a new package only after all eight splits match published results."""
    root, package = Path(root).resolve(), Path(package)
    entries = []
    for tag in TAGS:
        name = f"{tag}_time.npz"
        source = safe_file(root, f"data/processed/splits/{name}")
        split = load_time_split(source.parent, tag)
        if split is None:
            raise ValueError(f"{tag}: time split is unavailable")
        validate_publication(root, tag, split)
        entries.append({"name": name, "size": source.stat().st_size,
                        "sha256": sha256(source), "label_id": str(split["label_id"])})
    package.mkdir(parents=True, exist_ok=False)
    archive = package / "splits.tar.gz"
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for entry in entries:
                source = root / "data/processed/splits" / entry["name"]
                info = tarfile.TarInfo(entry["name"])
                info.size, info.mode = entry["size"], 0o644
                with source.open("rb") as stream:
                    tar.addfile(info, stream)
    manifest = {"schema": SCHEMA, "label_policy": LABEL_POLICY,
                "archive": {"name": archive.name, "size": archive.stat().st_size,
                            "sha256": sha256(archive)}, "files": entries}
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(package):
    manifest = json.loads(safe_file(package, "manifest.json").read_text())
    if (not isinstance(manifest, dict) or
            set(manifest) != {"schema", "label_policy", "archive", "files"} or
            manifest["schema"] != SCHEMA or manifest["label_policy"] != LABEL_POLICY):
        raise ValueError("unsupported frozen input manifest")
    archive = manifest["archive"]
    if not isinstance(archive, dict) or set(archive) != {"name", "size", "sha256"}:
        raise ValueError("invalid archive metadata")
    if archive["name"] != "splits.tar.gz":
        raise ValueError("unexpected archive name")
    entries = manifest["files"]
    if not isinstance(entries, list) or len(entries) != len(TAGS):
        raise ValueError("manifest must contain all eight time splits")
    for entry in [archive, *entries]:
        fields = {"name", "size", "sha256"} | ({"label_id"} if entry is not archive else set())
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("invalid file metadata")
        if (type(entry["size"]) is not int or not 0 < entry["size"] <= MAX_FILE_BYTES or
                not isinstance(entry["sha256"], str) or
                re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
            raise ValueError("invalid file size or SHA256")
        if entry is not archive and (not isinstance(entry["label_id"], str) or
                                     re.fullmatch(r"[0-9a-f]{64}", entry["label_id"]) is None):
            raise ValueError("invalid label ID")
    if {entry["name"] for entry in entries} != {f"{tag}_time.npz" for tag in TAGS}:
        raise ValueError("manifest split allowlist mismatch")
    path = safe_file(package, archive["name"])
    if path.stat().st_size != archive["size"] or sha256(path) != archive["sha256"]:
        raise ValueError("archive size or SHA256 mismatch")
    return manifest



def atomic_install(source, target, expected_sha256):
    """Publish a fully synced file with atomic no-clobber semantics.

    A killed copy can leave an ignored temporary file, never a truncated final
    split. link() is atomic on the target filesystem and refuses an existing
    path. Matching complete files make retries idempotent; conflicts are kept.
    """
    target = Path(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".time-restore-", suffix=".tmp",
                                         dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with Path(source).open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256(temporary) != expected_sha256:
            raise ValueError(f"staged split SHA256 mismatch: {target.name}")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or sha256(target) != expected_sha256:
                raise ValueError(f"refusing to overwrite conflicting local split: {target}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore(root, package):
    root, package = Path(root).resolve(), Path(package).resolve()
    manifest = read_manifest(package)
    entries = {entry["name"]: entry for entry in manifest["files"]}
    destination = root / "data/processed/splits"
    for parent in [root / "data", root / "data/processed", destination]:
        if parent.is_symlink():
            raise ValueError(f"symlink destination forbidden: {parent}")
    # Stage only allowlisted regular files; tarfile.extract is deliberately unused.
    with tempfile.TemporaryDirectory(prefix="qsar-time-inputs-") as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(package / "splits.tar.gz", mode="r:gz") as tar:
            for member in tar:
                if member.name not in entries or member.name in seen or not member.isfile():
                    raise ValueError(f"unexpected, duplicate or nonregular archive member: {member.name}")
                entry = entries[member.name]
                if member.size != entry["size"]:
                    raise ValueError(f"archive member size mismatch: {member.name}")
                with tar.extractfile(member) as source, (stage / member.name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                if sha256(stage / member.name) != entry["sha256"]:
                    raise ValueError(f"split SHA256 mismatch: {member.name}")
                seen.add(member.name)
        if seen != set(entries):
            raise ValueError("archive is missing required splits")
        for tag in TAGS:
            name = f"{tag}_time.npz"
            split = load_time_split(stage, tag)
            if str(split["label_id"]) != entries[name]["label_id"]:
                raise ValueError(f"{tag}: manifest and split label versions differ")
            validate_publication(root, tag, split)
            marker = destination / f"{tag}_time_NA.json"
            if marker.exists() or marker.is_symlink():
                raise ValueError(f"{tag}: existing NA marker; inspect it before restoring")
            target = destination / name
            if target.is_symlink() or (target.exists() and
                                      (not target.is_file() or sha256(target) != entries[name]["sha256"])):
                raise ValueError(f"{tag}: refusing to overwrite conflicting local split")
        destination.mkdir(parents=True, exist_ok=True)
        for name in entries:
            target = destination / name
            atomic_install(stage / name, target, entries[name]["sha256"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--package", type=Path, help="package directory (default: ROOT/results/time_split/inputs)")
    parser.add_argument("--pack", action="store_true", help="maintainer: create a NEW package from accepted local splits")
    args = parser.parse_args()
    package = args.package if args.package is not None else args.root / PACKAGE
    result = (pack if args.pack else restore)(args.root, package)
    print(f"{'Packed' if args.pack else 'Verified and restored'} {len(result['files'])} frozen time splits")


if __name__ == "__main__":
    main()
