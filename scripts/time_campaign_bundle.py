"""Create or verify offline, role-specific inputs for temporal retraining.

Creation requires an explicit, new --output directory. The deterministic archive
contains source, dependency specifications and exactly eight caches/splits; no
previous predictions, models or results. Verification never extracts an archive.
After extraction by the operator, use --verify-manifest manifest.json --root .
Hashes establish consistency, not authenticity; transfer the manifest securely.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

import numpy as np

from time_labels import load_time_split

TAGS = ("a2a", "abl1", "egfr", "egfr_full", "herg", "hivpr", "mpro", "vegfr2")
SPECS = ("requirements.txt", "requirements-dev.txt", "environment.yml", "pyproject.toml")
SCHEMA = "qsar-time-campaign-v1"
COMMON_SOURCES = {
    "scripts/run_time_campaign.py", "scripts/time_campaign_bundle.py",
    "scripts/time_labels.py", "scripts/qsar_common.py",
}


def required_sources(role):
    return COMMON_SOURCES | ({"scripts/04_train_and_evaluate.py"} if role == "rf" else {
        "scripts/gnn_03_train_gin.py", "scripts/gnn_graph_dataset.py"})


class HashingReader:
    """Hash exactly the bytes consumed by tarfile without buffering the cache."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.size += len(chunk)
        return chunk


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file(root, relative):
    """Only canonical relative regular-file paths, with no symlink ancestors."""
    if not isinstance(relative, str):
        raise ValueError("manifest path must be a string")
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts or
            str(path) != relative or "\\" in relative):
        raise ValueError(f"unsafe manifest path: {relative}")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"symlink input forbidden: {relative}")
    if not candidate.is_file() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing or nonregular input: {relative}")
    return candidate


def data_paths(role):
    suffix = {"rf": "fingerprints", "gin": "graphs"}[role]
    return {f"data/processed/{tag}_{suffix}.npz" for tag in TAGS} | {
        f"data/processed/splits/{tag}_time.npz" for tag in TAGS}


def allowed_path(relative, role):
    path = PurePosixPath(relative)
    return (relative in data_paths(role) or relative in SPECS or
            (path.parts[0] in ("scripts", "experiments") and path.suffix == ".py"))


def target_metadata(root, role):
    targets = {}
    suffix = {"rf": "fingerprints", "gin": "graphs"}[role]
    for tag in TAGS:
        cache_path = safe_file(root, f"data/processed/{tag}_{suffix}.npz")
        safe_file(root, f"data/processed/splits/{tag}_time.npz")
        with np.load(cache_path, allow_pickle=False) as cache:
            smiles, original_y = cache["smiles"], cache["y"]
            split = load_time_split(root / "data/processed/splits", tag, smiles, original_y)
        if split is None:
            raise ValueError(f"{tag}: temporal split is marked unavailable")
        targets[tag] = {
            "label_id": str(split["label_id"]),
            "label_policy": str(split["label_policy"]),
            "source_sha256": str(split["source_sha256"]),
            "dataset_signature": str(split["dataset_signature"]),
            "cutoff_valid": int(split["cutoff_valid"]),
            "cutoff_test": int(split["cutoff_test"]),
            "rows": {"total": len(smiles), **{
                key: len(split[f"{key}_idx"])
                for key in ("train", "valid", "test", "excluded")}},
        }
    return targets


def make_manifest(root, role):
    root = Path(root).resolve()
    top = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError("bundle source must be the Git worktree root, not a nested directory")
    paths = data_paths(role) | set(SPECS)
    for folder in ("scripts", "experiments"):
        paths.update(str(p.relative_to(root)) for p in (root / folder).rglob("*.py"))
    files = []
    for relative in sorted(paths):
        path = safe_file(root, relative)
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all",
         "--", "scripts", "experiments", *SPECS], text=True).splitlines()
    ignored = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "--ignored", "--exclude-standard", "-z",
         "--", "scripts", "experiments", *SPECS]).decode().split("\0")
    status.extend(f"!! {name}" for name in sorted(ignored) if name in paths)
    return {
        "schema": SCHEMA, "role": role,
        "source": {"git_head": head, "dirty": bool(status), "status": status,
                   "sha256": {f["path"]: f["sha256"] for f in files
                              if f["path"].split("/")[0] in ("scripts", "experiments")}},
        "dependency_sha256": {f["path"]: f["sha256"] for f in files if f["path"] in SPECS},
        "targets": target_metadata(root, role), "files": files,
    }


def verify_manifest(manifest, root):
    """Validate schema, path boundaries, every byte and label/cache alignment."""
    root = Path(root).resolve()
    fields = {"schema", "role", "source", "dependency_sha256", "targets", "files"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("invalid manifest fields")
    if manifest["schema"] != SCHEMA or manifest["role"] not in ("rf", "gin"):
        raise ValueError("unsupported manifest schema or role")
    source = manifest["source"]
    if (not isinstance(source, dict) or
            set(source) != {"git_head", "dirty", "status", "sha256"} or
            not isinstance(source["git_head"], str) or
            not re.fullmatch(r"[0-9a-f]{40,64}", source["git_head"]) or
            type(source["dirty"]) is not bool or not isinstance(source["status"], list) or
            not all(isinstance(item, str) for item in source["status"]) or
            source["dirty"] != bool(source["status"])):
        raise ValueError("invalid source provenance")
    if not isinstance(manifest["files"], list):
        raise ValueError("files must be a list")
    seen = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("invalid file entry")
        path = safe_file(root, entry["path"])
        relative = entry["path"]
        if relative in seen:
            raise ValueError(f"duplicate path: {relative}")
        if not allowed_path(relative, manifest["role"]):
            raise ValueError(f"file outside role input allowlist: {relative}")
        if (type(entry["size"]) is not int or entry["size"] < 0 or
                not isinstance(entry["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError(f"invalid file metadata: {relative}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"file changed: {relative}")
        seen[relative] = entry["sha256"]
    required = data_paths(manifest["role"]) | set(SPECS) | required_sources(manifest["role"])
    if not required.issubset(seen):
        raise ValueError("manifest missing required role inputs, dependency specs or runtime source: "
                         + ", ".join(sorted(required - set(seen))))
    expected_source = {p: h for p, h in seen.items()
                       if p.split("/")[0] in ("scripts", "experiments")}
    if not expected_source or source["sha256"] != expected_source:
        raise ValueError("source hash inventory differs from files")
    if manifest["dependency_sha256"] != {p: seen[p] for p in SPECS}:
        raise ValueError("dependency hash inventory differs from files")
    if manifest["targets"] != target_metadata(root, manifest["role"]):
        raise ValueError("target metadata differs from validated splits")
    return manifest


def write_bundle(root, output, manifest):
    """Stable ordering, timestamps, owners and gzip header; never overwrite."""
    root, output = Path(root).resolve(), Path(output).resolve()
    verify_manifest(manifest, root)
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "bundle.tar.gz").open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                info = tarfile.TarInfo("manifest.json")
                info.size, info.mode = len(payload), 0o644
                archive.addfile(info, io.BytesIO(payload))
                for entry in sorted(manifest["files"], key=lambda item: item["path"]):
                    path = safe_file(root, entry["path"])
                    # Recheck while packaging so concurrent preparation cannot
                    # silently produce a bundle inconsistent with its manifest.
                    info = tarfile.TarInfo(entry["path"])
                    info.size, info.mode = entry["size"], 0o644
                    with path.open("rb") as stream:
                        reader = HashingReader(stream)
                        archive.addfile(info, reader)
                        extra = stream.read(1)
                    if (extra or reader.size != entry["size"] or
                            reader.digest.hexdigest() != entry["sha256"]):
                        raise ValueError(f"input changed during packaging: {entry['path']}")
    (output / "manifest.json").write_bytes(payload)
    (output / "bundle.sha256").write_text(sha256(output / "bundle.tar.gz") + "  bundle.tar.gz\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--role", choices=("rf", "gin"))
    parser.add_argument("--output", type=Path, help="new directory for manifest and archive")
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.verify_manifest:
        if args.role or args.output:
            parser.error("verification cannot be combined with creation options")
        verify_manifest(json.loads(args.verify_manifest.read_text()), args.root)
        print(f"Verified {args.verify_manifest}")
    else:
        if not args.role or not args.output:
            parser.error("creation requires --role and --output")
        manifest = make_manifest(args.root, args.role)
        write_bundle(args.root, args.output, manifest)
        print(f"Created {args.role} inputs in {args.output}; source dirty={manifest['source']['dirty']}")


if __name__ == "__main__":
    main()
