"""Verify and unpack either supported return archive into a new directory.

Archive and member checks establish byte integrity against the supplied manifest,
not model acceptance or manifest authenticity. Study archives may use backward
hardlinks to earlier regular members; ordinary handoff archives cannot use links.
No checkpoint is loaded and no archived source is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tarfile
import tempfile


SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def digest_file(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("Member path must be a string")
    path = PurePosixPath(name)
    if (not name or name == "." or path.is_absolute() or ".." in path.parts
            or str(path) != name or "\\" in name or "\x00" in name
            or PureWindowsPath(name).drive):
        raise ValueError(f"Unsafe member path: {name!r}")
    return name


def checksum(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Invalid SHA256 in return manifest")
    return value


def nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid {name} in return manifest")
    return value


def read_manifest(path):
    raw = path.read_bytes()
    spec = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(spec, dict):
        raise ValueError("Return manifest must be an object")
    checksum(spec.get("sha256"))
    nonnegative_integer(spec.get("bytes"), "archive bytes")
    declared = spec.get("files")
    if type(declared) is int:
        nonnegative_integer(declared, "file count")
        hashes = spec.get("member_sha256")
        if not isinstance(hashes, dict) or len(hashes) != declared:
            raise ValueError("Study member inventory/count mismatch")
        entries = {safe_name(name): {"sha256": checksum(sha), "bytes": None}
                   for name, sha in hashes.items()}
        kind = "study-with-hardlinks"
        if "hardlinks" in spec:
            nonnegative_integer(spec["hardlinks"], "hardlink count")
    elif isinstance(declared, dict):
        if "member_sha256" in spec or "hardlinks" in spec:
            raise ValueError("Ambiguous return manifest format")
        entries = {}
        for name, item in declared.items():
            safe_name(name)
            if not isinstance(item, dict) or set(item) != {"sha256", "bytes"}:
                raise ValueError(f"Invalid regular member metadata: {name}")
            entries[name] = {"sha256": checksum(item["sha256"]),
                             "bytes": nonnegative_integer(item["bytes"], "member bytes")}
        kind = "regular-handoff"
    else:
        raise ValueError("Unsupported return manifest files field")
    for name in entries:
        if any(parent.as_posix() in entries for parent in PurePosixPath(name).parents):
            raise ValueError(f"Member path conflicts with a parent file: {name}")
    return spec, entries, kind, hashlib.sha256(raw).hexdigest()


def inspect_members(stream, entries, kind, spec):
    members = stream.getmembers()
    seen, regular = set(), set()
    links = 0
    for member in members:
        name = safe_name(member.name)
        if name in seen:
            raise ValueError(f"Duplicate archive member: {name}")
        if name not in entries:
            raise ValueError(f"Undeclared archive member: {name}")
        if member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
            if member.size < 0 or (entries[name]["bytes"] is not None
                                   and member.size != entries[name]["bytes"]):
                raise ValueError(f"Archive member size mismatch: {name}")
            regular.add(name)
        elif member.islnk() and kind == "study-with-hardlinks":
            target = safe_name(member.linkname)
            if target not in regular or member.size != 0:
                raise ValueError(f"Hardlink must target an earlier regular member: {name}")
            if entries[name]["sha256"] != entries[target]["sha256"]:
                raise ValueError(f"Hardlink member hash differs from target: {name}")
            links += 1
        else:
            raise ValueError(f"Forbidden archive member type: {name}")
        seen.add(name)
    if seen != set(entries) or len(members) != len(entries):
        raise ValueError("Archive member inventory/count mismatch")
    if "hardlinks" in spec and spec["hardlinks"] != links:
        raise ValueError("Archive hardlink count mismatch")
    return members, links


def verify_extracted(root, entries):
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink after extraction: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Nonregular file after extraction: {path}")
        actual.add(path.relative_to(root).as_posix())
    if actual != set(entries):
        raise ValueError("Extracted member inventory mismatch")
    for name, item in entries.items():
        path = root / name
        if item["bytes"] is not None and path.stat().st_size != item["bytes"]:
            raise ValueError(f"Extracted member size mismatch: {name}")
        if digest_file(path) != item["sha256"]:
            raise ValueError(f"Extracted member checksum mismatch: {name}")


def unpack(archive, manifest, out_dir):
    archive, manifest, out_dir = Path(archive), Path(manifest), Path(out_dir).absolute()
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError("Output directory must be new; refusing to overwrite")
    spec, entries, kind, manifest_sha = read_manifest(manifest)
    temporary = None
    try:
        # Keep one open archive handle from SHA verification through extraction.
        with archive.open("rb") as raw:
            actual_bytes = raw.seek(0, 2)
            raw.seek(0)
            if actual_bytes != spec["bytes"] or digest_stream(raw) != spec["sha256"]:
                raise ValueError("Return archive checksum/size mismatch")
            raw.seek(0)
            with tarfile.open(fileobj=raw, mode="r:gz") as stream:
                members, links = inspect_members(stream, entries, kind, spec)
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=".verified-return-", dir=out_dir.parent))
                stream.extractall(temporary, members=members, filter="data")
        verify_extracted(temporary, entries)
        if out_dir.exists() or out_dir.is_symlink():
            raise ValueError("Output directory appeared during verification; refusing to overwrite")
        temporary.rename(out_dir)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return {"state": "verified_return_archive", "format": kind,
            "archive_sha256": spec["sha256"], "archive_bytes": spec["bytes"],
            "manifest_sha256": manifest_sha, "files": len(entries), "hardlinks": links,
            "out_dir": str(out_dir),
            "acceptance_scope": "Archive/member byte integrity only; independent numerical replay remains required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.manifest, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
