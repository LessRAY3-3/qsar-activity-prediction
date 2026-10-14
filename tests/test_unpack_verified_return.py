"""Returned bytes must be complete, safe, verified and published only once."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

import unpack_verified_return as unpacker


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_manifest(path, data):
    path.write_text(json.dumps(data))
    return path


def custom_archive(root, members, kind="study"):
    """members are (name, tar type, data bytes, link target)."""
    archive = root / "returned.tar.gz"
    values = {}
    with tarfile.open(archive, "w:gz") as stream:
        for name, member_type, content, target in members:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = member_type, target
            info.size = len(content) if member_type == tarfile.REGTYPE else 0
            stream.addfile(info, io.BytesIO(content) if info.size else None)
            values[name] = content
    hashes = {name: sha(content) for name, content in values.items()}
    base = {"sha256": sha(archive.read_bytes()), "bytes": archive.stat().st_size}
    if kind == "study":
        base.update(files=len(hashes), member_sha256=hashes)
    else:
        base["files"] = {name: {"sha256": hashes[name], "bytes": len(content)}
                         for name, content in values.items()}
    return archive, save_manifest(root / "manifest.json", base), base


def assert_rejected_cleanly(root, archive, manifest, match=None):
    destination = root / "unpacked"
    with pytest.raises((ValueError, tarfile.TarError), match=match):
        unpacker.unpack(archive, manifest, destination)
    assert not destination.exists()
    assert not list(root.glob(".verified-return-*"))


def test_real_study_packager_hardlink_roundtrip(tmp_path, load_script):
    packager = load_script("package_time_budget_study")
    source = tmp_path / "source"
    source.mkdir()
    (source / "protocol.json").write_text('{"jobs":2}')
    (source / "study_status.json").write_text('{"state":"training_complete_pending_independent_audit"}')
    for job in ("one", "two"):
        folder = source / "control/lane0/results/time_campaign" / job
        folder.mkdir(parents=True)
        (folder / "status.json").write_text('{"state":"success"}')
        (folder / "checkpoint.pt").write_bytes(b"identical opaque checkpoint bytes")
    archive = tmp_path / "study.tar.gz"
    spec = packager.package(source, archive)
    assert spec["hardlinks"] >= 2
    manifest = save_manifest(tmp_path / "manifest.json", spec)
    out = tmp_path / "returned study"
    result = unpacker.unpack(archive, manifest, out)
    assert result["hardlinks"] == spec["hardlinks"]
    assert result["files"] == spec["files"]
    assert result["format"] == "study-with-hardlinks"
    for original in source.rglob("*"):
        if original.is_file():
            assert (out / "study" / original.relative_to(source)).read_bytes() == original.read_bytes()


def test_real_regular_packager_roundtrip_and_cli(tmp_path, load_script):
    packager = load_script("win4070_handoff")
    source = tmp_path / "source"
    source.mkdir()
    (source / "complete.json").write_text('{"state":"complete"}')
    (source / "attempt").mkdir()
    (source / "attempt/stdout.log").write_text("log contents\n")
    archive = tmp_path / "regular.tar.gz"
    spec = packager.regular_archive(source, archive)
    manifest = Path(str(archive) + ".manifest.json")
    out = tmp_path / "unpacked"
    proc = subprocess.run([sys.executable, unpacker.__file__, "--archive", str(archive),
                           "--manifest", str(manifest), "--out-dir", str(out)],
                          text=True, capture_output=True, check=True)
    report = json.loads(proc.stdout)
    assert report["format"] == "regular-handoff"
    assert report["hardlinks"] == 0
    assert report["files"] == len(spec["files"])
    assert report["state"] == "verified_return_archive"
    for name in spec["files"]:
        assert (out / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_bad_archive_metadata_fails_before_publish(tmp_path, field):
    archive, manifest, spec = custom_archive(tmp_path, [("file", tarfile.REGTYPE, b"x", "")])
    spec[field] = "0" * 64 if field == "sha256" else spec["bytes"] + 1
    save_manifest(manifest, spec)
    assert_rejected_cleanly(tmp_path, archive, manifest, "archive checksum/size")


@pytest.mark.parametrize("kind", ["study", "regular"])
def test_bad_member_hash_cleans_extracted_temp(tmp_path, kind):
    archive, manifest, spec = custom_archive(tmp_path, [("nested/file", tarfile.REGTYPE, b"x", "")], kind)
    if kind == "study":
        spec["member_sha256"]["nested/file"] = "0" * 64
    else:
        spec["files"]["nested/file"]["sha256"] = "0" * 64
    save_manifest(manifest, spec)
    assert_rejected_cleanly(tmp_path, archive, manifest, "member checksum")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/../../escape", "./file", "a//file",
                                 "C:/escape", "nested\\escape", "."])
def test_unsafe_manifest_or_tar_names_rejected(tmp_path, name):
    archive, manifest, _ = custom_archive(tmp_path, [(name, tarfile.REGTYPE, b"x", "")])
    assert_rejected_cleanly(tmp_path, archive, manifest, "Unsafe")
    assert not (tmp_path.parent / "escape").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE,
                                 tarfile.FIFOTYPE, tarfile.DIRTYPE])
def test_forbidden_member_types(tmp_path, kind):
    archive, manifest, _ = custom_archive(tmp_path, [("unsafe", kind, b"", "../elsewhere")])
    assert_rejected_cleanly(tmp_path, archive, manifest, "Forbidden")


def test_regular_manifest_never_allows_hardlinks(tmp_path):
    archive, manifest, _ = custom_archive(tmp_path, [("a", tarfile.REGTYPE, b"x", ""),
                                                    ("b", tarfile.LNKTYPE, b"x", "a")], "regular")
    assert_rejected_cleanly(tmp_path, archive, manifest, "Forbidden")


@pytest.mark.parametrize("members,match", [
    ([("a", tarfile.LNKTYPE, b"x", "b"), ("b", tarfile.REGTYPE, b"x", "")], "earlier regular"),
    ([("a", tarfile.REGTYPE, b"x", ""), ("b", tarfile.LNKTYPE, b"x", "a"),
      ("c", tarfile.LNKTYPE, b"x", "b")], "earlier regular"),
    ([("a", tarfile.REGTYPE, b"x", ""), ("b", tarfile.LNKTYPE, b"y", "a")], "hash differs"),
    ([("a", tarfile.REGTYPE, b"x", ""), ("b", tarfile.LNKTYPE, b"x", "../a")], "Unsafe"),
    ([("a", tarfile.LNKTYPE, b"x", "a")], "earlier regular"),
])
def test_invalid_hardlinks(tmp_path, members, match):
    archive, manifest, _ = custom_archive(tmp_path, members)
    assert_rejected_cleanly(tmp_path, archive, manifest, match)


def test_duplicate_members_rejected(tmp_path):
    archive, manifest, _ = custom_archive(tmp_path, [("a", tarfile.REGTYPE, b"x", ""),
                                                    ("a", tarfile.REGTYPE, b"x", "")])
    assert_rejected_cleanly(tmp_path, archive, manifest, "Duplicate archive")


@pytest.mark.parametrize("problem", ["extra_tar", "missing_tar", "wrong_count", "hardlink_count", "parent_file"])
def test_exact_inventory_and_counts_required(tmp_path, problem):
    members = [("a", tarfile.REGTYPE, b"x", "")]
    if problem == "parent_file":
        members += [("a/b", tarfile.REGTYPE, b"y", "")]
    archive, manifest, spec = custom_archive(tmp_path, members)
    if problem == "extra_tar":
        spec.update(files=0, member_sha256={})
    elif problem == "missing_tar":
        spec["member_sha256"]["missing"] = sha(b"x")
        spec["files"] += 1
    elif problem == "wrong_count":
        spec["files"] += 1
    elif problem == "hardlink_count":
        spec["hardlinks"] = 1
    save_manifest(manifest, spec)
    assert_rejected_cleanly(tmp_path, archive, manifest)


def test_regular_member_size_checked(tmp_path):
    archive, manifest, spec = custom_archive(tmp_path, [("a", tarfile.REGTYPE, b"x", "")], "regular")
    spec["files"]["a"]["bytes"] = 2
    save_manifest(manifest, spec)
    assert_rejected_cleanly(tmp_path, archive, manifest, "member size")


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_preserved(tmp_path, kind):
    archive, manifest, _ = custom_archive(tmp_path, [("a", tarfile.REGTYPE, b"x", "")])
    out = tmp_path / "unpacked"
    if kind == "directory":
        out.mkdir()
        (out / "keep").write_text("original")
    elif kind == "file":
        out.write_text("original")
    else:
        out.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        unpacker.unpack(archive, manifest, out)
    if kind == "directory":
        assert (out / "keep").read_text() == "original"
    elif kind == "file":
        assert out.read_text() == "original"
    else:
        assert out.is_symlink()


def test_duplicate_manifest_keys_refused(tmp_path):
    archive, manifest, _ = custom_archive(tmp_path, [("a", tarfile.REGTYPE, b"x", "")])
    manifest.write_text('{"files":1,"files":2}')
    assert_rejected_cleanly(tmp_path, archive, manifest, "Duplicate manifest")


def test_extraction_error_cleans_temporary_directory(tmp_path, monkeypatch):
    archive, manifest, _ = custom_archive(tmp_path, [("a", tarfile.REGTYPE, b"x", "")])
    def fail_after_partial(self, destination, **kwargs):
        assert kwargs["filter"] == "data"
        (destination / "partial").write_text("partial")
        raise tarfile.ExtractError("disk failure")
    monkeypatch.setattr(tarfile.TarFile, "extractall", fail_after_partial)
    assert_rejected_cleanly(tmp_path, archive, manifest, "disk failure")
