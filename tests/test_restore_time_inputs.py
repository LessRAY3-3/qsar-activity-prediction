"""Frozen inputs restore on a clean checkout and fail closed on mixed vintages."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import tarfile

import numpy as np
import pytest
import torch

import restore_time_inputs as restore
from time_labels import LABEL_POLICY, dataset_signature, label_signature


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "source"
    splits = root / "data/processed/splits"
    splits.mkdir(parents=True)
    (root / "results/gnn_models").mkdir(parents=True)
    smiles = np.array(["C", "CC", "CCC", "CCCC", "CCCCC"])
    original_y = np.array([6., 7., 8., 9., 10.])
    split = dict(smiles=smiles, original_y=original_y,
                 y=np.array([6., 7., 8., 9., np.nan]),
                 year=np.array([2000., 2010., 2020., 2021., np.nan]),
                 train_idx=np.array([0]), valid_idx=np.array([1]), test_idx=np.array([2, 3]),
                 excluded_idx=np.array([4]), cutoff_valid=2010, cutoff_test=2020,
                 label_policy=LABEL_POLICY, source_sha256="a" * 64,
                 dataset_signature=dataset_signature(smiles, original_y))
    split["label_id"] = label_signature(split)
    for tag in restore.TAGS:
        np.savez(splits / f"{tag}_time.npz", **split)
        for prefix, suf in [("rf", ""), ("gnn", ""), ("gnn", "_seed1"), ("gnn", "_seed2")]:
            np.savez(root / f"results/{prefix}_preds_{tag}_time{suf}.npz",
                     label_id=split["label_id"], test_idx=split["test_idx"], y_true=np.array([8., 9.]))
            metric = f"{prefix + '_' if prefix == 'gnn' else ''}metrics_{tag}{suf}.json"
            (root / "results" / metric).write_text(json.dumps({"time": {"label_id": split["label_id"]}}))
    # Actual torch serialization and weights-only loading; no mocked checkpoints.
    from gnn_03_train_gin import GINRegressor
    state = GINRegressor(hidden=8, num_layers=2, dropout=0.2).state_dict()
    for tag in restore.TAGS:
        for suffix, seed in (("", 42), ("_seed1", 1), ("_seed2", 2)):
            torch.save({"state_dict": state, "label_id": split["label_id"],
                        "args": {"tag": tag, "model": "gin", "split": "time",
                                 "seed": seed, "suffix": suffix}},
                       root / f"results/gnn_models/{tag}_gin_time{suffix}.pt")
    package = tmp_path / "package"
    restore.pack(root, package)
    fresh = tmp_path / "fresh"
    shutil.copytree(root / "results", fresh / "results")
    return root, fresh, package


def test_clean_checkout_restores_exact_bytes_and_is_idempotent(frozen):
    root, fresh, package = frozen
    restore.restore(fresh, package)
    restore.restore(fresh, package)
    for tag in restore.TAGS:
        relative = f"data/processed/splits/{tag}_time.npz"
        assert (fresh / relative).read_bytes() == (root / relative).read_bytes()
    assert not list((fresh / "data/processed").glob("*fingerprints*"))


@pytest.mark.parametrize("conflict", ["prediction", "metric", "na", "existing", "symlink"])
def test_conflicting_publication_or_local_state_writes_no_splits(frozen, conflict):
    _, fresh, package = frozen
    # Last target catches writes interleaved with validation of earlier targets.
    tag = restore.TAGS[-1]
    splits = fresh / "data/processed/splits"
    if conflict == "prediction":
        np.savez(fresh / f"results/gnn_preds_{tag}_time_seed2.npz", label_id="b" * 64)
    elif conflict == "metric":
        (fresh / f"results/gnn_metrics_{tag}_seed2.json").write_text('{"time":{"label_id":"old"}}')
    else:
        splits.mkdir(parents=True)
        if conflict == "na":
            (splits / f"{tag}_time_NA.json").write_text('{}')
        elif conflict == "existing":
            (splits / f"{tag}_time.npz").write_bytes(b"local data")
        else:
            (splits / f"{tag}_time.npz").symlink_to(package / "manifest.json")
    with pytest.raises(ValueError):
        restore.restore(fresh, package)
    assert not (splits / "a2a_time.npz").exists()
    if conflict == "existing":
        assert (splits / f"{tag}_time.npz").read_bytes() == b"local data"


def test_corrupt_archive_fails_hash_before_writes(frozen):
    _, fresh, package = frozen
    with (package / "splits.tar.gz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        restore.restore(fresh, package)
    assert not (fresh / "data").exists()


@pytest.mark.parametrize("kind", ["path", "symlink", "duplicate", "missing", "bytes"])
def test_rehashed_unsafe_or_incomplete_archive_is_rejected(frozen, kind):
    root, fresh, package = frozen
    archive = package / "splits.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for tag in restore.TAGS:
            if kind == "missing" and tag == restore.TAGS[-1]:
                continue
            data = (root / f"data/processed/splits/{tag}_time.npz").read_bytes()
            member = tarfile.TarInfo(f"{tag}_time.npz")
            member.size = len(data)
            if tag == "a2a":
                if kind == "path":
                    member.name = "../outside.npz"
                elif kind == "symlink":
                    member.type, member.linkname, member.size = tarfile.SYMTYPE, "../outside.npz", 0
                elif kind == "bytes":
                    data = b"x" * len(data)
            tar.addfile(member, io.BytesIO(data))
            if kind == "duplicate" and tag == "a2a":
                tar.addfile(member, io.BytesIO(data))
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["archive"].update(size=archive.stat().st_size,
                               sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        restore.restore(fresh, package)
    assert not (fresh / "data").exists()
    assert not (fresh.parent / "outside.npz").exists()


@pytest.mark.parametrize("change", ["label_id", "missing_label", "tag", "model", "split", "seed", "suffix"])
def test_real_torch_checkpoint_version_and_cell_gate_before_restore(frozen, change):
    _, fresh, package = frozen
    checkpoint = fresh / "results/gnn_models/vegfr2_gin_time_seed2.pt"
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert any(torch.is_tensor(value) for value in data["state_dict"].values())
    if change == "label_id":
        data[change] = "0" * 64
    elif change == "missing_label":
        data.pop("label_id")
    else:
        data["args"][change] = "wrong-cell"
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        restore.restore(fresh, package)
    assert not (fresh / "data/processed/splits/a2a_time.npz").exists()


def test_missing_checkpoint_is_not_silently_ignored(frozen):
    _, fresh, package = frozen
    (fresh / "results/gnn_models/vegfr2_gin_time_seed2.pt").unlink()
    with pytest.raises(ValueError, match="missing or nonregular"):
        restore.restore(fresh, package)
    assert not (fresh / "data/processed/splits/a2a_time.npz").exists()


def test_atomic_restore_copy_error_leaves_no_partial_target_and_retry_succeeds(tmp_path, monkeypatch):
    source, target = tmp_path / "source.npz", tmp_path / "final.npz"
    source.write_bytes(b"complete source content" * 1000)
    digest = restore.sha256(source)
    original_copy = restore.shutil.copyfileobj
    def fail_copy(input_file, output):
        output.write(input_file.read(17))
        raise OSError("injected disk write error")
    monkeypatch.setattr(restore.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="disk write"):
        restore.atomic_install(source, target, digest)
    assert not target.exists()
    assert not list(tmp_path.glob(".time-restore-*"))
    monkeypatch.setattr(restore.shutil, "copyfileobj", original_copy)
    restore.atomic_install(source, target, digest)
    assert target.read_bytes() == source.read_bytes()


def test_atomic_restore_refuses_a_conflict_created_after_preflight(tmp_path, monkeypatch):
    source, target = tmp_path / "source.npz", tmp_path / "final.npz"
    source.write_bytes(b"new split")
    original_link = restore.os.link
    def concurrent_create(src, dest):
        Path(dest).write_bytes(b"other process split")
        original_link(src, dest)
    monkeypatch.setattr(restore.os, "link", concurrent_create)
    with pytest.raises(ValueError, match="conflicting"):
        restore.atomic_install(source, target, restore.sha256(source))
    assert target.read_bytes() == b"other process split"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal interruption test")
def test_sigkill_during_restore_copy_never_publishes_truncated_npz(tmp_path):
    source, target = tmp_path / "source.npz", tmp_path / "final.npz"
    ready = tmp_path / "ready"
    source.write_bytes(b"complete frozen split" * 4096)
    code = """
import pathlib, signal, sys
sys.path.insert(0, sys.argv[1])
import restore_time_inputs as module
source, target, ready = map(pathlib.Path, sys.argv[2:])
def interrupted_copy(input_file, output):
    output.write(input_file.read(17)); output.flush()
    ready.touch()
    signal.pause()
module.shutil.copyfileobj = interrupted_copy
module.atomic_install(source, target, module.sha256(source))
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", code,
                                str(Path(restore.__file__).parent), str(source), str(target), str(ready)])
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None, "copy process exited before interruption point"
            time.sleep(0.02)
        assert ready.exists(), "copy process did not reach the interruption point"
        process.kill()
        process.wait(timeout=10)
        assert not target.exists()
        restore.atomic_install(source, target, restore.sha256(source))
        assert target.read_bytes() == source.read_bytes()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
