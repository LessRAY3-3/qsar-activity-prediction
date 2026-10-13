"""Offline campaign bundles must preserve labels and exclude old artifacts."""
import copy
import json
import subprocess
import tarfile

import numpy as np
import pytest

import time_campaign_bundle as bundle
from time_labels import LABEL_POLICY, dataset_signature, label_signature


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in bundle.SPECS:
        (root / name).write_text("# dependency fixture\n")
    for directory in ("scripts", "experiments", "models", "results", "data/processed/splits"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "scripts/train.py").write_text("# fixture source\n")
    (root / "experiments/summary.py").write_text("# fixture source\n")
    for path in bundle.required_sources("rf") | bundle.required_sources("gin"):
        (root / path).write_text("# fixture runtime source\n")
    (root / "models/old.pkl").write_bytes(b"old model")
    (root / "results/old_preds.npz").write_bytes(b"old predictions")
    smiles = np.array(["C", "CC", "CCC", "CCCC"])
    original_y = np.array([6., 7., 8., 9.])
    split = dict(
        smiles=smiles, original_y=original_y, y=np.array([6., 7., 8., np.nan]),
        year=np.array([2000., 2010., 2020., np.nan]),
        train_idx=np.array([0]), valid_idx=np.array([1]), test_idx=np.array([2]),
        excluded_idx=np.array([3]), cutoff_valid=2010, cutoff_test=2020,
        label_policy=LABEL_POLICY, source_sha256="a" * 64,
        dataset_signature=dataset_signature(smiles, original_y))
    split["label_id"] = label_signature(split)
    for tag in bundle.TAGS:
        for suffix in ("graphs", "fingerprints"):
            np.savez(root / f"data/processed/{tag}_{suffix}.npz", smiles=smiles, y=original_y)
        np.savez(root / f"data/processed/splits/{tag}_time.npz", **split)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    return root


def test_roles_share_labels_and_archives_are_deterministic(repo, tmp_path):
    manifests = {role: bundle.make_manifest(repo, role) for role in ("rf", "gin")}
    assert manifests["rf"]["targets"] == manifests["gin"]["targets"]
    for role, manifest in manifests.items():
        assert manifest["source"]["dirty"] is False
        assert bundle.verify_manifest(manifest, repo) == manifest
        first, second = tmp_path / f"{role}1", tmp_path / f"{role}2"
        bundle.write_bundle(repo, first, manifest)
        bundle.write_bundle(repo, second, manifest)
        assert bundle.sha256(first / "bundle.tar.gz") == bundle.sha256(second / "bundle.tar.gz")
        with tarfile.open(first / "bundle.tar.gz") as archive:
            names = archive.getnames()
            assert "manifest.json" in names
            assert all(info.isfile() for info in archive.getmembers())
            assert not any(name.startswith(("results/", "models/")) for name in names)
            assert not any(("graphs" if role == "rf" else "fingerprints") in p for p in names)
            assert json.load(archive.extractfile("manifest.json")) == manifest
        with pytest.raises(FileExistsError):
            bundle.write_bundle(repo, first, manifest)


def test_dirty_source_is_explicit_and_hash_verified(repo):
    (repo / "scripts/train.py").write_text("# uncommitted preflight fix\n")
    manifest = bundle.make_manifest(repo, "rf")
    assert manifest["source"]["dirty"] is True
    assert any("scripts/train.py" in line for line in manifest["source"]["status"])
    bundle.verify_manifest(manifest, repo)
    (repo / "scripts/train.py").write_text("# later changes\n")
    with pytest.raises(ValueError, match="file changed"):
        bundle.verify_manifest(manifest, repo)


@pytest.mark.parametrize("path", ["../escape", "/etc/passwd", "scripts/../escape",
                                 "./scripts/train.py", "scripts//train.py", "a\\b"])
def test_unsafe_paths_rejected(repo, path):
    manifest = bundle.make_manifest(repo, "rf")
    manifest["files"][0]["path"] = path
    with pytest.raises(ValueError, match="unsafe"):
        bundle.verify_manifest(manifest, repo)


def test_duplicate_and_extra_artifact_rejected(repo):
    manifest = bundle.make_manifest(repo, "rf")
    manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        bundle.verify_manifest(manifest, repo)
    manifest["files"][-1] = {"path": "models/old.pkl", "size": 9,
                             "sha256": bundle.sha256(repo / "models/old.pkl")}
    with pytest.raises(ValueError, match="allowlist"):
        bundle.verify_manifest(manifest, repo)


def test_symlink_and_missing_role_input_rejected(repo):
    manifest = bundle.make_manifest(repo, "rf")
    path = repo / "data/processed/a2a_fingerprints.npz"
    path.unlink()
    path.symlink_to(repo / "data/processed/abl1_fingerprints.npz")
    with pytest.raises(ValueError, match="symlink"):
        bundle.verify_manifest(manifest, repo)
    path.unlink()
    manifest["files"] = [f for f in manifest["files"] if f["path"] != str(path.relative_to(repo))]
    with pytest.raises(ValueError, match="missing required"):
        bundle.verify_manifest(manifest, repo)


def test_cache_misalignment_na_and_stale_labels_fail_before_packaging(repo):
    path = repo / "data/processed/a2a_fingerprints.npz"
    with np.load(path) as cache:
        smiles, y = cache["smiles"], cache["y"]
    np.savez(path, smiles=smiles[::-1], y=y)
    with pytest.raises(ValueError, match="row order"):
        bundle.make_manifest(repo, "rf")
    np.savez(path, smiles=smiles, y=y)
    marker = repo / "data/processed/splits/a2a_time_NA.json"
    marker.write_text('{"reason": "regeneration incomplete"}')
    with pytest.raises(ValueError, match="unavailable"):
        bundle.make_manifest(repo, "rf")
    marker.unlink()
    split_path = repo / "data/processed/splits/a2a_time.npz"
    with np.load(split_path) as archive:
        split = dict(archive)
    split["y"][0] += 1
    np.savez(split_path, **split)
    with pytest.raises(ValueError, match="signature mismatch"):
        bundle.make_manifest(repo, "rf")


def test_metadata_tamper_and_invalid_schema_rejected(repo):
    manifest = bundle.make_manifest(repo, "rf")
    manifest["targets"]["a2a"]["cutoff_valid"] = 1990
    with pytest.raises(ValueError, match="target metadata"):
        bundle.verify_manifest(manifest, repo)
    manifest["schema"] = "other"
    with pytest.raises(ValueError, match="schema"):
        bundle.verify_manifest(manifest, repo)


@pytest.mark.parametrize("role,path", [
    (role, path) for role in ("rf", "gin") for path in sorted(bundle.required_sources(role))
])
def test_role_runtime_source_cannot_be_omitted_even_with_consistent_hash_inventory(repo, role, path):
    manifest = bundle.make_manifest(repo, role)
    manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != path]
    del manifest["source"]["sha256"][path]
    with pytest.raises(ValueError, match="runtime source"):
        bundle.verify_manifest(manifest, repo)


def test_cli_requires_explicit_output():
    with pytest.raises(SystemExit) as error:
        bundle.main(["--role", "rf"])
    assert error.value.code == 2
