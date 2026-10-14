"""Extra seeds use the frozen worker API without rewriting its CLI/constants."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import time_campaign_bundle as bundle
from time_labels import LABEL_POLICY, dataset_signature, label_signature


@pytest.fixture
def extension_bundle(tmp_path):
    root = tmp_path / "inputs"
    original = Path(__file__).resolve().parents[1]
    names = bundle.required_sources("gin") | {
        "scripts/run_time_campaign_seed_extension.py", "scripts/run_time_budget_extension.py"}
    for name in names | set(bundle.SPECS):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)
    smiles = np.array(["C", "CC", "CCC", "CCCC"])
    y = np.array([5., 6., 7., 8.], dtype=np.float32)
    split = dict(smiles=smiles, y=y, original_y=y, label_policy=LABEL_POLICY,
                 year=np.array([2000., 2010., 2020., 2021.]),
                 train_idx=np.array([0]), valid_idx=np.array([1]), test_idx=np.array([2, 3]),
                 excluded_idx=np.array([], dtype=int), cutoff_valid=2010, cutoff_test=2020,
                 source_sha256="a" * 64, dataset_signature=dataset_signature(smiles, y))
    split["label_id"] = label_signature(split)
    (root / "data/processed/splits").mkdir(parents=True)
    for tag in bundle.TAGS:
        np.savez(root / f"data/processed/{tag}_graphs.npz", smiles=smiles, y=y)
        np.savez(root / f"data/processed/splits/{tag}_time.npz", **split)
    files = [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": bundle.sha256(p)}
             for p in sorted(root.rglob("*")) if p.is_file()]
    hashes = {entry["path"]: entry["sha256"] for entry in files}
    manifest = {"schema": bundle.SCHEMA, "role": "gin", "files": files,
                "source": {"git_head": "1" * 40, "dirty": False, "status": [],
                           "sha256": {p: h for p, h in hashes.items() if p.startswith("scripts/")}},
                "dependency_sha256": {p: hashes[p] for p in bundle.SPECS},
                "targets": bundle.target_metadata(root, "gin")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    bundle.verify_manifest(manifest, root)
    return root


def test_new_seeds_use_actual_worker_and_dry_run_is_read_only(extension_bundle, load_script, monkeypatch, capsys):
    wrapper = load_script("run_time_campaign_seed_extension")
    monkeypatch.setattr(wrapper, "ROOT", extension_bundle)
    before = {str(p): p.read_bytes() for p in extension_bundle.rglob("*") if p.is_file()}
    seeds_before = wrapper.worker.SEEDS
    wrapper.main(["--tags", "a2a", "--seeds", "3", "7", "--epochs", "200",
                  "--min-epochs", "60", "--patience", "40"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "dry-run"
    assert [job["id"] for job in plan["jobs"]] == ["gin_a2a_seed3", "gin_a2a_seed7"]
    for seed, job in zip((3, 7), plan["jobs"]):
        assert job["command"][-2:] == ["--suffix", f"_seed{seed}"]
        assert job["provenance"]["hyperparameters"]["seed"] == seed
        assert job["provenance"]["hyperparameters"]["epochs"] == 200
    assert wrapper.worker.SEEDS == seeds_before == (42, 1, 2)
    assert before == {str(p): p.read_bytes() for p in extension_bundle.rglob("*") if p.is_file()}
    assert not (extension_bundle / "results").exists()


def test_execute_delegates_jobs_with_role_lock_and_rerun(extension_bundle, load_script, monkeypatch):
    wrapper = load_script("run_time_campaign_seed_extension")
    monkeypatch.setattr(wrapper, "ROOT", extension_bundle)
    calls = []
    def run_job(job, args, root):
        assert (root / ".time_campaign_gin.lock").is_file()
        assert args.role == "gin" and args.rerun and args.device == "cuda"
        assert args.campaign_dir == root / "results/time_campaign"
        calls.append(job["id"])
    monkeypatch.setattr(wrapper.worker, "run_job", run_job)
    wrapper.main(["--tags", "a2a", "--seeds", "42", "3", "--device", "cuda", "--execute", "--rerun"])
    assert calls == ["gin_a2a_seed42", "gin_a2a_seed3"]
    assert not (extension_bundle / ".time_campaign_gin.lock").exists()


@pytest.mark.parametrize("args", [["--seeds", "0"], ["--seeds", "-1"], ["--seeds", "3", "3"],
                                 ["--epochs", "0"], ["--patience", "0"],
                                 ["--min-epochs", "101"], ["--min-epochs", "-1"]])
def test_wrapper_invalid_arguments_rejected(load_script, args):
    wrapper = load_script("run_time_campaign_seed_extension")
    with pytest.raises(SystemExit):
        wrapper.parse_args(args)


def test_wrapper_requires_manifest_coverage(extension_bundle, load_script):
    wrapper = load_script("run_time_campaign_seed_extension")
    path = extension_bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    name = wrapper.WRAPPER_PATH
    manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != name]
    del manifest["source"]["sha256"][name]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wrapper"):
        wrapper.verify_wrapper_bundle(extension_bundle)
