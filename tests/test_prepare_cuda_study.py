"""Frozen CUDA deployment must neither substitute sources nor start training."""
import hashlib
import io
import json
import subprocess
import tarfile

import pytest


def test_real_frozen_preparation_offline(tmp_path, load_script):
    module = load_script("prepare_cuda_study")
    target = tmp_path / "deployment with spaces"
    result = module.prepare(target)
    assert result["jobs"] == 128
    assert result["bundle_commit"] == module.BUNDLE_COMMIT
    assert not (target / "output").exists()
    verification = json.loads((target / "offline_verification.json").read_text())
    assert verification["state"] == "offline_verified"
    assert verification["plan"]["parallel_workers"] == 1
    assert verification["plan"]["thread_environment"]["QSAR_TORCH_THREADS"] == "4"
    manifest = json.loads((target / "input/manifest.json").read_text())
    for item in manifest["files"]:
        assert module.digest(target / "input" / item["path"]) == item["sha256"]
    assert "--rerun" not in (target / "run.sh").read_text()
    for name in ("run.sh", "package_return.sh"):
        subprocess.run(["bash", "-n", str(target / name)], check=True)
    with pytest.raises(ValueError, match="new output"):
        module.prepare(target)


def test_modified_archive_rejected_before_extraction(tmp_path, load_script):
    module = load_script("prepare_cuda_study")
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module.extract_input(archive, tmp_path / "input")
    assert not (tmp_path / "input").exists()


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("input", "symlink"),
                                       ("file", "duplicate")])
def test_unsafe_tar_rejected_even_with_matching_hash(tmp_path, load_script, name, kind):
    module = load_script("prepare_cuda_study")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo(name)
        if kind == "symlink":
            info.type, info.linkname = tarfile.SYMTYPE, "/tmp/elsewhere"
            stream.addfile(info)
        else:
            info.size = 1
            stream.addfile(info, io.BytesIO(b"x"))
            if kind == "duplicate":
                stream.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe|Duplicate"):
        module.extract_input(archive, tmp_path / "input", expected_sha=module.digest(archive))
    assert not (tmp_path / "input").exists()


def test_preflight_detects_altered_input_without_training(tmp_path, load_script):
    module = load_script("prepare_cuda_study")
    target = tmp_path / "study"
    module.prepare(target)
    trainer = target / "input/scripts/gnn_03_train_gin.py"
    trainer.write_text("raise RuntimeError('must not run')")
    import sys
    result = subprocess.run([sys.executable, str(target / "preflight.py"), "--offline"],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "Frozen input changed" in result.stderr
    assert not (target / "output").exists()


def fake_study(root):
    root.mkdir()
    (root / "protocol.json").write_text(json.dumps({"jobs": 128, "device": "cuda",
                                                  "seeds": [42, 1, 2, 3, 4, 5, 6, 7]}))
    (root / "study_status.json").write_text(json.dumps({"state": "training_complete_pending_independent_audit"}))
    lanes = (("a2a", "egfr"), ("egfr_full", "hivpr"), ("herg", "mpro"), ("abl1", "vegfr2"))
    for arm in ("control", "extended"):
        for index, tags in enumerate(lanes):
            for tag in tags:
                for seed in (42, 1, 2, 3, 4, 5, 6, 7):
                    job = root / arm / f"lane{index}/results/time_campaign/gin_{tag}_seed{seed}"
                    job.mkdir(parents=True)
                    (job / "status.json").write_text('{"state":"success"}')
                    (job / "train.log").write_text("recorded")
    lane = root / "control/lane0"
    for name in ("scripts/trainer.py", "data/processed/graphs.npz", "results/model.pt", "manifest.json"):
        path = lane / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence")


def test_collect_keeps_sources_results_logs_excludes_input_and_preserves_source(tmp_path, load_script):
    module = load_script("collect_cuda_study")
    root = tmp_path / "output"
    fake_study(root)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    target = tmp_path / "return"
    result = module.collect(root, target)
    assert result["jobs"] == 128
    assert not (target / "control/lane0/data").exists()
    for name in ("scripts/trainer.py", "results/model.pt", "results/time_campaign/gin_a2a_seed42/train.log", "manifest.json"):
        assert (target / "control/lane0" / name).is_file()
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("problem", ["incomplete", "wrong_job", "lock", "symlink", "inside"])
def test_collect_refuses_unfinished_or_unsafe_study(tmp_path, load_script, problem):
    module = load_script("collect_cuda_study")
    root = tmp_path / "output"
    fake_study(root)
    target = tmp_path / "return"
    if problem == "incomplete":
        (root / "control/lane0/results/time_campaign/gin_a2a_seed42/status.json").unlink()
    elif problem == "wrong_job":
        job = root / "control/lane0/results/time_campaign/gin_a2a_seed42"
        job.rename(job.with_name("unplanned_job"))
    elif problem == "lock":
        (root / ".time_campaign_budget.lock").write_text("owner")
    elif problem == "symlink":
        (root / "link").symlink_to(tmp_path)
    else:
        target = root / "return"
    with pytest.raises(ValueError):
        module.collect(root, target)
    assert not target.exists()
