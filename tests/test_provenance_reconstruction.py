"""Preparing a provenance rerun must not overwrite earlier experiments."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "provenance_planner", Path(__file__).parents[1] / "scripts/plan_provenance_reconstruction.py")
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


@pytest.fixture
def fixture_plan(tmp_path):
    source = tmp_path / "scripts/train.py"
    source.parent.mkdir()
    source.write_text("# frozen training source\n")
    data = tmp_path / "data/input.npz"
    data.parent.mkdir()
    data.write_bytes(b"frozen data")
    old = tmp_path / "results/metrics.json"
    old.parent.mkdir()
    old.write_text('{"r2": 0.5}')
    plan = {"source_sha256": {"scripts/train.py": planner.sha256(source)},
            "input_sha256": {"data/input.npz": planner.sha256(data)},
            "jobs": [{"id": "egfr_random_seed42", "inputs": ["data/input.npz"]}]}
    return tmp_path, plan, old


def test_prepare_isolated_copies_and_refuses_existing(fixture_plan):
    root, plan, old = fixture_plan
    before = old.read_bytes()
    dest = planner.prepare_job(root, plan, "egfr_random_seed42")
    assert dest == root / ".scratch/provenance_reconstruction/egfr_random_seed42"
    assert (dest / "scripts/train.py").read_bytes() == (root / "scripts/train.py").read_bytes()
    assert not (dest / "data/input.npz").is_symlink()
    (dest / "data/input.npz").write_bytes(b"independent copy")
    assert (root / "data/input.npz").read_bytes() == b"frozen data"
    with pytest.raises(FileExistsError):
        planner.prepare_job(root, plan, "egfr_random_seed42")
    assert old.read_bytes() == before


def test_changed_input_fails_before_creating_workspace(fixture_plan):
    root, plan, _ = fixture_plan
    (root / "data/input.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since plan"):
        planner.prepare_job(root, plan, "egfr_random_seed42")
    assert not (root / ".scratch").exists()


def test_rejects_job_and_input_path_escape(fixture_plan):
    root, plan, _ = fixture_plan
    with pytest.raises(ValueError, match="Invalid job id"):
        planner.prepare_job(root, plan, "../../results")
    plan["source_sha256"] = {"../outside.py": "0"}
    with pytest.raises(ValueError, match="Unsafe relative path"):
        planner.prepare_job(root, plan, "egfr_random_seed42")


def test_rejects_symlink_destination_parent(fixture_plan, tmp_path):
    root, plan, _ = fixture_plan
    (root / ".scratch").mkdir()
    target = root / "results"
    (root / ".scratch/provenance_reconstruction").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        planner.prepare_job(root, plan, "egfr_random_seed42")
    assert not (target / "egfr_random_seed42").exists()
