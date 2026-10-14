"""The supplement has independent roots and immutable complete-cohort protocols."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_time_campaign_seed_extension import extension_bundle as _extension_bundle

extension_bundle = _extension_bundle


def test_cpu80_and_cuda128_plans_cover_complete_cohorts(extension_bundle, load_script):
    study = load_script("run_time_budget_extension")
    for seeds, device, workers, count in [(list(range(3, 8)), "cpu", 4, 80),
                                         ([42, 1, 2, 3, 4, 5, 6, 7], "cuda", 1, 128)]:
        plan = study.plan(extension_bundle, seeds, device, workers)
        assert plan["jobs"] == count and plan["device"] == device
        assert plan["parallel_workers"] == workers and plan["seeds"] == seeds
        assert plan["wrapper_sha256"] == study.digest(extension_bundle / study.WRAPPER_PATH)
        cohort, roots = set(), set()
        for arm in study.ARMS:
            for lane, tags in enumerate(study.LANES):
                root = Path("output") / arm / f"lane{lane}"
                roots.add(root)
                command = study.worker_command(root, arm, tags, seeds, device)
                assert command[1] == str(root / study.WRAPPER_PATH)
                assert command[command.index("--device") + 1] == device
                for tag in tags:
                    for seed in seeds:
                        cohort.add((arm, tag, seed))
        assert len(cohort) == count and len(roots) == 8


def test_dry_run_does_not_create_output(extension_bundle, tmp_path, load_script, capsys):
    study = load_script("run_time_budget_extension")
    output = tmp_path / "new-study"
    study.main(["--bundle-root", str(extension_bundle), "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["jobs"] == 80
    assert not output.exists()


def test_prepare_failure_does_not_publish_partial_lane(extension_bundle, tmp_path, load_script, monkeypatch):
    study = load_script("run_time_budget_extension")
    root = tmp_path / "output/control/lane0"
    original = study.shutil.copy2
    def fail(*args, **kwargs):
        raise OSError("copy interrupted")
    monkeypatch.setattr(study.shutil, "copy2", fail)
    with pytest.raises(OSError, match="interrupted"):
        study.prepare_lane(extension_bundle, root)
    assert not root.exists() and not list(root.parent.glob(".lane0.prepare-*"))
    monkeypatch.setattr(study.shutil, "copy2", original)
    study.prepare_lane(extension_bundle, root)
    study.prepare_lane(extension_bundle, root)
    assert (root / "manifest.json").read_bytes() == (extension_bundle / "manifest.json").read_bytes()


def test_execution_isolated_roots_threads_resume_and_protocol_guard(extension_bundle, tmp_path, load_script, monkeypatch):
    study = load_script("run_time_budget_extension")
    monkeypatch.setattr(study, "environment", lambda: {"platform": "offline test fixture"})
    output = tmp_path / "extension"
    previous = tmp_path / "original-study"
    previous.mkdir()
    (previous / "accepted.json").write_bytes(b"unchanged original")
    calls = []
    def train(command, **kwargs):
        assert kwargs["env"] | study.THREAD_ENV == kwargs["env"]
        assert kwargs["env"]["QSAR_TORCH_THREADS"] == "4"
        assert kwargs["env"]["QSAR_TORCH_INTEROP_THREADS"] == "1"
        assert kwargs["cwd"].is_relative_to(output)
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess, "run", train)
    cli = ["--bundle-root", str(extension_bundle), "--output", str(output),
           "--parallel-workers", "1", "--execute"]
    study.main(cli)
    assert len(calls) == 8 and len({root for _, root in calls}) == 8
    assert all("/control/" in str(root) for _, root in calls[:4])
    assert all("/extended/" in str(root) for _, root in calls[4:])
    state = json.loads((output / "study_status.json").read_text())
    assert state["state"] == "training_complete_pending_independent_audit"
    study.main(cli + ["--rerun"])
    assert all("--rerun" in command for command, _ in calls[8:])
    with pytest.raises(ValueError, match="protocol changed"):
        study.main(cli + ["--seeds", "3", "4"])
    assert len(calls) == 16
    assert (previous / "accepted.json").read_bytes() == b"unchanged original"
    assert not (extension_bundle / "results").exists()


def test_failed_control_never_starts_extended(extension_bundle, tmp_path, load_script, monkeypatch):
    study = load_script("run_time_budget_extension")
    monkeypatch.setattr(study, "environment", lambda: {"platform": "offline test fixture"})
    output = tmp_path / "failed-study"
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="control worker failed"):
        study.main(["--bundle-root", str(extension_bundle), "--output", str(output), "--execute"])
    assert not (output / "extended").exists()
    assert json.loads((output / "study_status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("extra", [["--seeds", "3", "3"], ["--seeds", "0"],
                                   ["--parallel-workers", "0"], ["--parallel-workers", "5"]])
def test_invalid_schedule_rejected(tmp_path, load_script, extra):
    study = load_script("run_time_budget_extension")
    with pytest.raises(SystemExit):
        study.parse_args(["--bundle-root", str(tmp_path / "inputs"), "--output", str(tmp_path / "out"), *extra])


def test_nested_output_rejected_before_writes(tmp_path, load_script):
    study = load_script("run_time_budget_extension")
    source = tmp_path / "input"
    for output in (source, source / "child", tmp_path):
        with pytest.raises(SystemExit):
            study.main(["--bundle-root", str(source), "--output", str(output), "--execute"])
    assert not source.exists()
