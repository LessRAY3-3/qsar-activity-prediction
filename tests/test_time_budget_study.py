"""Isolation and fixed-comparison requirements for the CPU sensitivity study."""
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("budget_study", SCRIPTS / "run_time_budget_study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_complete_balanced_cohort_and_isolated_destinations(tmp_path):
    tags = [tag for lane in study.LANES for tag in lane]
    assert len(tags) == len(set(tags)) == 8
    destinations = []
    for arm, params in study.ARMS.items():
        for lane, lane_tags in enumerate(study.LANES):
            root = tmp_path / arm / f"lane{lane}"
            destinations.append(root)
            cmd = study.worker_command(root, arm, lane_tags)
            assert cmd[1] == str(root / "scripts/run_time_campaign.py")
            assert cmd[cmd.index("--device") + 1] == "cpu"
            assert cmd[cmd.index("--seeds") + 1:cmd.index("--seeds") + 4] == ["42", "1", "2"]
            for key, value in params.items():
                assert cmd[cmd.index("--" + key.replace("_", "-")) + 1] == str(value)
    assert len(set(destinations)) == 8
    assert study.ARMS["control"] == {"epochs": 100, "min_epochs": 0, "patience": 20}
    assert study.ARMS["extended"] == {"epochs": 200, "min_epochs": 60, "patience": 40}


def test_reject_nested_output_before_writing(tmp_path):
    import pytest
    bundle = tmp_path / "input"
    for output in (bundle, bundle / "outputs", tmp_path):
        with pytest.raises(SystemExit):
            study.main(["--bundle-root", str(bundle), "--output", str(output), "--execute"])
    assert not bundle.exists()
