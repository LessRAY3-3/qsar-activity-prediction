"""Copy completed CUDA study evidence without duplicating input graph caches.

Source files are never changed. Keep lane source, dependency specifications,
canonical outputs, job snapshots and every log. Packaging/replay remain separate.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile


def collect(root, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(root):
        raise ValueError("Return directory must be new and outside the study")
    status = json.loads((root / "study_status.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if status["state"] != "training_complete_pending_independent_audit" or protocol["jobs"] != 128:
        raise ValueError("Expected a completed 128-job study")
    if protocol["device"] != "cuda" or protocol["seeds"] != [42, 1, 2, 3, 4, 5, 6, 7]:
        raise ValueError("Unexpected device or seed cohort")
    if list(root.rglob(".time_campaign_*.lock")):
        raise ValueError("Study is still locked")
    jobs = set(root.glob("*/lane*/results/time_campaign/*/status.json"))
    lanes = (("a2a", "egfr"), ("egfr_full", "hivpr"),
             ("herg", "mpro"), ("abl1", "vegfr2"))
    expected = {root / arm / f"lane{index}/results/time_campaign/gin_{tag}_seed{seed}/status.json"
                for arm in ("control", "extended") for index, tags in enumerate(lanes)
                for tag in tags for seed in (42, 1, 2, 3, 4, 5, 6, 7)}
    if jobs != expected or any(json.loads(p.read_text())["state"] != "success" for p in jobs):
        raise ValueError("Missing or unsuccessful jobs")
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in returned study: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Nonregular evidence: {path}")
        relative = path.relative_to(root)
        # Frozen graphs and split inputs travel separately in the pinned bundle.
        if len(relative.parts) >= 3 and relative.parts[2] == "data":
            continue
        paths.append((path, relative))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".collect-study-", dir=output.parent))
    try:
        for path, relative in paths:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        temporary.rename(output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {"state": "collected_pending_independent_audit", "files": len(paths), "jobs": 128}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.study_root, args.output)))


if __name__ == "__main__":
    main()
