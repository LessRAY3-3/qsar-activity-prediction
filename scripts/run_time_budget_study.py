"""Run a predeclared CPU budget sensitivity study in eight isolated worker roots.

Requires an extracted, clean GIN input bundle. Plans by default; --execute
creates outputs and trains. A repeat invocation resumes verified successes;
failed/incompatible jobs require --rerun. Never writes into the input bundle.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_time_campaign import atomic_json, digest, environment, worker_lock
from time_campaign_bundle import verify_manifest

ARMS = {"control": {"epochs": 100, "min_epochs": 0, "patience": 20},
        "extended": {"epochs": 200, "min_epochs": 60, "patience": 40}}
LANES = (("a2a", "egfr"), ("egfr_full", "hivpr"),
         ("herg", "mpro"), ("abl1", "vegfr2"))
THREAD_ENV = {"QSAR_TORCH_THREADS": "4", "QSAR_TORCH_INTEROP_THREADS": "1",
              "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
              "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "4",
              "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}


def plan(bundle):
    manifest = verify_manifest(json.loads((bundle / "manifest.json").read_text()), bundle)
    if manifest["role"] != "gin" or manifest["source"]["dirty"]:
        raise ValueError("study requires a clean-source GIN bundle")
    return {"schema": 1, "manifest_sha256": digest(bundle / "manifest.json"),
            "source_commit": manifest["source"]["git_head"],
            "coordinator_sha256": digest(Path(__file__)),
            "arms": ARMS, "lanes": LANES, "seeds": [42, 1, 2], "jobs": 48,
            "device": "cpu", "parallel_workers": 4, "thread_environment": THREAD_ENV,
            "selection": "Lowest validation RMSE; unchanged scheduler and 1e-5 improvement tolerance",
            "primary_comparison": "Paired same-host best-checkpoint train MSE and validation RMSE",
            "common_flag": "best_epoch < 30 and best_checkpoint_train_mse > 0.5 in both arms",
            "test_policy": "Secondary complete-cohort report; never select budget, seed or stopping from test scores",
            "publication_policy": "Sensitivity study; preserve prior CUDA results and all RF results",
            "planned_extension": "2x2 environment x budget design: repeat both arms on RTX 4070 CUDA after user restores connection",
            "environment_caveat": "CPU/CUDA contrasts include OS, architecture and numerical libraries; not an isolated hardware causal effect",
            "interpretation": "Extra budget with unchanged scheduler; not a learning-rate or architecture search"}


def prepare_lane(bundle, root):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if root.exists():
        if (root / "manifest.json").read_bytes() != (bundle / "manifest.json").read_bytes():
            raise ValueError(f"incompatible existing workspace: {root}")
        verify_manifest(manifest, root)
        return
    root.mkdir(parents=True)
    for entry in manifest["files"]:
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / entry["path"], target)
    shutil.copy2(bundle / "manifest.json", root / "manifest.json")
    verify_manifest(manifest, root)


def worker_command(root, arm, tags, rerun=False):
    command = [sys.executable, str(root / "scripts/run_time_campaign.py"),
               "--role", "gin", "--device", "cpu", "--tags", *tags,
               "--seeds", "42", "1", "2", "--execute"]
    for key, value in ARMS[arm].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if rerun:
        command.append("--rerun")
    return command


def run_lane(bundle, output, arm, lane, tags, rerun):
    root = output / arm / f"lane{lane}"
    prepare_lane(bundle, root)
    command = worker_command(root, arm, tags, rerun)
    env = {**os.environ, **THREAD_ENV}
    with (root / "worker.log").open("a") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    status = {"arm": arm, "lane": lane, "tags": tags, "command": command,
              "returncode": result.returncode, "log": str(root / "worker.log")}
    atomic_json(root / "lane_status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    bundle, output = args.bundle_root.resolve(), args.output.resolve()
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        parser.error("input bundle and study output must be separate directories")
    protocol = plan(bundle)
    # JSON-normalize tuples before comparing the persisted protocol on resume.
    protocol = json.loads(json.dumps(protocol))
    print(json.dumps(protocol, indent=2), flush=True)
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=True)
    with worker_lock(output, "budget"):
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("study protocol changed; use a new output directory")
        atomic_json(protocol_path, protocol)
        state = {"started_at": datetime.now(timezone.utc).isoformat(),
                 "environment": environment(), "state": "running", "lanes": []}
        atomic_json(output / "study_status.json", state)
        try:
            # Both arms receive identical concurrency and thread settings.
            for arm in ARMS:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(run_lane, bundle, output, arm, lane, tags, args.rerun)
                               for lane, tags in enumerate(LANES)]
                    for future in as_completed(futures):
                        status = future.result()
                        state["lanes"].append(status)
                        atomic_json(output / "study_status.json", state)
                        print(json.dumps(status), flush=True)
                if any(item["returncode"] for item in state["lanes"]):
                    raise RuntimeError(f"{arm} worker failed; inspect logs before resuming")
            state["state"] = "training_complete_pending_independent_audit"
        except BaseException as exc:
            state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / "study_status.json", state)


if __name__ == "__main__":
    main()
