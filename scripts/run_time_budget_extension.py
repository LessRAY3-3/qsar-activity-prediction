"""Run extra seeds for the budget study without changing its training code.

Default: seeds 3/4/5/6/7, 80 jobs, CPU with four concurrent lanes. To run a
complete eight-seed CUDA cohort later, pass --device cuda --parallel-workers 1
--seeds 42 1 2 3 4 5 6 7. Requires a new clean GIN bundle containing both new
entry points. Dry-run is the default; --execute explicitly starts training.
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
import uuid

from run_time_campaign import atomic_json, digest, environment, worker_lock
from run_time_campaign_seed_extension import DEFAULT_SEEDS, WRAPPER_PATH, positive_integer
from time_campaign_bundle import verify_manifest

ARMS = {"control": {"epochs": 100, "min_epochs": 0, "patience": 20},
        "extended": {"epochs": 200, "min_epochs": 60, "patience": 40}}
LANES = (("a2a", "egfr"), ("egfr_full", "hivpr"),
         ("herg", "mpro"), ("abl1", "vegfr2"))
THREAD_ENV = {"QSAR_TORCH_THREADS": "4", "QSAR_TORCH_INTEROP_THREADS": "1",
              "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
              "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "4",
              "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}
COORDINATOR_PATH = "scripts/run_time_budget_extension.py"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=positive_integer, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--parallel-workers", type=positive_integer, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must be distinct")
    if args.parallel_workers > len(LANES):
        parser.error("--parallel-workers cannot exceed the four isolated lanes")
    bundle, output = args.bundle_root.resolve(), args.output.resolve()
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        parser.error("input bundle and extension output must be separate directories")
    args.bundle_root, args.output = bundle, output
    return args


def plan(bundle, seeds=DEFAULT_SEEDS, device="cpu", parallel_workers=4):
    manifest = verify_manifest(json.loads((bundle / "manifest.json").read_text()), bundle)
    if manifest["role"] != "gin" or manifest["source"]["dirty"]:
        raise ValueError("extension requires a clean-source GIN bundle")
    hashes = manifest["source"]["sha256"]
    for name, running in ((WRAPPER_PATH, Path(__file__).with_name(Path(WRAPPER_PATH).name)),
                          (COORDINATOR_PATH, Path(__file__))):
        if hashes.get(name) != digest(running):
            raise ValueError(f"running extension entry point differs from frozen manifest: {name}")
    return {"schema": 1, "study_kind": "time-budget-seed-extension",
            "manifest_sha256": digest(bundle / "manifest.json"),
            "source_commit": manifest["source"]["git_head"],
            "coordinator_sha256": hashes[COORDINATOR_PATH], "wrapper_sha256": hashes[WRAPPER_PATH],
            "arms": ARMS, "lanes": LANES, "seeds": list(seeds), "jobs": 16 * len(seeds),
            "device": device, "parallel_workers": parallel_workers, "thread_environment": THREAD_ENV,
            "selection": "Lowest validation RMSE; unchanged scheduler and 1e-5 improvement tolerance",
            "primary_comparison": "Paired same-host best-checkpoint train MSE and validation RMSE",
            "common_flag": "best_epoch < 30 and best_checkpoint_train_mse > 0.5 in both arms",
            "test_policy": "Complete-cohort secondary report; never select budget or seed from test scores",
            "publication_policy": "Preserve the original three-seed study and all main CUDA/RF results",
            "environment_caveat": "CPU/CUDA contrasts also include OS, architecture and numerical libraries"}


def prepare_lane(bundle, root):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if root.exists():
        if (root / "manifest.json").read_bytes() != (bundle / "manifest.json").read_bytes():
            raise ValueError(f"incompatible existing workspace: {root}")
        verify_manifest(manifest, root)
        return
    # Publish a complete lane at once, so interrupted preparation is resumable.
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name("." + root.name + ".prepare-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for entry in manifest["files"]:
            target = temporary / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle / entry["path"], target)
        shutil.copy2(bundle / "manifest.json", temporary / "manifest.json")
        verify_manifest(manifest, temporary)
        temporary.rename(root)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def worker_command(root, arm, tags, seeds, device, rerun=False):
    command = [sys.executable, str(root / WRAPPER_PATH), "--device", device,
               "--tags", *tags, "--seeds", *map(str, seeds), "--execute"]
    for key, value in ARMS[arm].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if rerun:
        command.append("--rerun")
    return command


def run_lane(bundle, output, arm, lane, tags, seeds, device, rerun):
    root = output / arm / f"lane{lane}"
    prepare_lane(bundle, root)
    command = worker_command(root, arm, tags, seeds, device, rerun)
    env = {**os.environ, **THREAD_ENV}
    with (root / "worker.log").open("a") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    status = {"arm": arm, "lane": lane, "tags": tags, "seeds": list(seeds), "device": device,
              "command": command, "returncode": result.returncode, "log": str(root / "worker.log")}
    atomic_json(root / "lane_status.json", status)
    return status


def main(argv=None):
    args = parse_args(argv)
    bundle, output = args.bundle_root, args.output
    protocol = json.loads(json.dumps(plan(bundle, args.seeds, args.device, args.parallel_workers)))
    print(json.dumps(protocol, indent=2), flush=True)
    if not args.execute:
        return
    if output.exists() and any(output.iterdir()) and not (output / "protocol.json").is_file():
        raise ValueError("refusing a nonempty output directory without an extension protocol")
    output.mkdir(parents=True, exist_ok=True)
    with worker_lock(output, "budget"):
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("extension protocol changed; use a new output directory")
        atomic_json(protocol_path, protocol)
        state = {"started_at": datetime.now(timezone.utc).isoformat(),
                 "environment": environment(), "state": "running", "lanes": []}
        atomic_json(output / "study_status.json", state)
        try:
            for arm in ARMS:
                with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
                    futures = [executor.submit(run_lane, bundle, output, arm, lane, tags,
                                               args.seeds, args.device, args.rerun)
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
