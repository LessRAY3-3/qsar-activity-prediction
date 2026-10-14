"""Plan or execute additional GIN seeds through the unchanged campaign worker.

Run from a clean extracted GIN bundle containing this wrapper. Dry-run is the
default; all training, validation, snapshots and resume decisions remain in
run_time_campaign.py. No seed constants or trainer defaults are modified.
"""
import argparse
import json
from pathlib import Path

import run_time_campaign as worker
from time_campaign_bundle import verify_manifest

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = (3, 4, 5, 6, 7)
WRAPPER_PATH = "scripts/run_time_campaign_seed_extension.py"


def positive_integer(raw):
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(role="gin")
    parser.add_argument("--tags", nargs="+", choices=worker.TAGS, default=list(worker.TAGS))
    parser.add_argument("--seeds", nargs="+", type=positive_integer, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=positive_integer, default=100)
    parser.add_argument("--patience", type=positive_integer, default=20)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    if not 0 <= args.min_epochs <= args.epochs:
        parser.error("require 0 <= --min-epochs <= --epochs")
    return args


def verify_wrapper_bundle(root):
    manifest = verify_manifest(json.loads((root / "manifest.json").read_text()), root)
    if manifest["role"] != "gin" or manifest["source"]["dirty"]:
        raise ValueError("seed extension requires a clean-source GIN bundle")
    expected = manifest["source"]["sha256"].get(WRAPPER_PATH)
    if expected != worker.digest(Path(__file__)):
        raise ValueError("running seed wrapper is missing from or differs from the frozen manifest")
    return manifest


def main(argv=None):
    args = parse_args(argv)
    root = ROOT.resolve()
    if args.campaign_dir is None:
        args.campaign_dir = root / "results/time_campaign"
    verify_wrapper_bundle(root)
    jobs = worker.build_jobs(args, root)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", "jobs": [
        {key: value for key, value in job.items() if key != "split"} for job in jobs]}, indent=2))
    if not args.execute:
        return
    with worker.worker_lock(root, "gin"):
        for job in jobs:
            worker.run_job(job, args, root)


if __name__ == "__main__":
    main()
