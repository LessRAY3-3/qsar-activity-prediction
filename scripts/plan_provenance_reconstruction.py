"""Prepare independent reruns for missing historical logs; never start training.

The plan preserves unknown historical provenance. It snapshots the *current*
recipe, not a claim that the historical executable/environment was recovered.
Preparation copies sources and inputs to a fresh per-job workspace so even
trainers with fixed BASE-relative output paths cannot overwrite old results.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ("hidden", "num_layers", "dropout", "lr", "batch_size", "epochs", "patience")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(root):
    jobs = []
    evidence = {}
    for seed in (42, 1, 2):
        suffix = "" if seed == 42 else f"_seed{seed}"
        metric_path = Path("results") / f"gnn_metrics_egfr{suffix}.json"
        metrics = json.loads((root / metric_path).read_text())
        evidence[str(metric_path)] = sha256(root / metric_path)
        for split in ("random", "scaffold"):
            params = metrics[split]["params"]
            argv = ["scripts/gnn_03_train_gin.py", "--tag", "egfr", "--model", "gin",
                    "--split", split, "--seed", str(seed), "--device", "cuda"]
            if suffix:
                argv += ["--suffix", suffix]
            for key in PARAMETERS:
                argv += ["--" + key.replace("_", "-"), str(params[key])]
            jobs.append({"id": f"egfr_gin_{split}_seed{seed}", "argv": argv,
                         "tags": ["egfr"], "split": split, "seed": seed,
                         "parameter_evidence": str(metric_path) + ":" + split + ".params",
                         "historical_device": None, "historical_log": None,
                         "historical_r2": metrics[split]["r2"]})
        for model, stem in (("mt", "mt_metrics"), ("pooled", "pooled_st_metrics")):
            metric_path = Path("results/multitask") / f"{stem}{suffix}.json"
            metrics = json.loads((root / metric_path).read_text())
            evidence[str(metric_path)] = sha256(root / metric_path)
            for split in ("random", "scaffold"):
                params = metrics[split]["_run"]["params"]
                argv = ["scripts/gnn_07_multitask.py", "--model", model, "--split", split,
                        "--seed", str(seed), "--tasks", str(params["tasks"]),
                        "--outdir", "results/multitask", "--figdir", "figures/multitask"]
                for key in PARAMETERS:
                    argv += ["--" + key.replace("_", "-"), str(params[key])]
                jobs.append({"id": f"{model}_{split}_seed{seed}", "argv": argv,
                             "tags": metrics[split]["_run"]["tags"], "split": split,
                             "seed": seed, "historical_device": None, "historical_log": None,
                             "parameter_evidence": str(metric_path) + ":" + split + "._run.params",
                             "historical_r2": {tag: metrics[split][tag]["r2"]
                                               for tag in metrics[split]["_run"]["tags"]}})
    for job in jobs:
        job["command"] = ["python", "-u", *job["argv"]]
        job["working_directory"] = ".scratch/provenance_reconstruction/" + job["id"]
    inputs = {}
    for job in jobs:
        names = []
        for tag in job["tags"]:
            names += [f"data/processed/{tag}_graphs.npz",
                      f"data/processed/splits/{tag}_{job['split']}.npz"]
        job["inputs"] = names
        for name in names:
            inputs[name] = sha256(root / name)
    return {
        "schema_version": 1,
        "status": "prepared_only_waiting_for_user_cuda_connection",
        "interpretation": "Independent reconstruction from saved parameters and current code/data; not recovered historical logs.",
        "historical_unknowns": ["original source revision", "device", "package environment",
                                "epoch trajectory", "stdout/stderr", "original RNG state beyond stored seed"],
        "scope": {"egfr_missing_log_fits": 6, "multitask_and_pooled_fits": 12,
                  "excluded_egfr_time_fits": 3,
                  "exclusion_reason": "Current time cells already have CUDA runtime and campaign provenance.",
                  "rf": "Use completed independent ten-fit RF supplement; no rerun needed."},
        "new_backend": "cuda",
        "cohort_boundary": "These 18 supplemental fits are separate from the frozen 128-job CUDA/CPU architecture comparison. Do not alter that experiment bundle or mix its results with these fits.",
        "execution_requirements": [
            "Wait for user to reconnect win4070; this planner never connects or trains.",
            "Run from each freshly prepared workspace, never the primary repository.",
            "Before training, require torch.cuda.is_available() and record selected device/name, torch/CUDA/PyG/numpy versions, Python/platform, thread counts and start/end UTC.",
            "multitask uses select_device(): CUDA is selected first only when available; do not permit silent CPU/MPS fallback.",
            "Capture complete stdout/stderr and exit code per job; hash resulting predictions/checkpoints/metrics.",
            "Compare new predictions/R2 to recorded baselines; report differences without replacing historical results.",
            "A matching score does not establish identity with the historical run."],
        "source_sha256": {str(p.relative_to(root)): sha256(p)
                          for p in sorted((root / "scripts").glob("*.py"))},
        "input_sha256": inputs, "historical_metrics_sha256": evidence, "jobs": jobs,
    }


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {name}")
    return path


def prepare_job(root, plan, job_id):
    """Copy to an exclusive workspace; any existing destination is an error."""
    if not re.fullmatch(r"[a-z0-9_]+", job_id):
        raise ValueError("Invalid job id")
    job = next(job for job in plan["jobs"] if job["id"] == job_id)
    files = dict(plan["source_sha256"])
    files.update({name: plan["input_sha256"][name] for name in job["inputs"]})
    for name, expected in files.items():
        path = root / safe_relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe input path: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Source/input changed since plan: {name}")
    parent = root / ".scratch" / "provenance_reconstruction"
    if parent.resolve() != root.resolve() / ".scratch" / "provenance_reconstruction":
        raise ValueError("Workspace parent must not traverse symlinks")
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / job_id
    destination.mkdir()  # Deliberately no exist_ok: never overwrite or resume silently.
    for name in files:
        path = safe_relative(name)
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / path, target)
    (destination / "reconstruction_job.json").write_text(json.dumps(job, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepare-job", help="Prepare this plan job; never train")
    args = parser.parse_args()
    if args.prepare_job:
        destination = prepare_job(ROOT, json.loads(args.plan.read_text()), args.prepare_job)
        print(destination)
    else:
        plan = build_plan(ROOT)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as handle:
            json.dump(plan, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {len(plan['jobs'])} planned jobs; no training started.")


if __name__ == "__main__":
    main()
