"""Prepare the immutable 128-job CUDA budget study without training or SSH.

The input archive is pinned independently of the working checkout. Generated
launchers reference only the extracted frozen coordinator and trainer.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "a212056556c7f3095c53bdaa0ed816f7d163e6617fe8f09c6ae094de6fab9ca3"
MANIFEST_SHA = "ab9c71cb1ff48505f14e8759c37d5e00c0c138cc50cf0d6e6baf752f96d9d802"
BUNDLE_COMMIT = "85210630c475e8733089a476b8d5bf4852dbf773"
CORE_COMMIT = "90a470050367cebc4cd83dfe8d3f7fb37d53e6de"
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def extract_input(archive, target, expected_sha=ARCHIVE_SHA, manifest_sha=MANIFEST_SHA):
    """Verify the pinned archive, exact regular-file inventory and every byte."""
    if digest(archive) != expected_sha:
        raise ValueError("Frozen archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or str(path) != member.name or "\\" in member.name):
                raise ValueError(f"Unsafe archive member: {member.name}")
        manifest_member = stream.extractfile("manifest.json")
        content = manifest_member.read()
        if hashlib.sha256(content).hexdigest() != manifest_sha:
            raise ValueError("Frozen manifest checksum mismatch")
        manifest = json.loads(content)
        entries = {item["path"]: item for item in manifest["files"]}
        if len(entries) != len(manifest["files"]) or set(names) != {"manifest.json", *entries}:
            raise ValueError("Frozen archive inventory differs from manifest")
        target.mkdir(parents=True, exist_ok=False)
        for member in members:
            destination = target / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            if member.name != "manifest.json":
                item = entries[member.name]
                if destination.stat().st_size != item["size"] or digest(destination) != item["sha256"]:
                    raise ValueError(f"Frozen member checksum mismatch: {member.name}")
    return manifest


PREFLIGHT = '''"""Verify immutable inputs and the historical CUDA runtime; never train."""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
spec = json.loads((ROOT / "manifest.json").read_text())
source = ROOT / "input"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if digest(source / "manifest.json") != spec["manifest_sha256"]:
    raise SystemExit("Frozen manifest changed")
manifest = json.loads((source / "manifest.json").read_text())
for entry in manifest["files"]:
    path = source / entry["path"]
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != ROOT.parent):
        raise SystemExit("Symlink in frozen input: " + entry["path"])
    if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
        raise SystemExit("Frozen input changed: " + entry["path"])
command = [sys.executable, str(source / "scripts/run_time_budget_extension.py"),
           "--bundle-root", str(source), "--output", str(ROOT / "output"),
           "--device", "cuda", "--parallel-workers", "1", "--seeds", "42", "1", "2", "3", "4", "5", "6", "7"]
plan = json.loads(subprocess.check_output(command, text=True))
if plan["jobs"] != 128 or plan["source_commit"] != spec["bundle_commit"]:
    raise SystemExit("Unexpected frozen study plan")
if "--offline" in sys.argv:
    print(json.dumps({"state": "offline_verified", "jobs": 128, "plan": plan}, indent=2))
    raise SystemExit(0)
if platform.python_version() != "3.12.15":
    raise SystemExit("Expected historical Python 3.12.15; environment drift requires review")
packages = {}
for line in (source / "requirements.txt").read_text().splitlines():
    if "==" not in line or line.lstrip().startswith("#"):
        continue
    name, version = line.split("==", 1)
    actual = metadata.version(name.strip())
    expected = "2.14.0+cu126" if name.strip() == "torch" else version.strip()
    if actual != expected:
        raise SystemExit(f"Dependency drift: {name} {actual}, expected {expected}")
    packages[name] = actual
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES=0")
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.free,memory.total", "--format=csv"],
                     capture_output=True, text=True, check=True)
props = torch.cuda.get_device_properties(0)
if "4070" not in props.name or torch.version.cuda != "12.6":
    raise SystemExit("Expected RTX 4070 / CUDA 12.6 historical runtime")
# Frozen real GIN/PyG operations and worst-size batches; no artifacts saved.
import argparse
import gc
import importlib.util
import numpy as np
sys.path.insert(0, str(source / "scripts"))
from torch_geometric.loader import DataLoader
from gnn_graph_dataset import MoleculeGraphDataset
from time_labels import load_time_split, require_prediction_version
trainer_spec = importlib.util.spec_from_file_location("frozen_gin", source / "scripts/gnn_03_train_gin.py")
trainer = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(trainer)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
args = argparse.Namespace(model="gin", hidden=128, num_layers=4, dropout=0.2)
checks = []
for tag in ("a2a", "egfr_full", "abl1", "egfr", "herg", "hivpr", "mpro", "vegfr2"):
    cache = source / f"data/processed/{tag}_graphs.npz"
    with np.load(cache, allow_pickle=False) as data:
        split = load_time_split(source / "data/processed/splits", tag, data["smiles"], data["y"])
        graph_size = data["n_nodes"] + data["n_edges"]
    y = split["y"]
    mean, std = float(y[split["train_idx"]].mean()), float(y[split["train_idx"]].std())
    normalized = (y.copy() - mean) / std
    reconstructed = normalized[split["test_idx"]] * std + mean
    require_prediction_version({"label_id": split["label_id"], "test_idx": split["test_idx"],
                                "y_true": reconstructed}, split)
    for pool, batch_size in (("train", 128), ("valid", 512), ("test", 512)):
        indices = split[pool + "_idx"]
        largest = indices[np.argsort(graph_size[indices])[-batch_size:]]
        dataset = MoleculeGraphDataset(cache, indices=largest)
        dataset.y = normalized
        batch = next(iter(DataLoader(dataset, batch_size=batch_size))).to("cuda")
        torch.manual_seed(42)
        model = trainer.build_model(args).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        if pool == "train":
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y)
            if not torch.isfinite(loss):
                raise SystemExit(f"Nonfinite CUDA smoke loss: {tag}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise SystemExit(f"Nonfinite CUDA smoke gradient: {tag}")
            optimizer.step()
            del optimizer, loss
        else:
            model.eval()
            with torch.no_grad():
                prediction = model(batch)
        if not torch.isfinite(prediction).all():
            raise SystemExit(f"Nonfinite CUDA smoke prediction: {tag}/{pool}")
        torch.cuda.synchronize()
        checks.append({"tag": tag, "pool": pool, "graphs": len(largest),
                       "nodes": batch.num_nodes, "edges": batch.num_edges,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
        print(json.dumps(checks[-1]), flush=True)
        del model, prediction, batch, dataset
        gc.collect()
        torch.cuda.empty_cache()
report = {"state": "cuda_preflight_passed", "time": datetime.now(timezone.utc).isoformat(),
          "python": platform.python_version(), "platform": platform.platform(),
          "packages": packages, "cuda": torch.version.cuda, "gpu": props.name,
          "total_memory": props.total_memory, "nvidia_smi": smi.stdout, "plan": plan, "checks": checks,
          "max_peak_reserved_mib": max(c["peak_reserved_mib"] for c in checks),
          "environment": {k: os.environ.get(k) for k in plan["thread_environment"]}}
logs = ROOT / "logs"
logs.mkdir(exist_ok=True)
(logs / ("preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")).write_text(json.dumps(report, indent=2) + "\\n")
print(json.dumps(report, indent=2))
'''

RUN = '''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${QSAR_PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QSAR_TORCH_THREADS=4 QSAR_TORCH_INTEROP_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=4 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
"$PY" "$HERE/preflight.py"
exec "$PY" "$HERE/input/scripts/run_time_budget_extension.py" \\
 --bundle-root "$HERE/input" --output "$HERE/output" --device cuda --parallel-workers 1 \\
 --seeds 42 1 2 3 4 5 6 7 --execute
'''

PACKAGE = '''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${QSAR_PYTHON:-python3}"
STAMP="$(date -u +%Y%m%dT%H%M%S)-$$"
"$PY" "$HERE/collect_cuda_study.py" --study-root "$HERE/output" --output "$HERE/returns/$STAMP/study"
"$PY" "$HERE/package_time_budget_study.py" --study-root "$HERE/returns/$STAMP/study" --output "$HERE/returns/$STAMP/study.tar.gz"
'''


def prepare(output, archive=None):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Preparation requires a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".study-prepare-", dir=output.parent))
    try:
        frozen = extract_input(archive or ROOT / "results/time_budget_sensitivity/extension_inputs.tar.gz",
                               temporary / "input")
        if frozen["source"]["git_head"] != BUNDLE_COMMIT or frozen["source"]["dirty"]:
            raise ValueError("Unexpected frozen source identity")
        replication = json.loads((ROOT / "results/time_budget_sensitivity/replication_extension.json").read_text())
        for name, checksum in replication["training_core_sha256"].items():
            if digest(temporary / "input" / name) != checksum:
                raise ValueError("Training core no longer matches accepted CPU study")
        manifest = {"schema": 1, "bundle_commit": BUNDLE_COMMIT, "training_core_commit": CORE_COMMIT,
                    "archive_sha256": ARCHIVE_SHA, "manifest_sha256": MANIFEST_SHA,
                    "jobs": 128, "seeds": SEEDS, "device": "cuda", "parallel_workers": 1,
                    "training_core_sha256": replication["training_core_sha256"],
                    "resume": "No --rerun. Frozen worker skips successful jobs only after signature and canonical/snapshot hash verification; failed/incomplete jobs restart from epoch 1.",
                    "interruption": "Frozen worker predates transaction recovery. Never auto-delete stale locks or backups; inspect owner and artifacts before recovery.",
                    "acceptance": "Training complete is not independently accepted; return archive for CPU checkpoint replay and paired environment report."}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name, content in (("preflight.py", PREFLIGHT), ("run.sh", RUN), ("package_return.sh", PACKAGE)):
            (temporary / name).write_text(content)
        for name in ("collect_cuda_study.py", "package_time_budget_study.py"):
            shutil.copy2(ROOT / "scripts" / name, temporary / name)
        check = subprocess.run([sys.executable, str(temporary / "preflight.py"), "--offline"],
                               capture_output=True, text=True, check=True)
        (temporary / "offline_verification.json").write_text(check.stdout)
        temporary.rename(output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
