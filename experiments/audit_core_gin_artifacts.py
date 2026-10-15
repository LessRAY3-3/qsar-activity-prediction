"""Replay 72 published core GIN checkpoints and bind today's evidence hashes.

Current source compatibility / model-to-prediction consistency is distinct from
historical training identity. The audit never claims that today's source hashes
were recorded at the time of an old training run. No fitting or artifact rewrite.
"""
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from artifact_guards import atomic_json, check_partition, check_prediction, training_scale  # noqa: E402
from time_labels import load_time_split, require_prediction_version  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments"))
from rf_core_confirmation import TAGS, sha  # noqa: E402


def check_checkpoint(saved, tag, kind, seed, mean, std, label_id=None):
    args = saved["args"]
    for key, expected in (("tag", tag), ("split", kind), ("seed", seed), ("model", "gin")):
        if args.get(key) != expected:
            raise ValueError(f"checkpoint {key} identity mismatch")
    if not np.allclose([saved["y_mean"], saved["y_std"]], [mean, std], rtol=0, atol=1e-6):
        raise ValueError("checkpoint train-only normalization differs")
    if label_id is not None and saved.get("label_id") != label_id:
        raise ValueError("checkpoint time label identity mismatch")
    if not saved["state_dict"] or any(not torch.isfinite(v).all() for v in saved["state_dict"].values()):
        raise ValueError("checkpoint weights invalid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be new")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    module = importlib.import_module("gnn_03_train_gin")
    hashes = {}
    for rel in ("experiments/audit_core_gin_artifacts.py", "experiments/rf_core_confirmation.py",
                "scripts/gnn_03_train_gin.py", "scripts/gnn_graph_dataset.py",
                "scripts/qsar_common.py", "scripts/time_labels.py", "scripts/artifact_guards.py"):
        hashes[rel] = sha(ROOT / rel)

    def pin(rel):
        hashes[rel] = sha(ROOT / rel)
        return ROOT / rel

    rows = []
    for tag in TAGS:
        graph = pin(f"data/processed/{tag}_graphs.npz")
        with np.load(graph) as z:
            original_y, smiles = z["y"], z["smiles"]
        for kind in ("random", "scaffold", "time"):
            path = pin(f"data/processed/splits/{tag}_{kind}.npz")
            with np.load(path) as z:
                split = dict(z)
            if kind == "time":
                split = load_time_split(path.parent, tag, smiles, original_y)
                y = split["y"]
            else:
                y = original_y
                check_partition(len(y), **{k: split[k] for k in ("train_idx", "valid_idx", "test_idx")})
                if not np.array_equal(smiles, split["smiles"]):
                    raise ValueError("graph cache/split order differs")
            mean, std = training_scale(y, split["train_idx"])
            dataset = module.MoleculeGraphDataset(graph, indices=split["test_idx"])
            dataset.y = (y.copy()-mean)/std
            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
            for seed, suffix in ((42, ""), (1, "_seed1"), (2, "_seed2")):
                model_path = pin(f"results/gnn_models/{tag}_gin_{kind}{suffix}.pt")
                pred_path = pin(f"results/gnn_preds_{tag}_{kind}{suffix}.npz")
                metrics_path = pin(f"results/gnn_metrics_{tag}{suffix}.json")
                stored_metrics = json.loads(metrics_path.read_text())[kind]
                saved = torch.load(model_path, map_location="cpu", weights_only=False)
                check_checkpoint(saved, tag, kind, seed, mean, std,
                                 str(split["label_id"]) if kind == "time" else None)
                for key, value in stored_metrics["params"].items():
                    if key in saved["args"] and saved["args"][key] != value:
                        raise ValueError(f"metric/checkpoint parameter mismatch: {key}")
                model = module.build_model(argparse.Namespace(**saved["args"]))
                model.load_state_dict(saved["state_dict"], strict=True)
                pred_raw, truth_raw = module.predict(model, loader, "cpu")
                pred, truth = pred_raw*std+mean, truth_raw*std+mean
                with np.load(pred_path) as z:
                    stored = dict(z)
                check_prediction(stored, split, y)
                if kind == "time":
                    require_prediction_version(stored, split)
                if not np.allclose(pred, stored["y_pred"], rtol=1e-5, atol=1e-4):
                    raise ValueError(f"checkpoint replay differs: {tag}/{kind}/{seed}")
                if not np.allclose(truth, stored["y_true"], rtol=0, atol=1e-5):
                    raise ValueError("checkpoint label replay differs")
                for measured in (module.metrics(truth, pred), module.metrics(stored["y_true"], stored["y_pred"])):
                    if any(not np.isclose(measured[k], stored_metrics[k], rtol=1e-5, atol=1e-5)
                           for k in ("r2", "rmse", "mae")):
                        raise ValueError("checkpoint metric replay differs")
                rows.append({"tag": tag, "split": kind, "seed": seed, "passed": True,
                             "model": str(model_path.relative_to(ROOT)),
                             "model_sha256": hashes[str(model_path.relative_to(ROOT))],
                             "max_prediction_abs_error": float(np.max(np.abs(pred-stored["y_pred"]))),
                             "checkpoint_has_runtime": "runtime" in saved,
                             "historical_training_identity_recovered_by_this_audit": False})
                print(f"PASS {tag}/{kind}/{seed} error={rows[-1]['max_prediction_abs_error']:.3g}", flush=True)
    if any(sha(ROOT / rel) != digest for rel, digest in hashes.items()):
        raise ValueError("input changed during audit")
    atomic_json(args.report, {"schema": "qsar-core-gin-current-artifact-audit-v1", "status": "passed",
                             "jobs": rows, "jobs_checked": len(rows), "training_performed": False,
                             "source_and_artifact_sha256": hashes,
                             "prediction_tolerance": {"atol": 1e-4, "rtol": 1e-5},
                             "metric_tolerance": {"atol": 1e-5, "rtol": 1e-5},
                             "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                             "pyg": torch_geometric.__version__, "numpy": np.__version__,
                                             "platform": platform.platform(), "device": "cpu", "threads": 2},
                             "interpretation": "Today's pinned inputs/source replay published checkpoints. This does not reconstruct historical training environments, trajectories, or source revisions."})


if __name__ == "__main__":
    main()
