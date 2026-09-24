"""GNN step 2: build the molecular graph cache ({TAG}_graphs.npz).

Reads SMILES in the row order of data/processed/{TAG}_fingerprints.npz
(the same order the split indices from gnn_01_make_splits.py refer to),
featurizes every molecule with gnn_graph_dataset.smiles_to_graph, and
writes one CSR-style npz (see that module for the format).

The scaffold labels are copied from the persisted split file so the cache
is self-contained for the later error analysis (colouring by chemotype).

Why a cache instead of featurizing on the fly: RDKit featurization costs
~1 ms/molecule; paying it once and loading arrays afterwards keeps every
training run fast and reproducible.
"""
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_graph_dataset import smiles_to_graph  # noqa: E402

TAG = os.environ.get("QSAR_TAG", "egfr")
IN_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_fingerprints.npz")
SPLIT_NPZ = os.path.join(BASE, "data", "processed", "splits", f"{TAG}_random.npz")
OUT_NPZ = os.path.join(BASE, "data", "processed", f"{TAG}_graphs.npz")


def main():
    d = np.load(IN_NPZ)
    smiles, y = d["smiles"], d["y"].astype(np.float32)
    split = np.load(SPLIT_NPZ)
    assert np.array_equal(split["smiles"], smiles), (
        "split file SMILES order differs from fingerprint file - indices would misalign"
    )
    scaffold_smiles = split["scaffold_smiles"]
    print(f"[{TAG}] featurizing {len(smiles)} molecules ...")

    t0 = time.time()
    node_feats, edge_indices, edge_feats = [], [], []
    n_nodes = np.zeros(len(smiles), dtype=np.int32)
    n_edges = np.zeros(len(smiles), dtype=np.int32)
    failed = []
    for i, smi in enumerate(smiles):
        g = smiles_to_graph(str(smi))
        if g is None:
            failed.append(i)
            g = {"node_feat": np.zeros((1, 9), dtype=np.int16),
                 "edge_index": np.zeros((2, 0), dtype=np.int32),
                 "edge_feat": np.zeros((0, 3), dtype=np.int16),
                 "n_nodes": 1, "n_edges": 0}
        node_feats.append(g["node_feat"])
        edge_indices.append(g["edge_index"])
        edge_feats.append(g["edge_feat"])
        n_nodes[i] = g["n_nodes"]
        n_edges[i] = g["n_edges"]

    node_feat = np.concatenate(node_feats)
    edge_index = np.concatenate(
        [e if e.shape[1] else np.zeros((2, 0), dtype=np.int32) for e in edge_indices], axis=1
    )
    edge_feat = np.concatenate(edge_feats)

    np.savez(OUT_NPZ,
             node_feat=node_feat, edge_index=edge_index, edge_feat=edge_feat,
             n_nodes=n_nodes, n_edges=n_edges, y=y,
             smiles=smiles, scaffold_smiles=scaffold_smiles)

    nn, ne = n_nodes.astype(float), n_edges.astype(float)
    print(f"done in {time.time() - t0:.1f}s -> {os.path.relpath(OUT_NPZ, BASE)}")
    print(f"  molecules={len(smiles)}  unparseable={len(failed)} {failed[:5]}")
    print(f"  nodes/mol: min={nn.min():.0f} mean={nn.mean():.1f} max={nn.max():.0f} | total={len(node_feat)}")
    print(f"  edges/mol: min={ne.min():.0f} mean={ne.mean():.1f} max={ne.max():.0f} | total={edge_index.shape[1]}")
    if failed:
        print("  WARNING: some molecules failed to parse; they got a single isolated dummy node.")


if __name__ == "__main__":
    main()
