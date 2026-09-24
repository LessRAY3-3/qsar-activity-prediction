"""GNN step 6 (bonus): prepare the MoleculeNet ESOL benchmark.

ESOL (Delaney 2004, 1128 molecules, aqueous solubility in log mol/L) is a
standard public regression benchmark for molecular GNNs. Running our GIN
on it shows the pipeline is not specific to the in-house EGFR dataset.

This script downloads the original delaney.csv (MoleculeNet mirror),
featurizes it with the SAME smiles_to_graph as the EGFR data, and writes
the same artifacts the training script expects:
  data/raw/esol_delaney.csv
  data/processed/esol_graphs.npz
  data/processed/splits/esol_{random,scaffold}.npz

Split protocol mirrors EGFR: random (seed 42, 80/10/10) and scaffold
(whole Bemis-Murcko scaffolds into test until ~20%, then 10% of train as
validation). No RF baseline exists for ESOL; the numbers are compared to
published ESOL results in the README instead.
MoleculeNet's official S3 mirror (deepchemdata) now returns 403, so we
use the identical delaney-processed.csv vendored in the DeepChem GitHub
repo (same MoleculeNet file, widely mirrored).
"""
import os
import sys
import time
import urllib.request

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_01_make_splits import RANDOM_STATE, scaffold_split  # noqa: E402
from gnn_02_build_graphs import smiles_to_graph  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
RAW_CSV = os.path.join(BASE, "data", "raw", "esol_delaney.csv")
OUT_NPZ = os.path.join(BASE, "data", "processed", "esol_graphs.npz")
SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")


def download():
    if os.path.exists(RAW_CSV):
        print(f"cached: {os.path.relpath(RAW_CSV, BASE)}")
        return
    for attempt in (1, 2, 3):
        try:
            print(f"downloading {URL} (attempt {attempt})")
            with urllib.request.urlopen(URL, timeout=60) as r, open(RAW_CSV, "wb") as f:
                f.write(r.read())
            break
        except Exception as e:
            print(f"  failed: {e}")
            time.sleep(3 * attempt)
    else:
        raise SystemExit("could not download ESOL; check network and re-run")
    print(f"saved -> {os.path.relpath(RAW_CSV, BASE)}")


def main():
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    download()
    df = pd.read_csv(RAW_CSV)
    smiles_col = [c for c in df.columns if c.lower() == "smiles"]
    y_candidates = [c for c in df.columns if "solubility" in c.lower()]
    y_col = next((c for c in y_candidates if "measured" in c.lower()), None)
    if y_col is None:  # fall back to any non-predicted solubility column
        y_col = next((c for c in y_candidates if "predicted" not in c.lower()), None)
    assert len(smiles_col) == 1 and y_col, f"unexpected columns: {df.columns.tolist()}"
    smiles_col = smiles_col[0]
    df = df[[smiles_col, y_col]].dropna().reset_index(drop=True)
    smiles = df[smiles_col].astype(str).to_numpy()
    y = df[y_col].astype(float).to_numpy(dtype=np.float32)
    print(f"ESOL: {len(smiles)} molecules | y range [{y.min():.2f}, {y.max():.2f}]")

    # drop molecules RDKit cannot parse (keeps indices aligned)
    keep = np.array([Chem.MolFromSmiles(s) is not None for s in smiles])
    smiles, y = smiles[keep], y[keep]
    print(f"after parse filter: {len(smiles)}")

    print("featurizing ...")
    node_feats, edge_indices, edge_feats, n_nodes, n_edges = [], [], [], [], []
    scafs = []
    for smi in smiles:
        g = smiles_to_graph(smi)
        mol = Chem.MolFromSmiles(smi)
        scafs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        node_feats.append(g["node_feat"])
        edge_indices.append(g["edge_index"])
        edge_feats.append(g["edge_feat"])
        n_nodes.append(g["n_nodes"])
        n_edges.append(g["n_edges"])
    node_feat = np.concatenate(node_feats)
    edge_index = np.concatenate(edge_indices, axis=1)
    edge_feat = np.concatenate(edge_feats)
    n_nodes, n_edges = np.array(n_nodes, np.int32), np.array(n_edges, np.int32)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, node_feat=node_feat, edge_index=edge_index, edge_feat=edge_feat,
             n_nodes=n_nodes, n_edges=n_edges, y=y,
             smiles=smiles, scaffold_smiles=np.array(scafs, dtype=str))
    print(f"saved -> {os.path.relpath(OUT_NPZ, BASE)}")

    os.makedirs(SPLIT_DIR, exist_ok=True)
    n = len(smiles)
    splits = {
        "random": train_test_split(np.arange(n), test_size=0.2, random_state=RANDOM_STATE),
        "scaffold": scaffold_split(smiles, test_size=0.2),
    }
    for name, (tr, te) in splits.items():
        tr_new, va = train_test_split(tr, test_size=0.1, random_state=RANDOM_STATE)
        assert len(set(tr_new) & set(va)) == 0 and len(set(tr_new) | set(va) | set(te)) == n
        out = os.path.join(SPLIT_DIR, f"esol_{name}.npz")
        np.savez(out, train_idx=tr_new, valid_idx=va, test_idx=te,
                 scaffold_smiles=np.array(scafs, dtype=str), smiles=smiles)
        print(f"saved -> {os.path.relpath(out, BASE)}  train={len(tr_new)} valid={len(va)} test={len(te)}")


if __name__ == "__main__":
    main()
