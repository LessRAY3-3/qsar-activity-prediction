"""Molecular graph featurization (RDKit) + PyG Dataset backed by a cached npz.

Used by gnn_02_build_graphs.py (writes the cache) and gnn_03_train_gin.py
(reads it). The featurizer follows the MoleculeNet/OGB convention of
categorical features that get embedded (AtomEncoder/BondEncoder style):

  atom (node), 9 features:
      atomic number | chirality tag | degree | formal charge
      | total num H | radical electrons | hybridization | aromatic | in ring
  bond (edge), 3 features:
      bond type | stereo | conjugated

All features are small non-negative integer indices; out-of-range values
are clamped into the last "misc" bucket, so ATOM_FEATURE_DIMS /
BOND_FEATURE_DIMS are exact embedding table sizes.

Cache format (data/processed/{TAG}_graphs.npz), CSR-style so thousands of
graphs live in one file:
      node_feat  (total_nodes, 9) int16
      edge_index (2, total_edges) int32   (node ids local to each molecule)
      edge_feat  (total_edges, 3) int16
      n_nodes    (n_graphs,) int32
      n_edges    (n_graphs,) int32
      y          (n_graphs,) float32
      smiles, scaffold_smiles (n_graphs,) str
"""
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

ATOM_FEATURE_DIMS = [119, 9, 12, 11, 9, 5, 10, 2, 2]  # 9 atom features
BOND_FEATURE_DIMS = [13, 7, 2]                        # 3 bond features


def _clamp_idx(value, dim):
    return min(max(int(value), 0), dim - 1)


def atom_features(atom):
    return [
        _clamp_idx(atom.GetAtomicNum(), 119),
        _clamp_idx(atom.GetChiralTag(), 9),
        _clamp_idx(atom.GetDegree(), 12),
        _clamp_idx(atom.GetFormalCharge() + 5, 11),
        _clamp_idx(atom.GetTotalNumHs(), 9),
        _clamp_idx(atom.GetNumRadicalElectrons(), 5),
        _clamp_idx(atom.GetHybridization(), 10),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


def bond_features(bond):
    return [
        _clamp_idx(bond.GetBondType(), 13),
        _clamp_idx(bond.GetStereo(), 7),
        int(bond.GetIsConjugated()),
    ]


def smiles_to_graph(smiles):
    """Convert one SMILES to graph arrays. Returns None if RDKit fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    node_feat = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.int16)
    edges, efeat = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append((i, j))
        efeat.append(bond_features(b))
        edges.append((j, i))  # undirected graph
        efeat.append(bond_features(b))
    if edges:
        edge_index = np.array(edges, dtype=np.int32).T
        edge_feat = np.array(efeat, dtype=np.int16)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_feat = np.zeros((0, len(BOND_FEATURE_DIMS)), dtype=np.int16)
    return {"node_feat": node_feat, "edge_index": edge_index, "edge_feat": edge_feat,
            "n_nodes": node_feat.shape[0], "n_edges": edge_index.shape[1]}


class MoleculeGraphDataset(Dataset):
    """PyG Dataset over the CSR graph cache.

    `indices` selects a subset (e.g. one split's train_idx) while node/edge
    ids in the returned Data objects stay local to each molecule, so any
    subset is safe to batch with the PyG DataLoader.
    """

    def __init__(self, npz_path, indices=None):
        d = np.load(npz_path)
        self.node_feat = d["node_feat"]
        self.edge_index = d["edge_index"]
        self.edge_feat = d["edge_feat"]
        self.n_nodes = d["n_nodes"]
        self.n_edges = d["n_edges"]
        self.y = d["y"].astype(np.float32)
        self.indices = np.arange(len(self.y)) if indices is None else np.asarray(indices)
        self.node_ptr = np.concatenate([[0], np.cumsum(self.n_nodes)])
        self.edge_ptr = np.concatenate([[0], np.cumsum(self.n_edges)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        g = int(self.indices[i])
        ns, ne = int(self.node_ptr[g]), int(self.node_ptr[g + 1])
        es, ee = int(self.edge_ptr[g]), int(self.edge_ptr[g + 1])
        return Data(
            x=torch.from_numpy(self.node_feat[ns:ne].astype(np.int64)),
            edge_index=torch.from_numpy(self.edge_index[:, es:ee].astype(np.int64)),
            edge_attr=torch.from_numpy(self.edge_feat[es:ee].astype(np.int64)),
            y=torch.tensor([self.y[g]]),
        )
