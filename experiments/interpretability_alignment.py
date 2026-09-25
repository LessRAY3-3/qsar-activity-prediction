"""Interpretability alignment: what parts of a molecule do the two models
actually look at, on the chemotypes where their failures are asymmetric?

Background (README 7.5, scaffold test set): the chromone series (n=37)
costs the GIN more (mean |err| 1.12 vs RF 0.64); the thienopyrimidine
series (n=47) costs the RF more (1.60 vs 1.16). This experiment asks:
do the two architectures attend to the same atoms?

Molecule selection: within each family (RDKit substructure match; the two
SMARTS below reproduce the documented 37 / 47 counts exactly), take the
top-N molecules by |err_GIN - err_RF| - the largest disagreement is where
"different parts" is most plausible.

  GNN side: PyG GNNExplainer (node-mask attribution, explanation of the
           model's own prediction) on the committed scaffold-split GIN
           checkpoint. Per-atom scores rendered as contour maps.
  RF side : RF retrained on the 4438 paired pool (deterministic, ~30 s),
           SHAP TreeExplainer bit attributions; each top bit is decoded
           to the set of atoms it covers via Morgan bitInfo spheres.
  Alignment: per molecule, what fraction of the GNN's top atoms falls
           inside the RF's top-bit atom coverage? Plus a core-vs-
           substituent split (do attentions sit on the scaffold core?).

Outputs:
  figures/interpretability/<family>_<i>_<short>_{gnn,rf}.png
  figures/interpretability/alignment_summary.png
  results/interpretability/alignment.json
  results/interpretability/findings.md

Partial success is allowed: a molecule that fails one method is recorded
in alignment.json and skipped in figures; findings.md reports what ran.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from gnn_03_train_gin import GINRegressor  # noqa: E402

# substructure queries - reproduce the README's 37 / 47 series sizes exactly
FAMILIES = {
    "chromone": "O=c1ccoc2ccccc12",
    "thienopyrimidine": "c1ncc2ccsc2n1",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=os.environ.get("QSAR_TAG", "egfr"))
    p.add_argument("--split", default="scaffold")
    p.add_argument("--n-per-family", type=int, default=5)
    p.add_argument("--top-bits", type=int, default=5)
    p.add_argument("--top-atom-frac", type=float, default=0.25)
    p.add_argument("--gnn-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(BASE, "results", "interpretability"))
    p.add_argument("--fig-dir", default=os.path.join(BASE, "figures", "interpretability"))
    return p.parse_args()


# ---------------------------------------------------------------- helpers
def morgan_bit_atoms(mol, bit, radius=2, n_bits=2048):
    """Atoms covered by one Morgan bit: union of spheres around the
    (center_atom, radius) entries RDKit reports via bitInfo."""
    from rdkit.Chem import AllChem

    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bi)
    covered = set()
    for center, r in bi.get(int(bit), []):
        sphere = {int(center)}
        frontier = {int(center)}
        for _ in range(int(r)):
            nxt = set()
            for a in frontier:
                atom = mol.GetAtomWithIdx(a)
                for b in atom.GetBonds():
                    nxt.add(b.GetOtherAtomIdx(a))
            nxt -= sphere
            sphere |= nxt
            frontier = nxt
        covered |= sphere
    return sorted(covered)


def load_gin(tag, split, device):
    import torch

    ckpt = torch.load(os.path.join(BASE, "results", "gnn_models",
                                   f"{tag}_gin_{split}.pt"),
                      map_location=device, weights_only=False)
    a = ckpt["args"]
    model = GINRegressor(a["hidden"], a["num_layers"], a["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _make_wrapper(model):
    """Adapt GINRegressor (takes a Data object) to the Explainer calling
    convention model(x=..., edge_index=..., batch=...)."""
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, edge_index, batch=None, edge_attr=None):
            data = Data(x=x.long(), edge_index=edge_index, batch=batch)
            return self.m(data)

    return Wrapper(model)


def explain_gnn_atoms(model, x, edge_index, device, epochs, seed):
    """GNNExplainer node attribution for one molecule. Returns per-atom
    scores (np.array, normalized to [0,1]) or None on failure."""
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer

    torch.manual_seed(seed)
    wrapper = _make_wrapper(model)
    n = x.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    try:
        # node masks cannot carry gradients through Embedding layers (integer
        # categorical features), so attribute via EDGE masks and aggregate
        # incident edge weight per atom.
        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(mode="regression", task_level="graph",
                              return_type="raw"))
        explanation = explainer(x=x.to(device), edge_index=edge_index.to(device),
                                batch=batch)
        m = explanation.edge_mask.detach().cpu().numpy().squeeze().astype(float)
        eidx = edge_index.cpu().numpy()
        s = np.zeros(n)
        for e in range(eidx.shape[1]):
            s[eidx[0, e]] += m[e]
            s[eidx[1, e]] += m[e]
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else None
    except Exception as e:
        print(f"    GNNExplainer failed: {e}")
        return None


def make_shap_explainer(rf, X_bg):
    """Interventional TreeExplainer: path-dependent mode returns garbage
    with current sklearn (additivity check fails), interventional verifies."""
    import shap

    return shap.TreeExplainer(rf, data=np.asarray(X_bg, dtype=np.float32),
                              feature_perturbation="interventional")


def shap_bit_attributions(explainer, X_sel):
    sv = explainer.shap_values(np.asarray(X_sel, dtype=np.float32))
    if hasattr(sv, "values"):  # newer shap returns an Explanation object
        sv = sv.values
    return np.asarray(sv)


# ---------------------------------------------------------------- figures
def gnn_figure(mol, weights, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

    d2d = rdMolDraw2D.MolDraw2DCairo(420, 420)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, [float(w) for w in weights], d2d, colorMap="jet", contourLines=8)
    d2d.FinishDrawing()
    img = mpimg.imread(io.BytesIO(d2d.GetDrawingText()), format="png")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def rf_figure(mol, top_bits, bit_shap, bit_atoms_map, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw

    fig, (ax_mol, ax_bar) = plt.subplots(1, 2, figsize=(9, 4),
                                         gridspec_kw={"width_ratios": [1, 1.2]})
    # structure panel: atoms covered by top bits, coloured by |shap|
    absvals = {b: abs(bit_shap[b]) for b in top_bits}
    vmax = max(absvals.values()) if absvals else 1.0
    colors, covered = {}, set()
    for b in top_bits:
        for a in bit_atoms_map[b]:
            covered.add(a)
            prev = colors.get(a)
            intensity = absvals[b] / vmax
            colors[a] = (intensity, 0.1, 1 - intensity) if prev is None else \
                (max(prev[0], intensity), 0.1, max(prev[2], 1 - intensity))
    img = Draw.MolToImage(mol, size=(360, 360), highlightAtoms=list(covered),
                          highlightAtomColors=colors, highlightBonds=[],
                          legend="top-bit coverage")
    ax_mol.imshow(img)
    ax_mol.axis("off")
    ax_mol.set_title("RF top-bit atom coverage", fontsize=10)
    # bar panel
    labels = [f"bit {b}" for b in top_bits]
    vals = [bit_shap[b] for b in top_bits]
    ax_bar.barh(range(len(vals))[::-1], vals,
                color=["tab:red" if v < 0 else "tab:blue" for v in vals])
    ax_bar.set_yticks(range(len(vals))[::-1], labels)
    ax_bar.axvline(0, color="k", lw=0.8)
    ax_bar.set_xlabel("SHAP value (pIC50)")
    ax_bar.set_title("top bits", fontsize=10)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    import torch
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    import pandas as pd
    err_csv = os.path.join(BASE, "results", f"error_analysis_{args.tag}_{args.split}.csv")
    df = pd.read_csv(err_csv)
    fp = np.load(os.path.join(BASE, "data", "processed",
                              f"{args.tag}_fingerprints.npz"))
    X, y = fp["X"], fp["y"]
    graphs = np.load(os.path.join(BASE, "data", "processed",
                                  f"{args.tag}_graphs.npz"))
    node_ptr = np.concatenate([[0], np.cumsum(graphs["n_nodes"])])
    edge_ptr = np.concatenate([[0], np.cumsum(graphs["n_edges"])])
    sp = np.load(os.path.join(BASE, "data", "processed", "splits",
                              f"{args.tag}_{args.split}.npz"))
    test_idx = sp["test_idx"]
    assert len(df) == len(test_idx), "error analysis csv does not match split"

    # ---- molecule selection: families by substructure, top-N disagreement
    selections = {}
    for fam, qsmi in FAMILIES.items():
        pat = Chem.MolFromSmiles(qsmi)
        mask = np.array([bool(Chem.MolFromSmiles(s) is not None and
                              Chem.MolFromSmiles(s).HasSubstructMatch(pat))
                         for s in df["smiles"]], dtype=bool)
        fam_df = df[mask].copy()
        fam_df["disagree"] = (fam_df["err_gin"] - fam_df["err_rf"]).abs()
        fam_df = fam_df.sort_values("disagree", ascending=False)
        selections[fam] = fam_df.head(args.n_per_family)
        print(f"[{fam}] family size={mask.sum()} (query {qsmi}), "
              f"selected top-{len(selections[fam])} by |err_gin - err_rf|")

    # ---- models
    mj = json.load(open(os.path.join(BASE, "results", f"metrics_{args.tag}.json")))
    bp = mj["best_params_random"]
    rf_params = {k: bp[k] for k in ("n_estimators", "max_depth", "min_samples_split")}
    from sklearn.ensemble import RandomForestRegressor
    t0 = time.time()
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
    rf.fit(X[sp["train_idx"]], y[sp["train_idx"]])
    print(f"RF retrained on paired pool ({time.time()-t0:.0f}s)")
    shap_exp = make_shap_explainer(rf, X[sp["train_idx"][:200]])

    gin = load_gin(args.tag, args.split, device)

    # ---- per-molecule analysis
    records = []
    for fam, fam_df in selections.items():
        pat = Chem.MolFromSmiles(FAMILIES[fam])
        for i, row in enumerate(fam_df.itertuples(), 1):
            global_pos = int(row.Index)          # position within test set
            gid = int(test_idx[global_pos])      # global molecule id
            smi = row.smiles
            mol = Chem.MolFromSmiles(smi)
            short = f"mol{i}"
            title = (f"{fam} {short}  y={row.y_true:.2f}  "
                     f"err RF={row.err_rf:.2f} / GIN={row.err_gin:.2f}")

            rec = {"family": fam, "id": short, "test_pos": global_pos,
                   "graph_id": gid, "smiles": smi, "y_true": float(row.y_true),
                   "err_rf": float(row.err_rf), "err_gin": float(row.err_gin)}
            print(f"[{fam}/{short}] gid={gid} {smi[:40]}")

            # ----- GNN side
            ns, ne = int(node_ptr[gid]), int(node_ptr[gid + 1])
            es, ee = int(edge_ptr[gid]), int(edge_ptr[gid + 1])
            x = torch_from_numpy_int(graphs["node_feat"][ns:ne])
            edge_index = torch_from_numpy_int(graphs["edge_index"][:, es:ee])
            t0 = time.time()
            gnn_scores = explain_gnn_atoms(gin, x, edge_index, device,
                                           args.gnn_epochs, args.seed)
            rec["gnn_seconds"] = round(time.time() - t0, 1)
            n_atoms = mol.GetNumAtoms()
            if gnn_scores is not None and len(gnn_scores) == n_atoms:
                rec["gnn_atom_scores"] = [round(float(v), 4) for v in gnn_scores]
                k = max(1, int(round(args.top_atom_frac * n_atoms)))
                top_atoms = set(np.argsort(-gnn_scores)[:k].tolist())
                rec["gnn_top_atoms"] = sorted(top_atoms)
                gnn_figure(mol, gnn_scores,
                           os.path.join(args.fig_dir,
                                        f"{fam}_{short}_gnn.png"),
                           f"GIN attention - {title}")
            else:
                rec["gnn_top_atoms"] = None
                rec["gnn_error"] = "explainer failed or size mismatch"

            # ----- RF side
            sv_one = shap_bit_attributions(shap_exp, X[gid:gid + 1])[0]
            top_bits = np.argsort(-np.abs(sv_one))[:args.top_bits].tolist()
            rec["rf_top_bits"] = [int(b) for b in top_bits]
            rec["rf_bit_shap"] = {str(int(b)): round(float(sv_one[b]), 4)
                                  for b in top_bits}
            bit_atoms_map = {int(b): morgan_bit_atoms(mol, int(b))
                             for b in top_bits}
            rec["rf_bit_atoms"] = {str(b): v for b, v in bit_atoms_map.items()}
            rf_figure(mol, [int(b) for b in top_bits],
                      {int(b): float(sv_one[b]) for b in top_bits},
                      bit_atoms_map,
                      os.path.join(args.fig_dir, f"{fam}_{short}_rf.png"),
                      f"RF SHAP bits - {title}")

            # ----- alignment
            if rec["gnn_top_atoms"] is not None:
                rf_covered = set().union(*bit_atoms_map.values()) \
                    if bit_atoms_map else set()
                gnn_top = set(rec["gnn_top_atoms"])
                inter = gnn_top & rf_covered
                union = gnn_top | rf_covered
                rec["overlap_fraction"] = round(len(inter) / max(1, len(gnn_top)), 4)
                rec["jaccard"] = round(len(inter) / max(1, len(union)), 4)
                core = set(a for a in mol.GetSubstructMatch(pat)) if pat else set()
                rec["core_atoms"] = sorted(core)
                rec["gnn_top_on_core"] = round(
                    len(gnn_top & core) / max(1, len(gnn_top)), 4)
            records.append(rec)

    # ---- summary figure
    summary_figure(records, os.path.join(args.fig_dir, "alignment_summary.png"))

    # ---- outputs
    out = {"tag": args.tag, "split": args.split,
           "families": FAMILIES,
           "selection": f"top-{args.n_per_family} per family by |err_gin - err_rf|",
           "params": {"top_bits": args.top_bits,
                      "top_atom_frac": args.top_atom_frac,
                      "gnn_epochs": args.gnn_epochs, "seed": args.seed},
           "molecules": records}
    json.dump(out, open(os.path.join(args.out, "alignment.json"), "w"), indent=2)
    write_findings(records, os.path.join(args.out, "findings.md"), args)
    print(f"records: {len(records)}  -> results/interpretability/")


def torch_from_numpy_int(arr):
    import torch

    return torch.from_numpy(np.asarray(arr).astype(np.int64))


def summary_figure(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in records if r.get("overlap_fraction") is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fams = sorted({r["family"] for r in ok})
    for fpos, fam in enumerate(fams):
        vals = [r["overlap_fraction"] for r in ok if r["family"] == fam]
        xs = np.arange(len(vals)) + fpos * 0.4
        ax.bar(xs, vals, width=0.38, label=f"{fam} (mean {np.mean(vals):.2f})")
    ax.axhline(np.mean([r["overlap_fraction"] for r in ok]), color="k",
               ls="--", lw=1)
    ax.set_ylabel("GNN top atoms inside RF top-bit coverage")
    ax.set_title("Cross-model atom-level alignment (per molecule)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_findings(records, path, args):
    ok = [r for r in records if r.get("overlap_fraction") is not None]
    fams = sorted({r["family"] for r in records})
    lines = ["# Interpretability alignment - findings", "",
             f"Generated by experiments/interpretability_alignment.py "
             f"(tag={args.tag}, split={args.split}, "
             f"top_bits={args.top_bits}, top_atom_frac={args.top_atom_frac}).",
             ""]
    if not ok:
        lines.append("No molecule completed both methods; see alignment.json "
                     "for per-method errors.")
    else:
        ov = [r["overlap_fraction"] for r in ok]
        core = [r["gnn_top_on_core"] for r in ok]
        lines += [
            f"## Numbers",
            f"- molecules analysed: {len(records)} "
            f"({', '.join(f'{f} x{sum(1 for r in records if r['family']==f)}' for f in fams)}); "
            f"both methods ok on {len(ok)}",
            f"- GNN top atoms inside RF top-bit coverage: "
            f"mean {np.mean(ov):.2f} (min {min(ov):.2f}, max {max(ov):.2f})",
            f"- GNN top atoms sitting on the scaffold core: "
            f"mean {np.mean(core):.2f}",
            "",
            "## Findings",
        ]
        # data-driven findings, each with molecule examples
        for fam in fams:
            sub = [r for r in ok if r["family"] == fam]
            if not sub:
                continue
            best = max(sub, key=lambda r: r["overlap_fraction"])
            worst = min(sub, key=lambda r: r["overlap_fraction"])
            lines.append(
                f"1. **{fam}**: cross-model alignment mean "
                f"{np.mean([r['overlap_fraction'] for r in sub]):.2f} "
                f"(n={len(sub)}). Highest: {best['id']} "
                f"(overlap {best['overlap_fraction']:.2f}, "
                f"`{best['smiles'][:50]}`); lowest: {worst['id']} "
                f"(overlap {worst['overlap_fraction']:.2f}, "
                f"`{worst['smiles'][:50]}`).")
        fam_bits = {}
        for r in records:
            for b in r.get("rf_top_bits", []):
                fam_bits.setdefault((r["family"], b), 0)
                fam_bits[(r["family"], b)] += 1
        for fam in fams:
            top = sorted(((b, n) for (f, b), n in fam_bits.items() if f == fam),
                         key=lambda t: -t[1])[:3]
            if top:
                lines.append(
                    f"2. **{fam} RF recurring bits**: "
                    + ", ".join(f"bit {b} (top-5 in {n}/{sum(1 for r in records if r['family']==fam)} mols)"
                                for b, n in top)
                    + " - recurring bits indicate the substructures driving the RF "
                      "on this chemotype.")
        lines += [
            f"3. **GNN attention on scaffold core**: mean fraction of GNN top "
            f"atoms on the family scaffold core = {np.mean(core):.2f} "
            f"(values " + ", ".join(f"{r['id']}:{r['gnn_top_on_core']:.2f}"
                                     for r in ok) + ").",
            "",
            "## Caveats",
            "- GNNExplainer attributions are post-hoc and approximate; read the",
            "  heatmaps as indicative, not causal.",
            "- SHAP bit values are exact for the retrained RF (deterministic);",
            "  the RF was retrained on the 4438 paired pool (no committed",
            "  checkpoint existed for this pool).",
            "- Cross-model atom alignment is judged by set overlap of two",
            "  different attribution vocabularies; treat low overlap as",
            "  'attends differently', not as either model being wrong.",
        ]
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
