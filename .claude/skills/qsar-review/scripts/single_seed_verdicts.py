"""Reproduce the PR #6 seed-drop sensitivity from committed predictions.

Read-only: reports the 48 random/scaffold cells by default. It changes only
GNN ensemble membership, so this is not an RF seed-robustness experiment.
Each prediction is checked against the RF test indices and labels. Where a
split NPZ is available, those indices are checked too; missing historical
split files are disclosed in the report, never silently called validated.

python .claude/skills/qsar-review/scripts/single_seed_verdicts.py --out /tmp/seed-drops.json
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[4]
TAGS = ('a2a', 'abl1', 'egfr', 'egfr_full', 'herg', 'hivpr', 'mpro', 'vegfr2')
MODELS = ('gin', 'gine', 'attentivefp')
SEEDS = (42, 1, 2)


def bootstrap_module(repo):
    path = Path(repo) / 'experiments/paired_bootstrap.py'
    spec = importlib.util.spec_from_file_location('review_paired_bootstrap', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verdict(result):
    return 'gnn' if result['ci_lo'] > 0 else 'rf' if result['ci_hi'] < 0 else 'tie'


def cell_sensitivity(repo, tag, split, model='gin', n_boot=10000, seed=42, bootstrap=None):
    repo = Path(repo)
    pb = bootstrap or bootstrap_module(repo)
    if split == 'time':
        # Temporal inputs need the frozen label-policy/version checks as well.
        pb.load_cell(tag, split, model)
    rf_path = repo / 'results' / f'rf_preds_{tag}_{split}.npz'
    with np.load(rf_path) as data:
        idx, y, rf = (np.array(data[key]) for key in ('test_idx', 'y_true', 'y_pred'))
    if idx.ndim != 1 or len(np.unique(idx)) != len(idx) or len(idx) != len(y):
        raise ValueError(f'{tag}/{split}: invalid RF test index vector')
    inputs = [rf_path]
    predictions = []
    for training_seed in SEEDS:
        suffix = '' if training_seed == 42 else f'_seed{training_seed}'
        path = repo / 'results' / pb.gnn_preds_name(tag, split, suffix, model)
        with np.load(path) as data:
            if not np.array_equal(data['test_idx'], idx):
                raise ValueError(f'{path.name}: test indices differ from RF')
            if not np.allclose(data['y_true'], y, rtol=0, atol=1e-4):
                raise ValueError(f'{path.name}: labels differ from RF')
            predictions.append(np.asarray(data['y_pred'], dtype=float))
        inputs.append(path)
    arrays = [y, rf, *predictions]
    if any(a.ndim != 1 or len(a) != len(y) or not np.isfinite(a).all() for a in arrays):
        raise ValueError(f'{tag}/{split}: non-finite or malformed prediction payload')
    split_path = repo / 'data/processed/splits' / f'{tag}_{split}.npz'
    split_checked = split_path.exists()
    if split_checked:
        with np.load(split_path) as data:
            if not np.array_equal(data['test_idx'], idx):
                raise ValueError(f'{tag}/{split}: test indices differ from split')
        inputs.append(split_path)
    preds = np.stack(predictions)
    full = pb.paired_bootstrap(y, preds.mean(axis=0), rf, n_boot=n_boot, seed=seed)
    full['verdict'] = verdict(full)
    drops = []
    for i, training_seed in enumerate(SEEDS):
        res = pb.paired_bootstrap(y, np.delete(preds, i, axis=0).mean(axis=0), rf,
                                  n_boot=n_boot, seed=seed)
        drops.append(dict(dropped_seed=training_seed, verdict=verdict(res), **res))
    return dict(tag=tag, split=split, model=model, full=full, drops=drops,
                conditional=any(d['verdict'] != full['verdict'] for d in drops),
                split_indices_checked=split_checked,
                inputs={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in inputs})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--tags', nargs='+', choices=TAGS, default=TAGS)
    p.add_argument('--splits', nargs='+', choices=('random', 'scaffold', 'time'),
                   default=('random', 'scaffold'))
    p.add_argument('--n-boot', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    if args.n_boot < 1:
        p.error('--n-boot must be positive')
    if args.out:
        dest = str(args.out.resolve()).lower()
        for name in ('data', 'results', 'models', 'figures'):
            protected = str((BASE / name).resolve()).lower()
            if dest == protected or dest.startswith(protected + '/'):
                p.error('--out may not write into data/results/models/figures')
    pb = bootstrap_module(BASE)
    rows = [cell_sensitivity(BASE, tag, split, model, args.n_boot, args.seed, pb)
            for model in args.models for tag in args.tags for split in args.splits
            if split != 'time' or model == 'gin']
    result = dict(n_boot=args.n_boot, bootstrap_seed=args.seed, training_seeds=list(SEEDS),
                  scope='GNN seed deletion with frozen RF; ensemble size also changes',
                  n_cells=len(rows), n_conditional=sum(r['conditional'] for r in rows), cells=rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
