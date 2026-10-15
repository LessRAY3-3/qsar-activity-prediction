"""Read-only audit of current random/scaffold analysis inputs (no training).

Writes only the explicitly requested report. Does not prove original model
training identity or replay model weights; checks cached identities and values.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_guards import (atomic_json, check_error_table, collect_seed_metrics,
                             require_rf_params)

TAGS = ('a2a', 'abl1', 'egfr', 'egfr_full', 'herg', 'hivpr', 'mpro', 'vegfr2')


def audit(root):
    root = Path(root)
    results = root / 'results'
    sys.path.insert(0, str(root / 'experiments'))
    summary = importlib.import_module('time_split_summary')
    summary.BASE = str(root)
    cells = []
    for tag in TAGS:
        metrics = json.loads((results / f'metrics_{tag}.json').read_text())
        seeds, _ = collect_seed_metrics(results, tag)
        summary.comparison_reference(tag)
        with np.load(root / f'data/processed/{tag}_fingerprints.npz') as fp:
            for split in ('random', 'scaffold'):
                require_rf_params(metrics, split)
                with np.load(root / f'data/processed/splits/{tag}_{split}.npz') as sp, \
                     np.load(results / f'rf_preds_{tag}_{split}.npz') as rf, \
                     np.load(results / f'gnn_preds_{tag}_{split}.npz') as gin:
                    if not np.array_equal(fp['smiles'], sp['smiles']):
                        raise ValueError(f'{tag}/{split}: fingerprint/split order mismatch')
                    error_file = results / f'error_analysis_{tag}_{split}.csv'
                    table = pd.read_csv(error_file)
                    check_error_table(table, sp, rf, gin, fp['y'])
                    cells.append(dict(tag=tag, split=split, molecules=len(sp['test_idx']),
                                      n_seeds=len(seeds[split]['r2']), status='passed'))
    return dict(status='passed', scope='cached prediction/label/index/error-table/metric consistency',
                limitations=['No model replay or retraining', 'No proof of original training identity'],
                cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.root)
    except Exception as exc:
        report = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
