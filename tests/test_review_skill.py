"""Behavioral checks for the published review tools (no network calls)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / '.claude/skills/qsar-review/scripts'


def load_skill(name):
    spec = importlib.util.spec_from_file_location(f'review_{name}', SKILL_SCRIPTS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_audit(tmp_path):
    processed = tmp_path / 'data/processed'
    (processed / 'splits').mkdir(parents=True)
    pd.DataFrame({'parent_smiles': ['CC', 'CCC', 'CCO'],
                  'canonical_smiles': ['CC', 'CCC', 'CCO'],
                  'pic50': [5., 6., 7.]}).to_csv(processed / 'toy_pic50_clean.csv', index=False)
    return processed / 'splits'


@pytest.mark.parametrize('fault', ['length', 'order', 'negative_index', 'overlap', 'missing_row'])
def test_allow_skip_never_accepts_existing_bad_split(tmp_path, fault, capsys):
    split_dir = prepare_audit(tmp_path)
    data = dict(smiles=np.array(['CC', 'CCC', 'CCO']), train_idx=np.array([0]),
                valid_idx=np.array([1]), test_idx=np.array([2]))
    if fault == 'length':
        data['smiles'] = data['smiles'][:2]
    elif fault == 'order':
        data['smiles'] = data['smiles'][::-1]
    elif fault == 'negative_index':
        data['test_idx'] = np.array([-1])
    elif fault == 'overlap':
        data['valid_idx'] = np.array([0])
    elif fault == 'missing_row':
        data['valid_idx'] = np.array([], dtype=int)
    np.savez(split_dir / 'toy_random.npz', **data)
    assert load_skill('qc_salts_censoring').main(
        ['--repo', str(tmp_path), '--tag', 'toy', '--allow-skip']) == 1
    assert 'FAIL' in capsys.readouterr().out


def test_allow_skip_accepts_only_absent_splits(tmp_path):
    prepare_audit(tmp_path)
    mod = load_skill('qc_salts_censoring')
    assert mod.main(['--repo', str(tmp_path), '--tag', 'toy']) == 1
    assert mod.main(['--repo', str(tmp_path), '--tag', 'toy', '--allow-skip']) == 0


def test_temporal_exclusions_are_valid_partition(tmp_path):
    split_dir = prepare_audit(tmp_path)
    np.savez(split_dir / 'toy_time.npz', smiles=np.array(['CC', 'CCC', 'CCO']),
             train_idx=np.array([0]), test_idx=np.array([1]), excluded_idx=np.array([2]))
    assert load_skill('qc_salts_censoring').main(
        ['--repo', str(tmp_path), '--tag', 'toy', '--allow-skip']) == 0


def test_censoring_audit_matches_stripped_relation_policy(tmp_path):
    raw = tmp_path / 'raw.csv'
    pd.DataFrame({'canonical_smiles': ['CC', 'CCC', 'CCO', 'CCCC'],
                  'standard_value': [10., 20., 30., 40.],
                  'standard_units': ['nM'] * 4,
                  'standard_relation': [' = ', ' > ', None, '= ']}).to_csv(raw, index=False)
    mod = load_skill('qc_salts_censoring')
    mod.RAW_CSV = str(raw)
    mod.BASE = str(tmp_path)
    out = mod.check_censoring()
    assert out['n_removed_by_relation_rule'] == 2
    assert out['n_censored_bounds'] == 1


def test_censoring_audit_empty_input_has_no_zero_division(tmp_path):
    raw = tmp_path / 'raw.csv'
    pd.DataFrame(columns=['canonical_smiles', 'standard_value', 'standard_relation']).to_csv(
        raw, index=False)
    mod = load_skill('qc_salts_censoring')
    mod.RAW_CSV = str(raw)
    mod.BASE = str(tmp_path)
    assert mod.check_censoring()['n_raw_rows'] == 0


def test_parent_censoring_overlap_keeps_padded_equals(tmp_path, monkeypatch):
    processed, raw = tmp_path / 'data/processed', tmp_path / 'data/raw'
    processed.mkdir(parents=True)
    raw.mkdir()
    pd.DataFrame({'parent_smiles': ['CC'], 'n_measurements': [1]}).to_csv(
        processed / 'toy_pic50_clean.csv', index=False)
    pd.DataFrame({'canonical_smiles': ['CC', 'CC'], 'standard_units': ['nM'] * 2,
                  'standard_value': [10., 20.], 'standard_relation': [' = ', ' > ']}).to_csv(
        raw / 'toy_activities.csv', index=False)
    mod = load_skill('chembl_censoring_counts')
    monkeypatch.setattr(mod, 'load_parent_fn', lambda repo: lambda s: (s, 1, True))
    result = mod.overlap('toy', str(tmp_path))
    assert result['n_exact_rows'] == 1
    assert result['n_modelled_mixed_exact_censored'] == 1


def seed_predictions(tmp_path):
    directory = tmp_path / 'results'
    directory.mkdir()
    y = np.linspace(4., 9., 60)
    error = np.tile([-1., 1.], 30)
    idx = np.arange(60)
    np.savez(directory / 'rf_preds_toy_random.npz', test_idx=idx, y_true=y,
             y_pred=y + .30 * error)
    for suffix, scale in [('', 0.), ('_seed1', .45), ('_seed2', .45)]:
        np.savez(directory / f'gnn_preds_toy_random{suffix}.npz', test_idx=idx,
                 y_true=y, y_pred=y + scale * error)
    return directory


def test_published_seed_check_detects_conditional_verdict(tmp_path):
    seed_predictions(tmp_path)
    mod = load_skill('single_seed_verdicts')
    result = mod.cell_sensitivity(tmp_path, 'toy', 'random', n_boot=100,
                                  bootstrap=mod.bootstrap_module(mod.BASE))
    assert result['conditional']
    assert result['drops'][0]['verdict'] == 'rf'
    assert result['drops'][1]['verdict'] == 'gnn'
    assert not result['split_indices_checked']  # absence explicitly disclosed
    assert len(result['inputs']) == 4


def test_published_seed_check_rejects_prediction_misalignment(tmp_path):
    directory = seed_predictions(tmp_path)
    path = directory / 'gnn_preds_toy_random_seed1.npz'
    with np.load(path) as d:
        arrays = {key: d[key] for key in d.files}
    arrays['test_idx'] = arrays['test_idx'][::-1]
    np.savez(path, **arrays)
    mod = load_skill('single_seed_verdicts')
    with pytest.raises(ValueError, match='test indices differ'):
        mod.cell_sensitivity(tmp_path, 'toy', 'random', n_boot=100,
                             bootstrap=mod.bootstrap_module(mod.BASE))


def test_readme_time_verifier_includes_constant_baseline():
    mod = load_skill('verify_readme_tables')
    mod.lines = (Path(mod.BASE) / 'README.md').read_text().splitlines()
    mod.check_t2()
    assert all(cell[4] for cell in mod.results)
    assert len([r for r in mod.results if r[1].endswith('/constant')]) == 8
    # A wrong constant must fail despite every RF/GIN value remaining valid.
    rows = mod.find_table('| Tag | RF rand | GIN rand | Constant |')
    row = rows[0]
    row[3] = '0.500'
    mod.find_table = lambda fragment: rows
    mod.results = []
    mod.check_t2()
    failures = [r for r in mod.results if not r[4]]
    assert len(failures) == 1 and failures[0][1].endswith('/constant')


@pytest.fixture
def environment_readme(tmp_path, load_experiment):
    """Complete accepted synthetic jobs with distinct means and flag counts."""
    import hashlib
    import json

    comparator = load_experiment('time_environment_comparison')
    mod = load_skill('verify_readme_tables')
    mod.BASE = str(tmp_path)
    header = '| Environment | Budget | Jobs | Train MSE | Validation RMSE | Test R² | Flags |'
    mod.lines = [header, '| --- | --- | --- | --- | --- | --- | --- |',
                 '| CPU | original | 64 | 0.139 | 0.239 | -0.362 | 0 |',
                 '| CPU | extended | 64 | 0.339 | 0.439 | -0.162 | 0 |',
                 '| CUDA | original | 64 | 0.539 | 0.639 | 0.039 | 63 |',
                 '| CUDA | extended | 64 | 0.739 | 0.839 | 0.239 | 64 |']
    roots = {}
    def digest(name):
        return hashlib.sha256(name.encode()).hexdigest()
    for device, directory in [('cpu', 'm3_eight_seed'), ('cuda', 'cuda_eight_seed')]:
        root = tmp_path / 'results/time_budget_sensitivity' / directory / 'analysis'
        root.mkdir(parents=True)
        roots[device] = root
        rows = []
        for arm in comparator.ARMS:
            for tag_i, tag in enumerate(comparator.TAGS):
                for seed_i, seed in enumerate(comparator.SEEDS):
                    offset = .01 * tag_i + .001 * seed_i
                    offset += .4 if device == 'cuda' else 0
                    offset += .2 if arm == 'extended' else 0
                    mse = .1 + offset
                    rows.append(dict(arm=arm, tag=tag, seed=seed, label_id=digest(tag),
                                     device=device, best_epoch=29, common_flag=mse > .5,
                                     train_mse=mse, valid_rmse_std=.2 + offset,
                                     test_r2=-.4 + offset, test_rmse=.6 + offset,
                                     test_mae=.5 + offset))
        pd.DataFrame(rows).to_csv(root / 'diagnostics.csv', index=False)
        for name in comparator.REPORT_FILES - {'diagnostics.csv'}:
            (root / name).write_text('fixture\n')
        provenance = {'source_sha256': {name: digest(name) for name in comparator.CODE_FILES},
                      'input_sha256': {name: digest(name) for name in comparator.INPUT_FILES},
                      'host': device + '-host', 'training_environment': {'platform': device},
                      'protocol': {'device': device, 'arms': comparator.ARMS,
                                   'seeds': list(comparator.SEEDS), 'jobs': 128}}
        report = {'schema': 'm3-time-budget-report-v1', 'status': 'accepted',
                  'training_device': device, 'training_host': provenance['host'],
                  'training_platform': device, 'seeds': list(comparator.SEEDS), 'n_seeds': 8,
                  'jobs_accepted': 128, 'paired_rows': 64, 'targets': 8, 'factorial_cells_complete': 2,
                  **comparator.OUTCOME_DEFINITIONS, 'provenance': provenance,
                  'output_sha256': {name: comparator.digest(root / name)
                                    for name in comparator.REPORT_FILES}}
        (root / 'report.json').write_text(json.dumps(report))
        # An intentionally wrong preaggregated table must not influence the audit.
        (root / 'cells.csv').write_text('all values are incorrect\n')
    return mod, roots


def test_environment_table_independently_checks_all_twenty_values(environment_readme):
    mod, _ = environment_readme
    mod.check_environment_table()
    assert len(mod.results) == 20
    assert all(cell[4] for cell in mod.results)
    assert {cell[0] for cell in mod.results} == {'T10_environment'}
    values = {cell[1]: cell[3] for cell in mod.results}
    assert values['CPU/original/train_mse'] == pytest.approx(.1385)
    assert values['CUDA/extended/valid_rmse_std'] == pytest.approx(.8385)
    assert values['CUDA/original/flags'] == 63


@pytest.mark.parametrize('column', range(2, 7))
def test_environment_table_catches_each_numeric_column(environment_readme, column):
    mod, _ = environment_readme
    cells = mod.lines[2].split('|')
    cells[column + 1] = ' 999 '
    mod.lines[2] = '|'.join(cells)
    mod.check_environment_table()
    bad = [cell for cell in mod.results if not cell[4]]
    assert len(bad) == 1
    assert bad[0][1].startswith('CPU/original/')


@pytest.mark.parametrize('fault', ['missing_report', 'pending', 'partial', 'hash', 'source'])
def test_environment_table_requires_accepted_complete_matching_reports(environment_readme, fault):
    import json

    mod, roots = environment_readme
    report_path = roots['cuda'] / 'report.json'
    report = json.loads(report_path.read_text())
    if fault == 'missing_report':
        report_path.unlink()
    elif fault == 'pending':
        report['status'] = 'pending'
    elif fault == 'partial':
        report['jobs_accepted'] = 48
    elif fault == 'hash':
        (roots['cuda'] / 'diagnostics.csv').write_text('changed\n')
    elif fault == 'source':
        key = next(iter(report['provenance']['source_sha256']))
        report['provenance']['source_sha256'][key] = '0' * 64
    if fault != 'missing_report':
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        mod.check_environment_table()
    assert mod.results == []


def test_environment_table_absent_is_explicitly_not_checked(tmp_path, capsys):
    mod = load_skill('verify_readme_tables')
    mod.BASE = str(tmp_path)
    mod.lines = ['# No four-cell table yet']
    mod.check_environment_table()
    assert mod.results == []
    assert 'no checks performed' in capsys.readouterr().out


def test_environment_table_rejects_duplicate_or_missing_cells(environment_readme):
    mod, _ = environment_readme
    mod.lines[-1] = mod.lines[-2]
    with pytest.raises(ValueError, match='exactly CPU/CUDA'):
        mod.check_environment_table()
    assert mod.results == []
