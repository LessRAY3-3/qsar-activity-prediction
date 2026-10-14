"""Architecture display names must not depend on legacy GNN storage keys."""
import json

import numpy as np
import pandas as pd
import pytest


def calibration():
    return {name: {'bin_mean_std': [.1, .3], 'bin_rmse': [.2, .5],
                   'slope': 1.5, 'spearman_std_vs_abs_err': .6}
            for name in ('rf', 'gin')}


@pytest.mark.parametrize('model_kind', ['gin', 'gine'])
def test_plot_labels_use_architecture_without_changing_values(
        load_experiment, tmp_path, monkeypatch, model_kind):
    import matplotlib.axes
    mod = load_experiment('uncertainty_ad')
    captured = []
    original = matplotlib.axes.Axes.plot
    def capture(self, x, y, *args, **kwargs):
        captured.append((kwargs.get('label', ''), np.asarray(x), np.asarray(y)))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', capture)
    mod.plot_calibration('tiny', 'random', calibration(), str(tmp_path / 'cal.png'),
                         model_kind=model_kind)
    mod.plot_ad_error('tiny', 'random', np.array([5., 6., 7.]),
                      [('rf', [5.2, 6.1, 7.4], 'orange'), ('gin', [5.1, 6.4, 7.3], 'green')],
                      {'rf': np.array([.2, .4, .9]), 'gin': np.array([.1, .3, .8])},
                      .4, str(tmp_path / 'ad.png'), model_kind=model_kind)
    architecture_rows = [r for r in captured if r[0].startswith(model_kind.upper() + ' ')]
    assert len(architecture_rows) == 2
    np.testing.assert_array_equal(architecture_rows[0][1], [.1, .3])
    np.testing.assert_array_equal(architecture_rows[0][2], [.2, .5])
    if model_kind == 'gine':
        assert not any(r[0].startswith('GIN ') for r in captured)
    assert sum(r[0].startswith('RF ') for r in captured) == 2


def test_figures_only_preserves_saved_numbers_and_never_runs_models(
        load_experiment, tmp_path, monkeypatch):
    mod = load_experiment('uncertainty_ad')
    out, fig = tmp_path / 'saved', tmp_path / 'figures'
    out.mkdir()
    summary = out / 'summary_tiny_gine.json'
    summary.write_text(json.dumps({'tag': 'tiny', 'model_kind': 'gine', 'ad_threshold': .4,
                                  'splits': {'random': {'n_test': 3,
                                                        'calibration': calibration()}}}))
    csv = out / 'tiny_random_gine.csv'
    frames = [pd.DataFrame({'model': name, 'test_idx': [3, 4, 5], 'y_true': [5., 6., 7.],
                            'pred': pred, 'max_sim': [.1, .3, .8]})
              for name, pred in [('rf', [5.2, 6.1, 7.4]), ('gin', [5.1, 6.4, 7.3])]]
    pd.concat(frames).to_csv(csv, index=False)
    before = {path: path.read_bytes() for path in (csv, summary)}
    def forbidden(*args, **kwargs):
        raise AssertionError('figure-only mode must not recompute predictions or AD')
    for name in ('process_cell', 'refresh_stored_ad', 'available_seed_suffixes',
                 'gnn_ensemble_predict'):
        monkeypatch.setattr(mod, name, forbidden)
    mod.main(['--model', 'gine', '--tags', 'tiny', '--splits', 'random',
              '--refresh-figures-only', '--out-dir', str(out), '--fig-dir', str(fig)])
    assert {path: path.read_bytes() for path in before} == before
    assert set(p.name for p in fig.iterdir()) == {'calib_tiny_random_gine.png',
                                                'ad_error_tiny_random_gine.png'}
