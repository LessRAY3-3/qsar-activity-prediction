import json
from pathlib import Path


def test_records_published_to_both_output_directories(load_experiment, tmp_path):
    mod = load_experiment('interpretability_alignment')
    result, figure = tmp_path / 'results', tmp_path / 'figures'
    record = {'tag': 'test', 'molecules': [{'gnn_atom_scores': [0.1, 0.9]}]}
    mod.write_alignment_record(record, result, figure)
    assert (result / 'alignment.json').read_bytes() == (figure / 'alignment.json').read_bytes()
    assert json.loads((result / 'alignment.json').read_text()) == record


def test_published_duplicate_records_match():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'results/interpretability/by_tag').glob('*/alignment.json'))
    assert len(paths) == 7
    for path in paths:
        other = root / 'figures/interpretability/by_tag' / path.parent.name / path.name
        assert path.read_bytes() == other.read_bytes(), path.parent.name


def test_default_output_is_tag_specific_and_keeps_legacy_records(
        load_experiment, tmp_path, monkeypatch):
    import sys
    mod = load_experiment('interpretability_alignment')
    monkeypatch.setattr(mod, 'BASE', str(tmp_path))
    legacy = tmp_path / 'results/interpretability/alignment.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('legacy record')
    for tag in ('a2a', 'herg'):
        monkeypatch.setattr(sys, 'argv', ['interpretability_alignment.py', '--tag', tag])
        args = mod.parse_args()
        assert Path(args.out) == tmp_path / 'results/interpretability/by_tag' / tag
        assert Path(args.fig_dir) == tmp_path / 'figures/interpretability/by_tag' / tag
        mod.write_alignment_record({'tag': tag}, args.out, args.fig_dir)
    assert legacy.read_text() == 'legacy record'
    for tag in ('a2a', 'herg'):
        path = tmp_path / 'results/interpretability/by_tag' / tag / 'alignment.json'
        assert json.loads(path.read_text()) == {'tag': tag}
