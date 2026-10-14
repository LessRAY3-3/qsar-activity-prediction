"""Audit actual release archives in an isolated scratch checkout, without training.

An optional --metadata JSON (GitHub release assets with browser_download_url)
lets an authenticated `gh api` discovery replace rate-limited anonymous API
lookup. Archive downloads, SHA verification and installation still use the real
fetcher. Reports include every archive member and the exact fetcher source hash.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args(argv)
    dest = args.work_dir.resolve()
    scratch = (ROOT / '.scratch').resolve()
    if not dest.is_relative_to(scratch) or dest == scratch:
        parser.error('--work-dir must be a dedicated directory below repo/.scratch')
    spec = importlib.util.spec_from_file_location('fetch', ROOT / 'scripts/fetch_artifacts.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source_hash = mod.sha256_file(ROOT / 'scripts/fetch_artifacts.py')
    (dest / 'data/raw').mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'data/raw/sha256sums.txt', dest / 'data/raw/sha256sums.txt')
    fetch = None
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
        urls = {a['name']: a['browser_download_url'] for a in metadata['assets']}
        def fetch(api_url):
            return urls
    mod.run(['models', 'raw', 'cache'], repo_root=dest, fetch_assets=fetch)
    bucket = dest / '.artifacts_cache/artifacts-v2'
    archives = []
    for name, _ in mod.ASSETS.values():
        path = bucket / name
        with tarfile.open(path) as archive:
            members = [{'name': m.name, 'size': m.size, 'type': m.type.decode('ascii')}
                       for m in archive.getmembers()]
        archives.append({'name': path.name, 'sha256': mod.sha256_file(path), 'members': members})
    report = {'status': 'passed', 'scope': 'actual release archive and isolated install audit; no training',
              'discovery': 'saved GitHub API metadata' if args.metadata else 'live GitHub API',
              'archives': archives, 'installed_models': len(list((dest / 'models').glob('*.joblib'))),
              'installed_time_models': len(list((dest / 'models').glob('rf_time*'))),
              'installed_caches': len(list((dest / 'data/processed').glob('*_fingerprints.npz'))),
              'raw_pins': len(mod.read_trust_anchor(dest)[0]), 'fetch_script_sha256': source_hash}
    assert len(archives) == 3 and report['installed_models'] == 18
    assert report['installed_time_models'] == 0 and report['installed_caches'] == 9
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'archives'}))


if __name__ == '__main__':
    main()
