"""Run a disclosed 16-case replacement suite inferred from PR #8's descriptions.

This is a new validation experiment, not recovery of the unpublished original
16 tooling mutants. It combines 10 real-artifact acceptance mutations with six
label-rule mutations explicitly motivated by T4. Every patch, source hash,
selected test and result is saved. Production files and experiment data are
never changed. Infrastructure/test-setup errors are inconclusive, not kills.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import check_time_tooling_mutations as acceptance

LABEL_SOURCE = "scripts/gnn_08_time_splits.py"
VERSION_SOURCE = "scripts/time_labels.py"
LABEL_TEST = "tests/test_time_splits.py"
BOUNDARIES = "test_asof_exact_record_boundaries_median_and_units"
VERSION = "test_prediction_version_checks_rows_and_actual_labels"
LABEL_CASES = {
    "train_inclusive_cutoff": (LABEL_SOURCE,
        'acts["document_year"] < cutoff_v,', 'acts["document_year"] <= cutoff_v,',
        BOUNDARIES, "Not equal to tolerance"),
    "train_uses_test_cutoff": (LABEL_SOURCE,
        'acts["document_year"] < cutoff_v,', 'acts["document_year"] < cutoff_t,',
        BOUNDARIES, "Not equal to tolerance"),
    "valid_inclusive_cutoff": (LABEL_SOURCE,
        'acts["document_year"] < cutoff_t,', 'acts["document_year"] <= cutoff_t,',
        BOUNDARIES, "Not equal to tolerance"),
    "mean_instead_of_median": (LABEL_SOURCE,
        'acts.loc[condition].groupby("parent")["pic50"].median()',
        'acts.loc[condition].groupby("parent")["pic50"].mean()',
        BOUNDARIES, "Not equal to tolerance"),
    "accept_non_nm_units": (LABEL_SOURCE,
        '& acts["standard_units"].eq("nM")', '& True',
        BOUNDARIES, "assert np.isnan(y[3])"),
    "remove_prediction_version_guard": (VERSION_SOURCE,
        '    if str(predictions.get("label_id", "")) != expected:\n'
        '        raise ValueError("time prediction label version differs from split; retrain all time models")',
        '    # MUTATION: accept stale or absent prediction label_id',
        VERSION, "DID NOT RAISE"),
}


def mutate_label(source, case):
    _, before, after, _, _ = LABEL_CASES[case]
    if source.count(before) != 1:
        raise ValueError(f"{case}: expected exactly one source anchor; review source drift")
    return source.replace(before, after, 1)


def classify(returncode, junit, name, marker):
    """A kill must be the intended test's call-phase assertion, never setup failure."""
    tree = ET.parse(junit)
    cases = tree.findall('.//testcase')
    if (tree.findall('.//error') or tree.findall('.//skipped') or len(cases) != 1
            or cases[0].get('name') != name):
        return 'inconclusive'
    failures = tree.findall('.//failure')
    if returncode == 0 and not failures:
        return 'survived'
    if returncode == 1 and len(failures) == 1 and marker in (
            failures[0].get('message', '') + (failures[0].text or '')):
        return 'caught'
    return 'inconclusive'


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    # Refuse stale anchors before creating output or running any training.
    source = (root / acceptance.WORKER).read_text()
    mutations = {name: (acceptance.WORKER, acceptance.mutate(source, name))
                 for name in acceptance.CASES}
    mutations.update({name: (spec[0], mutate_label((root / spec[0]).read_text(), name))
                      for name, spec in LABEL_CASES.items()})
    output.mkdir(parents=True, exist_ok=False)
    patches = output / 'patches'
    patches.mkdir()
    patch_records = []
    for name, (relative, modified) in mutations.items():
        original = (root / relative).read_text()
        patch = ''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                             fromfile=f'a/{relative}', tofile=f'b/{relative}'))
        (patches / f'{name}.patch').write_text(patch)
        patch_records.append({'case': name, 'source': relative,
                              'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                              'modified_sha256': hashlib.sha256(modified.encode()).hexdigest(),
                              'patch_sha256': hashlib.sha256(patch.encode()).hexdigest()})
    base = acceptance.run(root, output / 'acceptance')
    rows = [{**row, 'family': 'acceptance'} for row in base['rows'] if row['case'] != 'baseline']
    baseline_tests = [f'{LABEL_TEST}::{BOUNDARIES}', f'{LABEL_TEST}::{VERSION}']
    for name in ['label_baseline', *LABEL_CASES]:
        with tempfile.TemporaryDirectory(prefix='qsar-label-mutant-') as tmp:
            work = Path(tmp)
            shutil.copytree(root / 'scripts', work / 'scripts',
                            ignore=shutil.ignore_patterns('__pycache__', 'campaign_history'))
            (work / 'tests').mkdir()
            for rel in [LABEL_TEST, 'tests/conftest.py', 'pyproject.toml']:
                shutil.copy2(root / rel, work / rel)
            selected = baseline_tests
            if name != 'label_baseline':
                relative, modified = mutations[name]
                (work / relative).write_text(modified)
                selected = [f'{LABEL_TEST}::{LABEL_CASES[name][3]}']
            junit = output / f'{name}.xml'
            with (output / f'{name}.log').open('w') as log:
                result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *selected,
                                         f'--junitxml={junit}'], cwd=work, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if name == 'label_baseline':
                baseline_ok = result.returncode == 0
                if not baseline_ok:
                    break
                continue
            status = classify(result.returncode, junit, *LABEL_CASES[name][3:])
            rows.append({'case': name, 'family': 'labels', 'status': status,
                         'returncode': result.returncode, 'test': selected[0]})
            print(f'{name}: {status}', flush=True)
    report = {'schema': 'inferred-review-replacement-v1',
              'scope': 'New 16-case suite (10 acceptance + 6 T4 label rules); NOT original 16 tooling mutants',
              'selection_basis': ['PR8 T4 described label/version mutations',
                                  'PR8 Low described mocked acceptance/checkpoint checks'],
              'assumptions': ['No original patches or verifier notes available.',
                              'Count 16 is a disclosed replacement scope, not evidence of historical identity.',
                              'Targeted test kills do not constitute exhaustive mutation coverage.'],
              'baseline_acceptance_passed': base['rows'][0]['status'] == 'passed',
              'baseline_labels_passed': baseline_ok, 'rows': rows, 'patches': patch_records,
              'source_hashes': {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                                for rel in [acceptance.WORKER, LABEL_SOURCE, VERSION_SOURCE,
                                            acceptance.TEST, LABEL_TEST,
                                            'scripts/check_time_tooling_mutations.py',
                                            'scripts/reconstruct_review_mutations.py']},
              'all_caught': base['all_caught'] and baseline_ok and len(rows) == 16
                            and all(row['status'] == 'caught' for row in rows)}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.root, args.out_dir)['all_caught'] else 1)


if __name__ == '__main__':
    main()
