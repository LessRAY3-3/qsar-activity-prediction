"""The replacement suite must not count broken fixtures as caught mutations."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import reconstruct_review_mutations as audit


def report(tmp_path, *, failure=None, error=None, skipped=False, name='intended'):
    root = ET.Element('testsuite')
    case = ET.SubElement(root, 'testcase', name=name)
    if failure is not None:
        ET.SubElement(case, 'failure', message=failure)
    if error is not None:
        ET.SubElement(case, 'error', message=error)
    if skipped:
        ET.SubElement(case, 'skipped')
    path = tmp_path / 'test.xml'
    ET.ElementTree(root).write(path)
    return path


@pytest.mark.parametrize('kwargs,code', [
    ({'error': 'DID NOT RAISE'}, 1),
    ({'failure': 'ModuleNotFoundError'}, 1),
    ({'skipped': True}, 0),
    ({'failure': 'DID NOT RAISE', 'name': 'unrelated'}, 1),
    ({'failure': 'DID NOT RAISE'}, 2),
])
def test_infrastructure_wrong_test_or_skip_is_not_a_kill(tmp_path, kwargs, code):
    assert audit.classify(code, report(tmp_path, **kwargs), 'intended', 'DID NOT RAISE') == 'inconclusive'


def test_only_expected_assertion_is_caught(tmp_path):
    path = report(tmp_path, failure='DID NOT RAISE <class ValueError>')
    assert audit.classify(1, path, 'intended', 'DID NOT RAISE') == 'caught'
    assert audit.classify(0, report(tmp_path), 'intended', 'DID NOT RAISE') == 'survived'


def test_source_drift_refuses_missing_or_ambiguous_anchors():
    source = (Path(__file__).resolve().parents[1] / audit.LABEL_SOURCE).read_text()
    for name, (relative, before, *_) in audit.LABEL_CASES.items():
        if relative != audit.LABEL_SOURCE:
            continue
        with pytest.raises(ValueError, match='source drift'):
            audit.mutate_label(source.replace(before, 'changed'), name)
        with pytest.raises(ValueError, match='source drift'):
            audit.mutate_label(source + '\n' + before, name)
