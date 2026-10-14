"""Delivery integrity and fail-closed queue sequencing, without GPU training."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import types
import time
import sys

import pytest

import win4070_handoff as handoff


@pytest.fixture
def payload(tmp_path):
    (tmp_path / 'study').mkdir()
    (tmp_path / 'study/run.sh').write_text('#!/bin/sh\nexit 0\n')
    manifest = {'jobs': {'budget_comparison': 128, 'provenance_reconstruction': 18},
                'files': handoff.inventory(tmp_path)}
    handoff.json_write(tmp_path / 'delivery_manifest.json', manifest)
    return tmp_path


def test_hash_change_and_symlink_rejected(payload):
    assert handoff.verify(payload)['jobs']['budget_comparison'] == 128
    path = payload / 'study/run.sh'
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        handoff.verify(payload)
    path.unlink()
    other = payload / 'other.sh'
    other.write_text('#!/bin/sh\nexit 0\n')
    path.symlink_to(other)
    with pytest.raises(ValueError, match='symlink'):
        handoff.verify(payload)


def test_archive_roundtrip_and_existing_archive_refused(payload):
    import tarfile
    archive = payload.parent / (payload.name + '.tar.gz')
    result = handoff.regular_archive(payload, archive)
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == set(result['files'])
        assert all(p.isfile() for p in stream.getmembers())
    with pytest.raises(FileExistsError):
        handoff.regular_archive(payload, archive)


def test_queue_failure_does_not_start_next_stage(payload, monkeypatch):
    (payload / 'study/run.sh').write_text('#!/bin/sh\nexit 17\n')
    handoff.json_write(payload / 'delivery_manifest.json', {
        'jobs': {'budget_comparison': 128, 'provenance_reconstruction': 18},
        'files': {'study/run.sh': {'sha256': handoff.digest(payload / 'study/run.sh'),
                                'bytes': (payload / 'study/run.sh').stat().st_size}}})
    monkeypatch.setattr(handoff.shutil, 'disk_usage', lambda root: types.SimpleNamespace(free=20 * 1024**3))
    with pytest.raises(RuntimeError, match='budget_comparison failed'):
        handoff.execute(payload)
    state = json.loads((payload / 'queue_state.json').read_text())
    assert state['state'] == 'failed'
    assert len(state['stages']) == 1
    assert state['stages'][0]['exit_code'] == 17
    assert state['phase']['state'] == 'exited'
    assert state['phase']['child_pid'] == state['phase']['process_group']


def test_disk_shortage_never_launches(payload, monkeypatch):
    monkeypatch.setattr(handoff.shutil, 'disk_usage', lambda root: types.SimpleNamespace(free=100))
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='free space'):
        handoff.execute(payload)


def test_one_global_queue_lock(payload):
    with handoff.lock(payload):
        with pytest.raises(RuntimeError, match='Another full CUDA'):
            with handoff.lock(payload):
                pytest.fail('duplicate queue')


def test_no_detach_without_idle_inhibitor(payload, monkeypatch):
    monkeypatch.setattr(handoff.shutil, 'which', lambda name: None)
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='sleep policy'):
        handoff.detach(payload)


def refresh_manifest(root):
    handoff.json_write(root / 'delivery_manifest.json', {
        'jobs': {'budget_comparison': 128, 'provenance_reconstruction': 18},
        'files': {'study/run.sh': {'sha256': handoff.digest(root / 'study/run.sh'),
                                 'bytes': (root / 'study/run.sh').stat().st_size}}})


@pytest.mark.parametrize('kind', ['live_pid', 'unknown_pid', 'foreign_host', 'legacy_running',
                                  'frozen_lock', 'reconstruction_child', 'ambiguous_reconstruction'])
def test_stale_phase_blocks_before_any_cuda_smoke(payload, monkeypatch, kind):
    old = {'state': 'running', 'host': socket.gethostname(),
           'phase': {'state': 'running', 'child_pid': os.getpid(), 'process_group': os.getpgrp()}}
    if kind == 'unknown_pid':
        old['phase'].update(state='launching', child_pid=None, process_group=None)
    elif kind == 'foreign_host':
        old['host'] = 'different-machine'
    elif kind == 'legacy_running':
        old.pop('phase')
    elif kind == 'frozen_lock':
        old.update(state='failed', phase={'state': 'exited'})
        (payload / 'study/output').mkdir()
        (payload / 'study/output/.time_campaign_budget.lock').write_text('{}')
    elif kind in ('reconstruction_child', 'ambiguous_reconstruction'):
        old.update(state='failed', phase={'state': 'exited'})
        path = payload / 'supplement/output/job/attempt-test'
        path.mkdir(parents=True)
        handoff.json_write(path / 'status.json', {
            'status': 'running', 'child_pid': os.getpid() if kind == 'reconstruction_child' else None})
    handoff.json_write(payload / 'queue_state.json', old)
    monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must refuse before smoke'))
    monkeypatch.setattr(handoff.shutil, 'disk_usage', lambda root: types.SimpleNamespace(free=20 * 1024**3))
    with pytest.raises(RuntimeError):
        handoff.execute(payload)
    assert json.loads((payload / 'queue_state.json').read_text()) == old


def test_unknown_process_permission_is_not_treated_as_dead(monkeypatch):
    def denied(*args):
        raise PermissionError('not owned')
    monkeypatch.setattr(handoff.os, 'kill', denied)
    with pytest.raises(RuntimeError, match='Cannot inspect'):
        handoff.process_exists(12345)


def test_verified_dead_previous_group_can_resume_check(payload, monkeypatch):
    handoff.json_write(payload / 'queue_state.json', {
        'state': 'failed', 'host': socket.gethostname(),
        'phase': {'state': 'interrupted', 'child_pid': 12345, 'process_group': 12345}})
    def absent(*args):
        raise ProcessLookupError
    monkeypatch.setattr(handoff.os, 'kill', absent)
    monkeypatch.setattr(handoff.os, 'killpg', absent)
    handoff.require_previous_stopped(payload)


def start_dummy_supervisor(root):
    (root / 'study/run.sh').write_text(
        '#!/bin/bash\nsleep 90 &\nchild=$!\n'
        'trap \'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143\' TERM\n'
        'wait "$child"\n')
    refresh_manifest(root)
    code = (
        'import sys, types; from pathlib import Path; '
        'sys.path.insert(0, sys.argv[1]); import win4070_handoff as h; '
        'h.shutil.disk_usage=lambda root: types.SimpleNamespace(free=20*1024**3); '
        'h.execute(Path(sys.argv[2]))')
    proc = subprocess.Popen([sys.executable, '-c', code, str(Path(handoff.__file__).parent), str(root)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        path = root / 'queue_state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if (state.get('phase') or {}).get('state') == 'running':
            return proc, state['phase']['process_group']
        if proc.poll() is not None:
            raise AssertionError(proc.communicate())
        time.sleep(.02)
    proc.kill()
    proc.wait()
    raise AssertionError('Dummy supervisor did not publish child ownership')


def cleanup_dummy(proc, group):
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass


def test_sigterm_reaps_owned_phase_and_marks_failed(payload):
    proc, group = start_dummy_supervisor(payload)
    try:
        proc.terminate()
        proc.communicate(timeout=35)
        state = json.loads((payload / 'queue_state.json').read_text())
        assert state['state'] == 'failed'
        assert state['phase']['state'] == 'interrupted'
        assert state['phase']['cleanup']['group_still_present'] is False
        assert not handoff.process_exists(group, group=True)
        assert len(state['stages']) == 0
    finally:
        cleanup_dummy(proc, group)


def test_sigkill_survivor_refuses_resume_before_smoke(payload, monkeypatch):
    proc, group = start_dummy_supervisor(payload)
    try:
        proc.kill()
        proc.communicate(timeout=5)
        assert handoff.process_exists(group, group=True)
        monkeypatch.setattr(handoff.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not smoke'))
        monkeypatch.setattr(handoff.shutil, 'disk_usage', lambda root: types.SimpleNamespace(free=20 * 1024**3))
        with pytest.raises(RuntimeError, match='still alive'):
            handoff.execute(payload)
        assert json.loads((payload / 'queue_state.json').read_text())['state'] == 'running'
    finally:
        cleanup_dummy(proc, group)
