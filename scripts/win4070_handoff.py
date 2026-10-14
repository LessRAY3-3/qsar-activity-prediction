"""Prepare/check or explicitly launch the two independent CUDA queues.

Preparation/checks never connect to a host or train. The portable handoff keeps
frozen 128-job training inputs separate from the 18-job reconstruction source.
Use --execute only after connecting the CUDA host; --detach survives SSH exit.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe manifest path: {name}')
    path = root / relative
    if path.resolve() != root.resolve() / relative or not path.is_file():
        raise ValueError(f'Missing/nonregular/symlink file: {name}')
    return path


def inventory(root):
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink not allowed: {p}')
        if p.is_dir():
            continue
        if not p.is_file():
            raise ValueError(f'Nonregular file: {p}')
        if '__pycache__' in p.parts:
            continue
        files[p.relative_to(root).as_posix()] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    return files


def verify(root):
    manifest = json.loads((root / 'delivery_manifest.json').read_text())
    if manifest['jobs'] != {'budget_comparison': 128, 'provenance_reconstruction': 18}:
        raise ValueError('Unexpected queue cohort')
    for name, expected in manifest['files'].items():
        p = safe_file(root, name)
        if p.stat().st_size != expected['bytes'] or digest(p) != expected['sha256']:
            raise ValueError(f'Delivery changed: {name}')
    return manifest


def regular_archive(root, destination):
    """Archive regular files only; preserve a full content inventory beside it."""
    files = inventory(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for name in files:
                p = root / name
                info = tarfile.TarInfo(name)
                info.size = p.stat().st_size
                info.mode = 0o644
                with p.open('rb') as content:
                    archive.addfile(info, content)
    result = {'sha256': digest(destination), 'bytes': destination.stat().st_size, 'files': files}
    json_write(Path(str(destination) + '.manifest.json'), result)
    return result


def prepare(repo, output):
    if output.exists() or output.is_symlink():
        raise ValueError('Preparation requires a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.win4070-prepare-', dir=output.parent))
    try:
        prepare_study = module(repo / 'scripts/prepare_cuda_study.py', 'prepare_study')
        prepare_study.prepare(temporary / 'study')
        plan_path = repo / 'results/provenance_reconstruction/plan.json'
        plan = json.loads(plan_path.read_text())
        target = temporary / 'supplement/source'
        source_files = {**plan['source_sha256'], **plan['input_sha256'], **plan['historical_metrics_sha256']}
        source_files.update({name: digest(repo / name) for name in
                             ['results/provenance_reconstruction/plan.json', 'requirements.txt', 'environment.yml']})
        for name, expected in source_files.items():
            src = safe_file(repo, name)
            if digest(src) != expected:
                raise ValueError(f'Reconstruction source/input no longer matches plan: {name}')
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        (temporary / 'tools').mkdir()
        shutil.copy2(repo / 'scripts/run_provenance_reconstruction.py', temporary / 'tools/run_provenance_reconstruction.py')
        shutil.copy2(Path(__file__), temporary / 'handoff.py')
        commands = {'ssh_alias': 'win4070',
                    'suggested_remote_root': '/home/lessray/qsar-campaigns/cuda-ready-20261014',
                    'historical_python_candidate': '/home/lessray/qsar-campaigns/time-asof-20261013-7ce15c1-gin/.venv/bin/python',
                    'check_on_cuda_host': ['<verified-python>', 'handoff.py', '--root', '.', '--check'],
                    'launch_detached_after_user_connects': ['<verified-python>', 'handoff.py', '--root', '.', '--execute', '--detach'],
                    'resume_after_inspecting_failure': ['<verified-python>', 'handoff.py', '--root', '.', '--execute', '--detach'],
                    'return_files': ['study/returns/*/study.tar.gz', 'study/returns/*/study.tar.gz.manifest.json',
                                     'returns/provenance-*.tar.gz', 'returns/provenance-*.tar.gz.manifest.json',
                                     'queue_state.json', 'logs/', 'study/logs/'],
                    'local_acceptance': [
                        'Verify returned archive/member SHA256 before safe extraction to a new directory.',
                        'python experiments/time_budget_report.py --study-root <returned-study> --input-root <delivery>/study/input --seeds 42 1 2 3 4 5 6 7 --training-device cuda --threads 2 --out-dir <new-cuda-report>',
                        'python experiments/time_environment_comparison.py --cpu-report results/time_budget_sensitivity/m3_eight_seed/analysis --cuda-report <new-cuda-report> --out-dir <new-environment-report>',
                        'python experiments/provenance_reconstruction_report.py --run-dir <returned-provenance> --plan <delivery>/supplement/source/results/provenance_reconstruction/plan.json --runner <delivery>/tools/run_provenance_reconstruction.py --out-dir <new-provenance-report>'],
                    'policy': 'No SSH performed by this tool. No auto dependency/driver upgrades. Frozen worker stale locks require inspection; never delete them automatically.'}
        json_write(temporary / 'commands.json', commands)
        # Paths are illustrative until a real target directory is selected.
        json_write(temporary / 'delivery_manifest.json', {
            'schema': 'qsar-win4070-delivery-v1', 'created_utc': now(),
            'jobs': {'budget_comparison': 128, 'provenance_reconstruction': 18},
            'order': ['budget_comparison', 'provenance_reconstruction'],
            'files': inventory(temporary)})
        verify(temporary)
        result = subprocess.run([sys.executable, str(temporary / 'handoff.py'), '--root', str(temporary), '--check'],
                                text=True, capture_output=True, check=True)
        # This is mutable operation evidence, not part of immutable delivery sources.
        json_write(temporary / 'offline_check.json', {'stdout': result.stdout, 'stderr': result.stderr,
                                                     'returncode': result.returncode, 'no_training': True})
        temporary.rename(output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {'state': 'prepared_not_connected_or_trained', 'directory': str(output), 'jobs': 146,
            'delivery_manifest_sha256': digest(output / 'delivery_manifest.json')}


def check(root):
    manifest = verify(root)
    for command in ([sys.executable, str(root / 'study/preflight.py'), '--offline'],
                    [sys.executable, str(root / 'tools/run_provenance_reconstruction.py'),
                     '--root', str(root / 'supplement/source'), '--run-dir', str(root / 'supplement/output'),
                     '--validate-only']):
        subprocess.run(command, check=True)
    print(json.dumps({'state': 'offline_ready', 'jobs': manifest['jobs'], 'no_cuda_or_training': True}))


@contextmanager
def lock(root):
    with (root / '.queue.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another full CUDA queue is running') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def package_supplement(root):
    runner = module(root / 'tools/run_provenance_reconstruction.py', 'reconstruction_runner')
    plan = json.loads((root / 'supplement/source/results/provenance_reconstruction/plan.json').read_text())
    run_dir = root / 'supplement/output'
    expected_identity = {'plan_sha256': digest(root / 'supplement/source/results/provenance_reconstruction/plan.json'),
                         'runner_sha256': digest(root / 'tools/run_provenance_reconstruction.py'),
                         'threads': 4, 'backend': 'cuda'}
    # Do not trust only the aggregate count: verify every completed receipt again.
    with runner.queue_lock(run_dir):
        for job in plan['jobs']:
            if not runner.resume_valid(run_dir / job['id'], expected_identity):
                raise ValueError('Incomplete reconstruction queue')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        return regular_archive(run_dir, root / 'returns' / f'provenance-{stamp}.tar.gz')


def process_exists(pid, group=False):
    """Absence is the only safe automatic recovery decision."""
    if type(pid) is not int or pid < 2:
        raise RuntimeError('Missing or invalid prior process identity; inspect before resume')
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError('Cannot inspect prior process; refuse automatic resume') from exc
    return True


def require_previous_stopped(root):
    """Run under the queue lock, before any CUDA smoke or new supervisor state."""
    path = root / 'queue_state.json'
    if path.exists():
        previous = json.loads(path.read_text())
        phase = previous.get('phase')
        if phase and phase.get('state') != 'exited':
            if previous.get('host') != socket.gethostname():
                raise RuntimeError('Prior phase belongs to an unknown/foreign host; inspect before resume')
            if phase.get('state') == 'launching' or not phase.get('child_pid') or not phase.get('process_group'):
                raise RuntimeError('Prior phase has unknown launch ownership; inspect before resume')
            if process_exists(phase['child_pid']) or process_exists(phase['process_group'], group=True):
                raise RuntimeError('Prior phase process/group is still alive; refuse resume before CUDA smoke')
        elif previous.get('state') == 'running' and not phase:
            raise RuntimeError('Previous running supervisor has no phase identity; inspect before resume')
    locks = list((root / 'study/output').rglob('.time_campaign_*.lock'))
    if locks:
        raise RuntimeError(f'Frozen study locks remain; inspect owners before resume: {locks}')
    # The reconstruction runner deliberately gives trainers separate sessions;
    # a killed runner may disappear while one of those CUDA trainers survives.
    for path in (root / 'supplement/output').glob('*/attempt-*/status.json'):
        status = json.loads(path.read_text())
        if status.get('status') != 'running':
            continue
        if process_exists(status.get('child_pid')) or process_exists(status['child_pid'], group=True):
            raise RuntimeError(f'Prior reconstruction trainer is still alive: {path}')


def terminate_phase(child):
    """Stop the owned process group; let the reconstruction runner reap its child."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Reconstruction has a 15-second grace period for its separate trainer group.
    try:
        child.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    # A coordinator may exit before its descendants. Never leave its group live.
    deadline = time.monotonic() + 2
    while process_exists(child.pid, group=True) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_exists(child.pid, group=True):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {'exit_code': child.returncode,
            'group_still_present': process_exists(child.pid, group=True)}


def run_phase(root, state, name, command, env):
    state['current_stage'] = name
    state['phase'] = {'name': name, 'state': 'launching', 'command': command,
                      'child_pid': None, 'process_group': None}
    json_write(root / 'queue_state.json', state)
    child = None
    try:
        with (root / 'logs' / f'{name}.log').open('a') as log:
            print(f'{now()} starting {name}', flush=True)
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state['phase'].update(state='running', child_pid=child.pid, process_group=child.pid)
            json_write(root / 'queue_state.json', state)
            returncode = child.wait()
        if process_exists(child.pid, group=True):
            raise RuntimeError('Phase leader exited with surviving descendants; stopping its process group')
        state['phase'].update(state='exited', exit_code=returncode, end_utc=now())
        state['stages'].append({'name': name, 'exit_code': returncode, 'end_utc': now()})
        json_write(root / 'queue_state.json', state)
        if returncode:
            raise RuntimeError(f'{name} failed ({returncode}); inspect logs before resume')
    except BaseException:
        if child is not None and state['phase'].get('state') != 'exited':
            # A second ordinary stop request must not interrupt child cleanup.
            previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                cleanup = terminate_phase(child)
                state['phase'].update(state='interrupted', cleanup=cleanup, end_utc=now())
            finally:
                signal.signal(signal.SIGTERM, previous)
                json_write(root / 'queue_state.json', state)
        elif child is None:
            # Popen failed before returning ownership. Ambiguous interruptions
            # retain launching; ordinary launch failures are safe to retry.
            error = sys.exc_info()[1]
            if isinstance(error, OSError):
                state['phase'].update(state='exited', launch_error=str(error), end_utc=now())
        raise


def execute(root):
    verify(root)
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise RuntimeError('Require 10 GiB free space for isolated jobs, logs and return packages')
    (root / 'logs').mkdir(exist_ok=True)
    with lock(root):
        require_previous_stopped(root)
        state = {'state': 'running', 'start_utc': now(), 'pid': os.getpid(),
                 'host': socket.gethostname(), 'phase': None,
                 'delivery_manifest_sha256': digest(root / 'delivery_manifest.json'), 'stages': []}
        json_write(root / 'queue_state.json', state)
        env = {**os.environ, 'QSAR_PYTHON': sys.executable, 'CUDA_VISIBLE_DEVICES': '0',
               'PYTHONUNBUFFERED': '1'}
        phases = [('budget_comparison', ['bash', str(root / 'study/run.sh')]),
                  ('package_budget', ['bash', str(root / 'study/package_return.sh')]),
                  ('provenance_reconstruction', [sys.executable, str(root / 'tools/run_provenance_reconstruction.py'),
                    '--root', str(root / 'supplement/source'), '--run-dir', str(root / 'supplement/output')])]
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            for name, command in phases:
                verify(root)
                run_phase(root, state, name, command, env)
            state['current_stage'] = 'package_provenance'
            package_supplement(root)
            state['state'] = 'completed_pending_independent_replay'
        except BaseException as exc:
            state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            state['end_utc'] = now()
            json_write(root / 'queue_state.json', state)
            signal.signal(signal.SIGTERM, old_handler)


def detach(root):
    verify(root)
    inhibitor = shutil.which('systemd-inhibit')
    if not inhibitor:
        raise RuntimeError('systemd-inhibit unavailable; inspect sleep policy before unattended launch')
    prefix = [inhibitor, '--what=idle:sleep', '--mode=block', '--who=qsar-146', '--why=QSAR CUDA experiment queues']
    subprocess.run([*prefix, 'true'], check=True)
    (root / 'logs').mkdir(exist_ok=True)
    command = [*prefix, sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--execute']
    with (root / 'logs/supervisor.log').open('a') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, cwd=root)
    print(json.dumps({'state': 'dispatched_not_yet_validated', 'pid': child.pid,
                      'command': command, 'status': str(root / 'queue_state.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.root or args.execute or args.detach or args.check:
            parser.error('--prepare is separate from operation flags')
        output = args.prepare.absolute()
        result = prepare(Path(__file__).resolve().parents[1], output)
        archive = Path(str(output) + '.tar.gz')
        packed = regular_archive(output, archive)
        result.update(archive=str(archive), archive_sha256=packed['sha256'], archive_bytes=packed['bytes'])
        print(json.dumps(result, indent=2))
    else:
        if args.root is None or (args.detach and not args.execute) or (args.check and args.execute):
            parser.error('--root required; --detach requires --execute; --check cannot execute')
        root = args.root.resolve()
        if args.execute:
            detach(root) if args.detach else execute(root)
        else:
            check(root)


if __name__ == '__main__':
    main()
