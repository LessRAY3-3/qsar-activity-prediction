"""One-off launch after the broad inhibitor was denied; no power policy edits."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

root = Path('/home/lessray/qsar-campaigns/cuda-ready-20261014')
get = lambda key: subprocess.check_output(['gsettings', 'get', 'org.gnome.settings-daemon.plugins.power', key], text=True).strip()
policy = {key: get(key) for key in ('sleep-inactive-ac-type', 'sleep-inactive-ac-timeout',
                                   'sleep-inactive-battery-type', 'sleep-inactive-battery-timeout')}
power = {str(p): p.read_text().strip() for p in Path('/sys/class/power_supply').glob('*/online')}
if policy['sleep-inactive-ac-timeout'] != '0' or '1' not in power.values():
    raise SystemExit('Require AC online and existing automatic AC suspend timeout=0; no settings changed')
prefix = ['systemd-inhibit', '--what=idle', '--mode=block', '--who=qsar-146',
          '--why=QSAR CUDA queues; existing AC automatic suspend disabled']
subprocess.run([*prefix, 'true'], check=True)
command = [*prefix, sys.executable, '-u', str(root/'handoff.py'), '--root', str(root), '--execute']
(root/'logs').mkdir(exist_ok=True)
with (root/'logs/supervisor.log').open('a') as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=root, env={**os.environ, 'PYTHONUNBUFFERED':'1'})
record = {'state':'dispatched_not_yet_validated', 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'pid':process.pid, 'command':command, 'existing_power_policy':policy, 'power_online':power,
          'global_settings_changed':False, 'inhibitor':'idle only; idle:sleep request denied by host policy',
          'limitation':'Requires remaining on AC; does not block explicit suspend',
          'handoff_sha256':hashlib.sha256((root/'handoff.py').read_bytes()).hexdigest()}
(root/'logs/launch_power_policy.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
