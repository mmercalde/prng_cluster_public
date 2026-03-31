#!/usr/bin/env python3
"""
apply_s158d_b_systemd_run_launcher.py  v2

Team Beta S158D-B: Replace nohup SSH launch with systemd-run --user.

Root cause of worker launch failure:
  Bash disables job-control in non-interactive SSH shells.
  disown is a job-control builtin -- unreliable in this context.
  nohup only ignores SIGHUP, it is not a service manager.

Fix:
  systemd-run --user creates a transient user service managed by systemd.
  Workers survive SSH session teardown reliably.
  loginctl enable-linger keeps user manager alive after logout.

Also includes S158D-E: Zeus CUDA workers get isolated env at Popen.

Prerequisites (run once on each rig):
  ssh rrig6600  'sudo loginctl enable-linger michael'
  ssh rrig6600b 'sudo loginctl enable-linger michael'
  ssh rrig6600c 'sudo loginctl enable-linger michael'

Deploy:
  scp apply_s158d_b_systemd_run_launcher.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s158d_b_systemd_run_launcher.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/zmq_sqlite_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s158d_b")


NEW_METHOD = (
    "    def _launch_workers(self):\n"
    "        \"\"\"\n"
    "        SSH to each rig ONCE. Launch workers as systemd-run --user transient services.\n"
    "        Workers survive SSH session teardown. systemd owns the process, not the shell.\n"
    "        Zeus local workers launched via subprocess.Popen with isolated env per GPU.\n"
    "\n"
    "        Prerequisite (one-time per rig):\n"
    "            sudo loginctl enable-linger michael\n"
    "        \"\"\"\n"
    "        if self._workers_launched:\n"
    "            return\n"
    "\n"
    "        import subprocess\n"
    "        import shlex\n"
    "\n"
    "        def _abs(p, username):\n"
    "            return '/home/' + username + '/' + p[2:] if p.startswith('~/') else p\n"
    "\n"
    "        for node in self._nodes:\n"
    "            host      = node.get('hostname', '')\n"
    "            username  = node.get('username', 'michael')\n"
    "            gpu_count = node.get('gpu_count', 0)\n"
    "            if not host or host in ('localhost', '127.0.0.1') or gpu_count == 0:\n"
    "                continue\n"
    "\n"
    "            py_env      = _abs(node.get('python_env',  '~/rocm_env/bin/python3'),    username)\n"
    "            script_path = _abs(node.get('script_path', '~/distributed_prng_analysis'), username)\n"
    "            worker_script = script_path + '/zmq_sqlite_worker.py'\n"
    "\n"
    "            for gpu_id in range(gpu_count):\n"
    "                worker_id = host + ':gpu' + str(gpu_id)\n"
    "                unit      = 'zmq-worker-gpu' + str(gpu_id)\n"
    "                log_path  = '/tmp/zmq_worker_gpu' + str(gpu_id) + '.log'\n"
    "\n"
    "                worker_cmd = ' '.join([\n"
    "                    'cd', shlex.quote(script_path), '&&',\n"
    "                    'exec', shlex.quote(py_env), '-u', shlex.quote(worker_script),\n"
    "                    '--zeus-host', shlex.quote(self._zeus_ip),\n"
    "                    '--job-port', str(self.zmq_job_port),\n"
    "                    '--result-port', str(self.zmq_result_port),\n"
    "                    '--worker-id', shlex.quote(worker_id),\n"
    "                    '--gpu-id', str(gpu_id),\n"
    "                    '>>' + shlex.quote(log_path), '2>&1',\n"
    "                ])\n"
    "\n"
    "                remote_lines = [\n"
    "                    'set -e',\n"
    "                    'linger=$(loginctl show-user \"$USER\" -p Linger --value 2>/dev/null || echo no)',\n"
    "                    'if [ \"$linger\" != yes ]; then',\n"
    "                    '  echo \"ERROR: linger not enabled -- run: sudo loginctl enable-linger $USER\" >&2',\n"
    "                    '  exit 42',\n"
    "                    'fi',\n"
    "                    'systemctl --user stop ' + shlex.quote(unit) + ' >/dev/null 2>&1 || true',\n"
    "                    'systemctl --user reset-failed ' + shlex.quote(unit) + ' >/dev/null 2>&1 || true',\n"
    "                    ('systemd-run --user'\n"
    "                     + ' --unit=' + shlex.quote(unit)\n"
    "                     + ' --collect'\n"
    "                     + ' --property=Type=exec'\n"
    "                     + ' --property=Restart=always'\n"
    "                     + ' --property=RestartSec=2'\n"
    "                     + ' --setenv=ROCR_VISIBLE_DEVICES=' + str(gpu_id)\n"
    "                     + ' --setenv=CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_' + str(gpu_id)\n"
    "                     + ' --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0'\n"
    "                     + ' /bin/bash -lc ' + shlex.quote(worker_cmd)),\n"
    "                    'sleep 1',\n"
    "                    'systemctl --user is-active --quiet ' + shlex.quote(unit),\n"
    "                ]\n"
    "                remote_script = '\\n'.join(remote_lines)\n"
    "\n"
    "                try:\n"
    "                    proc = subprocess.run(\n"
    "                        ['ssh', '-q',\n"
    "                         '-o', 'StrictHostKeyChecking=no',\n"
    "                         '-o', 'BatchMode=yes',\n"
    "                         '-o', 'ConnectTimeout=10',\n"
    "                         username + '@' + host,\n"
    "                         'bash', '-lc', remote_script],\n"
    "                        capture_output=True, text=True, timeout=30,\n"
    "                    )\n"
    "                    if proc.returncode == 0:\n"
    "                        self.logger.info('[ZMQ] systemd-run worker active: ' + host + ' gpu' + str(gpu_id))\n"
    "                    elif proc.returncode == 42:\n"
    "                        self.logger.error('[ZMQ] linger not enabled on ' + host)\n"
    "                    else:\n"
    "                        self.logger.error(\n"
    "                            '[ZMQ] systemd-run failed on ' + host + ' gpu' + str(gpu_id) +\n"
    "                            ' rc=' + str(proc.returncode) +\n"
    "                            ' stderr=' + proc.stderr.strip()[:200]\n"
    "                        )\n"
    "                except subprocess.TimeoutExpired:\n"
    "                    self.logger.error('[ZMQ] SSH timeout on ' + host + ' gpu' + str(gpu_id))\n"
    "                except Exception as e:\n"
    "                    self.logger.error('[ZMQ] Launch failed ' + host + ' gpu' + str(gpu_id) + ': ' + str(e))\n"
    "\n"
    "        # Zeus local CUDA workers -- isolated env per GPU (S158D-E)\n"
    "        import subprocess as sp\n"
    "        import os as _os\n"
    "        for gpu_id in range(2):\n"
    "            worker_id = 'localhost:gpu' + str(gpu_id)\n"
    "            env = _os.environ.copy()\n"
    "            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)\n"
    "            env['CUPY_CACHE_DIR']       = '/tmp/cupy_cache_zeus_gpu' + str(gpu_id)\n"
    "            env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')\n"
    "            try:\n"
    "                sp.Popen(\n"
    "                    ['python3', 'zmq_sqlite_worker.py',\n"
    "                     '--zeus-host',   'localhost',\n"
    "                     '--job-port',    str(self.zmq_job_port),\n"
    "                     '--result-port', str(self.zmq_result_port),\n"
    "                     '--worker-id',   worker_id,\n"
    "                     '--gpu-id',      '0',\n"
    "                     '--cuda'],\n"
    "                    env=env,\n"
    "                    stdout=open('/tmp/zmq_zeus_gpu' + str(gpu_id) + '.log', 'w'),\n"
    "                    stderr=sp.STDOUT,\n"
    "                )\n"
    "                self.logger.info(\n"
    "                    '[ZMQ] Zeus CUDA worker launched (' + worker_id +\n"
    "                    ' CUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' logical_gpu=0)'\n"
    "                )\n"
    "            except Exception as e:\n"
    "                self.logger.error('[ZMQ] Zeus GPU' + str(gpu_id) + ' launch failed: ' + str(e))\n"
    "\n"
    "        time.sleep(WORKER_SETTLE_S)\n"
    "        self._workers_launched = True\n"
    "        self.logger.info('[ZMQ] All workers launched and settled')\n"
)


def apply():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        return False

    content = TARGET.read_text()

    start = content.find("    def _launch_workers(self):")
    if start < 0:
        print("ERROR: _launch_workers() not found")
        return False

    end = content.find("\n    def run_sieve_pass", start)
    if end < 0:
        print("ERROR: run_sieve_pass not found after _launch_workers()")
        return False

    new_content = content[:start] + NEW_METHOD + "\n" + content[end + 1:]

    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"ERROR: syntax error at line {e.lineno}: {e.msg}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(new_content)
    ast.parse(TARGET.read_text())

    assert "systemd-run" in new_content
    assert "nohup" not in new_content.split("def _launch_workers")[1].split("def run_sieve_pass")[0]

    print("SUCCESS: S158D-B v2 systemd-run --user launcher applied")
    print("SUCCESS: S158D-E Zeus CUDA isolation included")
    print()
    print("Prerequisites (one-time on each rig):")
    print("  ssh rrig6600  'sudo loginctl enable-linger michael'")
    print("  ssh rrig6600b 'sudo loginctl enable-linger michael'")
    print("  ssh rrig6600c 'sudo loginctl enable-linger michael'")
    print()
    print("Git commit:")
    print("  git add zmq_sqlite_coordinator.py apply_s158d_b_systemd_run_launcher.py")
    print("  git commit -m 'fix(s158d-b): systemd-run --user launcher + Zeus CUDA isolation'")
    print("  git push origin main && git push public main")
    return True


if __name__ == "__main__":
    print("Applying S158D-B v2: systemd-run --user worker launcher...")
    apply()
