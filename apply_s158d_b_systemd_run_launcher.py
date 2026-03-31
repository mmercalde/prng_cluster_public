#!/usr/bin/env python3
"""
apply_s158d_b_systemd_run_launcher.py

Team Beta S158D-B: Replace nohup SSH launch with systemd-run --user.

Addresses: rig workers not surviving SSH session teardown from Python subprocess.
Root cause: Bash job-control disabled in non-interactive SSH shells. nohup only
ignores SIGHUP — it is not a service manager.

Fix: systemd-run --user creates a transient user service with proper cgroup
ownership. Workers survive SSH session teardown reliably.

Prerequisites (run once on each rig):
  ssh rrig6600  'sudo loginctl enable-linger michael'
  ssh rrig6600b 'sudo loginctl enable-linger michael'
  ssh rrig6600c 'sudo loginctl enable-linger michael'

Verify:
  ssh rrig6600 'loginctl show-user michael -p Linger --value'
  → should print: yes

Zeus local workers unchanged — they use subprocess.Popen directly.

Deploy:
  scp ~/Downloads/apply_s158d_b_systemd_run_launcher.py rzeus:~/distributed_prng_analysis/
  ssh rzeus 'cd ~/distributed_prng_analysis && python3 apply_s158d_b_systemd_run_launcher.py'
"""

import ast
import shutil
from pathlib import Path

TARGET = Path("/home/michael/distributed_prng_analysis/zmq_sqlite_coordinator.py")
BACKUP = TARGET.with_suffix(".py.bak_s158d_b")

NEW_LAUNCH_WORKERS = '''    def _launch_workers(self):
        """
        SSH to each rig ONCE to launch workers as transient systemd user services.
        Workers survive SSH session teardown because systemd owns the process, not
        the SSH shell. Uses loginctl linger so user services persist after logout.

        Zeus local workers launched directly via subprocess.Popen (unchanged).
        """
        if self._workers_launched:
            return

        import subprocess
        import shlex

        def _abs(p: str, username: str) -> str:
            return f"/home/{username}/" + p[2:] if p.startswith("~/") else p

        for node in self._nodes:
            host      = node.get("hostname", "")
            username  = node.get("username", "michael")
            gpu_count = node.get("gpu_count", 0)
            if not host or host in ("localhost", "127.0.0.1") or gpu_count == 0:
                continue

            py_env      = _abs(node.get("python_env",  "~/rocm_env/bin/python3"),    username)
            script_path = _abs(node.get("script_path", "~/distributed_prng_analysis"), username)
            worker_script = f"{script_path}/zmq_sqlite_worker.py"

            for gpu_id in range(gpu_count):
                worker_id = f"{host}:gpu{gpu_id}"
                unit      = f"zmq-worker-gpu{gpu_id}"
                log_path  = f"/tmp/zmq_worker_gpu{gpu_id}.log"

                # Command the service will run inside the rig
                worker_cmd = (
                    f"cd {shlex.quote(script_path)} && "
                    f"exec {shlex.quote(py_env)} -u {shlex.quote(worker_script)} "
                    f"--zeus-host {shlex.quote(self._zeus_ip)} "
                    f"--job-port {self.zmq_job_port} "
                    f"--result-port {self.zmq_result_port} "
                    f"--worker-id {shlex.quote(worker_id)} "
                    f"--gpu-id {gpu_id} "
                    f">>{shlex.quote(log_path)} 2>&1"
                )

                # Remote bash script: check linger, stop stale unit, start fresh
                remote_script = (
                    f"set -e\n"
                    f"linger=$(loginctl show-user \"$USER\" -p Linger --value 2>/dev/null || echo no)\n"
                    f"if [ \"$linger\" != yes ]; then\n"
                    f"  echo 'ERROR: linger disabled — run: sudo loginctl enable-linger $USER' >&2\n"
                    f"  exit 42\n"
                    f"fi\n"
                    f"systemctl --user stop {shlex.quote(unit)} >/dev/null 2>&1 || true\n"
                    f"systemctl --user reset-failed {shlex.quote(unit)} >/dev/null 2>&1 || true\n"
                    f"systemd-run --user \\\n"
                    f"  --unit={shlex.quote(unit)} \\\n"
                    f"  --collect \\\n"
                    f"  --property=Type=exec \\\n"
                    f"  --property=Restart=always \\\n"
                    f"  --property=RestartSec=2 \\\n"
                    f"  --setenv=ROCR_VISIBLE_DEVICES={gpu_id} \\\n"
                    f"  --setenv=CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id} \\\n"
                    f"  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 \\\n"
                    f"  /bin/bash -lc {shlex.quote(worker_cmd)}\n"
                    f"sleep 1\n"
                    f"systemctl --user is-active --quiet {shlex.quote(unit)}\n"
                )

                try:
                    proc = subprocess.run(
                        ["ssh", "-q",
                         "-o", "StrictHostKeyChecking=no",
                         "-o", "BatchMode=yes",
                         "-o", "ConnectTimeout=10",
                         f"{username}@{host}",
                         "bash", "-lc", remote_script],
                        capture_output=True, text=True, timeout=30,
                    )
                    if proc.returncode == 0:
                        self.logger.info(
                            f"[ZMQ] systemd-run worker active: {host} gpu{gpu_id}"
                        )
                    elif proc.returncode == 42:
                        self.logger.error(
                            f"[ZMQ] linger not enabled on {host} — "
                            f"run: ssh {host} 'sudo loginctl enable-linger {username}'"
                        )
                    else:
                        self.logger.error(
                            f"[ZMQ] systemd-run failed on {host} gpu{gpu_id}: "
                            f"rc={proc.returncode} stderr={proc.stderr.strip()[:200]}"
                        )
                except subprocess.TimeoutExpired:
                    self.logger.error(
                        f"[ZMQ] SSH timeout launching worker on {host} gpu{gpu_id}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[ZMQ] Failed to launch worker on {host} gpu{gpu_id}: {e}"
                    )

        # Zeus local CUDA workers — subprocess.Popen with isolated env per GPU
        import subprocess as sp
        import os as _os
        for gpu_id in range(2):
            worker_id = f"localhost:gpu{gpu_id}"
            env = _os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["CUPY_CACHE_DIR"]       = f"/tmp/cupy_cache_zeus_gpu{gpu_id}"
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            try:
                sp.Popen(
                    ["python3", "zmq_sqlite_worker.py",
                     "--zeus-host",   "localhost",
                     "--job-port",    str(self.zmq_job_port),
                     "--result-port", str(self.zmq_result_port),
                     "--worker-id",   worker_id,
                     "--gpu-id",      "0",   # logical 0 inside masked process
                     "--cuda"],
                    env=env,
                    stdout=open(f"/tmp/zmq_zeus_gpu{gpu_id}.log", "w"),
                    stderr=sp.STDOUT,
                )
                self.logger.info(
                    f"[ZMQ] Zeus CUDA worker launched "
                    f"({worker_id} CUDA_VISIBLE_DEVICES={gpu_id} logical_gpu=0)"
                )
            except Exception as e:
                self.logger.error(f"[ZMQ] Zeus GPU{gpu_id} launch failed: {e}")

        time.sleep(WORKER_SETTLE_S)
        self._workers_launched = True
        self.logger.info("[ZMQ] All workers launched and settled")
'''


def apply():
    if not TARGET.exists():
        print(f"ERROR: target not found: {TARGET}")
        return False

    content = TARGET.read_text()

    # Find exact method boundaries
    start = content.find("    def _launch_workers(self):")
    if start < 0:
        print("ERROR: _launch_workers() not found")
        return False

    end = content.find("\n    def run_sieve_pass", start)
    if end < 0:
        print("ERROR: run_sieve_pass not found after _launch_workers()")
        return False

    new_content = content[:start] + NEW_LAUNCH_WORKERS + "\n" + content[end + 1:]

    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"ERROR: syntax error at line {e.lineno}: {e.msg}")
        return False

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    TARGET.write_text(new_content)

    # Verify
    ast.parse(TARGET.read_text())
    assert "systemd-run" in TARGET.read_text()
    assert "nohup" not in TARGET.read_text().split("def _launch_workers")[1].split("def run_sieve_pass")[0]
    print("✅ S158D-B: systemd-run --user launcher applied")
    print("✅ AST verified")
    print()
    print("Prerequisites on each rig:")
    print("  ssh rrig6600  'sudo loginctl enable-linger michael'")
    print("  ssh rrig6600b 'sudo loginctl enable-linger michael'")
    print("  ssh rrig6600c 'sudo loginctl enable-linger michael'")
    print()
    print("Verify:")
    print("  ssh rrig6600 'loginctl show-user michael -p Linger --value'")
    print("  → yes")
    print()
    print("Git commit:")
    print("  git add zmq_sqlite_coordinator.py apply_s158d_b_systemd_run_launcher.py")
    print("  git commit -m 'fix(s158d-b): systemd-run --user worker launcher'")
    print("  git push origin main && git push public main")
    return True


if __name__ == "__main__":
    print("Applying S158D-B: systemd-run --user worker launcher...")
    apply()
