#!/usr/bin/env python3
"""
apply_s161_tcp_launch_workers.py
==================================
Adds _tcp_launch_workers() to PersistentWorkerCoordinator.startup().

TB-approved S161 Gate 1 requirements:
  1. Bind TCP server first (already done in startup())
  2. Kill stale pwc_worker_service processes on each rig
  3. nohup launch pwc_worker_service per GPU with per-GPU log file
  4. Start 180s connect deadline AFTER last launch issued
  5. ROCm env vars same as _spawn_worker()

Apply:
    python3 apply_s161_tcp_launch_workers.py --dry-run
    python3 apply_s161_tcp_launch_workers.py
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_tcp_launch")

# ---------------------------------------------------------------------------
# Patch 1: Add _tcp_launch_workers() method after _spawn_worker()
# ---------------------------------------------------------------------------

OLD_ENSURE = '''    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:'''

NEW_METHOD = '''    def _tcp_launch_workers(self) -> None:
        """
        S161: SSH-launch pwc_worker_service on each AMD rig GPU via nohup.
        TB Gate 1 requirements:
          - TCP server already bound before this is called
          - Kill stale pwc_worker_service processes first
          - nohup launch per GPU with per-GPU log file
          - ROCm env vars same as _spawn_worker()
          - 180s connect deadline starts AFTER last launch issued
        """
        import subprocess as _sp
        import time as _t

        total_launched = 0

        for node in self.nodes:
            if self._is_localhost(node.hostname):
                continue
            if not self._is_rocm(node):
                continue

            host = node.hostname
            user = node.username
            pool = min(self.worker_pool_size, node.gpu_count)
            ssh_prefix = (
                f"ssh -q -o StrictHostKeyChecking=no -o BatchMode=yes "
                f"-o ConnectTimeout=10 {user}@{host}"
            )

            # Step 1: Kill stale pwc_worker_service processes
            kill_cmd = (
                f"{ssh_prefix} "
                f"'pkill -9 -f pwc_worker_service 2>/dev/null; sleep 1; echo killed'"
            )
            try:
                _sp.run(kill_cmd, shell=True, capture_output=True, timeout=10)
                self.logger.info(f"[PWC-TCP] {host}: stale workers killed")
            except Exception as e:
                self.logger.warning(f"[PWC-TCP] {host}: kill failed: {e}")

            # Step 2: Launch one worker per GPU via nohup
            activate_path = node.python_env.replace("/bin/python", "/bin/activate")

            for gpu_id in range(pool):
                rocm_env = " ".join(ROCM_ENV_VARS + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])
                log_file = f"/tmp/pwc_tcp_worker_{host.replace('.', '_')}_gpu{gpu_id}.log"
                worker_id = f"{host.replace('.', '_')}_gpu{gpu_id}"

                launch_cmd = (
                    f"{ssh_prefix} "
                    f"'cd {node.script_path} && "
                    f"source {activate_path} && "
                    f"nohup env PYTHONPATH=. {rocm_env} "
                    f"{node.python_env} -m persistent.pwc_worker_service "
                    f"--host 192.168.3.127 --port {self.pwc_port} "
                    f"--gpu-id {gpu_id} --worker-id {worker_id} "
                    f">> {log_file} 2>&1 & echo PID=$!'"
                )
                try:
                    r = _sp.run(
                        launch_cmd, shell=True, capture_output=True,
                        text=True, timeout=15
                    )
                    pid_line = r.stdout.strip()
                    self.logger.info(
                        f"[PWC-TCP] {host}:GPU{gpu_id} launched — {pid_line} "
                        f"log={log_file}"
                    )
                    total_launched += 1
                except Exception as e:
                    self.logger.error(f"[PWC-TCP] {host}:GPU{gpu_id} launch failed: {e}")

                # Stagger to prevent simultaneous HIP init
                if gpu_id < pool - 1:
                    _t.sleep(ROCM_SPAWN_STAGGER_S)

        self.logger.info(
            f"[PWC-TCP] {total_launched} workers launched across all rigs — "
            f"waiting up to 180s for connections"
        )

    def _ensure_worker_alive(self, handle: WorkerHandle) -> bool:'''

# ---------------------------------------------------------------------------
# Patch 2: Call _tcp_launch_workers() in startup() after TCP server starts,
#          and replace 60s wait with 180s wait starting after launch
# ---------------------------------------------------------------------------

OLD_STARTUP_TCP = '''        if self._tcp_transport is not None:
                self._tcp_transport.start()
                self.logger.info(
                    f"[PWC-TCP] transport started on {self.pwc_host}:{self.pwc_port}"
                )
            return'''

NEW_STARTUP_TCP = '''        if self._tcp_transport is not None:
                self._tcp_transport.start()
                self.logger.info(
                    f"[PWC-TCP] transport started on {self.pwc_host}:{self.pwc_port}"
                )
            # S161: SSH-launch workers on all AMD rigs now that TCP server is bound
            self._tcp_launch_workers()
            return'''

# ---------------------------------------------------------------------------
# Patch 3: Replace 60s wait with 180s in run_sieve_pass() worker-wait loop
# ---------------------------------------------------------------------------

OLD_WAIT = '''            while self._tcp_transport.worker_count() == 0:
                if time.time() - _wait_start > 60.0:
                    self.logger.error("[PWC-TCP] No workers connected after 60s — aborting")
                    return {"status": "error", "survivor_count": 0,
                            "survivors": [], "failed_chunks": 1, "total_chunks": 1}
                time.sleep(0.5)
            self.logger.info(f"[PWC-TCP] {self._tcp_transport.worker_count()} worker(s) connected — dispatching")'''

NEW_WAIT = '''            while self._tcp_transport.worker_count() == 0:
                if time.time() - _wait_start > 180.0:
                    self.logger.error("[PWC-TCP] No workers connected after 180s — aborting")
                    return {"status": "error", "survivor_count": 0,
                            "survivors": [], "failed_chunks": 1, "total_chunks": 1}
                time.sleep(0.5)
            self.logger.info(f"[PWC-TCP] {self._tcp_transport.worker_count()} worker(s) connected — dispatching")'''

PATCHES = [
    ("_tcp_launch_workers method",  OLD_ENSURE,       NEW_METHOD),
    ("startup TCP launch call",     OLD_STARTUP_TCP,  NEW_STARTUP_TCP),
    ("180s connect timeout",        OLD_WAIT,         NEW_WAIT),
]

def apply(dry_run: bool = False) -> None:
    content = TARGET.read_text(encoding="utf-8")
    for name, old, new in PATCHES:
        count = content.count(old)
        if count == 0:
            print(f"ERROR: anchor not found for [{name}]")
            sys.exit(1)
        if count > 1:
            print(f"ERROR: anchor matches {count} times for [{name}]")
            sys.exit(1)
        print(f"OK anchor: [{name}]")

    if dry_run:
        print("DRY RUN — no files modified")
        return

    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    for name, old, new in PATCHES:
        content = content.replace(old, new, 1)
        print(f"Applied: [{name}]")

    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"AST FAIL: {e}")
        sys.exit(1)
    print("AST OK")

    TARGET.write_text(content, encoding="utf-8")
    print(f"Written: {TARGET}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
