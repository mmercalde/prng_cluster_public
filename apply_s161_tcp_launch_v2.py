#!/usr/bin/env python3
"""
apply_s161_tcp_launch_v2.py
============================
Replaces _tcp_launch_workers() with a version that avoids SSH quoting issues.

Fix: Instead of inline shell command with nested quotes and & operator,
write a temp bash script to the rig via ssh, then execute it.
This avoids all quoting/escaping problems completely.

Apply:
    python3 apply_s161_tcp_launch_v2.py --dry-run
    python3 apply_s161_tcp_launch_v2.py
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_tcp_launch_v2")

OLD = '''    def _tcp_launch_workers(self) -> None:
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
        )'''

NEW = '''    def _tcp_launch_workers(self) -> None:
        """
        S161 v2: SSH-launch pwc_worker_service on each AMD rig GPU.
        Writes a temp bash script to the rig to avoid SSH quoting issues.
        TB Gate 1: nohup, per-GPU log, 180s deadline after last launch.
        """
        import subprocess as _sp
        import time as _t

        total_launched = 0

        for node in self.nodes:
            if self._is_localhost(node.hostname):
                continue
            if not self._is_rocm(node):
                continue

            host     = node.hostname
            user     = node.username
            pool     = min(self.worker_pool_size, node.gpu_count)
            ssh_base = ["ssh", "-q",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10",
                        f"{user}@{host}"]
            activate_path = node.python_env.replace("/bin/python", "/bin/activate")

            # Step 1: Kill stale pwc_worker_service processes
            try:
                _sp.run(
                    ssh_base + ["pkill -9 -f pwc_worker_service 2>/dev/null; echo killed"],
                    capture_output=True, timeout=10
                )
                self.logger.info(f"[PWC-TCP] {host}: stale workers killed")
            except Exception as e:
                self.logger.warning(f"[PWC-TCP] {host}: kill failed: {e}")

            # Step 2: Write launch script to rig, then execute it
            for gpu_id in range(pool):
                log_file  = f"/tmp/pwc_tcp_worker_{host.replace('.', '_')}_gpu{gpu_id}.log"
                worker_id = f"{host.replace('.', '_')}_gpu{gpu_id}"
                script    = f"/tmp/pwc_tcp_launch_gpu{gpu_id}.sh"

                # Build script content — no quoting issues since it's a file
                script_lines = [
                    "#!/bin/bash",
                    f"source {activate_path}",
                    f"cd {node.script_path}",
                ]
                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                script_lines += [
                    f"export ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                    f"export PYTHONPATH=.",
                    (
                        f"nohup {node.python_env} -m persistent.pwc_worker_service "
                        f"--host 192.168.3.127 --port {self.pwc_port} "
                        f"--gpu-id {gpu_id} --worker-id {worker_id} "
                        f">> {log_file} 2>&1 &"
                    ),
                    "echo PID=$!",
                ]
                script_content = "\n".join(script_lines)

                # Write script to rig via stdin pipe
                try:
                    write_r = _sp.run(
                        ssh_base + [f"cat > {script} && chmod +x {script}"],
                        input=script_content.encode(),
                        capture_output=True, timeout=10
                    )
                    if write_r.returncode != 0:
                        self.logger.error(f"[PWC-TCP] {host}:GPU{gpu_id} script write failed")
                        continue

                    # Execute the script
                    exec_r = _sp.run(
                        ssh_base + [f"bash {script}"],
                        capture_output=True, text=True, timeout=10
                    )
                    pid_line = exec_r.stdout.strip()
                    self.logger.info(
                        f"[PWC-TCP] {host}:GPU{gpu_id} launched — {pid_line} log={log_file}"
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
        )'''


def apply(dry_run: bool = False) -> None:
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: anchor matches {count} times")
        sys.exit(1)
    print("OK anchor: [_tcp_launch_workers v2]")
    if dry_run:
        print("DRY RUN — no files modified")
        return
    shutil.copy(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")
    content = content.replace(OLD, NEW, 1)
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
