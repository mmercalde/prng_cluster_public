#!/usr/bin/env python3
"""
apply_s161_tcp_launch_v3.py
============================
Replaces _tcp_launch_workers() on Zeus with temp-script approach.
Targets exact v1 content currently on Zeus.

Apply:
    python3 apply_s161_tcp_launch_v3.py --dry-run
    python3 apply_s161_tcp_launch_v3.py
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_tcp_launch_v3")

# Unique anchor from the v1 code on Zeus
OLD = '''                launch_cmd = (
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
                    self.logger.error(f"[PWC-TCP] {host}:GPU{gpu_id} launch failed: {e}")'''

NEW = '''                script    = f"/tmp/pwc_tcp_launch_gpu{gpu_id}.sh"

                # Build script content — no quoting issues since it is a file
                script_lines = [
                    "#!/bin/bash",
                    "source " + activate_path,
                    "cd " + node.script_path,
                ]
                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),
                    "export PYTHONPATH=.",
                    (
                        "nohup " + node.python_env + " -m persistent.pwc_worker_service "
                        "--host 192.168.3.127 --port " + str(self.pwc_port) + " "
                        "--gpu-id " + str(gpu_id) + " --worker-id " + worker_id + " "
                        ">> " + log_file + " 2>&1 &"
                    ),
                    "echo PID=$!",
                ]
                script_content = chr(10).join(script_lines)

                try:
                    # Write script to rig via stdin pipe
                    write_r = _sp.run(
                        ssh_base + ["cat > " + script + " && chmod +x " + script],
                        input=script_content.encode(),
                        capture_output=True, timeout=10
                    )
                    if write_r.returncode != 0:
                        self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " script write failed")
                        continue

                    # Execute the script
                    exec_r = _sp.run(
                        ssh_base + ["bash " + script],
                        capture_output=True, text=True, timeout=10
                    )
                    pid_line = exec_r.stdout.strip()
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " launched — " + pid_line + " log=" + log_file
                    )
                    total_launched += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))'''

# Also fix ssh_prefix to use list instead of string
OLD_SSH = '''            ssh_prefix = (
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
            activate_path = node.python_env.replace("/bin/python", "/bin/activate")'''

NEW_SSH = '''            ssh_base = ["ssh", "-q",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10",
                        user + "@" + host]
            activate_path = node.python_env.replace("/bin/python", "/bin/activate")

            # Step 1: Kill stale pwc_worker_service processes
            try:
                _sp.run(
                    ssh_base + ["pkill -9 -f pwc_worker_service 2>/dev/null; echo killed"],
                    capture_output=True, timeout=10
                )
                self.logger.info("[PWC-TCP] " + host + ": stale workers killed")
            except Exception as e:
                self.logger.warning("[PWC-TCP] " + host + ": kill failed: " + str(e))

            # Step 2: Launch one worker per GPU via temp bash script'''

PATCHES = [
    ("ssh_prefix to list + kill fix", OLD_SSH, NEW_SSH),
    ("launch_cmd to temp script",     OLD,     NEW),
]

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    for name, old, new in PATCHES:
        count = content.count(old)
        if count == 0:
            print(f"ERROR: anchor not found for [{name}]")
            sys.exit(1)
        if count > 1:
            print(f"ERROR: {count} matches for [{name}]")
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
        print(f"AST FAIL line {e.lineno}: {e.msg}")
        lines = content.splitlines()
        for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
            print(f"{i+1}: {lines[i]}")
        sys.exit(1)
    print("AST OK")
    TARGET.write_text(content, encoding="utf-8")
    print(f"Written: {TARGET}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
