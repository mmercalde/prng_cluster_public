#!/usr/bin/env python3
"""
apply_s161_tcp_deadline_fix.py
================================
Fix TCP sequential launch deadline: start the timeout clock from when
the previous worker connected, not from when current worker was launched.

SSH-PWC works because it blocks on stdout reading the ready signal directly.
TCP-PWC must poll worker_count(), so GPU1's ROCm init doesn't start until
GPU0's context is established — but our clock was starting at GPU1 launch time.

Fix: record _connect_time after each worker confirms, start next worker's
deadline from _connect_time + WORKER_HEARTBEAT_TIMEOUT_S.

Apply:
    python3 apply_s161_tcp_deadline_fix.py --dry-run
    python3 apply_s161_tcp_deadline_fix.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_tcp_deadline_fix")

OLD = '''                # S161: Mirror SSH-PWC — wait for this worker to connect before
                # launching next GPU. Prevents simultaneous ROCm context competition.
                _prev_count = self._tcp_transport.worker_count()
                _deadline = _t.time() + WORKER_HEARTBEAT_TIMEOUT_S
                _connected = False
                while _t.time() < _deadline:
                    if self._tcp_transport.worker_count() > _prev_count:
                        _connected = True
                        break
                    _t.sleep(0.5)
                if _connected:
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " ready (" + str(self._tcp_transport.worker_count()) + " total connected)"
                    )
                else:
                    self.logger.warning(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " did not connect within " + str(WORKER_HEARTBEAT_TIMEOUT_S) + "s"
                    )'''

NEW = '''                # S161: Mirror SSH-PWC — wait for this worker to connect before
                # launching next GPU. Prevents simultaneous ROCm context competition.
                # CRITICAL: deadline starts from _launch_time not from previous connect,
                # because ROCm init on this GPU starts immediately after launch script runs.
                # Each GPU takes ~WORKER_HEARTBEAT_TIMEOUT_S to init independently.
                _prev_count = self._tcp_transport.worker_count()
                _deadline = _launch_time + WORKER_HEARTBEAT_TIMEOUT_S
                while _t.time() < _deadline:
                    if self._tcp_transport.worker_count() > _prev_count:
                        _connected = True
                        break
                    _t.sleep(0.5)
                if _connected:
                    _launch_time = _t.time()  # reset for next GPU
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " ready (" + str(self._tcp_transport.worker_count()) + " total connected)"
                    )
                else:
                    self.logger.warning(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " did not connect within " + str(WORKER_HEARTBEAT_TIMEOUT_S) + "s"
                    )'''

# Also need to add _launch_time tracking before the exec_r call
OLD_EXEC = '''                    # Execute the script
                    exec_r = _sp.run(
                        ssh_base + ["bash " + script],
                        capture_output=True, text=True, timeout=10
                    )
                    pid_line = exec_r.stdout.strip()
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " launched \u2014 " + pid_line + " log=" + log_file
                    )'''

NEW_EXEC = '''                    # Execute the script
                    exec_r = _sp.run(
                        ssh_base + ["bash " + script],
                        capture_output=True, text=True, timeout=10
                    )
                    _launch_time = _t.time()  # S161: start deadline from actual launch
                    pid_line = exec_r.stdout.strip()
                    self.logger.info(
                        "[PWC-TCP] " + host + ":GPU" + str(gpu_id) +
                        " launched \u2014 " + pid_line + " log=" + log_file
                    )'''

PATCHES = [
    ("launch_time from exec",        OLD_EXEC, NEW_EXEC),
    ("deadline from launch_time",    OLD,      NEW),
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
        sys.exit(1)
    print("AST OK")
    TARGET.write_text(content, encoding="utf-8")
    print(f"Written: {TARGET}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    apply(args.dry_run)
