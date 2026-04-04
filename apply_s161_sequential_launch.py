#!/usr/bin/env python3
"""
apply_s161_sequential_launch.py
================================
Mirrors original PWC SSH launch: launch one GPU, wait for it to connect,
then launch next GPU. Prevents simultaneous ROCm context competition.

Apply:
    python3 apply_s161_sequential_launch.py --dry-run
    python3 apply_s161_sequential_launch.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_sequential_launch")

# Unique anchor — the stagger block that follows launch
OLD = '''                # Stagger \u2014 TCP path uses 1s (SSH is bootstrap only, not compute)
                if gpu_id < pool - 1:
                    _t.sleep(TCP_SPAWN_STAGGER_S)'''

NEW = '''                # S161: Mirror SSH-PWC — wait for this worker to connect before
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

# Also fix total_launched to only count confirmed-connected workers
OLD_TOTAL = '''                    total_launched += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))'''

NEW_TOTAL = '''                    if _connected:
                        total_launched += 1
                except Exception as e:
                    self.logger.error("[PWC-TCP] " + host + ":GPU" + str(gpu_id) + " launch failed: " + str(e))'''

PATCHES = [
    ("sequential connect wait", OLD,       NEW),
    ("total_launched on connect", OLD_TOTAL, NEW_TOTAL),
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
