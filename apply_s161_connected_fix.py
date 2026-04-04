#!/usr/bin/env python3
"""
apply_s161_connected_fix.py
============================
Fix: _connected variable referenced before assignment in _tcp_launch_workers.
The variable is set inside the wait loop but referenced in the except block
before the loop runs if an exception occurs during launch.

Fix: initialize _connected = False before the try block.

Apply:
    python3 apply_s161_connected_fix.py --dry-run
    python3 apply_s161_connected_fix.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_connected_fix")

OLD = '''                try:
                    # Write script to rig via stdin pipe
                    write_r = _sp.run(
                        ssh_base + ["cat > " + script + " && chmod +x " + script],'''

NEW = '''                _connected = False  # S161: init before try so except block can reference it
                try:
                    # Write script to rig via stdin pipe
                    write_r = _sp.run(
                        ssh_base + ["cat > " + script + " && chmod +x " + script],'''

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [_connected init before try]")
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
