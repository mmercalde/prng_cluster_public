#!/usr/bin/env python3
"""
apply_s161_tcp_count_fix.py
============================
Fixes _get_available_workers() in persistent_worker_coordinator.py.

Bug: max(1, worker_count()) forces pool_size=1 even with 0 connected workers,
bypassing the worker-wait loop in _dispatch_to_tcp().

Fix: remove max(1, ...) so 0 connected workers → 0 slots → dispatch blocks
until a worker connects.

Apply:
    python3 apply_s161_tcp_count_fix.py --dry-run
    python3 apply_s161_tcp_count_fix.py
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_tcp_count_fix")

OLD = '''            tcp_count = max(1, self._tcp_transport.worker_count())'''
NEW = '''            tcp_count = self._tcp_transport.worker_count()'''

def apply(dry_run: bool = False) -> None:
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: anchor matches {count} times — not unique")
        sys.exit(1)
    print("OK anchor: [tcp_count max(1) removal]")
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
