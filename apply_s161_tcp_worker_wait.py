#!/usr/bin/env python3
"""
apply_s161_tcp_worker_wait.py
==============================
Adds worker-wait loop to _dispatch_to_tcp() in persistent_worker_coordinator.py.

Problem: coordinator dispatches job immediately with 0 connected TCP workers,
gets broken pipe, exits. Fix: poll worker_count() until >= 1 before submit.

Apply:
    python3 apply_s161_tcp_worker_wait.py --dry-run
    python3 apply_s161_tcp_worker_wait.py
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_worker_wait")

OLD = '''        self._tcp_transport.submit_job(job)'''

NEW = '''        # S161: wait for at least one TCP worker to connect before dispatching
        _wait_start = time.time()
        while self._tcp_transport.worker_count() == 0:
            if time.time() - _wait_start > 60.0:
                return {"status": "error", "message": "TCP worker connect timeout (60s)"}
            time.sleep(0.5)
        self._tcp_transport.submit_job(job)'''

def apply(dry_run: bool = False) -> None:
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: anchor matches {count} times — not unique")
        sys.exit(1)
    print("OK anchor: [worker-wait loop]")
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
