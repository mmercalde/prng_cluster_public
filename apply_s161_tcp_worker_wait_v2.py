#!/usr/bin/env python3
"""
apply_s161_tcp_worker_wait_v2.py
=================================
Fixes IndexError in run_sieve_pass() when TCP mode has 0 connected workers.

Bug: all_workers=[] but num_workers=max(1,0)=1, then all_workers[0] crashes.
Fix: in TCP mode, wait for at least 1 worker to connect before building chunks.

Apply:
    python3 apply_s161_tcp_worker_wait_v2.py --dry-run
    python3 apply_s161_tcp_worker_wait_v2.py
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_worker_wait_v2")

OLD = '''        # Build chunk list — divide total seeds across all available workers
        all_workers = self._get_available_workers()
        num_workers = max(1, len(all_workers))'''

NEW = '''        # Build chunk list — divide total seeds across all available workers
        # S161: in TCP mode, wait for at least 1 worker to connect before dispatch
        if self._tcp_transport is not None:
            _wait_start = time.time()
            while self._tcp_transport.worker_count() == 0:
                if time.time() - _wait_start > 60.0:
                    self.logger.error("[PWC-TCP] No workers connected after 60s — aborting")
                    return {"status": "error", "survivor_count": 0,
                            "survivors": [], "failed_chunks": 1, "total_chunks": 1}
                time.sleep(0.5)
            self.logger.info(f"[PWC-TCP] {self._tcp_transport.worker_count()} worker(s) connected — dispatching")
        all_workers = self._get_available_workers()
        num_workers = max(1, len(all_workers))'''

def apply(dry_run: bool = False) -> None:
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: anchor matches {count} times — not unique")
        sys.exit(1)
    print("OK anchor: [tcp worker wait before chunk build]")
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
