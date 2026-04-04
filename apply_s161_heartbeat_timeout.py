#!/usr/bin/env python3
"""
apply_s161_heartbeat_timeout.py
================================
Increase WORKER_HEARTBEAT_TIMEOUT_S from 90s to 120s.

Each GPU on rrig6600 takes ~90s to initialize ROCm sequentially.
The timeout was exactly at the edge causing workers to be skipped.
120s gives 30s headroom — same one-time cost paid at session startup.

Apply:
    python3 apply_s161_heartbeat_timeout.py --dry-run
    python3 apply_s161_heartbeat_timeout.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_heartbeat_timeout")

OLD = 'WORKER_HEARTBEAT_TIMEOUT_S = 90   # [S152] seconds to wait for worker heartbeat on startup \u2014 90s allows ROCm init on fresh-booted rigs'
NEW = 'WORKER_HEARTBEAT_TIMEOUT_S = 120  # [S161] 120s allows ROCm sequential init on 8-GPU rigs (90s was edge case)'

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    print("OK anchor: [WORKER_HEARTBEAT_TIMEOUT_S]")
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
