#!/usr/bin/env python3
"""
apply_s161_smoke_dataset_v2.py
================================
Fixes smoke test dataset_path — should be daily3.json not the survivors NPZ.

Apply:
    python3 apply_s161_smoke_dataset_v2.py --dry-run
    python3 apply_s161_smoke_dataset_v2.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_smoke_dataset_v2")

OLD = '        dataset_path = "bidirectional_survivors_binary.npz",'
NEW = '        dataset_path = "daily3.json",'

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [smoke dataset daily3.json]")
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
