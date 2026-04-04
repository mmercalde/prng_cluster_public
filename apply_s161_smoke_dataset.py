#!/usr/bin/env python3
"""
apply_s161_smoke_dataset.py
============================
Adds dataset_path to smoke test run_sieve_pass() call.

Apply:
    python3 apply_s161_smoke_dataset.py --dry-run
    python3 apply_s161_smoke_dataset.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_smoke_dataset")

OLD = '''    result = pwc.run_sieve_pass(
        prng_type   = args.prng_type,
        residues    = residues,
        total_seeds = args.total_seeds,
        threshold   = 0.30,
        window_size = 8,
        output_file = "/tmp/pwc_smoke_test.json",
    )'''

NEW = '''    result = pwc.run_sieve_pass(
        prng_type    = args.prng_type,
        residues     = residues,
        total_seeds  = args.total_seeds,
        threshold    = 0.30,
        window_size  = 8,
        dataset_path = "bidirectional_survivors_binary.npz",
        output_file  = "/tmp/pwc_smoke_test.json",
    )'''

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [smoke dataset_path]")
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
