#!/usr/bin/env python3
"""
apply_s161_dashboard_worker_id.py
===================================
Fix: _dispatch_to_tcp() strips worker_id/hostname from result dict.
Dashboard log_gpu_result() gets res with no worker_id to parse.

Fix: Preserve worker_id in _dispatch_to_tcp() return dict.
Dashboard code already correctly parses worker_id → IP.

Apply:
    python3 apply_s161_dashboard_worker_id.py --dry-run
    python3 apply_s161_dashboard_worker_id.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_dashboard_worker_id")

OLD = '''        return {
            "status":         "ok",
            "job_id":         job["job_id"],
            "survivors":      survivors,
            "match_rates":    match_rates,
            "skip_sequences": skip_seqs,
            "strategy_ids":   strat_ids,
        }

    def _dispatch_local_sieve(self, job: Dict[str, Any], node: WorkerNode) -> Dict[str, Any]:'''

NEW = '''        return {
            "status":         "ok",
            "job_id":         job["job_id"],
            "survivors":      survivors,
            "match_rates":    match_rates,
            "skip_sequences": skip_seqs,
            "strategy_ids":   strat_ids,
            # S161 dashboard: preserve worker identity for ProgressWriter
            "worker_id":      result.get("worker_id", ""),
            "hostname":       result.get("hostname", ""),
            "gpu_id":         result.get("gpu_id", 0),
        }

    def _dispatch_local_sieve(self, job: Dict[str, Any], node: WorkerNode) -> Dict[str, Any]:'''

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [_dispatch_to_tcp preserve worker_id]")
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
