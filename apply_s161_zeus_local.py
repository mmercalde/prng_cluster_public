#!/usr/bin/env python3
"""
apply_s161_zeus_local.py
=========================
Fix: Zeus local GPUs (2x RTX 3080Ti) not participating in TCP mode.

In TCP mode _get_available_workers() returns only [None]*tcp_count,
excluding Zeus local nodes entirely. _run_once() then always routes
to _dispatch_to_tcp() skipping the local sieve path.

Fix 1: Add Zeus local nodes to available list even in TCP mode.
Fix 2: _run_once() checks isinstance(wh, WorkerNode) to route local
       jobs to _dispatch_local_sieve() regardless of TCP mode.

Apply:
    python3 apply_s161_zeus_local.py --dry-run
    python3 apply_s161_zeus_local.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_zeus_local")

OLD_GET_WORKERS = '''        if self._tcp_transport is not None:
            tcp_count = self._tcp_transport.ready_count()  # S161 v2: only ready workers
            # Return synthetic placeholder list sized to connected TCP workers
            # Actual dispatch goes through _dispatch_to_tcp(), not these handles
            return [None] * tcp_count'''

NEW_GET_WORKERS = '''        if self._tcp_transport is not None:
            tcp_count = self._tcp_transport.ready_count()  # S161 v2: only ready workers
            # TCP workers as synthetic placeholders + Zeus local nodes
            available = [None] * tcp_count
            for node in self.nodes:
                if self._is_localhost(node.hostname):
                    available.append(node)
            return available'''

OLD_RUN_ONCE = '''                if self._tcp_transport is not None:
                    return self._dispatch_to_tcp(job)
                if isinstance(wh, WorkerHandle):'''

NEW_RUN_ONCE = '''                if self._tcp_transport is not None and not isinstance(wh, WorkerNode):
                    return self._dispatch_to_tcp(job)
                if isinstance(wh, WorkerHandle):'''

PATCHES = [
    ("get_workers includes Zeus local in TCP mode", OLD_GET_WORKERS, NEW_GET_WORKERS),
    ("run_once routes local nodes to local sieve",  OLD_RUN_ONCE,    NEW_RUN_ONCE),
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
