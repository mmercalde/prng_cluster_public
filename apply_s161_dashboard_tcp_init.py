#!/usr/bin/env python3
"""
apply_s161_dashboard_tcp_init.py
==================================
Fix: TCP startup() returns early before ProgressWriter is initialized.
SSH-PWC path initializes ProgressWriter after worker spawn — TCP path
skips this entirely with an early return.

Fix: Initialize ProgressWriter in TCP startup path before return.

Apply:
    python3 apply_s161_dashboard_tcp_init.py --dry-run
    python3 apply_s161_dashboard_tcp_init.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_dashboard_tcp_init")

OLD = '''            # S161: SSH-launch workers on all AMD rigs now that TCP server is bound
            self._tcp_launch_workers()
            return'''

NEW = '''            # S161: SSH-launch workers on all AMD rigs now that TCP server is bound
            self._tcp_launch_workers()

            # Initialize ProgressWriter for web dashboard — must be done in TCP path
            # SSH-PWC does this after spawn loop but TCP returns early above
            try:
                from progress_display import ProgressWriter
                self._progress_writer = ProgressWriter("Forward Sieve", total_jobs=100, total_seeds=0)
                for node in self.nodes:
                    if self._is_localhost(node.hostname):
                        self._progress_writer.register_node("localhost", "RTX 3080 Ti", 2)
                    else:
                        self._progress_writer.register_node(node.hostname, node.gpu_type, node.gpu_count)
                self.logger.info("[PWC-TCP] ProgressWriter initialized for web dashboard")
            except Exception as e:
                self.logger.warning(f"ProgressWriter unavailable: {e}")
                self._progress_writer = None

            return'''

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [TCP startup ProgressWriter init]")
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
