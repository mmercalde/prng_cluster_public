#!/usr/bin/env python3
"""
apply_s161_dashboard_tcp.py
============================
Fix: web dashboard gets no data from TCP workers because
worker_handle_or_node is None in TCP mode, causing silent crash.

Fix: extract hostname/gpu_id/gpu_type from result payload for TCP workers.
TCP results contain hostname and gpu_id from pwc_worker_service._execute_job().

Apply:
    python3 apply_s161_dashboard_tcp.py --dry-run
    python3 apply_s161_dashboard_tcp.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_dashboard_tcp")

OLD = '''            # Log to ProgressWriter for web dashboard (mirrors coordinator.py line 1611)
            if self._progress_writer and res.get("status") == "ok" and elapsed > 0:
                try:
                    hostname = worker_handle_or_node.node.hostname if isinstance(worker_handle_or_node, WorkerHandle) else worker_handle_or_node.hostname
                    gpu_type = worker_handle_or_node.node.gpu_type if isinstance(worker_handle_or_node, WorkerHandle) else worker_handle_or_node.gpu_type
                    gpu_id   = worker_handle_or_node.gpu_id if isinstance(worker_handle_or_node, WorkerHandle) else 0
                    self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)
                except Exception:
                    pass'''

NEW = '''            # Log to ProgressWriter for web dashboard (mirrors coordinator.py line 1611)
            if self._progress_writer and res.get("status") == "ok" and elapsed > 0:
                try:
                    if worker_handle_or_node is None:
                        # S161 v2: TCP worker — extract from result payload
                        _payload = res.get("result", {}).get("payload", res.get("result", {}))
                        hostname = _payload.get("hostname", res.get("hostname", "tcp-worker"))
                        gpu_id   = _payload.get("gpu_id", 0)
                        gpu_type = "RX 6600"
                    elif isinstance(worker_handle_or_node, WorkerHandle):
                        hostname = worker_handle_or_node.node.hostname
                        gpu_type = worker_handle_or_node.node.gpu_type
                        gpu_id   = worker_handle_or_node.gpu_id
                    else:
                        hostname = worker_handle_or_node.hostname
                        gpu_type = worker_handle_or_node.gpu_type
                        gpu_id   = 0
                    self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)
                except Exception:
                    pass'''

def apply(dry_run=False):
    content = TARGET.read_text(encoding="utf-8")
    count = content.count(OLD)
    if count == 0:
        print("ERROR: anchor not found")
        sys.exit(1)
    if count > 1:
        print(f"ERROR: {count} matches")
        sys.exit(1)
    print("OK anchor: [dashboard TCP worker logging]")
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
