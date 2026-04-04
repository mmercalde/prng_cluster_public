#!/usr/bin/env python3
"""
apply_s161_dashboard_hostname.py
==================================
Fix: TCP worker results report socket.gethostname() (e.g. "rig-6600")
but ProgressWriter registered nodes by IP (e.g. "192.168.3.120").
log_gpu_result() lookup fails silently — dashboard shows no TCP data.

Fix 1: Build _hostname_to_ip map from self.nodes at startup.
Fix 2: Use map to resolve TCP worker hostnames before log_gpu_result().

Standalone — no external dependencies.

Apply:
    python3 apply_s161_dashboard_hostname.py --dry-run
    python3 apply_s161_dashboard_hostname.py
"""
import argparse, ast, shutil, sys
from pathlib import Path

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.pre_s161_dashboard_hostname")

# Patch 1: Build hostname→IP reverse map after nodes are loaded
OLD_NODE_MAP = '''        except Exception as e:
            self.logger.warning(f"ProgressWriter unavailable: {e}")
            self._progress_writer = None'''

NEW_NODE_MAP = '''        except Exception as e:
            self.logger.warning(f"ProgressWriter unavailable: {e}")
            self._progress_writer = None

        # S161: build reverse map hostname→IP for TCP dashboard fix
        # TCP workers report socket.gethostname() but nodes registered by IP
        self._hostname_to_ip: dict = {}
        for _n in self.nodes:
            if not self._is_localhost(_n.hostname):
                self._hostname_to_ip[_n.hostname] = _n.hostname  # IP→IP passthrough
        # Populated lazily when first TCP result arrives with a non-IP hostname'''

# Patch 2: Use hostname map in dashboard log call
OLD_DASH = '''                    if worker_handle_or_node is None:
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
                    self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)'''

NEW_DASH = '''                    if worker_handle_or_node is None:
                        # S161 v2: TCP worker — extract from result payload
                        _payload = res.get("result", {}).get("payload", res.get("result", {}))
                        _raw_host = _payload.get("hostname", res.get("hostname", "tcp-worker"))
                        gpu_id    = _payload.get("gpu_id", 0)
                        gpu_type  = "RX 6600"
                        # S161 dashboard fix: map short hostname to registered IP
                        # TCP workers report socket.gethostname() e.g. "rig-6600"
                        # but ProgressWriter registered nodes by IP e.g. "192.168.3.120"
                        if _raw_host not in self._hostname_to_ip:
                            # First time we see this hostname — find matching node by IP lookup
                            for _n in self.nodes:
                                if not self._is_localhost(_n.hostname):
                                    # Try to resolve short hostname to node IP
                                    try:
                                        import socket as _sock
                                        _resolved = _sock.gethostbyname(_raw_host)
                                        if _resolved == _n.hostname:
                                            self._hostname_to_ip[_raw_host] = _n.hostname
                                            gpu_type = _n.gpu_type
                                            break
                                    except Exception:
                                        pass
                        hostname = self._hostname_to_ip.get(_raw_host, _raw_host)
                        # Also get correct gpu_type from node if mapped
                        for _n in self.nodes:
                            if _n.hostname == hostname:
                                gpu_type = _n.gpu_type
                                break
                    elif isinstance(worker_handle_or_node, WorkerHandle):
                        hostname = worker_handle_or_node.node.hostname
                        gpu_type = worker_handle_or_node.node.gpu_type
                        gpu_id   = worker_handle_or_node.gpu_id
                    else:
                        hostname = worker_handle_or_node.hostname
                        gpu_type = worker_handle_or_node.gpu_type
                        gpu_id   = 0
                    self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed, success=True)'''

PATCHES = [
    ("hostname-to-IP reverse map", OLD_NODE_MAP, NEW_NODE_MAP),
    ("dashboard hostname resolution", OLD_DASH, NEW_DASH),
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
