#!/usr/bin/env python3
"""
apply_s162_dispatch_semaphore.py
=================================
S162 fix: Add dispatch semaphore to TCPWorkerTransport to prevent
simultaneous job send flood on Zeus's 1GbE NIC (Intel I210).

Root cause: 24 handler threads all call conn.send_obj() simultaneously
when chunks complete at the same time. Zeus's 1GbE NIC (enp2s0, I210)
cannot drain 24 concurrent TCP sends fast enough. rrig6600c workers
(last rig spawned) receive delayed/stalled job payloads → GPU page fault
→ cascade crash.

Fix: threading.Semaphore(8) around job payload send_obj() calls.
Limits concurrent outbound job dispatches to 8 at a time (one rig's worth).
Empty-queue responses are NOT throttled — only real job payloads.

TCP buffer tuning (applied separately on Zeus via sysctl):
  net.core.wmem_max=16777216
  net.core.wmem_default=1048576
  net.core.rmem_max=16777216
  net.core.rmem_default=1048576
  net.ipv4.tcp_wmem=4096 1048576 16777216
  net.ipv4.tcp_rmem=4096 1048576 16777216
"""

import ast
import sys
import os

TARGET = "persistent/pwc_transport_tcp.py"

# ── Patch 1: Add semaphore to __init__ ──────────────────────────────────────

OLD_INIT = '''        self._accept_thread: Optional[threading.Thread] = None
        self._lease_thread: Optional[threading.Thread] = None'''

NEW_INIT = '''        self._accept_thread: Optional[threading.Thread] = None
        self._lease_thread: Optional[threading.Thread] = None

        # S162: Dispatch semaphore — limits concurrent job payload sends to 8.
        # Prevents 24 simultaneous conn.send_obj() calls from flooding Zeus's
        # 1GbE NIC (Intel I210) when all workers request jobs at the same time.
        # Value=8 = one rig's worth of workers — allows full rig parallelism
        # while preventing cross-rig NIC saturation.
        self._dispatch_semaphore = threading.Semaphore(8)'''

# ── Patch 2: Wrap job payload send_obj with semaphore ───────────────────────

OLD_DISPATCH = '''                    lease_id = self._lease_job(job, worker_id)
                    conn.send_obj({
                        "message_type":     "job_assign",
                        "protocol_version": 1,
                        "worker_id":        "coordinator",
                        "timestamp":        time.time(),
                        "job_id":           job["job_id"],
                        "lease_id":         lease_id,
                        "attempt":          job.get("attempt", 0),
                        "payload":          job,
                    })'''

NEW_DISPATCH = '''                    lease_id = self._lease_job(job, worker_id)
                    # S162: Semaphore limits concurrent job sends to 8 at a time.
                    # Prevents NIC flood on Zeus's 1GbE (Intel I210) when all
                    # 24 workers request jobs simultaneously.
                    with self._dispatch_semaphore:
                        conn.send_obj({
                            "message_type":     "job_assign",
                            "protocol_version": 1,
                            "worker_id":        "coordinator",
                            "timestamp":        time.time(),
                            "job_id":           job["job_id"],
                            "lease_id":         lease_id,
                            "attempt":          job.get("attempt", 0),
                            "payload":          job,
                        })'''


def apply():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found. Run from distributed_prng_analysis/")
        sys.exit(1)

    with open(TARGET) as f:
        src = f.read()

    # Validate anchors exist exactly once
    for name, anchor in [("OLD_INIT", OLD_INIT), ("OLD_DISPATCH", OLD_DISPATCH)]:
        count = src.count(anchor)
        if count != 1:
            print(f"ERROR: {name} found {count} times (expected 1). Aborting.")
            sys.exit(1)

    # Check not already patched
    if "_dispatch_semaphore" in src:
        print("Already patched — _dispatch_semaphore already present. Skipping.")
        sys.exit(0)

    # Apply patches
    src = src.replace(OLD_INIT, NEW_INIT)
    src = src.replace(OLD_DISPATCH, NEW_DISPATCH)

    # AST validate
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: AST validation failed: {e}")
        sys.exit(1)

    # Write
    with open(TARGET, "w") as f:
        f.write(src)

    print("✅ Patch applied successfully.")
    print()
    print("Changes made to persistent/pwc_transport_tcp.py:")
    print("  1. Added self._dispatch_semaphore = threading.Semaphore(8) to __init__")
    print("  2. Wrapped job payload conn.send_obj() with semaphore context manager")
    print()
    print("This file runs on ZEUS only — no rig deployment needed.")
    print()
    print("Verify:")
    print("  grep -n '_dispatch_semaphore' persistent/pwc_transport_tcp.py")


if __name__ == "__main__":
    apply()
