#!/usr/bin/env python3
"""
apply_s152_heartbeat_timeout.py
================================
Patch: Increase WORKER_HEARTBEAT_TIMEOUT_S from 30s to 90s.

PROBLEM
-------
After a rig reboots, ROCm/CuPy initialization takes >30 seconds when
8 GPUs are initializing simultaneously. The coordinator's spawn timeout
is only 30 seconds — so all 8 workers get quarantined on the first trial
after any reboot, even a clean one.

Sequence:
  1. Rig reboots (for any reason)
  2. Run 1 coordinator tries to spawn workers
  3. Worker starts, imports CuPy, hits cp.zeros(1) GPU warmup
  4. ROCm init on 8 GPUs takes >30s
  5. Coordinator times out → quarantines all 8 workers
  6. Rig stays quarantined for rest of run

FIX
---
Increase WORKER_HEARTBEAT_TIMEOUT_S from 30 to 90 seconds.
90s gives ROCm enough time to initialize on a freshly booted rig
with 8 GPUs and the 4s stagger between spawns.

This does not slow down normal operation — workers that boot
successfully still return their heartbeat in 2-5 seconds.
The 90s timeout is only hit in failure cases.

Files patched
-------------
  persistent_worker_coordinator.py

Backup: persistent_worker_coordinator.py.bak_s152_heartbeat
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.bak_s152_heartbeat")

OLD_TIMEOUT = "WORKER_HEARTBEAT_TIMEOUT_S = 30   # seconds to wait for worker heartbeat on startup"
NEW_TIMEOUT = "WORKER_HEARTBEAT_TIMEOUT_S = 90   # [S152] seconds to wait for worker heartbeat on startup — 90s allows ROCm init on fresh-booted rigs"


def apply():
    src = TARGET.read_text()

    if "WORKER_HEARTBEAT_TIMEOUT_S = 90" in src:
        print("⚠️  Already at 90s — patch already applied. Aborting.")
        return

    if OLD_TIMEOUT not in src:
        print(f"❌ Anchor not found: {OLD_TIMEOUT!r}")
        return

    patched = src.replace(OLD_TIMEOUT, NEW_TIMEOUT, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  90s timeout present: {'WORKER_HEARTBEAT_TIMEOUT_S = 90' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print(f"  WORKER_HEARTBEAT_TIMEOUT_S: 30s → 90s")
    print()
    print("Effect: Workers on freshly rebooted rigs have 90s to return")
    print("heartbeat — enough time for ROCm to initialize 8 GPUs.")


if __name__ == "__main__":
    apply()
