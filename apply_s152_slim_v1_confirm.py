#!/usr/bin/env python3
"""
apply_s152_slim_v1_confirm.py
==============================
Patch: Add one log line to each IPC path in persistent_worker_coordinator.py
so slim_v1 vs legacy format is confirmed in every run's log.

slim_v1 fast path logs:  [slim_v1] GPU{id} chunk → N survivors
Legacy path logs:        [legacy-ipc] GPU{id} chunk → N survivors (WARN: expected slim_v1)

The legacy log is a WARNING because if the rig workers have been updated to
emit slim_v1 and we're still hitting the legacy path, something is wrong.

Files patched
-------------
  persistent_worker_coordinator.py

Backup: persistent_worker_coordinator.py.bak_s152_slim_confirm
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.bak_s152_slim_confirm")

# ── Patch 1: slim_v1 fast path — add log after survivors list is built ────────
OLD_SLIM_PATH = '''\
                    if inner.get("format") == "slim_v1":
                        # Fast path — parallel arrays (TB approved Option A)
                        survivors   = [int(s) for s in inner.get("seeds", [])]
                        match_rates = list(inner.get("match_rates", []))
                        n = len(survivors)'''

NEW_SLIM_PATH = '''\
                    if inner.get("format") == "slim_v1":
                        # Fast path — parallel arrays (TB approved Option A)
                        survivors   = [int(s) for s in inner.get("seeds", [])]
                        match_rates = list(inner.get("match_rates", []))
                        n = len(survivors)
                        # [S152-IPC] Confirm slim_v1 fast path active
                        _gpu_tag = f"{job.get('hostname','?')}:GPU{job.get('gpu_id','?')}"
                        self.logger.debug(f"[slim_v1] {_gpu_tag} chunk → {n} survivors")'''

# ── Patch 2: legacy path — add WARNING log ────────────────────────────────────
OLD_LEGACY_PATH = '''\
                    else:
                        # Legacy path — list of dicts (kept for rollout safety)
                        raw_survivors = inner.get("survivors", [])
                        survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]'''

NEW_LEGACY_PATH = '''\
                    else:
                        # Legacy path — list of dicts (kept for rollout safety)
                        # [S152-IPC] WARN: expected slim_v1 from updated workers
                        _gpu_tag = f"{job.get('hostname','?')}:GPU{job.get('gpu_id','?')}"
                        self.logger.warning(f"[legacy-ipc] {_gpu_tag} — worker sent legacy dict-list format (expected slim_v1)")
                        raw_survivors = inner.get("survivors", [])
                        survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]'''


def apply():
    src = TARGET.read_text()

    # Idempotency
    if "[S152-IPC]" in src:
        print("⚠️  [S152-IPC] marker already present — patch already applied. Aborting.")
        return

    # Validate anchors
    missing = []
    if OLD_SLIM_PATH not in src:
        missing.append("slim_v1 fast path anchor")
    if OLD_LEGACY_PATH not in src:
        missing.append("legacy path anchor")
    if missing:
        print(f"❌ Anchors not found: {missing}")
        return

    patched = src.replace(OLD_SLIM_PATH,   NEW_SLIM_PATH,   1)
    patched = patched.replace(OLD_LEGACY_PATH, NEW_LEGACY_PATH, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  [S152-IPC] in patched:      {'[S152-IPC]' in patched}")
        print(f"  slim_v1 log present:        {'[slim_v1]' in patched}")
        print(f"  legacy-ipc warn present:    {'[legacy-ipc]' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Verification:")
    print(f"  [slim_v1] log line:     {'[slim_v1]' in patched}")
    print(f"  [legacy-ipc] warn line: {'[legacy-ipc]' in patched}")
    print()
    print("What to look for in next run log:")
    print("  DEBUG [slim_v1] 192.168.3.120:GPU0 chunk → 150 survivors  ← GOOD")
    print("  WARN  [legacy-ipc] 192.168.3.120:GPU0 — worker sent legacy  ← BAD (rig not updated)")


if __name__ == "__main__":
    apply()
