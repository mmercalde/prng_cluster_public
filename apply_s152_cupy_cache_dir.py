#!/usr/bin/env python3
"""
apply_s152_cupy_cache_dir.py
=============================
Patch: Give each worker its own CUPY_CACHE_DIR (TB first-choice fix).

ROOT CAUSE (TB ruling S152)
---------------------------
slim_v1 uses cp.cuda.Device(gpu_id) with all 8 GPUs visible per worker.
When 8 workers spawn simultaneously and each calls cp.RawKernel() for the
first time, they all try to write to the same ~/.cupy/kernel_cache directory
concurrently. CuPy's kernel cache is not safe for concurrent multi-process
writes on ROCm/HIP. This race condition crashes one or more workers,
bringing down the entire rig.

TB RULING
---------
First choice: give each worker its own CUPY_CACHE_DIR.
"That is a documented, supported knob, and it directly removes shared
on-disk cache contention without disabling caching entirely."

FIX
---
In _spawn_worker(), add CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}
to the per-worker environment. Each of the 8 workers on a rig gets a
completely isolated cache directory under /tmp. No shared writes.
No race condition.

Per-worker cache dirs are in /tmp so they are:
- Unique per GPU (no contention)
- Wiped on reboot (clean state after every reboot — no stale cache artifacts)
- Fast (tmpfs on most Linux systems)

The first time each worker starts after a reboot, it compiles once and
caches to its own /tmp dir. Subsequent runs reuse the cache (within the
same boot). After reboot, each worker recompiles once — acceptable cost
given the 90s heartbeat timeout.

Files patched
-------------
  persistent_worker_coordinator.py

Backup: persistent_worker_coordinator.py.bak_s152_cupy_cache_dir
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("persistent_worker_coordinator.py")
BACKUP = Path("persistent_worker_coordinator.py.bak_s152_cupy_cache_dir")

OLD_ROCM_ENV = '''\
        rocm_env = " ".join(ROCM_ENV_VARS + [
            # [S149-B] HIP/CUDA per-worker masking removed
            # Workers see all GPUs; Device(gpu_id) selects directly in worker
        ])'''

NEW_ROCM_ENV = '''\
        # [S152-TB] Per-worker CUPY_CACHE_DIR — TB first-choice fix for slim_v1 crash.
        # Each worker gets an isolated kernel cache under /tmp/cupy_cache_gpu_N.
        # Eliminates concurrent multi-process writes to shared ~/.cupy/kernel_cache,
        # which causes a ROCm/HIP race condition when 8 workers start simultaneously.
        # /tmp is used so cache is wiped on reboot (clean state, no stale artifacts).
        _per_worker_cache = f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}"
        rocm_env = " ".join(ROCM_ENV_VARS + [
            _per_worker_cache,
            # [S149-B] HIP/CUDA per-worker masking removed
            # Workers see all GPUs; Device(gpu_id) selects directly in worker
        ])'''


def apply():
    src = TARGET.read_text()

    if "[S152-TB] Per-worker CUPY_CACHE_DIR" in src:
        print("⚠️  Already patched — aborting.")
        return

    if OLD_ROCM_ENV not in src:
        print("❌ Anchor not found.")
        return

    patched = src.replace(OLD_ROCM_ENV, NEW_ROCM_ENV, 1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  [S152-TB] marker present:      {'[S152-TB] Per-worker CUPY_CACHE_DIR' in patched}")
        print(f"  _per_worker_cache present:     {'_per_worker_cache' in patched}")
        print(f"  cupy_cache_gpu_ present:       {'cupy_cache_gpu_' in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Each worker now uses: CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_N")
    print("  GPU 0 → /tmp/cupy_cache_gpu_0")
    print("  GPU 1 → /tmp/cupy_cache_gpu_1")
    print("  ...")
    print("  GPU 7 → /tmp/cupy_cache_gpu_7")
    print()
    print("No shared kernel cache writes — race condition eliminated.")
    print("Cache lives in /tmp — wiped on reboot for clean state.")


if __name__ == "__main__":
    apply()
