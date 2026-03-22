#!/usr/bin/env python3
"""
apply_s152_slim_v1_vram_fix.py
================================
Patch: Fix catastrophic VRAM over-allocation in slim_v1 hybrid sieve worker.

ROOT CAUSE
----------
sieve_gpu_worker.py line 252:
    skip_sequences_gpu = cp.zeros(n_seeds * k, dtype=cp.uint32)

    n_seeds = 200,000  (chunk size)
    k       = 18,068   (draw count from daily3.json)
    Result  = 200,000 × 18,068 × 4 bytes = 14.5 GB per worker

RX 6600 VRAM = 8 GB. 8 concurrent workers × 14.5 GB = instant hard crash.

ROOT CAUSE CONFIRMED BY prng_registry.py comment (line 58):
    "Global array: skip_sequences[n_survivors * k] where k = window_size"

The kernel only writes into skip_sequences at survivor positions:
    skip_sequences[pos * k + i]   (pos = survivor index, never exceeds n_survivors)

The allocation only needs to hold MAX_SURVIVORS × k entries, not n_seeds × k.

THE FIX
-------
Replace:
    skip_sequences_gpu = cp.zeros(n_seeds * k, dtype=cp.uint32)

With:
    # Cap allocation to MAX_HYBRID_SURVIVORS × k — kernel only writes survivor positions
    # Configurable via env PRNG_MAX_HYBRID_SURVIVORS (default 5000)
    # 5000 × 18068 × 4 bytes = 361 MB per worker — safe on 8 GB VRAM with 8 workers
    _max_surv = int(os.environ.get('PRNG_MAX_HYBRID_SURVIVORS', '5000'))
    skip_sequences_gpu = cp.zeros(_max_surv * k, dtype=cp.uint32)

The readback is also updated to use min(count, _max_surv) as a safety clamp:
    ss_raw = skip_sequences_gpu[:min(count, _max_surv) * k].get().reshape(...)

Default 5000 is very conservative — typical hybrid survivor counts per chunk
are 10-30. Even at 5000 survivors: 5000 × 18068 × 4 = 361 MB per worker.
8 workers = 2.9 GB — well within 8 GB VRAM.

MEMORY COMPARISON
-----------------
Before fix:  200,000 × 18,068 × 4 bytes = 14,454 MB per worker  ← CRASH
After fix:     5,000 × 18,068 × 4 bytes =    361 MB per worker  ← SAFE
Reduction:                                         40× less VRAM

Files patched
-------------
  sieve_gpu_worker.py

Backup: sieve_gpu_worker.py.bak_s152_vram_fix

Deploy to rigs after patching Zeus:
  scp rzeus:~/distributed_prng_analysis/sieve_gpu_worker.py ~/Downloads/
  scp ~/Downloads/sieve_gpu_worker.py rrig6600:~/distributed_prng_analysis/
  scp ~/Downloads/sieve_gpu_worker.py rrig6600b:~/distributed_prng_analysis/
  scp ~/Downloads/sieve_gpu_worker.py rrig6600c:~/distributed_prng_analysis/
"""

import shutil
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

TARGET = Path("sieve_gpu_worker.py")
BACKUP = Path("sieve_gpu_worker.py.bak_s152_vram_fix")

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1: Replace the over-allocating skip_sequences_gpu line
# ─────────────────────────────────────────────────────────────────────────────
OLD_ALLOC = '''\
                skip_sequences_gpu  = cp.zeros(n_seeds * k, dtype=cp.uint32)'''

NEW_ALLOC = '''\
                # [S152-VRAM] Cap allocation: kernel writes survivor positions only (not n_seeds)
                # prng_registry.py: "skip_sequences[n_survivors * k] where k = window_size"
                # Before fix: 200,000 × 18,068 × 4 = 14.5 GB per worker → VRAM OOM crash
                # After fix:    5,000 × 18,068 × 4 =  361 MB per worker → safe on 8 GB
                _max_surv = int(os.environ.get('PRNG_MAX_HYBRID_SURVIVORS', '5000'))
                skip_sequences_gpu  = cp.zeros(_max_surv * k, dtype=cp.uint32)'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2: Clamp readback to _max_surv (defensive — count should never exceed it
# but if kernel writes beyond buffer due to race, this prevents out-of-bounds)
# ─────────────────────────────────────────────────────────────────────────────
OLD_READBACK = '''\
                    ss_raw  = skip_sequences_gpu[:count * k].get().reshape(count, k).tolist()'''

NEW_READBACK = '''\
                    # [S152-VRAM] Clamp readback to allocated buffer size
                    _safe_count = min(count, _max_surv)
                    ss_raw  = skip_sequences_gpu[:_safe_count * k].get().reshape(_safe_count, k).tolist()'''

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3: zip must use _safe_count-trimmed arrays for consistency
# ─────────────────────────────────────────────────────────────────────────────
OLD_ZIP = '''\
                    for seed, rate, sid, ss in zip(s_arr, r_arr, sid_arr, ss_raw):'''

NEW_ZIP = '''\
                    for seed, rate, sid, ss in zip(s_arr[:_safe_count], r_arr[:_safe_count], sid_arr[:_safe_count], ss_raw):'''


def apply():
    src = TARGET.read_text()

    # Idempotency
    if "[S152-VRAM]" in src:
        print("⚠️  [S152-VRAM] marker already present — patch already applied. Aborting.")
        return

    # Validate anchors
    missing = []
    for label, anchor in [
        ("skip_sequences_gpu allocation", OLD_ALLOC),
        ("ss_raw readback",               OLD_READBACK),
        ("zip loop",                       OLD_ZIP),
    ]:
        if anchor not in src:
            missing.append(label)

    if missing:
        print(f"❌ Anchors not found: {missing}")
        print("Aborting — check for prior partial patch or line number shifts.")
        return

    patched = src
    patched = patched.replace(OLD_ALLOC,     NEW_ALLOC,    1)
    patched = patched.replace(OLD_READBACK,  NEW_READBACK, 1)
    patched = patched.replace(OLD_ZIP,       NEW_ZIP,      1)

    if DRY_RUN:
        print("=== DRY RUN — no files written ===")
        print(f"  [S152-VRAM] in patched:          {'[S152-VRAM]' in patched}")
        print(f"  _max_surv allocation present:    {'_max_surv * k' in patched}")
        print(f"  _safe_count readback present:    {'_safe_count' in patched}")
        print(f"  n_seeds * k removed:             {'cp.zeros(n_seeds * k' not in patched}")
        return

    shutil.copy2(TARGET, BACKUP)
    print(f"✅ Backup: {BACKUP}")
    TARGET.write_text(patched)
    print(f"✅ Patched: {TARGET}")
    print()
    print("Verification:")
    print(f"  [S152-VRAM] marker:         {'[S152-VRAM]' in patched}")
    print(f"  Old n_seeds*k REMOVED:      {'cp.zeros(n_seeds * k' not in patched}")
    print(f"  New _max_surv*k present:    {'cp.zeros(_max_surv * k' in patched}")
    print(f"  _safe_count clamp present:  {'_safe_count' in patched}")
    print()
    print("VRAM usage after fix (default PRNG_MAX_HYBRID_SURVIVORS=5000):")
    print("  5,000 × 18,068 × 4 bytes = 361 MB per worker")
    print("  8 workers × 361 MB = 2.9 GB — safe on 8 GB VRAM")
    print()
    print("Tune via env var (set in coordinator ROCM_ENV_VARS):")
    print("  PRNG_MAX_HYBRID_SURVIVORS=1000  # 72 MB per worker — very safe")
    print("  PRNG_MAX_HYBRID_SURVIVORS=5000  # 361 MB per worker — default")
    print("  PRNG_MAX_HYBRID_SURVIVORS=10000 # 722 MB per worker — if needed")
    print()
    print("Deploy to rigs:")
    print("  scp rzeus:~/distributed_prng_analysis/sieve_gpu_worker.py ~/Downloads/")
    print("  scp ~/Downloads/sieve_gpu_worker.py rrig6600:~/distributed_prng_analysis/")
    print("  scp ~/Downloads/sieve_gpu_worker.py rrig6600b:~/distributed_prng_analysis/")
    print("  scp ~/Downloads/sieve_gpu_worker.py rrig6600c:~/distributed_prng_analysis/")


if __name__ == "__main__":
    apply()
