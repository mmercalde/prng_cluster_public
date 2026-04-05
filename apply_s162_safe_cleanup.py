#!/usr/bin/env python3
"""
apply_s162_safe_cleanup.py
===========================
S162 fix: Replace _best_effort_gpu_cleanup() in TCP worker with safe
gc-only cleanup.

Root cause confirmed via netconsole:
  First fault: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231
  Faulty client: SQC (inst) — shader INSTRUCTION cache fault
  This means the GPU tried to fetch a kernel instruction that was
  unmapped. _best_effort_gpu_cleanup() calls free_all_blocks() which
  on rrig6600c's ROCm driver aggressively unmaps GPU memory including
  live instruction mappings, causing the next chunk's kernel launch
  to fault.

  rrig6600 and rrig6600b are not affected — their ROCm driver handles
  free_all_blocks() more conservatively.

Fix: Replace _best_effort_gpu_cleanup() call with inline gc.collect()
only. This releases Python object references without touching the CuPy
memory pool or GPU instruction mappings.

The memory accumulation concern (S160) is addressed by gc.collect()
alone — explicit del of GPU arrays in the sieve job + gc.collect()
is sufficient without free_all_blocks().
"""

import ast
import sys
import os

TARGET = "persistent/pwc_worker_service.py"

OLD_CLEANUP = '''            # S162: best-effort GPU cleanup between chunks — prevents memory
            # accumulation crash (page fault → qcm timeout → rig reboot).
            # Same fix as S160 ZMQ worker. Placement: after result delivery,
            # before next job fetch so cleanup never delays or suppresses results.
            try:
                from sieve_filter import _best_effort_gpu_cleanup
                _best_effort_gpu_cleanup()
            except Exception as _cleanup_exc:
                log.debug(f\"[{self.worker_id}] GPU cleanup skipped: {_cleanup_exc}\")'''

NEW_CLEANUP = '''            # S162b: Safe gc-only cleanup between chunks.
            # _best_effort_gpu_cleanup() (which calls free_all_blocks()) caused
            # SQC instruction cache faults on rrig6600c — ROCm driver on that rig
            # aggressively unmaps GPU memory including live kernel instruction
            # mappings, causing next chunk kernel launch to fault.
            # Fix: gc.collect() only — releases Python refs without touching
            # CuPy memory pool or GPU instruction cache.
            try:
                import gc
                gc.collect()
            except Exception:
                pass'''


def apply():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found. Run from distributed_prng_analysis/")
        sys.exit(1)

    with open(TARGET) as f:
        src = f.read()

    # Validate anchor exists exactly once
    count = src.count(OLD_CLEANUP)
    if count != 1:
        print(f"ERROR: anchor found {count} times (expected 1). Aborting.")
        sys.exit(1)

    # Check not already patched
    if "S162b" in src:
        print("Already patched — S162b marker present. Skipping.")
        sys.exit(0)

    src = src.replace(OLD_CLEANUP, NEW_CLEANUP)

    # AST validate
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: AST validation failed: {e}")
        sys.exit(1)

    with open(TARGET, "w") as f:
        f.write(src)

    print("✅ Patch applied successfully.")
    print()
    print("Changes made to persistent/pwc_worker_service.py:")
    print("  Replaced _best_effort_gpu_cleanup() with inline gc.collect() only")
    print("  Eliminates free_all_blocks() which unmaps GPU instruction cache on rrig6600c")
    print()
    print("Deploy to ALL 3 rigs:")
    print("  scp persistent/pwc_worker_service.py rrig6600:~/distributed_prng_analysis/persistent/")
    print("  scp persistent/pwc_worker_service.py rrig6600b:~/distributed_prng_analysis/persistent/")
    print("  scp persistent/pwc_worker_service.py rrig6600c:~/distributed_prng_analysis/persistent/")


if __name__ == "__main__":
    apply()
