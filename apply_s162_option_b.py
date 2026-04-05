#!/usr/bin/env python3
"""
apply_s162_option_b.py
========================
TB-approved Option B diagnostic: Add AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3
to rrig6600c workers ONLY.

TB Ruling (S162):
  "TB approves AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 on rrig6600c only
  as the next targeted diagnostic. ROCm explicitly recommends these variables
  to force synchronous execution/copies so the true faulting kernel site is
  easier to identify. This is exactly the right next step for a
  concurrency-sensitive crash."

Background:
  rrig6600c crashes only under full 3-rig simultaneous load. Latest crash
  shows SQC (inst) fault — shader instruction cache fetch failure. TB diagnosis:
  this is a 3-rig concurrency-triggered system bug whose first visible casualty
  is rrig6600c, not a rig-specific weakness.

  AMD_SERIALIZE_KERNEL=3: Wait for completion before AND after each kernel enqueue
  AMD_SERIALIZE_COPY=3:   Wait for completion before AND after each copy enqueue

  This forces synchronous GPU execution on rrig6600c workers, eliminating async
  overlap as a potential crash trigger. If this stabilizes the run, the remaining
  bug is confirmed as async/concurrency-timing sensitive.

  This is a DIAGNOSTIC setting — not a permanent production setting.
  Throughput on rrig6600c will be reduced (~50-80% per GPU). Acceptable for
  one diagnostic run.

Patch: In TCP worker launch sequence in persistent_worker_coordinator.py,
add diagnostic env vars to rocm_env only when host == '192.168.3.162'.
"""

import ast
import sys
import os

TARGET = "persistent_worker_coordinator.py"

# The injection point: rocm_env construction in TCP launch loop
# We add per-host conditional after the base rocm_env is built

OLD_TCP_ENV = '''            for gpu_id in range(pool):
                rocm_env = " ".join(ROCM_ENV_VARS + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])'''

NEW_TCP_ENV = '''            for gpu_id in range(pool):
                # S162 Option B (TB-approved diagnostic): AMD_SERIALIZE_KERNEL=3
                # AMD_SERIALIZE_COPY=3 forces synchronous GPU execution on rrig6600c
                # only. Diagnostic for 3-rig concurrency-triggered SQC fault.
                # TB ruling: "not a permanent production setting."
                _diag_vars = []
                if host == "192.168.3.162":
                    _diag_vars = [
                        "AMD_SERIALIZE_KERNEL=3",
                        "AMD_SERIALIZE_COPY=3",
                    ]
                    self.logger.info(
                        "[PWC-TCP] [S162-OPT-B] 192.168.3.162: "
                        "AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 "
                        "diagnostic enabled (TB-approved)"
                    )
                rocm_env = " ".join(ROCM_ENV_VARS + _diag_vars + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])'''

# Also need to export the diag vars in the bash script_lines section
OLD_SCRIPT_VARS = '''                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),'''

NEW_SCRIPT_VARS = '''                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                # S162 Option B: export diagnostic vars for rrig6600c
                for var in _diag_vars:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),'''


def apply():
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found. Run from distributed_prng_analysis/")
        sys.exit(1)

    with open(TARGET) as f:
        src = f.read()

    # Check not already patched
    if "S162-OPT-B" in src:
        print("Already patched — S162-OPT-B marker present. Skipping.")
        sys.exit(0)

    # Validate anchors
    for name, anchor in [("OLD_TCP_ENV", OLD_TCP_ENV), ("OLD_SCRIPT_VARS", OLD_SCRIPT_VARS)]:
        count = src.count(anchor)
        if count != 1:
            print(f"ERROR: {name} found {count} times (expected 1). Aborting.")
            sys.exit(1)

    src = src.replace(OLD_TCP_ENV, NEW_TCP_ENV)
    src = src.replace(OLD_SCRIPT_VARS, NEW_SCRIPT_VARS)

    # AST validate
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: AST validation failed: {e}")
        sys.exit(1)

    with open(TARGET, "w") as f:
        f.write(src)

    print("✅ Option B patch applied successfully.")
    print()
    print("Changes made to persistent_worker_coordinator.py:")
    print("  - AMD_SERIALIZE_KERNEL=3 added to rrig6600c (192.168.3.162) workers ONLY")
    print("  - AMD_SERIALIZE_COPY=3 added to rrig6600c (192.168.3.162) workers ONLY")
    print("  - All other rigs unchanged")
    print()
    print("This file runs on ZEUS only — no rig deployment needed.")
    print()
    print("Verify:")
    print("  grep -n 'S162-OPT-B\\|AMD_SERIALIZE' persistent_worker_coordinator.py")
    print()
    print("Expected log line during startup:")
    print("  [PWC-TCP] [S162-OPT-B] 192.168.3.162: AMD_SERIALIZE_KERNEL=3 ...")
    print()
    print("REMINDER: This is a diagnostic run only.")
    print("  - rrig6600c throughput will be ~50-80% lower than normal")
    print("  - If 3-rig run stabilizes → async overlap confirmed as crash trigger")
    print("  - If still crashes → proceed to Option A on rrig6600c only")
    print("  - Remove this patch after diagnostic is complete")


if __name__ == "__main__":
    apply()
