#!/usr/bin/env python3
"""
apply_s162_exp1_scratch_reclaim.py
====================================
TB-directed Experiment 1: Add HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
to rrig6600c workers ONLY. All other rigs unchanged.

TB Ruling (S162 Option B follow-up):
  "Best next experiment: test ROCR scratch-reclaim behavior on rrig6600c
  only, while keeping the full 3-rig / 26-GPU topology.
  AMD's ROCR docs say HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1 allows ROCr,
  on device-memory allocation failure, to reclaim scratch memory assigned
  to all queues and retry allocation. That is exactly the kind of
  queue-associated memory behavior that could surface only under full-cluster
  concurrency."

Background:
  Option B (AMD_SERIALIZE_KERNEL=3) did not prevent the crash. Serial kernel
  execution makes no difference — async overlap is ruled out. The crash still
  begins with near-simultaneous SQC (inst) faults on two GPUs simultaneously,
  followed by full KFD queue collapse and CPU soft lockup.

  New hypothesis: ROCr scratch memory reclaim under full 3-rig concurrency
  is invalidating GPU virtual address mappings (including instruction cache
  pages), triggering the SQC (inst) fault. Setting
  HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 disables this reclaim behavior on
  rrig6600c workers only.

  This patch REPLACES the Option B AMD_SERIALIZE_KERNEL patch on Zeus.
  It must be applied on top of the Zeus live coordinator (which has Option B).

Experiment result interpretation:
  - Crash disappears → scratch async reclaim is the trigger → strong signal
  - Still crashes → proceed to Experiment 2 (HSA_NO_SCRATCH_RECLAIM=1)

NOTE: This patch handles two cases:
  1. Applied to clean repo version (no Option B)
  2. Applied to Zeus live version (has Option B from apply_s162_option_b.py)
"""

import ast
import sys
import os

TARGET = "persistent_worker_coordinator.py"


# ── Case 1: Clean repo anchor (no Option B) ──────────────────────────────────

CLEAN_OLD = '''            for gpu_id in range(pool):
                rocm_env = " ".join(ROCM_ENV_VARS + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])'''

CLEAN_NEW = '''            for gpu_id in range(pool):
                # S162-EXP1 (TB-directed): HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
                # on rrig6600c (192.168.3.162) only.
                # Disables ROCr async scratch reclaim which may invalidate GPU
                # VA mappings (including instruction cache) under full 3-rig
                # concurrency. TB experiment 1 of scratch-reclaim probe series.
                _exp1_vars = []
                if host == "192.168.3.162":
                    _exp1_vars = ["HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0"]
                    self.logger.info(
                        "[PWC-TCP] [S162-EXP1] 192.168.3.162: "
                        "HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 active (TB Exp 1)"
                    )
                rocm_env = " ".join(ROCM_ENV_VARS + _exp1_vars + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])'''

CLEAN_OLD_SCRIPT = '''                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),'''

CLEAN_NEW_SCRIPT = '''                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                # S162-EXP1: export scratch reclaim disable for rrig6600c
                for var in _exp1_vars:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),'''


# ── Case 2: Zeus live version anchor (has Option B) ──────────────────────────

OPT_B_OLD = '''                _diag_vars = []
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

OPT_B_NEW = '''                # S162-EXP1 (TB-directed): Replaces Option B diagnostic.
                # HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 on rrig6600c only.
                # Option B (AMD_SERIALIZE_KERNEL=3) ruled out async overlap.
                # Now testing ROCr scratch reclaim as crash trigger.
                _exp1_vars = []
                if host == "192.168.3.162":
                    _exp1_vars = ["HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0"]
                    self.logger.info(
                        "[PWC-TCP] [S162-EXP1] 192.168.3.162: "
                        "HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 active (TB Exp 1)"
                    )
                rocm_env = " ".join(ROCM_ENV_VARS + _exp1_vars + [
                    f"ROCR_VISIBLE_DEVICES={gpu_id}",
                    f"CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}",
                ])'''

OPT_B_OLD_SCRIPT = '''                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                # S162 Option B: export diagnostic vars for rrig6600c
                for var in _diag_vars:
                    script_lines.append("export " + var)
                script_lines += [
                    "export ROCR_VISIBLE_DEVICES=" + str(gpu_id),
                    "export CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_" + str(gpu_id),'''

OPT_B_NEW_SCRIPT = '''                for var in ROCM_ENV_VARS:
                    script_lines.append("export " + var)
                # S162-EXP1: export scratch reclaim disable for rrig6600c
                for var in _exp1_vars:
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
    if "S162-EXP1" in src:
        print("Already patched — S162-EXP1 marker present. Skipping.")
        sys.exit(0)

    # Determine which case we're in
    has_opt_b = "S162-OPT-B" in src
    has_clean = CLEAN_OLD in src

    if has_opt_b:
        print("Detected Option B on Zeus — replacing with Experiment 1...")
        env_count = src.count(OPT_B_OLD)
        script_count = src.count(OPT_B_OLD_SCRIPT)
        if env_count != 1 or script_count != 1:
            print(f"ERROR: Option B anchors found {env_count}/{script_count} times. Aborting.")
            sys.exit(1)
        src = src.replace(OPT_B_OLD, OPT_B_NEW)
        src = src.replace(OPT_B_OLD_SCRIPT, OPT_B_NEW_SCRIPT)
    elif has_clean:
        print("Detected clean repo — applying Experiment 1 fresh...")
        env_count = src.count(CLEAN_OLD)
        script_count = src.count(CLEAN_OLD_SCRIPT)
        if env_count != 1 or script_count != 1:
            print(f"ERROR: Clean anchors found {env_count}/{script_count} times. Aborting.")
            sys.exit(1)
        src = src.replace(CLEAN_OLD, CLEAN_NEW)
        src = src.replace(CLEAN_OLD_SCRIPT, CLEAN_NEW_SCRIPT)
    else:
        print("ERROR: Could not detect repo state (neither clean nor Option B). Aborting.")
        sys.exit(1)

    # AST validate
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: AST validation failed: {e}")
        sys.exit(1)

    with open(TARGET, "w") as f:
        f.write(src)

    print("✅ Experiment 1 patch applied successfully.")
    print()
    print("Changes made to persistent_worker_coordinator.py:")
    print("  HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 added to rrig6600c (192.168.3.162) ONLY")
    print("  All other rigs: unchanged")
    print()
    print("This file runs on ZEUS only — no rig deployment needed.")
    print()
    print("Verify:")
    print("  grep -n 'S162-EXP1\\|SCRATCH_ASYNC' persistent_worker_coordinator.py")
    print()
    print("Expected log line during startup:")
    print("  [PWC-TCP] [S162-EXP1] 192.168.3.162: HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 ...")
    print()
    print("Experiment result interpretation:")
    print("  STABLE → scratch async reclaim is the trigger → proceed to fix")
    print("  CRASH  → proceed to Experiment 2 (HSA_NO_SCRATCH_RECLAIM=1)")


if __name__ == "__main__":
    apply()
