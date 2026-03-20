#!/usr/bin/env python3
"""
verify_s149b_device_fix.py  (v2 — TB revised ruling)
======================================================
Post-fix verification for S149-B Direct GPU Selection architecture.

Verifies sieve_gpu_worker.py:
  1. Device(0) hardcoding fully removed
  2. Device(gpu_id) present in warmup and run_sieve_job
  3. run_sieve_job signature accepts gpu_id parameter
  4. gpu_id mismatch assertion present
  5. run_sieve_job called with gpu_id in run_worker
  6. Design comment updated

Verifies persistent_worker_coordinator.py:
  7. HIP_VISIBLE_DEVICES per-worker masking REMOVED
  8. CUDA_VISIBLE_DEVICES per-worker masking REMOVED
  9. ROCR_VISIBLE_DEVICES still absent
  10. HSA vars unchanged (HSA_OVERRIDE_GFX_VERSION, HSA_ENABLE_SDMA)
  11. S146 invariants preserved

Usage:
  python3 verify_s149b_device_fix.py
"""

import re
import sys

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    if condition:
        PASS += 1
    else:
        FAIL += 1
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load(path):
    with open(path) as f:
        return f.read()


WORKER = load("sieve_gpu_worker.py")
PWC = load("persistent_worker_coordinator.py")

# ---------------------------------------------------------------------------
# Section 1 — Device(0) fully removed from worker
# ---------------------------------------------------------------------------

section("1 — Device(0) hardcoding removed from sieve_gpu_worker.py")

device0_refs = list(re.finditer(r'cp\.cuda\.Device\(0\)', WORKER))
device_gpuid_refs = list(re.finditer(r'cp\.cuda\.Device\(gpu_id\)', WORKER))

check(
    f"cp.cuda.Device(0) fully removed",
    len(device0_refs) == 0,
    detail=f"Found {len(device0_refs)} occurrence(s) — expect 0"
)
check(
    f"cp.cuda.Device(gpu_id) present ({len(device_gpuid_refs)} occurrence(s))",
    len(device_gpuid_refs) >= 3,
    detail="Expect ≥3: warmup (×2) + run_sieve_job"
)
check(
    "getDeviceProperties(0) removed",
    "getDeviceProperties(0)" not in WORKER,
)
check(
    "getDeviceProperties(gpu_id) present",
    "getDeviceProperties(gpu_id)" in WORKER,
)

# ---------------------------------------------------------------------------
# Section 2 — run_sieve_job signature and call
# ---------------------------------------------------------------------------

section("2 — run_sieve_job(job, gpu_id) signature and call")

check(
    "run_sieve_job accepts gpu_id parameter",
    "def run_sieve_job(job: dict, gpu_id: int" in WORKER,
    detail="TB ruling: explicit parameter preferred over job.get('gpu_id')"
)
check(
    "gpu_id mismatch assertion present",
    "gpu_id mismatch" in WORKER,
    detail="Fail loudly if job gpu_id contradicts worker gpu_id"
)
check(
    "run_sieve_job called with gpu_id in run_worker",
    "run_sieve_job(job, gpu_id)" in WORKER,
    detail="Worker passes its own gpu_id to job execution"
)

# ---------------------------------------------------------------------------
# Section 3 — Design comment updated
# ---------------------------------------------------------------------------

section("3 — Design comment reflects direct-selection architecture")

check(
    "Old ROCR comment removed",
    "ROCR_VISIBLE_DEVICES has isolated the GPU" not in WORKER,
)
check(
    "S149-B comment present",
    "S149-B" in WORKER,
)
check(
    "No HIP/CUDA masking comment present",
    "no HIP/CUDA" in WORKER or "no HIP/CUDA/ROCR" in WORKER,
    detail="Documents the coherent architecture"
)

# ---------------------------------------------------------------------------
# Section 4 — Spawner: HIP/CUDA masking removed, HSA unchanged
# ---------------------------------------------------------------------------

section("4 — Spawner: per-worker masking removed, HSA vars intact")

spawn_block_match = re.search(
    r'rocm_env\s*=\s*" "\.join\(ROCM_ENV_VARS\s*\+\s*\[(.*?)\]\)',
    PWC, re.DOTALL
)
spawn_block = spawn_block_match.group(1) if spawn_block_match else ""

check(
    "HIP_VISIBLE_DEVICES REMOVED from _spawn_worker",
    "HIP_VISIBLE_DEVICES" not in spawn_block,
    detail="TB ruling: remove per-worker masking — workers see all GPUs"
)
check(
    "CUDA_VISIBLE_DEVICES REMOVED from _spawn_worker",
    "CUDA_VISIBLE_DEVICES" not in spawn_block,
    detail="TB ruling: coherent architecture — no masking + direct Device(gpu_id)"
)
check(
    "ROCR_VISIBLE_DEVICES still absent from spawner",
    "ROCR_VISIBLE_DEVICES" not in PWC,
    detail="Live test: causes hipErrorNoDevice on this stack"
)

rocm_env_match = re.search(r'ROCM_ENV_VARS\s*=\s*\[(.*?)\]', PWC, re.DOTALL)
rocm_env_block = rocm_env_match.group(1) if rocm_env_match else ""

check(
    "HSA_OVERRIDE_GFX_VERSION still in ROCM_ENV_VARS",
    "HSA_OVERRIDE_GFX_VERSION" in rocm_env_block,
    detail="Unchanged — TB: don't mix changes"
)
check(
    "HSA_ENABLE_SDMA still in ROCM_ENV_VARS",
    "HSA_ENABLE_SDMA" in rocm_env_block,
    detail="Unchanged"
)

# Confirm Zeus local dispatch path untouched (line ~421)
check(
    "Zeus local dispatch CUDA_VISIBLE_DEVICES still present (_dispatch_local_sieve)",
    'env["CUDA_VISIBLE_DEVICES"]' in PWC,
    detail="This is sieve_filter.py path on Zeus — not touched by S149-B"
)

# ---------------------------------------------------------------------------
# Section 5 — S146 invariants preserved
# ---------------------------------------------------------------------------

section("5 — S146 invariants preserved")

check("JOB_TIMEOUT_S still 600", "JOB_TIMEOUT_S = 600" in PWC)
check("_localhost_semaphore(2) still present", "threading.Semaphore(2)" in PWC)
check(
    "ROCM_SPAWN_STAGGER_S still 4.0s",
    bool(re.search(r'ROCM_SPAWN_STAGGER_S\s*=\s*4\.0', PWC)),
    detail="TB: keep for first validation pass"
)
check(
    "worker_pool_size configurable",
    bool(re.search(r'worker_pool_size\s*(?::\s*int)?\s*=\s*\d+', PWC)),
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS}/{total} checks passed")
print(f"{'='*60}")

if FAIL == 0:
    print("""
  ✓ S149-B Direct GPU Selection fix verified clean.

  Architecture confirmed coherent:
    - Workers see all GPUs (no HIP/CUDA masking in spawner)
    - Workers bind via Device(gpu_id) directly
    - run_sieve_job(job, gpu_id) explicit parameter
    - Mismatch assertion guards against coordinator bugs

  Proceed to live validation (TB sequence):

  STEP 1 — Manual per-device spawn (no masking env vars):
    ssh rrig6600 and run:
      HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \\
      python3 sieve_gpu_worker.py --gpu-id 0 --persistent
    Then separately:
      HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_ENABLE_SDMA=0 \\
      python3 sieve_gpu_worker.py --gpu-id 7 --persistent
    Both must print 'GPU ready' and emit {status: ready}

  STEP 2 — Single-rig 8-worker smoke (worker_pool_size=8):
    □ All 8 workers heartbeat — no quarantines
    □ rocm-smi --showuse shows GPU% > 0 on all 8

  STEP 3 — Short soak (sweep_preprod.sh, 5 trials, all rigs)

  STEP 4 — Commit + production relaunch
""")
else:
    print(f"\n  {FAIL} check(s) FAILED — review above before proceeding.")

sys.exit(0 if FAIL == 0 else 1)
