#!/usr/bin/env python3
"""
verify_s149a_rocr_fix.py
=========================
Post-fix verification harness for the S149-A ROCR isolation patch.

TB note: "The cleanest post-fix success condition would be:
  old harness fails in the expected two places,
  new verify harness passes 100%."

This harness verifies:
  - ROCR_VISIBLE_DEVICES IS present in _spawn_worker()
  - HIP_VISIBLE_DEVICES still present (not removed)
  - CUDA_VISIBLE_DEVICES still present (not removed)
  - No unintended changes to ROCM_ENV_VARS block
  - sieve_gpu_worker.py still uses Device(0) — correct by design
  - No other device references changed
  - Spawn stagger still 4.0s (TB: keep for first validation pass)
  - worker_pool_size default still accessible (not hardcoded to 4)

Usage:
  python3 verify_s149a_rocr_fix.py
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


PWC    = load("persistent_worker_coordinator.py")
WORKER = load("sieve_gpu_worker.py")

# ---------------------------------------------------------------------------
# Section 1 — ROCR fix is present
# ---------------------------------------------------------------------------

section("1 — ROCR_VISIBLE_DEVICES fix applied")

spawn_block_match = re.search(
    r'rocm_env\s*=\s*" "\.join\(ROCM_ENV_VARS\s*\+\s*\[(.*?)\]\)',
    PWC, re.DOTALL
)
spawn_block = spawn_block_match.group(1) if spawn_block_match else ""

check(
    "_spawn_worker env construction block found",
    bool(spawn_block_match),
)
check(
    "ROCR_VISIBLE_DEVICES={gpu_id} IS present in _spawn_worker",
    "ROCR_VISIBLE_DEVICES" in spawn_block,
    detail="This is the S149-A fix — authoritative HSA-level isolation"
)
check(
    "HIP_VISIBLE_DEVICES={gpu_id} still present (not removed)",
    "HIP_VISIBLE_DEVICES" in spawn_block,
    detail="Keep — covers HIP runtime layer per TB ruling"
)
check(
    "CUDA_VISIBLE_DEVICES={gpu_id} still present (not removed)",
    "CUDA_VISIBLE_DEVICES" in spawn_block,
    detail="Keep — covers CUDA compat layer"
)

# Verify all three are in the right place (per-spawn, not ROCM_ENV_VARS)
rocm_env_match = re.search(r'ROCM_ENV_VARS\s*=\s*\[(.*?)\]', PWC, re.DOTALL)
rocm_env_block = rocm_env_match.group(1) if rocm_env_match else ""

check(
    "ROCR_VISIBLE_DEVICES NOT in ROCM_ENV_VARS (correctly per-spawn only)",
    "ROCR_VISIBLE_DEVICES" not in rocm_env_block,
    detail="Should be per-spawn with gpu_id, not a static global"
)
check(
    "ROCM_ENV_VARS still contains HSA_OVERRIDE_GFX_VERSION",
    "HSA_OVERRIDE_GFX_VERSION" in rocm_env_block,
    detail="Unchanged — TB: don't mix changes"
)
check(
    "ROCM_ENV_VARS still contains HSA_ENABLE_SDMA",
    "HSA_ENABLE_SDMA" in rocm_env_block,
    detail="Unchanged — orthogonal to visibility per TB ruling"
)

# ---------------------------------------------------------------------------
# Section 2 — sieve_gpu_worker.py unchanged
# ---------------------------------------------------------------------------

section("2 — sieve_gpu_worker.py Device(0) design intact")

worker_device0 = list(re.finditer(r'cp\.cuda\.Device\(0\)', WORKER))
worker_device_gpuid = list(re.finditer(r'cp\.cuda\.Device\(gpu_id\)', WORKER))

check(
    f"cp.cuda.Device(0) still hardcoded ({len(worker_device0)} occurrence(s))",
    len(worker_device0) >= 2,
    detail="Correct by design — ROCR remaps device 0 to assigned GPU"
)
check(
    "cp.cuda.Device(gpu_id) NOT introduced (design unchanged)",
    len(worker_device_gpuid) == 0,
    detail="Worker should not be changed — spawner provides isolation"
)

# Confirm design intent comment still present
check(
    "Design intent comment still present in sieve_gpu_worker.py",
    "ROCR_VISIBLE_DEVICES has isolated the GPU" in WORKER,
    detail="Comment confirms Device(0) is intentional"
)

# ---------------------------------------------------------------------------
# Section 3 — Spawn stagger unchanged (TB: keep 4.0s for first validation)
# ---------------------------------------------------------------------------

section("3 — Spawn stagger unchanged (TB ruling: keep 4.0s)")

stagger_match = re.search(r'ROCM_SPAWN_STAGGER_S\s*=\s*([\d.]+)', PWC)
stagger_val = float(stagger_match.group(1)) if stagger_match else None

check(
    "ROCM_SPAWN_STAGGER_S found in PWC",
    stagger_match is not None,
)
check(
    "ROCM_SPAWN_STAGGER_S is still 4.0s (TB: do not reduce in same patch)",
    stagger_val == 4.0,
    detail=f"Current value: {stagger_val}s — reduce to 1.0s only after 8-worker stability proven"
)

# ---------------------------------------------------------------------------
# Section 4 — worker_pool_size not hardcoded
# ---------------------------------------------------------------------------

section("4 — worker_pool_size configurable (not hardcoded to 4)")

# Check that worker_pool_size parameter exists and has a default
pool_size_param = re.search(r'worker_pool_size\s*(?::\s*int)?\s*=\s*(\d+)', PWC)
pool_default = int(pool_size_param.group(1)) if pool_size_param else None

check(
    "worker_pool_size parameter found in PWC",
    pool_size_param is not None,
)
check(
    "worker_pool_size default is 8 (allows full rig utilization)",
    pool_default == 8,
    detail=f"Current default: {pool_default} — production uses manifest value (4), default is fallback"
)

# Confirm pool = min(worker_pool_size, node.gpu_count) cap is present
check(
    "pool = min(worker_pool_size, gpu_count) cap still present",
    "min(self.worker_pool_size, node.gpu_count)" in PWC,
    detail="Safety cap — prevents spawning more workers than GPUs"
)

# ---------------------------------------------------------------------------
# Section 5 — No unintended changes elsewhere
# ---------------------------------------------------------------------------

section("5 — No unintended changes (blast radius check)")

check(
    "JOB_TIMEOUT_S still 600",
    "JOB_TIMEOUT_S = 600" in PWC,
    detail="S146 invariant"
)
check(
    "_localhost_semaphore(2) still present",
    "threading.Semaphore(2)" in PWC,
    detail="S146 invariant — Zeus local dispatch limit"
)
check(
    "dispatch_lock per-worker lock still present",
    "dispatch_lock: threading.Lock" in PWC,
    detail="Per-worker serialization unchanged"
)
check(
    "WORKER_HEARTBEAT_TIMEOUT_S still present",
    "WORKER_HEARTBEAT_TIMEOUT_S" in PWC,
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
  ✓ S149-A ROCR isolation fix verified clean.

  Source analysis complete. Proceed to live validation:

  STEP 1 — Single-rig smoke test (rrig6600 only):
    On Zeus, temporarily set worker_pool_size=8 in manifest or
    pass directly. Spawn workers on rrig6600 only. Confirm:
      □ All 8 workers heartbeat within WORKER_HEARTBEAT_TIMEOUT_S
      □ No worker exits or quarantine during startup
      □ rocm-smi --showuse shows GPU% > 0 on all 8 GPUs during a job
      □ Jobs complete and return valid survivor counts
      □ No HSA init errors in log

  STEP 2 — Short soak (sweep_preprod.sh, worker_pool_size=8):
      □ 50M seeds, 5 trials, all rigs
      □ Monitor rig throughput — expect ~6M s/s per rig
      □ No worker crashes or quarantines

  STEP 3 — If both pass: restore production manifest + relaunch.
""")
else:
    print(f"\n  {FAIL} check(s) FAILED — review above before proceeding.")

sys.exit(0 if FAIL == 0 else 1)
