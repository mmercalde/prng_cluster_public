#!/usr/bin/env python3
"""
apply_s149b_device_gpu_id.py  (v2 — TB revised ruling)
========================================================
S149-B: Direct GPU selection architecture.

TB revised ruling:
  "Move the fix to sieve_gpu_worker.py, but pair it with one coherent
   architecture. Remove per-worker HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
   masking. Use cp.cuda.Device(gpu_id) everywhere. Pass gpu_id explicitly
   into run_sieve_job(job, gpu_id) rather than reading from job dict."

Two files changed:

FILE 1: sieve_gpu_worker.py
  - run_sieve_job(job) → run_sieve_job(job, gpu_id=0)
  - cp.cuda.Device(0) → cp.cuda.Device(gpu_id) (warmup ×2 + synchronize)
  - getDeviceProperties(0) → getDeviceProperties(gpu_id)
  - run_sieve_job call in run_worker: pass gpu_id explicitly
  - Assert job['gpu_id'] matches if present — fail loudly on mismatch
  - Design comment updated

FILE 2: persistent_worker_coordinator.py
  - Remove f"CUDA_VISIBLE_DEVICES={gpu_id}" from _spawn_worker
  - Remove f"HIP_VISIBLE_DEVICES={gpu_id}" from _spawn_worker
  - Keep HSA_OVERRIDE_GFX_VERSION, HSA_ENABLE_SDMA, paths, etc.
  - Note: line ~421 CUDA_VISIBLE_DEVICES in _dispatch_local_sieve is
    Zeus-local sieve_filter.py path — NOT touched

Live test evidence:
  ROCR_VISIBLE_DEVICES=1 → hipErrorNoDevice on this CuPy/ROCm stack
  HIP/CUDA masking + Device(gpu_id) = contradiction (TB ruling)
  Solution: workers see all GPUs, bind via Device(gpu_id) directly

Usage:
  python3 apply_s149b_device_gpu_id.py [--dry-run]
"""

import argparse
import shutil
import os
import sys

DRY_RUN = False


def log(msg):
    print(msg)


def backup(path, tag):
    if DRY_RUN:
        log(f"  [DRY-RUN] would create backup {path}.bak_{tag}")
        return
    bak = f"{path}.bak_{tag}"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")


def replace_exact(content, old, new, label):
    if old not in content:
        log(f"  ERROR: anchor not found — {label!r}")
        log(f"         (already patched, formatting drifted, or upstream refactor)")
        return content, False
    count = content.count(old)
    if count != 1:
        log(f"  WARNING: {count} occurrences of {label!r} — replacing first only")
    result = content.replace(old, new, 1)
    log(f"  patched: {label}")
    return result, True


def patch_worker(path):
    log(f"\n[FILE 1] {path}")
    backup(path, "s149b")

    with open(path) as f:
        content = f.read()

    ok_count = 0

    # Fix 1: run_sieve_job signature — add gpu_id parameter
    content, ok = replace_exact(
        content,
        'def run_sieve_job(job: dict) -> dict:',
        'def run_sieve_job(job: dict, gpu_id: int = 0) -> dict:',
        "run_sieve_job signature — add gpu_id parameter"
    )
    ok_count += ok

    # Fix 2: run_sieve_job body — replace Device(0) + add assertion
    content, ok = replace_exact(
        content,
        '    device = cp.cuda.Device(0)',
        (
            '    # [S149-B] Direct GPU selection — workers see all GPUs, bind via gpu_id\n'
            '    _job_gpu_id = job.get(\'gpu_id\', None)\n'
            '    if _job_gpu_id is not None and int(_job_gpu_id) != gpu_id:\n'
            '        raise ValueError(f"gpu_id mismatch: worker={gpu_id}, job={_job_gpu_id}")\n'
            '    device = cp.cuda.Device(gpu_id)'
        ),
        "run_sieve_job Device(0) → Device(gpu_id) + mismatch assertion"
    )
    ok_count += ok

    # Fix 3: run_worker warmup block
    content, ok = replace_exact(
        content,
        '    with cp.cuda.Device(0):\n        _ = cp.zeros(1, dtype=cp.float32)\n        cp.cuda.Device(0).synchronize()',
        '    with cp.cuda.Device(gpu_id):\n        _ = cp.zeros(1, dtype=cp.float32)\n        cp.cuda.Device(gpu_id).synchronize()',
        "run_worker warmup Device(0) → Device(gpu_id)"
    )
    ok_count += ok

    # Fix 4: getDeviceProperties(0) → getDeviceProperties(gpu_id)
    content, ok = replace_exact(
        content,
        "        device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()",
        "        device_name = cp.cuda.runtime.getDeviceProperties(gpu_id)['name'].decode()",
        "getDeviceProperties(0) → getDeviceProperties(gpu_id)"
    )
    ok_count += ok

    # Fix 5: run_sieve_job call — pass gpu_id
    content, ok = replace_exact(
        content,
        '                result = run_sieve_job(job)',
        '                result = run_sieve_job(job, gpu_id)',
        "run_sieve_job call — pass gpu_id explicitly"
    )
    ok_count += ok

    # Fix 6: Update design comment
    content, ok = replace_exact(
        content,
        '    - Always uses device 0 (ROCR_VISIBLE_DEVICES has isolated the GPU)',
        (
            '    - Uses cp.cuda.Device(gpu_id) for direct GPU selection [S149-B]\n'
            '    - Workers see all GPUs — no HIP/CUDA/ROCR visibility masking in spawner\n'
            '    - ROCR_VISIBLE_DEVICES not viable on this CuPy/ROCm stack'
        ),
        "design comment updated"
    )
    ok_count += ok

    log(f"\n  sieve_gpu_worker.py: {ok_count}/6 patches applied")

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")

    return ok_count == 6


def patch_coordinator(path):
    log(f"\n[FILE 2] {path}")
    backup(path, "s149b")

    with open(path) as f:
        content = f.read()

    ok_count = 0

    # Remove HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES per-worker masking
    content, ok = replace_exact(
        content,
        (
            '            f"CUDA_VISIBLE_DEVICES={gpu_id}",\n'
            '            f"HIP_VISIBLE_DEVICES={gpu_id}",\n'
            '        ])'
        ),
        (
            '            # [S149-B] HIP/CUDA per-worker masking removed\n'
            '            # Workers see all GPUs; Device(gpu_id) selects directly in worker\n'
            '        ])'
        ),
        "remove HIP/CUDA per-worker masking from _spawn_worker"
    )
    ok_count += ok

    log(f"\n  persistent_worker_coordinator.py: {ok_count}/1 patches applied")

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")

    return ok_count == 1


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S149-B Direct GPU Selection Fix (v2 — TB revised ruling)")
    print("Architecture: workers see all GPUs, bind via Device(gpu_id) directly")
    print("=" * 60)

    r1 = patch_worker(os.path.join(args.base_dir, "sieve_gpu_worker.py"))
    r2 = patch_coordinator(os.path.join(args.base_dir, "persistent_worker_coordinator.py"))

    print("\n" + "=" * 60)
    if r1 and r2:
        print("✓ S149-B patch COMPLETE — 2 files patched")
        print()
        print("Next steps (per TB revised ruling):")
        print("  1. python3 verify_s149b_device_fix.py")
        print("  2. Manual per-device spawn: --gpu-id 0 then --gpu-id 7 on rrig6600")
        print("     (no HIP/CUDA/ROCR env vars — bare HSA vars only)")
        print("  3. Single-rig 8-worker smoke (worker_pool_size=8)")
        print("  4. bash sweep_preprod.sh (5 trials, all rigs)")
        print("  5. Commit + relaunch production")
        print()
        print("Commit:")
        print("  git add sieve_gpu_worker.py persistent_worker_coordinator.py")
        print("  git commit -m 'fix(s149b): direct Device(gpu_id) — remove HIP/CUDA masking'")
        print("  git push origin main && git push public main")
        return True
    else:
        print(f"✗ S149-B patch INCOMPLETE — review errors above")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
