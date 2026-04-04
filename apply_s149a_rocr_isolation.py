#!/usr/bin/env python3
"""
apply_s149a_rocr_isolation.py
==============================
S149-A: Add ROCR_VISIBLE_DEVICES={gpu_id} to worker spawn environment.

TB ruling (S148/S149):
  "Approve the ROCR_VISIBLE_DEVICES={gpu_id} spawn fix in
   persistent_worker_coordinator.py. On Linux ROCm, AMD recommends
   ROCR_VISIBLE_DEVICES as the authoritative GPU-isolation mechanism;
   HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES are narrower runtime-layer
   controls. The worker's Device(0) design is correct and should remain
   unchanged."

Change:
  File: persistent_worker_coordinator.py
  Function: _spawn_worker()
  One line added to per-spawn env construction.

Usage:
  python3 apply_s149a_rocr_isolation.py [--dry-run]
"""

import argparse
import shutil
import os
import sys

DRY_RUN = False


def log(msg):
    print(msg)


def backup(path):
    if DRY_RUN:
        log(f"  [DRY-RUN] would create backup {path}.bak_s149a")
        return
    bak = path + ".bak_s149a"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        log(f"  backup → {bak}")


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    print(f"{'[DRY-RUN] ' if DRY_RUN else ''}S149-A ROCR Isolation Fix")
    print("TB ruling: approve ROCR_VISIBLE_DEVICES={gpu_id} in _spawn_worker()")
    print("=" * 60)

    path = os.path.join(args.base_dir, "persistent_worker_coordinator.py")
    backup(path)

    with open(path) as f:
        content = f.read()

    # Exact anchor — the two lines currently present
    old = (
        '            f"CUDA_VISIBLE_DEVICES={gpu_id}",\n'
        '            f"HIP_VISIBLE_DEVICES={gpu_id}",\n'
        '        ])'
    )
    new = (
        '            f"CUDA_VISIBLE_DEVICES={gpu_id}",\n'
        '            f"HIP_VISIBLE_DEVICES={gpu_id}",\n'
        '            f"ROCR_VISIBLE_DEVICES={gpu_id}",   # [S149-A] AMD authoritative HSA isolation\n'
        '        ])'
    )

    if old not in content:
        print("  ERROR: anchor not found — file may already be patched,")
        print("  formatting may have drifted, or an upstream refactor changed this block.")
        print("  Run verify_s149a_rocr_fix.py to check current state.")
        return False

    count = content.count(old)
    if count != 1:
        print(f"  ERROR: anchor found {count} times — expected exactly 1")
        return False

    content = content.replace(old, new, 1)
    log("  patched: ROCR_VISIBLE_DEVICES={gpu_id} added to _spawn_worker()")

    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
        log(f"  wrote {path}")
    else:
        log(f"  [DRY-RUN] would write {path}")

    print("\n" + "=" * 60)
    print("✓ S149-A patch COMPLETE")
    print()
    print("Next steps (per TB ruling):")
    print("  1. python3 verify_s149a_rocr_fix.py          # post-fix harness")
    print("  2. Single-rig live smoke: 8 workers on rrig6600")
    print("     ssh rrig6600 — confirm all 8 GPU heartbeats, jobs execute,")
    print("     rocm-smi --showuse shows nonzero GPU% on all 8")
    print("  3. bash sweep_preprod.sh (worker_pool_size=8, 5 trials)")
    print("  4. Only then restore production manifest")
    print()
    print("Commit after smoke test passes:")
    print("  git add persistent_worker_coordinator.py")
    print("  git commit -m 'fix(s149a): ROCR_VISIBLE_DEVICES isolation — enable 8 workers/rig'")
    print("  git push origin main && git push public main")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
