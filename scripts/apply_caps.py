#!/usr/bin/env python3
"""
apply_caps.py — S128 GPU Throughput Cap Update Tool
====================================================
Updates coordinator.py and gpu_optimizer.py with measured ceiling values
from Phase A + Phase B probe results.

Run AFTER Phase B (and optionally Phase C) is complete.
Reads measured values from command-line args.

Usage:
    python3 apply_caps.py \
        --rtx-phaseA-sps 33000 \
        --amd-phaseA-sps 11500 \
        --rtx-phaseB-ceiling 2000000 \
        --amd-phaseB-ceiling 400000 \
        --safety-factor 0.85

    Automatically computes:
        seed_cap_nvidia  = rtx_phaseB_ceiling × safety_factor
        seed_cap_amd     = amd_phaseB_ceiling × safety_factor
        scaling_factor   = rtx_phaseA_sps / amd_phaseA_sps
"""

import argparse
import re
import shutil
import os
from datetime import datetime

PRNG_DIR = os.path.expanduser("~/distributed_prng_analysis")
COORDINATOR_FILE = os.path.join(PRNG_DIR, "coordinator.py")
GPU_OPT_FILE = os.path.join(PRNG_DIR, "gpu_optimizer.py")


def make_backup(filepath):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{filepath}.bak_s128_{ts}"
    shutil.copy2(filepath, backup)
    print(f"  Backup: {backup}")
    return backup


def patch_coordinator(rtx_cap, amd_cap):
    """Patch line 233 seed_cap constants in coordinator.py"""
    print(f"\nPatching {COORDINATOR_FILE}")
    with open(COORDINATOR_FILE, "r") as f:
        content = f.read()

    # Match the line containing seed_cap_nvidia, seed_cap_amd, seed_cap_default
    old_pattern = re.compile(
        r'(seed_cap_nvidia\s*:\s*int\s*=\s*)\d+(,\s*seed_cap_amd\s*:\s*int\s*=\s*)\d+(,\s*seed_cap_default\s*:\s*int\s*=\s*)\d+'
    )
    match = old_pattern.search(content)
    if not match:
        raise ValueError("Could not find seed_cap line in coordinator.py — verify line 233 manually")

    old_str = match.group(0)
    new_str = f"seed_cap_nvidia: int = {rtx_cap}, seed_cap_amd: int = {amd_cap}, seed_cap_default: int = {amd_cap}"
    print(f"  OLD: {old_str}")
    print(f"  NEW: {new_str}")

    make_backup(COORDINATOR_FILE)
    new_content = content.replace(old_str, new_str)
    with open(COORDINATOR_FILE, "w") as f:
        f.write(new_content)
    print("  ✅ coordinator.py patched")


def patch_gpu_optimizer(rtx_sps, amd_sps, scaling_factor):
    """Patch seeds_per_second and scaling_factor in gpu_optimizer.py"""
    print(f"\nPatching {GPU_OPT_FILE}")
    with open(GPU_OPT_FILE, "r") as f:
        content = f.read()

    # Patch RTX 3080 Ti seeds_per_second
    rtx_sps_pattern = re.compile(
        r'("RTX 3080 Ti"[^}]*?"seeds_per_second"\s*:\s*)\d+'
    )
    match = rtx_sps_pattern.search(content)
    if not match:
        raise ValueError("Could not find RTX 3080 Ti seeds_per_second in gpu_optimizer.py")
    old = match.group(0)
    new = match.group(1) + str(rtx_sps)
    print(f"  RTX sps: {old} → {new}")
    content = content.replace(old, new)

    # Patch RTX 3080 Ti scaling_factor
    rtx_sf_pattern = re.compile(
        r'("RTX 3080 Ti"[^}]*?"scaling_factor"\s*:\s*)[\d.]+'
    )
    match = rtx_sf_pattern.search(content)
    if not match:
        raise ValueError("Could not find RTX 3080 Ti scaling_factor in gpu_optimizer.py")
    old = match.group(0)
    new = match.group(1) + f"{scaling_factor:.1f}"
    print(f"  RTX scaling_factor: {old} → {new}")
    content = content.replace(old, new)

    # Patch RX 6600 seeds_per_second
    amd_sps_pattern = re.compile(
        r'("RX 6600"[^}]*?"seeds_per_second"\s*:\s*)\d+'
    )
    match = amd_sps_pattern.search(content)
    if not match:
        raise ValueError("Could not find RX 6600 seeds_per_second in gpu_optimizer.py")
    old = match.group(0)
    new = match.group(1) + str(amd_sps)
    print(f"  AMD sps: {old} → {new}")
    content = content.replace(old, new)

    make_backup(GPU_OPT_FILE)
    with open(GPU_OPT_FILE, "w") as f:
        f.write(content)
    print("  ✅ gpu_optimizer.py patched")


def main():
    parser = argparse.ArgumentParser(description="Apply measured GPU caps to coordinator.py + gpu_optimizer.py")
    parser.add_argument("--rtx-phaseA-sps", type=int, required=True,
                        help="RTX 3080 Ti measured seeds/sec from Phase A (single card)")
    parser.add_argument("--amd-phaseA-sps", type=int, required=True,
                        help="RX 6600 measured seeds/sec from Phase A (single card)")
    parser.add_argument("--rtx-phaseB-ceiling", type=int, required=True,
                        help="RTX highest passing seed count from Phase B (concurrent)")
    parser.add_argument("--amd-phaseB-ceiling", type=int, required=True,
                        help="AMD highest passing seed count from Phase B (concurrent)")
    parser.add_argument("--safety-factor", type=float, default=0.85,
                        help="Safety factor applied to Phase B ceiling (default: 0.85)")
    args = parser.parse_args()

    rtx_cap = int(args.rtx_phaseB_ceiling * args.safety_factor)
    amd_cap = int(args.amd_phaseB_ceiling * args.safety_factor)
    scaling_factor = args.rtx_phaseA_sps / args.amd_phaseA_sps

    print("=" * 56)
    print("  S128 GPU Cap Update")
    print("=" * 56)
    print(f"  Phase A RTX sps:       {args.rtx_phaseA_sps:,}")
    print(f"  Phase A AMD sps:       {args.amd_phaseA_sps:,}")
    print(f"  Phase B RTX ceiling:   {args.rtx_phaseB_ceiling:,}")
    print(f"  Phase B AMD ceiling:   {args.amd_phaseB_ceiling:,}")
    print(f"  Safety factor:         {args.safety_factor}")
    print(f"  → seed_cap_nvidia:     {rtx_cap:,}")
    print(f"  → seed_cap_amd:        {amd_cap:,}")
    print(f"  → scaling_factor:      {scaling_factor:.2f}x  (RTX/AMD ratio)")
    print("=" * 56)

    patch_coordinator(rtx_cap, amd_cap)
    patch_gpu_optimizer(args.rtx_phaseA_sps, args.amd_phaseA_sps, scaling_factor)

    print("\n✅ ALL PATCHES APPLIED")
    print("\nNext steps:")
    print("  1. Review diffs: git diff coordinator.py gpu_optimizer.py")
    print("  2. Run Phase C stability test with new caps")
    print("  3. If stable: git add coordinator.py gpu_optimizer.py")
    print('  4.             git commit -m "feat(S128): update GPU seed caps from throughput probe"')
    print("  5. Dual-push to private + public repos")


if __name__ == "__main__":
    main()
