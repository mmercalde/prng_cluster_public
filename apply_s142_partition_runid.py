#!/usr/bin/env python3
"""
apply_s142_partition_runid.py
S142 — TB Fix: partition-scoped run_id eliminates INSERT OR IGNORE collision

Root cause (TB diagnosis confirmed):
  Both NP2 partition workers write to step1_trial_history with identical
  run_id=f"step1_{prng_base_w}_{int(seed_start_w)}". Optuna assigns globally
  unique trial numbers across the shared study, but both workers may attempt
  writes for the same trial_number if one partition's _worker_obj races the
  other. The UNIQUE(run_id, trial_number) constraint + INSERT OR IGNORE means
  whichever partition writes first wins; the other is silently discarded.
  Result: ~50% of COMPLETE trial rows missing, no exception, no print.

  Confirmed: all 4 DB rows share run_id=step1_java_lcg_15000000.
  Missing trials T3,T5,T6,T9 were silently dropped by INSERT OR IGNORE.

Fix:
  Append partition index to run_id in _worker_obj:
    run_id=f"step1_{prng_base_w}_{int(seed_start_w)}_p{partition_idx}"

  This makes (run_id, trial_number) globally unique across partitions.
  get_best_step1_params() queries by prng_type not run_id — warm-start
  still sees all rows from both partitions. No schema change needed.

File: window_optimizer_integration_final.py (one line)
"""

import shutil
import os
import sys

BASE = os.path.expanduser("~/distributed_prng_analysis")
TARGET = os.path.join(BASE, "window_optimizer_integration_final.py")

OLD = """                                _db_th.write_step1_trial(
                                    run_id=f"step1_{prng_base_w}_{int(seed_start_w)}","""

NEW = """                                _db_th.write_step1_trial(
                                    run_id=f"step1_{prng_base_w}_{int(seed_start_w)}_p{partition_idx}",  # [S142-TB] partition-scoped: eliminates INSERT OR IGNORE collision"""


def main():
    print("=" * 60)
    print("S142 TB fix — partition-scoped run_id")
    print("=" * 60)

    with open(TARGET, 'r') as f:
        src = f.read()

    if OLD not in src:
        print("ERROR: anchor not found.")
        print("First 80 chars:", repr(OLD[:80]))
        sys.exit(1)

    count = src.count(OLD)
    if count > 1:
        print(f"ERROR: anchor appears {count} times — must be unique")
        sys.exit(1)

    # Backup
    bak = TARGET + ".bak_s142_tb"
    if not os.path.exists(bak):
        shutil.copy2(TARGET, bak)
        print(f"Backup: {bak}")
    else:
        print(f"Backup already exists: {bak}")

    patched = src.replace(OLD, NEW, 1)
    with open(TARGET, 'w') as f:
        f.write(patched)

    lines_before = src.count('\n')
    lines_after  = patched.count('\n')
    print(f"Patch applied: {lines_before} → {lines_after} lines")

    with open(TARGET) as f:
        n = sum(1 for _ in f)
    print(f"Final line count: {n}")

    print("\n✅ Patch complete.")
    print("\nVerification — clear history and re-run 10-trial debug:")
    print("  Expected: 8 rows (10 trials - 2 PRUNED), 0 NULL sessions")


if __name__ == "__main__":
    main()
