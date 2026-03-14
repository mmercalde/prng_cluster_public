#!/usr/bin/env python3
"""
apply_s142_worker_retry.py
S142 — P2 Fix: SQLite lock contention in _worker_obj trial history writes

Problem:
  Both NP2 partition processes (P0, P1) call DistributedPRNGDatabase().write_step1_trial()
  simultaneously. They hammer the same prng_analysis.db. SQLite throws an
  OperationalError: database is locked, which is silently swallowed by the bare
  except block. Result: ~half the trial history rows missing (P1's writes lost).

Fix:
  Replace the bare try/except in _worker_obj with a 3-attempt retry loop using
  random jitter backoff (0.1–0.5s). SQLite lock contention is transient — a
  short wait is sufficient for the other partition to release the write lock.

File: window_optimizer_integration_final.py
"""

import shutil
import os
import sys

BASE = os.path.expanduser("~/distributed_prng_analysis")
TARGET = os.path.join(BASE, "window_optimizer_integration_final.py")

OLD_WRITE = """                        # [S140b-NP2] Trial history — child-local DB connection
                        try:
                            from database_system import DistributedPRNGDatabase as _DBTH
                            _db_th = _DBTH()
                            _sess = (",".join(cfg.sessions)
                                     if isinstance(cfg.sessions, (list, tuple))
                                     else str(cfg.sessions))
                            _db_th.write_step1_trial(
                                run_id=f"step1_{prng_base_w}_{int(seed_start_w)}",
                                study_name=study_name_w,
                                trial_number=int(trial.number),
                                prng_type=str(prng_base_w),
                                seed_range_start=int(seed_start_w),
                                seed_range_end=int(seed_start_w + seed_count_w - 1),
                                params={
                                    'window_size': cfg.window_size,
                                    'offset': cfg.offset,
                                    'skip_min': cfg.skip_min,
                                    'skip_max': cfg.skip_max,
                                    'time_of_day': _sess,
                                    'forward_threshold': cfg.forward_threshold,
                                    'reverse_threshold': cfg.reverse_threshold,
                                },
                                trial_score=float(score),
                                forward_survivors=int(
                                    getattr(result, "forward_count", 0)),
                                reverse_survivors=int(
                                    getattr(result, "reverse_count", 0)),
                                bidirectional_survivors=int(
                                    getattr(result, "bidirectional_count", 0)),
                                pruned=False
                            )
                        except Exception as _th_e:
                            print(f"   [P{partition_idx}] trial-history write "
                                  f"failed (non-fatal): {_th_e}")"""

NEW_WRITE = """                        # [S140b-NP2] Trial history — child-local DB connection
                        # [S142] Retry loop: SQLite lock contention when P0+P1 write simultaneously
                        import time as _time_th
                        import random as _rand_th
                        _sess = (",".join(cfg.sessions)
                                 if isinstance(cfg.sessions, (list, tuple))
                                 else str(cfg.sessions))
                        _th_written = False
                        for _th_attempt in range(3):
                            try:
                                from database_system import DistributedPRNGDatabase as _DBTH
                                _db_th = _DBTH()
                                _db_th.write_step1_trial(
                                    run_id=f"step1_{prng_base_w}_{int(seed_start_w)}",
                                    study_name=study_name_w,
                                    trial_number=int(trial.number),
                                    prng_type=str(prng_base_w),
                                    seed_range_start=int(seed_start_w),
                                    seed_range_end=int(seed_start_w + seed_count_w - 1),
                                    params={
                                        'window_size': cfg.window_size,
                                        'offset': cfg.offset,
                                        'skip_min': cfg.skip_min,
                                        'skip_max': cfg.skip_max,
                                        'time_of_day': _sess,
                                        'forward_threshold': cfg.forward_threshold,
                                        'reverse_threshold': cfg.reverse_threshold,
                                    },
                                    trial_score=float(score),
                                    forward_survivors=int(
                                        getattr(result, "forward_count", 0)),
                                    reverse_survivors=int(
                                        getattr(result, "reverse_count", 0)),
                                    bidirectional_survivors=int(
                                        getattr(result, "bidirectional_count", 0)),
                                    pruned=False
                                )
                                _th_written = True
                                break
                            except Exception as _th_e:
                                if _th_attempt < 2:
                                    _time_th.sleep(0.1 + _rand_th.random() * 0.4)
                                else:
                                    print(f"   [P{partition_idx}] trial-history write "
                                          f"failed after 3 attempts: {_th_e}")"""

def main():
    print("=" * 60)
    print("S142 P2 — _worker_obj SQLite retry fix")
    print("=" * 60)

    with open(TARGET, 'r') as f:
        src = f.read()

    if OLD_WRITE not in src:
        print("ERROR: anchor not found in target file.")
        print("First 80 chars of anchor:", repr(OLD_WRITE[:80]))
        sys.exit(1)

    count = src.count(OLD_WRITE)
    if count > 1:
        print(f"ERROR: anchor appears {count} times — must be unique")
        sys.exit(1)

    # Backup
    bak = TARGET + ".bak_s142_p2"
    if not os.path.exists(bak):
        shutil.copy2(TARGET, bak)
        print(f"Backup: {bak}")

    patched = src.replace(OLD_WRITE, NEW_WRITE, 1)
    with open(TARGET, 'w') as f:
        f.write(patched)

    lines_before = src.count('\n')
    lines_after  = patched.count('\n')
    print(f"Patch applied: {lines_before} → {lines_after} lines")

    with open(TARGET) as f:
        n = sum(1 for _ in f)
    print(f"Final line count: {n}")

    print("\n✅ Patch complete.")
    print("\nVerification — re-run 10-trial NP2 debug run then check DB:")
    print("  Expected: 9 rows (10 minus T7 PRUNED which has no session)")
    print("  NULL sessions: 0")


if __name__ == "__main__":
    main()
