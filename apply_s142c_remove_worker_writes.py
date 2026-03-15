#!/usr/bin/env python3
"""
apply_s142c_remove_worker_writes.py
S142-C — TB Option A: remove _worker_obj trial history writes

TB Ruling:
  _worker_obj is an incomplete execution path and must not write to the
  canonical step1_trial_history table. Backfill from shared Optuna study
  is the only writer. Remove _worker_obj write block entirely.
  Also: drop _backfill suffix from canonical run_id.

Patch 1 — Remove _worker_obj write block from window_optimizer_integration_final.py
  Replaces the entire retry+write block with a comment only.
  `return score` is preserved (it's the Optuna objective return value).

Patch 2 — Drop _backfill suffix from backfill run_id
  run_id=f"step1_{prng_base}_{int(seed_start)}_backfill"
  → run_id=f"step1_{prng_base}_{int(seed_start)}"

Post-patch DB cleanup:
  DELETE all rows with run_id ending in _p0, _p1, _backfill (old duplicates).
"""

import shutil
import os
import sys
import subprocess

BASE = os.path.expanduser("~/distributed_prng_analysis")
TARGET = os.path.join(BASE, "window_optimizer_integration_final.py")

# ─── Patch 1: Remove _worker_obj write block ─────────────────────────────────

OLD_WORKER_WRITE = """                        # [S140b-NP2] Trial history — child-local DB connection
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
                                    run_id=f"step1_{prng_base_w}_{int(seed_start_w)}_p{partition_idx}",  # [S142-TB] partition-scoped: eliminates INSERT OR IGNORE collision
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
                                          f"failed after 3 attempts: {_th_e}")
                        return score"""

NEW_WORKER_WRITE = """                        # [S142-C] _worker_obj trial history writes removed per TB ruling.
                        # Canonical step1_trial_history is written by backfill from
                        # the shared Optuna study after all partition workers complete.
                        return score"""

# ─── Patch 2: Drop _backfill suffix from run_id ──────────────────────────────

OLD_RUNID = '                        run_id=f"step1_{prng_base}_{int(seed_start)}_backfill",'
NEW_RUNID = '                        run_id=f"step1_{prng_base}_{int(seed_start)}",  # [S142-C] canonical run_id, no suffix'


def main():
    print("=" * 60)
    print("S142-C — Remove _worker_obj writes, clean backfill run_id")
    print("=" * 60)

    with open(TARGET, 'r') as f:
        src = f.read()

    patches = [
        ("P1-worker-write", OLD_WORKER_WRITE, NEW_WORKER_WRITE),
        ("P2-runid-suffix", OLD_RUNID,        NEW_RUNID),
    ]

    for label, old, new in patches:
        if old not in src:
            print(f"ERROR: anchor '{label}' not found.")
            print(f"  First 80 chars: {repr(old[:80])}")
            sys.exit(1)
        count = src.count(old)
        if count > 1:
            print(f"ERROR: anchor '{label}' appears {count} times")
            sys.exit(1)
        print(f"  Anchor '{label}': OK (unique)")

    bak = TARGET + ".bak_s142c"
    if not os.path.exists(bak):
        shutil.copy2(TARGET, bak)
        print(f"Backup: {bak}")
    else:
        print(f"Backup already exists: {bak}")

    patched = src
    for label, old, new in patches:
        patched = patched.replace(old, new, 1)

    with open(TARGET, 'w') as f:
        f.write(patched)

    lines_before = src.count('\n')
    lines_after  = patched.count('\n')
    print(f"Patch applied: {lines_before} → {lines_after} lines ({lines_after - lines_before:+d})")

    with open(TARGET) as f:
        n = sum(1 for _ in f)
    print(f"Final line count: {n}")

    # AST check
    result = subprocess.run(
        ["python3", "-c", f"import ast; ast.parse(open('{TARGET}').read()); print('AST: CLEAN')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"AST ERROR: {result.stderr.strip()}")
        sys.exit(1)

    # DB cleanup — remove all rows with _p0, _p1, _backfill run_ids
    print("\n─── DB cleanup ────────────────────────────────────────")
    db_path = os.path.join(BASE, "prng_analysis.db")
    import sqlite3
    conn = sqlite3.connect(db_path)
    before = conn.execute("SELECT COUNT(*) FROM step1_trial_history").fetchone()[0]
    conn.execute("DELETE FROM step1_trial_history WHERE run_id LIKE '%_p0' OR run_id LIKE '%_p1' OR run_id LIKE '%_backfill'")
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM step1_trial_history").fetchone()[0]
    print(f"  Rows before: {before}")
    print(f"  Rows after:  {after} (removed {before - after} duplicates)")
    conn.close()

    print("\n✅ Patch + DB cleanup complete.")
    print("\nNext: re-run 10-trial debug, expect:")
    print("  DB: 7-8 rows, run_id=step1_java_lcg_*, 0 NULL sessions, no duplicates")


if __name__ == "__main__":
    main()
