#!/usr/bin/env python3
"""
apply_s142_null_session_fix.py
S142 — Issue 1 Fix: NULL-session collision in step1_trial_history

Problem:
  save_best_so_far callback fires during n_parallel=2 runs using trial.params
  which has session_idx (int), not time_of_day (str). These NULL-session rows
  are written first via INSERT OR IGNORE, silently blocking the correct NP2
  writes (from _worker_obj) for the same (run_id, trial_number).

Fix (two files, one guard each):
  1. window_optimizer_integration_final.py
     Add 'n_parallel_gt1': (n_parallel > 1) to _trial_history_ctx so the
     callback knows it should stand down.

  2. window_optimizer_bayesian.py
     In save_best_so_far, skip the [S140b] per-trial history write block
     when trial_history_context.get('n_parallel_gt1') is True.
     The NP2 _worker_obj handles writes with correct session strings.
"""

import shutil
import os
import sys

BASE = os.path.expanduser("~/distributed_prng_analysis")

FILES = {
    "integration": os.path.join(BASE, "window_optimizer_integration_final.py"),
    "bayesian":    os.path.join(BASE, "window_optimizer_bayesian.py"),
}

# ─── Patch 1: window_optimizer_integration_final.py ─────────────────────────
# Add 'n_parallel_gt1' flag to _trial_history_ctx

OLD_CTX = """        _trial_history_ctx = {
            'run_id':     f"step1_{prng_base}_{int(seed_start)}",
            'study_name': study_name,
            'prng_type':  prng_base,
            'seed_start': seed_start,
            'seed_end':   seed_start + seed_count,
        }"""

NEW_CTX = """        _trial_history_ctx = {
            'run_id':       f"step1_{prng_base}_{int(seed_start)}",
            'study_name':   study_name,
            'prng_type':    prng_base,
            'seed_start':   seed_start,
            'seed_end':     seed_start + seed_count,
            'n_parallel_gt1': n_parallel > 1,  # [S142] guard: NP2 owns writes
        }"""

# ─── Patch 2: window_optimizer_bayesian.py ──────────────────────────────────
# Skip per-trial history write in save_best_so_far when n_parallel > 1

OLD_GUARD = """        # [S140b] per-trial history write
        if trial_history_context:"""

NEW_GUARD = """        # [S140b] per-trial history write
        # [S142] Skip when n_parallel>1 — _worker_obj owns writes with correct session strings.
        # If we write here too, INSERT OR IGNORE silently blocks the NP2 write (same PK, NULL session).
        if trial_history_context and not trial_history_context.get('n_parallel_gt1'):"""


def apply_patch(path, old, new, label):
    with open(path, 'r') as f:
        src = f.read()

    if old not in src:
        print(f"  ERROR: anchor not found in {path} — '{label}'")
        print(f"  First 80 chars of anchor: {repr(old[:80])}")
        sys.exit(1)

    count = src.count(old)
    if count > 1:
        print(f"  ERROR: anchor appears {count} times in {path} — must be unique")
        sys.exit(1)

    # Backup
    bak = path + ".bak_s142"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  Backup: {bak}")

    patched = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(patched)

    lines_before = src.count('\n')
    lines_after  = patched.count('\n')
    print(f"  {label}: OK  ({lines_before} → {lines_after} lines)")


def main():
    print("=" * 60)
    print("S142 P1 — NULL-session collision fix")
    print("=" * 60)

    # Patch 1
    print(f"\n[1/2] window_optimizer_integration_final.py")
    apply_patch(FILES["integration"], OLD_CTX, NEW_CTX,
                "_trial_history_ctx n_parallel_gt1 flag")

    # Patch 2
    print(f"\n[2/2] window_optimizer_bayesian.py")
    apply_patch(FILES["bayesian"], OLD_GUARD, NEW_GUARD,
                "save_best_so_far NP2 guard")

    # Final line counts
    print("\n─── Line counts ───────────────────────────────────────")
    for label, path in FILES.items():
        with open(path) as f:
            n = sum(1 for _ in f)
        print(f"  {os.path.basename(path)}: {n} lines")

    print("\n✅  Patch complete.")
    print("\nVerification steps:")
    print("  # Run 10-trial NP2 debug run:")
    print("  ssh rzeus \"cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \\")
    print("    PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \\")
    print("    --params '{\\\"window_trials\\\":10,\\\"n_parallel\\\":2}' \\")
    print("    > /tmp/s142_debug.log 2>&1 &\"")
    print()
    print("  # Check DB — all sessions should be non-NULL:")
    print("  ssh rzeus \"cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \\")
    print("    PYTHONPATH=. python3 -c \\\"")
    print("import sqlite3; conn = sqlite3.connect('prng_analysis.db')")
    print("rows = conn.execute('SELECT trial_number, session, trial_score FROM step1_trial_history ORDER BY trial_number').fetchall()")
    print("print(f'{len(rows)} rows:'); [print(f'  T{r[0]}: sess={r[1]} score={r[2]:.0f}') for r in rows]")
    print("null_cnt = sum(1 for r in rows if r[1] is None)")
    print("print(f'NULL sessions: {null_cnt} (should be 0)')\\\"\"")


if __name__ == "__main__":
    main()
