#!/usr/bin/env python3
"""
S138B Patch: Fix trial count ceiling not enforced on resume.

ROOT CAUSE:
    When resuming a study with study_name specified, _trials_per_worker is
    calculated as max_iterations // n_parallel, ignoring already-completed
    trials in the DB. Each partition worker then runs its full quota on top
    of existing trials, causing overrun.

    Example: 51 existing trials, max_iterations=100, n_parallel=2
    Old: each partition runs 50 → total = 51 + 50 + 50 = 151 (overrun by 51)
    New: remaining = 100 - 51 = 49 → each partition runs 24/25 → total ~100

FIX:
    Before dividing trials per worker, query the study DB for existing
    completed trial count. Subtract from max_iterations to get remaining.
    If already at or over ceiling, skip workers entirely.

CHANGE:
    window_optimizer_integration_final.py:
    - After study ready block, query existing completed trials from DB
    - Compute remaining = max(0, max_iterations - existing_complete)
    - Divide remaining across workers instead of max_iterations
    - If remaining == 0, skip worker launch and go straight to results
"""

import sys
import shutil
from datetime import datetime

TARGET = sys.argv[1] if len(sys.argv) > 1 else '/home/michael/distributed_prng_analysis/window_optimizer_integration_final.py'

# Backup
BACKUP = TARGET + f'.bak_s138b_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy2(TARGET, BACKUP)
print(f"✅ Backup saved: {BACKUP}")

with open(TARGET, 'r') as f:
    src = f.read()

# ============================================================================
# CHANGE: Replace naive _trials_per_worker calculation with ceiling-aware version
# ============================================================================
OLD_TRIALS = """            # ----------------------------------------------------------------
            # Divide trials and launch worker processes
            # ----------------------------------------------------------------
            # S137: Initialize accumulator, bounds, optimizer so they exist in n_parallel path
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
            bounds = SearchBounds.from_config()      # S137-D: needed for session_options after best trial
            optimizer = WindowOptimizer(self, dataset_path)  # S137-E: needed for save_results

            _trials_per_worker = [max_iterations // n_parallel] * n_parallel
            for _ri in range(max_iterations % n_parallel):
                _trials_per_worker[_ri] += 1"""

NEW_TRIALS = """            # ----------------------------------------------------------------
            # Divide trials and launch worker processes
            # ----------------------------------------------------------------
            # S137: Initialize accumulator, bounds, optimizer so they exist in n_parallel path
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
            bounds = SearchBounds.from_config()      # S137-D: needed for session_options after best trial
            optimizer = WindowOptimizer(self, dataset_path)  # S137-E: needed for save_results

            # S138B: Enforce trial ceiling — subtract already-completed trials
            try:
                import optuna as _ocount
                _count_storage = _ocount.storages.RDBStorage(
                    url=_mp_storage_url,
                    engine_kwargs={"connect_args": {"timeout": 20}}
                )
                _count_study = _ocount.load_study(
                    study_name=_mp_study_name,
                    storage=_count_storage,
                )
                _existing_complete = len([
                    t for t in _count_study.trials
                    if t.state == _ocount.trial.TrialState.COMPLETE
                ])
                print(f"   [n_parallel] Existing complete trials: {_existing_complete}")
            except Exception as _ce:
                print(f"   [n_parallel] Could not query existing trials: {_ce} -- assuming 0")
                _existing_complete = 0

            _remaining_trials = max(0, max_iterations - _existing_complete)
            print(f"   [n_parallel] Remaining trials to run: {_remaining_trials} "
                  f"(ceiling={max_iterations}, existing={_existing_complete})")

            if _remaining_trials == 0:
                print(f"   [n_parallel] Trial ceiling already reached -- skipping workers")
                _trials_per_worker = [0] * n_parallel
            else:
                _trials_per_worker = [_remaining_trials // n_parallel] * n_parallel
                for _ri in range(_remaining_trials % n_parallel):
                    _trials_per_worker[_ri] += 1"""

assert OLD_TRIALS in src, "ABORT: trials_per_worker block not found"
src = src.replace(OLD_TRIALS, NEW_TRIALS, 1)
print("✅ Change 1: trials_per_worker now ceiling-aware")

# ============================================================================
# Also guard the worker launch — skip if all workers have 0 trials
# ============================================================================
OLD_LAUNCH = """            print(f\"\\n{'='*60}\")
            print(f\"LAUNCHING {n_parallel} PARTITION WORKERS (multiprocessing.Process)\")
            for _pi in range(n_parallel):
                print(f\"   P{_pi}: {_PARALLEL_PARTITIONS[_pi]}  -> {_trials_per_worker[_pi]} trials\")
            print(f\"   Study: {_mp_study_name}\")"""

NEW_LAUNCH = """            if _remaining_trials == 0:
                print(f\"\\n   [n_parallel] No workers launched — ceiling already met\")
            else:
                print(f\"\\n{'='*60}\")
                print(f\"LAUNCHING {n_parallel} PARTITION WORKERS (multiprocessing.Process)\")
                for _pi in range(n_parallel):
                    print(f\"   P{_pi}: {_PARALLEL_PARTITIONS[_pi]}  -> {_trials_per_worker[_pi]} trials\")
                print(f\"   Study: {_mp_study_name}\")"""

assert OLD_LAUNCH in src, "ABORT: launch block not found"
src = src.replace(OLD_LAUNCH, NEW_LAUNCH, 1)
print("✅ Change 2: worker launch guarded by _remaining_trials > 0")

# ============================================================================
# Write patched file
# ============================================================================
with open(TARGET, 'w') as f:
    f.write(src)

print("\n✅ All changes applied successfully")
print(f"   Target: {TARGET}")
print(f"\nVerification:")
print(f"   'S138B' occurrences: {src.count('S138B')}")
print(f"   '_remaining_trials' occurrences: {src.count('_remaining_trials')}")
print(f"   '_existing_complete' occurrences: {src.count('_existing_complete')}")
