#!/usr/bin/env python3
"""
apply_s147_trial_stats_fix.py
Fixes two bugs in progress_display.py:
1. ProgressWriter.__init__ never initializes self.trial_stats (always {})
2. Fresh ProgressWriter on each trial wipes previous trial_stats from JSON

Fix: On __init__, read back existing trial_stats from progress file so
     a new PWC instance doesn't wipe the last trial's survivor counts.
     Also initialize self.trial_stats = {} in __init__ so getattr fallback
     is never needed.
"""
import os, shutil, sys, argparse

TARGET = os.path.expanduser("~/distributed_prng_analysis/progress_display.py")

# ── Patch 1: initialize self.trial_stats in __init__ AND read back from file ──
OLD_INIT = """    def __init__(self, step_name: str, total_jobs: int = 100, total_seeds: int = 0):
        self.step_name = step_name
        self.total_jobs = total_jobs
        self.total_seeds = total_seeds
        self.jobs_completed = 0
        self.seeds_completed = 0
        self.start_time = time.time()
        self.nodes = {}
        self.finished = False
        self._write()"""

NEW_INIT = """    def __init__(self, step_name: str, total_jobs: int = 100, total_seeds: int = 0):
        self.step_name = step_name
        self.total_jobs = total_jobs
        self.total_seeds = total_seeds
        self.jobs_completed = 0
        self.seeds_completed = 0
        self.start_time = time.time()
        self.nodes = {}
        self.finished = False
        # S147 fix: initialize trial_stats and preserve across fresh instances
        # Read back existing trial_stats from file so a new PWC instance per
        # trial doesn't wipe the previous trial's survivor counts from the dashboard.
        self.trial_stats = {}
        try:
            import json
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, 'r') as _f:
                    _existing = json.load(_f)
                existing_ts = _existing.get('trial_stats', {})
                if existing_ts and existing_ts.get('trial_num', 0) > 0:
                    self.trial_stats = existing_ts
        except Exception:
            pass
        self._write()"""

def apply_patch(dry_run):
    if not os.path.exists(TARGET):
        print(f"ERROR: {TARGET} not found")
        sys.exit(1)

    with open(TARGET, 'r') as f:
        content = f.read()

    count = content.count(OLD_INIT)
    if count == 0:
        print("SKIP: anchor not found — already patched or file differs")
        sys.exit(0)
    if count > 1:
        print(f"ERROR: anchor matches {count} times — ambiguous")
        sys.exit(1)

    before = len(content.splitlines())
    new_content = content.replace(OLD_INIT, NEW_INIT, 1)
    after = len(new_content.splitlines())

    if dry_run:
        print(f"DRY RUN: would patch {TARGET}")
        print(f"  Lines: {before} → {after}")
        return

    bak = TARGET + ".bak_s147"
    if not os.path.exists(bak):
        shutil.copy(TARGET, bak)
        print(f"BAK: {bak}")

    with open(TARGET, 'w') as f:
        f.write(new_content)

    print(f"OK: patched {TARGET} ({before} → {after} lines)")
    print()
    print("Restart the dashboard to pick up the fix:")
    print("  pkill -f web_dashboard.py")
    print("  cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate")
    print("  nohup python3 web_dashboard.py --port 5002 > logs/dashboard.log 2>&1 &")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.dry_run)
