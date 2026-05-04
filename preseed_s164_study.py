#!/usr/bin/env python3
"""
preseed_s164_study.py
=====================
Pre-seeds an Optuna study with the S162 VICTORY config:
  W6_O64_evening_S3-37  FT=0.68  RT=0.70
  887 bidirectional survivors — best confirmed survivor-producing trial.

Run this on Zeus BEFORE launching the S164 production run.
The run must then pass study_name='window_opt_s164_preseed' in --params.

Session context:
  - S164: validation run, NPZ accumulator confirmation
  - Card0 reseat complete (16.0 GT/s)
  - Coverage reset confirmed
  - 23,765 accumulated survivors in bidirectional_survivors_all.npz

Usage:
  cd ~/distributed_prng_analysis
  source ~/venvs/torch/bin/activate
  python3 preseed_s164_study.py [--dry-run]

Then launch:
  PYTHONPATH=. nohup python3 agents/watcher_agent.py \\
    --run-pipeline --start-step 1 --end-step 1 --force-step 1 \\
    --params '{"min_workers": 24, "seed_cap_amd": 100000,
               "window_trials": 5, "study_name": "window_opt_s164_preseed"}' \\
    > logs/s164_production_HHMM.log 2>&1 &
"""

import sys
import os
import argparse

STUDY_NAME = "window_opt_s164_preseed"

# S162 VICTORY config — W6_O64_evening_S3-37_FT0.68_RT0.70
# session_options ordering (from window_optimizer.py SearchBounds):
#   idx=0 : ['midday', 'evening']
#   idx=1 : ['midday']
#   idx=2 : ['evening']   <-- S162 victory used evening only
S162_VICTORY_PARAMS = {
    "window_size":        6,
    "offset":             64,
    "session_idx":        2,    # evening
    "skip_min":           3,
    "skip_max":           37,
    "forward_threshold":  0.68,
    "reverse_threshold":  0.70,
}


def main():
    parser = argparse.ArgumentParser(description="Pre-seed S164 Optuna study")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing the DB")
    parser.add_argument("--study-dir", default="optuna_studies",
                        help="Directory for Optuna DBs (default: optuna_studies)")
    args = parser.parse_args()

    db_path = os.path.join(args.study_dir, f"{STUDY_NAME}.db")
    storage_url = f"sqlite:///{db_path}"

    print(f"S164 Optuna Pre-seed")
    print(f"=" * 50)
    print(f"Study name : {STUDY_NAME}")
    print(f"DB path    : {db_path}")
    print(f"Storage    : {storage_url}")
    print()
    print(f"Enqueueing S162 VICTORY trial:")
    for k, v in S162_VICTORY_PARAMS.items():
        print(f"  {k:22s} = {v}")
    print()

    if args.dry_run:
        print("DRY RUN — no DB written.")
        print()
        print("Launch command (after real run):")
        _print_launch()
        return

    # Validate study dir exists
    os.makedirs(args.study_dir, exist_ok=True)

    # Check if DB already exists
    if os.path.exists(db_path):
        print(f"WARNING: {db_path} already exists.")
        resp = input("Overwrite? (y/N): ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            sys.exit(1)
        os.remove(db_path)
        print(f"Removed existing DB.")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: optuna not installed. Activate venv first.")
        sys.exit(1)

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 20}}
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=False,
    )

    study.enqueue_trial(S162_VICTORY_PARAMS)
    print(f"Trial enqueued. Study has {len(study.trials)} queued trial(s).")
    print()

    # Verify
    trials = study.trials
    if len(trials) == 1 and trials[0].state.name == "WAITING":
        print(f"VERIFY: 1 WAITING trial confirmed in study.")
        print(f"  Params: {trials[0].params}")
        print()
        print(f"Study ready. DB at: {db_path}")
        print()
        _print_launch()
    else:
        print(f"ERROR: Unexpected study state: {[(t.number, t.state) for t in trials]}")
        sys.exit(1)


def _print_launch():
    print("=" * 60)
    print("LAUNCH COMMAND:")
    print()
    print('PYTHONPATH=. nohup python3 agents/watcher_agent.py \\')
    print('  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \\')
    print(f'  --params \'{{"min_workers": 24, "seed_cap_amd": 100000,')
    print(f'             "window_trials": 5, "study_name": "{STUDY_NAME}"}}\' \\')
    print('  > logs/s164_production_$(date +%H%M).log 2>&1 &')
    print('echo PID: $!')
    print()
    print("Trials 2-5 will be explored by Optuna Bayesian from the")
    print("warm-started S162 victory config baseline.")
    print()
    print("After Step 1 completes, run Step 2:")
    print()
    print('PYTHONPATH=. nohup python3 agents/watcher_agent.py \\')
    print('  --run-pipeline --start-step 2 --end-step 2 \\')
    print('  > logs/s164_step2_$(date +%H%M).log 2>&1 &')


if __name__ == "__main__":
    main()
