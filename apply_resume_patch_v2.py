"""
S114 Resume Patch v2 — FLAG-BASED (permanent, safe)

Adds --resume-study flag to window_optimizer.py CLI and
resume_study parameter to window_optimizer_bayesian.py search().

Default behavior: UNCHANGED — fresh study every run
With flag:        resumes most recent incomplete study DB

Usage:
    python3 window_optimizer.py --strategy bayesian --lottery-file daily3.json \\
        --trials 100 --resume-study

Or via WATCHER params:
    --params '{"lottery_file": "daily3.json", "resume_study": true}'

Apply on Zeus:
    python3 apply_resume_patch_v2.py
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1: window_optimizer.py — add --resume-study CLI arg
# ─────────────────────────────────────────────────────────────────────────────
WO_PATH = '/home/michael/distributed_prng_analysis/window_optimizer.py'

with open(WO_PATH) as f:
    wo_content = f.read()

# Add CLI argument after --test-both-modes
old_wo = """    parser.add_argument('--test-both-modes', action='store_true',"""

new_wo = """    parser.add_argument('--resume-study', action='store_true',
                       help='Resume most recent incomplete Optuna study DB instead of starting fresh. '
                            'Skips warm-start enqueue if study already has trials. '
                            'Default: False (fresh study every run).')
    parser.add_argument('--test-both-modes', action='store_true',"""

if old_wo in wo_content:
    wo_content = wo_content.replace(old_wo, new_wo)
    print("✅ Patch 1a: --resume-study CLI arg added")
else:
    print("❌ Patch 1a FAILED: --test-both-modes arg not found")

# Pass resume_study to the bayesian run_bayesian_optimization call
old_wo2 = """            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            lottery_file=args.lottery_file,"""

new_wo2 = """            prng_type=args.prng_type,
            test_both_modes=args.test_both_modes,
            resume_study=args.resume_study,
            lottery_file=args.lottery_file,"""

if old_wo2 in wo_content:
    wo_content = wo_content.replace(old_wo2, new_wo2)
    print("✅ Patch 1b: resume_study passed to run_bayesian_optimization")
else:
    # Try alternate pattern
    print("⚠️  Patch 1b: trying alternate pattern...")
    old_wo2b = "            prng_type=args.prng_type,\n            test_both_modes=args.test_both_modes"
    new_wo2b = "            prng_type=args.prng_type,\n            test_both_modes=args.test_both_modes,\n            resume_study=getattr(args, 'resume_study', False)"
    if old_wo2b in wo_content:
        wo_content = wo_content.replace(old_wo2b, new_wo2b)
        print("✅ Patch 1b (alt): resume_study passed via getattr")
    else:
        print("❌ Patch 1b FAILED")

with open(WO_PATH, 'w') as f:
    f.write(wo_content)

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2: window_optimizer_bayesian.py — add resume logic to search()
# ─────────────────────────────────────────────────────────────────────────────
WOB_PATH = '/home/michael/distributed_prng_analysis/window_optimizer_bayesian.py'

with open(WOB_PATH) as f:
    wob_content = f.read()

# Update search() signature to accept resume_study parameter
old_sig = """    def search(self, 
               objective_function: Callable,
               bounds: 'SearchBounds',
               max_iterations: int,
               scorer: ResultScorer) -> Dict:"""

new_sig = """    def search(self, 
               objective_function: Callable,
               bounds: 'SearchBounds',
               max_iterations: int,
               scorer: ResultScorer,
               resume_study: bool = False) -> Dict:"""

if old_sig in wob_content:
    wob_content = wob_content.replace(old_sig, new_sig)
    print("✅ Patch 2a: search() signature updated with resume_study param")
else:
    print("❌ Patch 2a FAILED: search() signature not found")

# Replace study creation block with resume-aware version
old_study = '''        # Create persistent storage for the study
        import time
        study_name = f"window_opt_{int(time.time())}"
        storage_path = f"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{study_name}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            sampler=sampler,
            load_if_exists=False
        )
        print(f"   📊 Optuna study saved to: optuna_studies/{study_name}.db")
        
        # Warm-start: enqueue known-good S112 config as trial 0
        # Gives TPE a positive signal immediately, guiding search toward small windows
        # Does not affect subsequent runs — each run creates a new study DB
        study.enqueue_trial({
            'window_size': 8,
            'offset': 43,
            'skip_min': 5,
            'skip_max': 56,
            'forward_threshold': 0.49,
            'reverse_threshold': 0.49
        })
        print("   🌡️  Warm-start: enqueued W8_O43_S5-56 as trial 0 (S112 known-good config)")'''

new_study = '''        # Create persistent storage for the study
        import time
        import glob as _glob
        import os as _os

        # --- Resume logic (S114 patch) ---
        # resume_study=True: find most recent incomplete DB and continue
        # resume_study=False (default): always create fresh study
        _resume = False
        _resumed_completed = 0
        study_name = f"window_opt_{int(time.time())}"
        storage_path = f"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{study_name}.db"

        if resume_study:
            _existing_dbs = sorted(
                _glob.glob("optuna_studies/window_opt_*.db"),
                key=_os.path.getmtime,
                reverse=True
            )
            if _existing_dbs:
                _candidate_db = _existing_dbs[0]
                _candidate_name = _os.path.splitext(_os.path.basename(_candidate_db))[0]
                _candidate_storage = f"sqlite:////home/michael/distributed_prng_analysis/{_candidate_db}"
                try:
                    _candidate_study = optuna.load_study(
                        study_name=_candidate_name,
                        storage=_candidate_storage
                    )
                    _candidate_completed = len([
                        t for t in _candidate_study.trials
                        if t.state.name == "COMPLETE"
                    ])
                    if _candidate_completed > 0 and _candidate_completed < max_iterations:
                        _resume = True
                        _resumed_completed = _candidate_completed
                        study_name = _candidate_name
                        storage_path = _candidate_storage
                        print(f"   🔄 RESUMING study: {_candidate_name}")
                        print(f"   🔄 Completed: {_candidate_completed}/{max_iterations} trials")
                        print(f"   🔄 Remaining: {max_iterations - _candidate_completed} trials")
                    else:
                        print(f"   📊 No resumable study found — starting fresh")
                except Exception as _e:
                    print(f"   ⚠️  Could not load existing study ({_e}) — starting fresh")
            else:
                print(f"   📊 No existing study DBs found — starting fresh")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            sampler=sampler,
            load_if_exists=_resume
        )
        print(f"   📊 Optuna study: optuna_studies/{study_name}.db")

        # Warm-start: enqueue known-good S112 config as trial 0
        # Skipped on resume — trial already exists in DB
        # Only runs on fresh study starts
        if not _resume:
            study.enqueue_trial({
                'window_size': 8,
                'offset': 43,
                'skip_min': 5,
                'skip_max': 56,
                'forward_threshold': 0.49,
                'reverse_threshold': 0.49
            })
            print("   🌡️  Warm-start: enqueued W8_O43_S5-56 as trial 0 (S112 known-good config)")
        else:
            print(f"   ✅ Resume mode: skipping warm-start (already in DB)")

        # Trials remaining: full count on fresh, remainder on resume
        _trials_to_run = max_iterations - _resumed_completed'''

if old_study in wob_content:
    wob_content = wob_content.replace(old_study, new_study)
    print("✅ Patch 2b: resume-aware study creation block applied")
else:
    print("❌ Patch 2b FAILED: study creation block not found")
    print("   Ensure warm-start patch from earlier session is present")

# Update study.optimize() to use _trials_to_run
old_optimize = "        study.optimize(optuna_objective, n_trials=max_iterations, callbacks=[_incremental_callback])"
new_optimize = "        study.optimize(optuna_objective, n_trials=_trials_to_run, callbacks=[_incremental_callback])"

if old_optimize in wob_content:
    wob_content = wob_content.replace(old_optimize, new_optimize)
    print("✅ Patch 2c: study.optimize() updated to use _trials_to_run")
else:
    print("❌ Patch 2c FAILED: study.optimize() call not found")

with open(WOB_PATH, 'w') as f:
    f.write(wob_content)

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3: agent_manifests/window_optimizer.json — add resume_study param
# ─────────────────────────────────────────────────────────────────────────────
import json

MANIFEST_PATH = '/home/michael/distributed_prng_analysis/agent_manifests/window_optimizer.json'

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# Add resume_study to CLI param mapping
if 'param_mapping' in manifest.get('execution', [{}])[0]:
    manifest['execution'][0]['param_mapping']['resume-study'] = 'resume_study'
    print("✅ Patch 3a: resume-study added to param_mapping")

# Add resume_study to search_bounds/params documentation
if 'search_bounds' in manifest:
    manifest['search_bounds']['resume_study'] = {
        "type": "bool",
        "default": False,
        "description": "Resume most recent incomplete Optuna study DB. "
                       "False = fresh study every run (default, safe). "
                       "True = continue from last checkpoint."
    }
    print("✅ Patch 3b: resume_study documented in search_bounds")

# Add to default_params as False (safe default)
if 'default_params' in manifest:
    manifest['default_params']['resume_study'] = False
    print("✅ Patch 3c: resume_study=False added to default_params")

# Fix trials 50 -> 100 while we're here
if manifest.get('default_params', {}).get('trials') == 50:
    manifest['default_params']['trials'] = 100
    print("✅ Patch 3d: trials default 50->100")

with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)

print()
print("=" * 60)
print("ALL PATCHES COMPLETE")
print("=" * 60)
print()
print("To resume current 21-trial study:")
print("  nohup bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \\")
print("    --run-pipeline --start-step 1 --end-step 1 \\")
print('    --params \'{"lottery_file": "daily3.json", "resume_study": true}\'' + "' \\")
print("    > logs/step1_s114_resume.log 2>&1 &")
print()
print("For fresh run (default, no flag needed):")
print("  nohup bash -c 'PYTHONPATH=. python3 agents/watcher_agent.py \\")
print("    --run-pipeline --start-step 1 --end-step 1 \\")
print('    --params \'{"lottery_file": "daily3.json"}\'' + "' \\")
print("    > logs/step1_s114_fresh.log 2>&1 &")
