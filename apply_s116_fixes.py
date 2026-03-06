#!/usr/bin/env python3
"""
S116 Fix Script - verified against live Zeus file content
Fix 1: window_optimizer_bayesian.py - add study_name to search(), update resume logic
Fix 2: window_optimizer.py - add study_name to run_bayesian_optimization(), pass to search()
Fix 3: agent_manifests/window_optimizer.json - rename 'trials'->'window_trials', add study_name
"""

import json, sys

# ============================================================================
# FIX 1: window_optimizer_bayesian.py
# ============================================================================
print("Fix 1: window_optimizer_bayesian.py")

with open('window_optimizer_bayesian.py', 'r') as f:
    content = f.read()

# 1a: Add study_name to search() signature
old_sig = "               resume_study: bool = False) -> Dict:"
new_sig = "               resume_study: bool = False,\n               study_name: str = '') -> Dict:"
if old_sig in content:
    content = content.replace(old_sig, new_sig)
    print("  ✅ 1a: Added study_name to search() signature")
elif "study_name: str = '') -> Dict:" in content:
    print("  ✅ 1a: Already has study_name - skipping")
else:
    print("  ❌ 1a: Pattern not found"); sys.exit(1)

# 1b: Replace resume logic to support specific study_name
old_resume = """        if resume_study:
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
                print(f"   📊 No existing study DBs found — starting fresh")"""

new_resume = """        if resume_study:
            # specific study_name takes priority over auto-select
            if study_name:
                _candidate_dbs = [f"optuna_studies/{study_name}.db"]
                print(f"   🔄 Requested study: {study_name}")
            else:
                _candidate_dbs = sorted(
                    _glob.glob("optuna_studies/window_opt_*.db"),
                    key=_os.path.getmtime,
                    reverse=True
                )
            _resume_found = False
            for _candidate_db in _candidate_dbs:
                if not _os.path.exists(_candidate_db):
                    print(f"   ⚠️  Study DB not found: {_candidate_db}")
                    continue
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
                        _resume_found = True
                        break
                    else:
                        print(f"   📊 Study {_candidate_name}: {_candidate_completed} trials — skipping")
                except Exception as _e:
                    print(f"   ⚠️  Could not load {_candidate_name} ({_e})")
            if not _resume_found:
                print(f"   📊 No resumable study found — starting fresh")"""

if old_resume in content:
    content = content.replace(old_resume, new_resume)
    print("  ✅ 1b: Resume logic updated to support specific study_name")
else:
    print("  ❌ 1b: Resume logic block not found"); sys.exit(1)

with open('window_optimizer_bayesian.py', 'w') as f:
    f.write(content)
print("✅ Fix 1 complete\n")

# ============================================================================
# FIX 2: window_optimizer.py
# ============================================================================
print("Fix 2: window_optimizer.py")

with open('window_optimizer.py', 'r') as f:
    content = f.read()

# 2a: Add study_name to run_bayesian_optimization() signature
old_rbsig = "    resume_study: bool = False\n) -> Dict[str, Any]:"
new_rbsig = "    resume_study: bool = False,\n    study_name: str = ''\n) -> Dict[str, Any]:"
if old_rbsig in content:
    content = content.replace(old_rbsig, new_rbsig)
    print("  ✅ 2a: Added study_name to run_bayesian_optimization()")
elif "study_name: str = ''" in content:
    print("  ✅ 2a: study_name already in run_bayesian_optimization()")
else:
    print("  ❌ 2a: run_bayesian_optimization() signature not found"); sys.exit(1)

# 2b: Pass resume_study and study_name to strategy.search()
old_search = "        return strategy.search(objective, bounds, max_iterations, scorer)"
new_search = "        return strategy.search(objective, bounds, max_iterations, scorer, resume_study=resume_study, study_name=study_name)"
if old_search in content:
    content = content.replace(old_search, new_search)
    print("  ✅ 2b: resume_study and study_name passed to strategy.search()")
elif "resume_study=resume_study" in content:
    print("  ✅ 2b: Already passing resume_study to strategy.search()")
else:
    print("  ❌ 2b: strategy.search() call not found"); sys.exit(1)

with open('window_optimizer.py', 'w') as f:
    f.write(content)
print("✅ Fix 2 complete\n")

# ============================================================================
# FIX 3: agent_manifests/window_optimizer.json
# ============================================================================
print("Fix 3: agent_manifests/window_optimizer.json")

with open('agent_manifests/window_optimizer.json', 'r') as f:
    manifest = json.load(f)

dp = manifest.get('default_params', {})

if 'trials' in dp and 'window_trials' not in dp:
    dp['window_trials'] = dp.pop('trials')
    print("  ✅ 3a: Renamed 'trials' -> 'window_trials' in default_params")
elif 'window_trials' in dp:
    print("  ✅ 3a: Already uses 'window_trials'")
else:
    print("  ❌ 3a: Neither 'trials' nor 'window_trials' found"); sys.exit(1)

if 'study_name' not in dp:
    dp['study_name'] = ''
    print("  ✅ 3b: Added 'study_name': '' to default_params")
else:
    print("  ✅ 3b: study_name already in default_params")

manifest['default_params'] = dp

args_map = manifest.get('args_map', {})
if 'study-name' not in args_map:
    args_map['study-name'] = 'study_name'
    manifest['args_map'] = args_map
    print("  ✅ 3c: Added 'study-name' -> 'study_name' to args_map")
else:
    print("  ✅ 3c: study-name already in args_map")

pd = manifest.get('param_docs', {})
if 'study_name' not in pd:
    pd['study_name'] = {
        "type": "str",
        "default": "",
        "description": "Optuna study DB name to resume (e.g. 'window_opt_1772507547'). Empty = auto-select most recent incomplete study.",
        "optimized_by": "Manual"
    }
    manifest['param_docs'] = pd
    print("  ✅ 3d: Added study_name to param_docs")

with open('agent_manifests/window_optimizer.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print("✅ Fix 3 complete\n")

print("=" * 60)
print("ALL FIXES COMPLETE")
print("\nVerify syntax:")
print("  python3 -c \"import py_compile; [py_compile.compile(f, doraise=True) for f in ['window_optimizer.py','window_optimizer_bayesian.py']]; print('Syntax OK')\"")
print("\nResume specific study:")
print("  --params '{\"window_trials\": 50, \"resume_study\": true, \"study_name\": \"window_opt_1772507547\"}'")
