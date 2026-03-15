#!/usr/bin/env python3
"""
apply_s145r1_progressive_sweep.py
==================================
S145-R1 — Progressive Empirical Sweep of Seed IDs 0→2^32 for CA/java_lcg

Applies all TB-approved changes:
  1. window_optimizer_integration_final.py  — survivor accumulator (per-seed score merge)
  2. agent_manifests/window_optimizer.json  — manifest field corrections (4 values)
  3. agents/watcher_agent.py                — fresh-study invariant conditionalized
  4. agents/watcher_agent.py                — Step 1 timeout override 480→900
  5. .gitignore                             — accumulator JSON exception

All files backed up to <file>.s145r1_backup before modification.
Line-count verification after each patch.

Usage:
    python3 apply_s145r1_progressive_sweep.py [--dry-run]
"""

import re
import sys
import shutil
from pathlib import Path

DRY_RUN = '--dry-run' in sys.argv

PROJECT_ROOT = Path('/home/michael/distributed_prng_analysis')

FILES = {
    'integration': PROJECT_ROOT / 'window_optimizer_integration_final.py',
    'manifest':    PROJECT_ROOT / 'agent_manifests/window_optimizer.json',
    'watcher':     PROJECT_ROOT / 'agents/watcher_agent.py',
    'gitignore':   PROJECT_ROOT / '.gitignore',
}

# ─────────────────────────────────────────────────────────────────────────────

def read(path):
    return Path(path).read_text(encoding='utf-8')

def write(path, content):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {path}")
        return
    Path(path).write_text(content, encoding='utf-8')

def backup(path):
    bak = Path(str(path) + '.s145r1_backup')
    if DRY_RUN:
        print(f"  [DRY-RUN] would backup {path} → {bak.name}")
        return
    shutil.copy2(path, bak)
    print(f"  ✅ Backup: {bak.name}")

def verify_line_count(path, original_lines, added=0, removed=0):
    current = len(read(path).splitlines())
    expected = original_lines + added - removed
    if current == expected:
        print(f"  ✅ Line count OK: {current} lines")
    else:
        print(f"  ⚠️  Line count mismatch: expected {expected}, got {current}")

def apply_patch(content, old, new, description):
    if old not in content:
        print(f"  ❌ ANCHOR NOT FOUND: {description}")
        print(f"     Looking for: {repr(old[:80])}")
        return content, False
    result = content.replace(old, new, 1)
    print(f"  ✅ Patched: {description}")
    return result, True

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — window_optimizer_integration_final.py
# Replace NPZ conversion call with accumulator block + redirected conversion
# ─────────────────────────────────────────────────────────────────────────────

def patch_integration(path):
    print("\n[1/5] window_optimizer_integration_final.py — Survivor Accumulator")
    content = read(path)
    original_lines = len(content.splitlines())
    backup(path)

    OLD = '''            # Convert to NPZ binary format (required by Step 2)
            from subprocess import run as subprocess_run, CalledProcessError
            try:
                subprocess_run(
                    ["python3", "convert_survivors_to_binary.py", "bidirectional_survivors.json"],
                    check=True
                )
                print(f"✅ Converted to bidirectional_survivors_binary.npz")
            except CalledProcessError as e:
                print(f"❌ NPZ conversion failed: {e}")
                raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")'''

    NEW = '''            # [S145-R1] SURVIVOR ACCUMULATOR — merge into persistent cross-run store
            # Merge policy: best per-seed score wins on conflict (TB ruling S145-R1)
            # bidirectional_survivors.json still written above — no change to existing output
            import os as _os_s145
            _accum_path = 'bidirectional_survivors_all.json'
            try:
                if _os_s145.path.exists(_accum_path):
                    with open(_accum_path) as _af:
                        _prior_survivors = json.load(_af)
                else:
                    _prior_survivors = []
                _prior_count = len(_prior_survivors)
                # Merge — best per-seed score wins on conflict
                _merged = {s['seed']: s for s in _prior_survivors}
                for s in bidirectional_deduped:
                    if s['seed'] not in _merged or \
                       float(s.get('score', 0)) > float(_merged[s['seed']].get('score', 0)):
                        _merged[s['seed']] = s
                _merged_list = sorted(_merged.values(), key=lambda x: x['seed'])
                with open(_accum_path, 'w') as _af:
                    json.dump(_merged_list, _af)
                _net_new = len(_merged_list) - _prior_count
                print(f"\\n[S145-R1][ACCUMULATOR] {len(_merged_list):,} total survivors across all runs")
                print(f"   This run: +{len(bidirectional_deduped):,} candidates | Net new: +{_net_new:,}")
                print(f"   Accumulator: {_accum_path}")
            except Exception as _accum_err:
                print(f"\\n⚠️  [S145-R1][ACCUMULATOR] Failed (non-fatal): {_accum_err}")
                print(f"   Falling back to per-run NPZ conversion")
                _accum_path = 'bidirectional_survivors.json'

            # Convert accumulated set to NPZ binary format (required by Step 2)
            # Uses accumulator if available, falls back to per-run file on accumulator error
            from subprocess import run as subprocess_run, CalledProcessError
            try:
                subprocess_run(
                    ["python3", "convert_survivors_to_binary.py", _accum_path],
                    check=True
                )
                print(f"✅ Converted {_accum_path} to bidirectional_survivors_binary.npz")
            except CalledProcessError as e:
                print(f"❌ NPZ conversion failed: {e}")
                raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")'''

    content, ok = apply_patch(content, OLD, NEW, "survivor accumulator + redirected NPZ conversion")
    if ok:
        write(path, content)
        verify_line_count(path, original_lines, added=27, removed=9)
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — agent_manifests/window_optimizer.json
# Four value changes against live confirmed fields
# ─────────────────────────────────────────────────────────────────────────────

def patch_manifest(path):
    print("\n[2/5] agent_manifests/window_optimizer.json — Four field corrections")
    content = read(path)
    original_lines = len(content.splitlines())
    backup(path)
    all_ok = True

    # 2a — action timeout_minutes 240 → 900
    content, ok = apply_patch(
        content,
        '"timeout_minutes": 240',
        '"timeout_minutes": 900',
        "action.timeout_minutes 240 → 900"
    )
    all_ok = all_ok and ok

    # 2b — default_params.max_seeds 10000000 → 1073741824 (quarter of 2^32)
    content, ok = apply_patch(
        content,
        '"max_seeds": 10000000,',
        '"max_seeds": 1073741824,  // [S145-R1] quarter-space sweep target',
        "default_params.max_seeds 10000000 → 1073741824"
    )
    all_ok = all_ok and ok

    # 2c — default_params.enable_pruning false → true
    content, ok = apply_patch(
        content,
        '"enable_pruning": false,',
        '"enable_pruning": true,   // [S145-R1] prune zero-survivor trials (~17min saved each)',
        "default_params.enable_pruning false → true"
    )
    all_ok = all_ok and ok

    # 2d — default_params.window_trials 100 → 50 (recommended for 4-session split)
    # NOTE: operator may override per session. This sets the default.
    content, ok = apply_patch(
        content,
        '"window_trials": 100,',
        '"window_trials": 50,      // [S145-R1] 50 trials × ~17min = ~14hr per session',
        "default_params.window_trials 100 → 50"
    )
    all_ok = all_ok and ok

    if all_ok:
        write(path, content)
        verify_line_count(path, original_lines)  # same line count, value replacements only
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — agents/watcher_agent.py
# Conditionalize fresh-study invariant on study_name presence
# ─────────────────────────────────────────────────────────────────────────────

def patch_watcher_invariant(path):
    print("\n[3/5] agents/watcher_agent.py — Fresh-study invariant conditionalized")
    content = read(path)
    original_lines = len(content.splitlines())
    backup(path)

    OLD = '''                if _next_start > 0:
                    final_params['seed_start'] = _next_start
                    final_params['resume_study'] = False   # INVARIANT: new range = fresh study
                    final_params['study_name'] = ''        # force fresh study name
                    logger.info(
                        f"[COVERAGE] Step 1: advancing seed_start to {_next_start:,} "
                        f"for {_prng_type} — forcing fresh study"
                    )'''

    NEW = '''                if _next_start > 0:
                    final_params['seed_start'] = _next_start
                    # [S145-R1] Conditionalize fresh-study invariant on study_name presence.
                    # If operator explicitly provides study_name in default_params, preserve
                    # Optuna continuity across seed range boundaries (cross-session resume).
                    # Default behavior (no study_name) unchanged — fresh study on range advance.
                    _explicit_study = final_params.get('study_name', '')
                    if not _explicit_study:
                        final_params['resume_study'] = False   # INVARIANT: new range = fresh study
                        final_params['study_name'] = ''        # force fresh study name
                        logger.info(
                            f"[COVERAGE] Step 1: advancing seed_start to {_next_start:,} "
                            f"for {_prng_type} — forcing fresh study (no explicit study_name)"
                        )
                    else:
                        final_params['resume_study'] = True
                        logger.info(
                            f"[COVERAGE] Step 1: advancing seed_start to {_next_start:,} "
                            f"for {_prng_type} — preserving Optuna continuity "
                            f"(study_name='{_explicit_study}' explicitly set) [S145-R1]"
                        )'''

    content, ok = apply_patch(content, OLD, NEW,
        "fresh-study invariant — conditionalized on study_name presence")
    if ok:
        write(path, content)
        verify_line_count(path, original_lines, added=13, removed=6)
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 — agents/watcher_agent.py
# Step 1 timeout override 480 → 900
# ─────────────────────────────────────────────────────────────────────────────

def patch_watcher_timeout(path):
    print("\n[4/5] agents/watcher_agent.py — Step 1 timeout 480 → 900")
    content = read(path)
    original_lines = len(content.splitlines())
    # Already backed up in patch 3 — skip backup

    OLD = '        step_timeout_overrides={0: 1, 1: 480, 5: 360}'
    NEW = '        step_timeout_overrides={0: 1, 1: 900, 5: 360}  # [S145-R1] 900min = 50 trials × ~17min + buffer'

    content, ok = apply_patch(content, OLD, NEW,
        "step_timeout_overrides Step 1: 480 → 900")
    if ok:
        write(path, content)
        verify_line_count(path, original_lines)
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 5 — .gitignore
# Add accumulator JSON exception after line 44
# ─────────────────────────────────────────────────────────────────────────────

def patch_gitignore(path):
    print("\n[5/5] .gitignore — accumulator JSON exception")
    content = read(path)
    original_lines = len(content.splitlines())
    backup(path)

    # Check if already patched
    if 'bidirectional_survivors_all.json' in content:
        print("  ℹ️  Already contains exception — skipping")
        return True

    OLD = '!schema_*.json'
    NEW = ('!schema_*.json\n'
           '!bidirectional_survivors_all.json   '
           '# [S145-R1] persistent cross-run survivor accumulator')

    content, ok = apply_patch(content, OLD, NEW,
        "accumulator JSON exception")
    if ok:
        write(path, content)
        verify_line_count(path, original_lines, added=1)
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("apply_s145r1_progressive_sweep.py")
    print("S145-R1 — Progressive Empirical Sweep — CA/java_lcg")
    if DRY_RUN:
        print("MODE: DRY RUN — no files will be modified")
    print("=" * 70)

    # Verify all target files exist
    print("\nVerifying target files...")
    all_exist = True
    for name, path in FILES.items():
        if path.exists():
            lines = len(read(path).splitlines())
            print(f"  ✅ {path.name} ({lines} lines)")
        else:
            print(f"  ❌ NOT FOUND: {path}")
            all_exist = False
    if not all_exist:
        print("\nAbort — one or more target files missing.")
        sys.exit(1)

    # Apply patches
    results = []
    results.append(patch_integration(FILES['integration']))
    results.append(patch_manifest(FILES['manifest']))
    results.append(patch_watcher_invariant(FILES['watcher']))
    results.append(patch_watcher_timeout(FILES['watcher']))
    results.append(patch_gitignore(FILES['gitignore']))

    # Summary
    print("\n" + "=" * 70)
    passed = sum(results)
    print(f"RESULT: {passed}/5 patches applied successfully")

    if passed == 5:
        print("\n✅ ALL PATCHES APPLIED")
        print("\nNext steps:")
        print("  1. Run smoke test: python3 agents/watcher_agent.py --run-pipeline "
              "--start-step 1 --end-step 1  (100k seeds, 2 trials)")
        print("  2. Verify [S145-R1][ACCUMULATOR] log line appears")
        print("  3. Verify bidirectional_survivors_all.json created")
        print("  4. Commit: git add -f bidirectional_survivors_binary.npz "
              "bidirectional_survivors_all.json")
        print("  5. git add agent_manifests/window_optimizer.json "
              "agents/watcher_agent.py .gitignore "
              "window_optimizer_integration_final.py")
        print("  6. git commit -m 'feat(s145-r1): progressive sweep — "
              "accumulator + resume invariant + manifest corrections'")
        print("  7. git push origin main && git push public main")
        print("  8. Sync to ser8 + upload to Claude Project")
    else:
        print("\n⚠️  SOME PATCHES FAILED — review anchors above")
        print("   Do NOT commit until all 5 patches pass")
        sys.exit(1)

if __name__ == '__main__':
    main()
