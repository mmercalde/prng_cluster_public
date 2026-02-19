#!/usr/bin/env python3
"""
S93 Bug Fixes — Bug A + Bug B + Bug C
======================================

Bug A (CRITICAL): WATCHER Retry-Without-Rerun
    When health check returns RETRY, freshness gate blocks Step 5 re-execution
    but skip_registry still increments. NN hits SKIP_MODEL after 1 real run
    counted 3 times.
    
    FIX: Invalidate freshness (touch primary input) before setting current_step=5.

Bug B: Health Check Evaluates Requested Model, Not Winner
    After compare-models selects catboost, health check still evaluates neural_net
    because it reads from the CLI-requested model_type's diagnostics.
    
    FIX: Read winner from compare_models_summary.json sidecar and override
    the diagnostics evaluation target.

Bug C: diagnostics_llm_analyzer.py History Loop Type Guard
    History loop catches JSONDecodeError + KeyError but not AttributeError.
    When a history file contains a bare string instead of dict, hist.get() crashes.
    
    FIX: Add isinstance(hist, dict) guard and broaden exception catch.

Team Beta: Option 1 (best) for Bug A — force re-execution via freshness invalidation.

Usage:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    python3 apply_s93_bug_a_b_c_fixes.py

Author: Team Alpha (S93)
Date: 2026-02-15
"""

import os
import sys
import shutil
import re
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

WATCHER_PATH = "agents/watcher_agent.py"
HEALTH_CHECK_PATH = "training_health_check.py"
ANALYZER_PATH = "diagnostics_llm_analyzer.py"

BACKUP_SUFFIX = f".pre_s93_bug_fixes"

PATCHES_APPLIED = []
PATCHES_FAILED = []


# =============================================================================
# HELPERS
# =============================================================================

def backup_file(filepath):
    """Create backup of file."""
    backup = filepath + BACKUP_SUFFIX
    if os.path.exists(backup):
        print(f"  ⚠️  Backup already exists: {backup}")
        return True
    shutil.copy2(filepath, backup)
    print(f"  📦 Backup: {backup}")
    return True


def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()


def write_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)


def verify_syntax(filepath):
    """Verify Python syntax."""
    import py_compile
    try:
        py_compile.compile(filepath, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"  ❌ SYNTAX ERROR: {e}")
        return False


def restore_from_backup(filepath):
    """Restore file from backup."""
    backup = filepath + BACKUP_SUFFIX
    if os.path.exists(backup):
        shutil.copy2(backup, filepath)
        print(f"  🔄 Restored from backup: {backup}")
        return True
    return False


def apply_patch(name, filepath, old_str, new_str):
    """Apply a single string replacement patch."""
    content = read_file(filepath)
    
    if old_str not in content:
        print(f"  ❌ PATCH {name}: Target string NOT FOUND in {filepath}")
        PATCHES_FAILED.append(name)
        return False
    
    count = content.count(old_str)
    if count > 1:
        print(f"  ❌ PATCH {name}: Target string found {count} times (expected 1)")
        PATCHES_FAILED.append(name)
        return False
    
    content = content.replace(old_str, new_str)
    write_file(filepath, content)
    print(f"  ✅ PATCH {name}: Applied")
    PATCHES_APPLIED.append(name)
    return True


# =============================================================================
# BUG A: Freshness Invalidation on RETRY
# =============================================================================

def apply_bug_a():
    """
    Bug A Fix: When health check returns RETRY, invalidate freshness
    before setting current_step=5 so Step 5 actually re-executes.
    
    Two patches:
    A1: Add _invalidate_step_freshness() method to WatcherAgent
    A2: Call it in the RETRY branch of the health check loop
    """
    print("\n" + "="*60)
    print("BUG A: WATCHER Retry-Without-Rerun (CRITICAL)")
    print("="*60)
    
    content = read_file(WATCHER_PATH)
    
    # --- PATCH A1: Add _invalidate_step_freshness method ---
    # Insert after _get_max_training_retries method
    
    anchor_a1 = '        return 2\n\n    def _run_step_streaming'
    
    new_method = '''        return 2

    def _invalidate_step_freshness(self, step: int) -> None:
        """
        Invalidate freshness for a step so it re-executes on next run_step() call.
        
        Bug A fix (S93): When health check requests RETRY, the existing output is
        'fresh' but unhealthy. Touch the primary input to make output appear stale,
        forcing run_step() to actually re-execute instead of skipping.
        
        Strategy: Touch the first required input to current time.
        This makes check_output_freshness() return (False, "STALE: ...").
        """
        try:
            required_inputs, primary_output = get_step_io_from_manifest(step)
            if required_inputs and os.path.exists(required_inputs[0]):
                target = required_inputs[0]
                old_mtime = os.path.getmtime(target)
                os.utime(target, None)  # Touch to current time
                new_mtime = os.path.getmtime(target)
                logger.info(
                    "[S93][BUG-A] Invalidated freshness for Step %d: "
                    "touched %s (mtime %.0f -> %.0f)",
                    step, target, old_mtime, new_mtime
                )
            else:
                logger.warning(
                    "[S93][BUG-A] Could not invalidate freshness: "
                    "no inputs found for Step %d", step
                )
        except Exception as e:
            logger.warning(
                "[S93][BUG-A] Freshness invalidation failed (non-fatal): %s", e
            )

    def _run_step_streaming'''
    
    if anchor_a1 not in content:
        print(f"  ❌ PATCH A1: Anchor not found (_get_max_training_retries → _run_step_streaming)")
        PATCHES_FAILED.append("A1-add-invalidate-method")
        return False
    
    content = content.replace(anchor_a1, new_method)
    print(f"  ✅ PATCH A1: Added _invalidate_step_freshness() method")
    PATCHES_APPLIED.append("A1-add-invalidate-method")
    
    # --- PATCH A2: Call invalidation before setting current_step=5 ---
    # In the RETRY branch of the health check loop in run_pipeline()
    
    anchor_a2 = '''                            # Stay on Step 5 -- override current_step back
                            # (_handle_proceed already advanced to 6)
                            self.current_step = 5'''
    
    replacement_a2 = '''                            # Stay on Step 5 -- override current_step back
                            # (_handle_proceed already advanced to 6)
                            # S93 Bug A fix: Invalidate freshness so run_step()
                            # actually re-executes instead of skipping
                            self._invalidate_step_freshness(5)
                            self.current_step = 5'''
    
    if anchor_a2 not in content:
        print(f"  ❌ PATCH A2: Anchor not found (RETRY current_step=5 block)")
        PATCHES_FAILED.append("A2-call-invalidate-on-retry")
        return False
    
    content = content.replace(anchor_a2, replacement_a2)
    print(f"  ✅ PATCH A2: Added freshness invalidation before RETRY loop-back")
    PATCHES_APPLIED.append("A2-call-invalidate-on-retry")
    
    write_file(WATCHER_PATH, content)
    return True


# =============================================================================
# BUG A DEFENSE-IN-DEPTH: Skip Registry Real-Attempts Guard
# =============================================================================

def apply_bug_a_defense():
    """
    Team Beta micro-suggestion: Only increment consecutive_critical when
    Step 5 actually executed (not skipped by freshness). This makes the
    skip registry count real training attempts, not health-check passes.
    
    Patch location: run_pipeline() health check block in watcher_agent.py.
    Guard: Check if the run_step(5) result had "skipped": True. If so,
    don't call check_training_health() at all — stale diagnostics have
    no new information.
    """
    print("\n" + "="*60)
    print("BUG A DEFENSE-IN-DEPTH: Skip Registry Real-Attempts Guard")
    print("="*60)
    
    content = read_file(WATCHER_PATH)
    
    # The health check block starts with this condition.
    # We add a guard: if results.get("skipped"), skip health check entirely.
    
    anchor_a3 = '''                # -- Session 76: Post-Step-5 training health check ----------
                if (step == 5
                        and decision.recommended_action == "proceed"
                        and TRAINING_HEALTH_CHECK_AVAILABLE):

                    # Single health check call -- cached for both helpers
                    _health = check_training_health()'''
    
    replacement_a3 = '''                # -- Session 76: Post-Step-5 training health check ----------
                # S93 defense-in-depth: Only run health check if Step 5
                # actually executed (not skipped by freshness gate).
                # Stale diagnostics = no new evidence = don't increment counter.
                _step5_skipped = (results or {}).get("skipped", False)
                if _step5_skipped and step == 5:
                    logger.info(
                        "[S93][A3] Step 5 was freshness-skipped — "
                        "bypassing health check (no new training evidence)"
                    )
                if (step == 5
                        and decision.recommended_action == "proceed"
                        and TRAINING_HEALTH_CHECK_AVAILABLE
                        and not _step5_skipped):

                    # Single health check call -- cached for both helpers
                    _health = check_training_health()'''
    
    if anchor_a3 not in content:
        print(f"  ❌ PATCH A3: Anchor not found (Session 76 health check block)")
        PATCHES_FAILED.append("A3-skip-registry-guard")
        return False
    
    content = content.replace(anchor_a3, replacement_a3)
    print(f"  ✅ PATCH A3: Added skipped-execution guard on health check")
    PATCHES_APPLIED.append("A3-skip-registry-guard")
    
    write_file(WATCHER_PATH, content)
    return True

def apply_bug_b():
    """
    Bug B Fix: Health check should evaluate the winner model type,
    not the CLI-requested model type.
    
    The issue is in training_health_check.py's _check_training_health_impl().
    When diagnostics are in single-model format but compare-models was used,
    the diagnostics file only has the last model's data. We need to check
    if a compare_models_summary exists and use the winner from there.
    
    Patch: Add winner-aware logic to _check_training_health_impl().
    """
    print("\n" + "="*60)
    print("BUG B: Health Check Model Mismatch")
    print("="*60)
    
    content = read_file(HEALTH_CHECK_PATH)
    
    # The fix: After loading diagnostics, check for compare_models_summary.json
    # If it exists and has a winner, override the model_type for evaluation.
    # This goes in _check_training_health_impl() after loading the diagnostics.
    
    anchor_b = '''    # — Determine if multi-model or single-model format ———————————
    if 'models' in diag:
        # Multi-model format (from --compare-models)
        return _evaluate_multi_model(diag, policies, metric_bounds)
    else:
        # Single-model format
        return _evaluate_single_model(diag, policies, metric_bounds)'''
    
    replacement_b = '''    # — S93 Bug B: Check for compare-models winner override ————————
    # When compare-models was used, the winner model type should be
    # evaluated, not whatever model_type is in the diagnostics file.
    # The compare_models_summary.json sidecar has the authoritative winner.
    _compare_summary_path = os.path.join(
        os.path.dirname(diagnostics_path), "compare_models_summary.json"
    )
    if os.path.isfile(_compare_summary_path):
        try:
            with open(_compare_summary_path) as _csf:
                _cs = json.load(_csf)
            _winner = _cs.get("winner_model_type", _cs.get("winner"))
            if _winner and isinstance(_winner, str):
                _diag_model = diag.get("model_type", "unknown")
                if _diag_model != _winner:
                    logger.info(
                        "[S93][BUG-B] Overriding diagnostics model_type "
                        "'%s' with compare-models winner '%s'",
                        _diag_model, _winner
                    )
                    diag["model_type"] = _winner
        except (json.JSONDecodeError, IOError) as _cs_err:
            logger.debug("Could not read compare_models_summary: %s", _cs_err)

    # — Determine if multi-model or single-model format ———————————
    if 'models' in diag:
        # Multi-model format (from --compare-models)
        return _evaluate_multi_model(diag, policies, metric_bounds)
    else:
        # Single-model format
        return _evaluate_single_model(diag, policies, metric_bounds)'''
    
    if anchor_b not in content:
        # Try with different dash encoding (UTF-8 em-dash vs plain)
        anchor_b_alt = anchor_b.replace('—', '\xe2\x80\x94')
        if anchor_b_alt in content:
            anchor_b = anchor_b_alt
            replacement_b = replacement_b  # replacement uses ASCII comments
        else:
            print(f"  ❌ PATCH B1: Anchor not found (multi/single model format check)")
            print(f"     Searching for partial match...")
            if "Determine if multi-model or single-model format" in content:
                print(f"     Found partial match — trying broader search")
            PATCHES_FAILED.append("B1-winner-override")
            return False
    
    count = content.count(anchor_b)
    if count != 1:
        print(f"  ❌ PATCH B1: Anchor found {count} times (expected 1)")
        PATCHES_FAILED.append("B1-winner-override")
        return False
    
    content = content.replace(anchor_b, replacement_b)
    print(f"  ✅ PATCH B1: Added compare-models winner override in health check")
    PATCHES_APPLIED.append("B1-winner-override")
    
    write_file(HEALTH_CHECK_PATH, content)
    return True


# =============================================================================
# BUG C: diagnostics_llm_analyzer.py History Type Guard
# =============================================================================

def apply_bug_c():
    """
    Bug C Fix: History loop in build_diagnostics_prompt() catches
    JSONDecodeError + KeyError but not AttributeError. When a history
    file contains a bare string, hist.get() crashes.
    
    Fix: Add isinstance(hist, dict) guard and broaden exception to Exception.
    """
    print("\n" + "="*60)
    print("BUG C: diagnostics_llm_analyzer.py History Type Guard")
    print("="*60)
    
    content = read_file(ANALYZER_PATH)
    
    # The history loop
    anchor_c = '''                try:
                    with open(hp) as f:
                        hist = json.load(f)
                    recent.append({
                        'run_id': os.path.basename(hp).replace('.json', ''),
                        'severity': hist.get('diagnosis', {}).get('severity',
                                    hist.get('watcher_severity', 'unknown')),
                        'model_type': hist.get('model_type', 'unknown'),
                        'archived_at': hist.get('archived_at', ''),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue'''
    
    replacement_c = '''                try:
                    with open(hp) as f:
                        hist = json.load(f)
                    # S93 Bug C: Guard against non-dict entries (e.g. bare strings)
                    if not isinstance(hist, dict):
                        logger.warning(
                            "Skipping non-dict history entry in %s: %s",
                            hp, type(hist).__name__
                        )
                        continue
                    recent.append({
                        'run_id': os.path.basename(hp).replace('.json', ''),
                        'severity': hist.get('diagnosis', {}).get('severity',
                                    hist.get('watcher_severity', 'unknown')),
                        'model_type': hist.get('model_type', 'unknown'),
                        'archived_at': hist.get('archived_at', ''),
                    })
                except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
                    continue'''
    
    if anchor_c not in content:
        print(f"  ❌ PATCH C1: Anchor not found (history loop in build_diagnostics_prompt)")
        PATCHES_FAILED.append("C1-history-type-guard")
        return False
    
    content = content.replace(anchor_c, replacement_c)
    print(f"  ✅ PATCH C1: Added isinstance guard + broadened exception catch")
    PATCHES_APPLIED.append("C1-history-type-guard")
    
    write_file(ANALYZER_PATH, content)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("S93 Bug Fixes — A + B + C")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Verify all files exist
    for filepath in [WATCHER_PATH, HEALTH_CHECK_PATH, ANALYZER_PATH]:
        if not os.path.isfile(filepath):
            print(f"\n❌ FATAL: File not found: {filepath}")
            print(f"   Run from: ~/distributed_prng_analysis/")
            sys.exit(1)
    
    # Create backups
    print("\n--- Creating backups ---")
    for filepath in [WATCHER_PATH, HEALTH_CHECK_PATH, ANALYZER_PATH]:
        backup_file(filepath)
    
    # Apply patches
    bug_a_ok = apply_bug_a()
    bug_a_defense_ok = apply_bug_a_defense()
    bug_b_ok = apply_bug_b()
    bug_c_ok = apply_bug_c()
    
    # Verify syntax on all modified files
    print("\n--- Syntax verification ---")
    all_syntax_ok = True
    for filepath in [WATCHER_PATH, HEALTH_CHECK_PATH, ANALYZER_PATH]:
        if verify_syntax(filepath):
            print(f"  ✅ {filepath}: syntax OK")
        else:
            print(f"  ❌ {filepath}: SYNTAX FAILED — restoring from backup")
            restore_from_backup(filepath)
            all_syntax_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Patches applied: {len(PATCHES_APPLIED)}")
    for p in PATCHES_APPLIED:
        print(f"    ✅ {p}")
    if PATCHES_FAILED:
        print(f"  Patches FAILED:  {len(PATCHES_FAILED)}")
        for p in PATCHES_FAILED:
            print(f"    ❌ {p}")
    print(f"  Syntax check:    {'PASSED' if all_syntax_ok else 'FAILED'}")
    
    if all_syntax_ok and not PATCHES_FAILED:
        print("\n✅ ALL PATCHES APPLIED SUCCESSFULLY")
        print("\nNext steps:")
        print("  1. Test Bug A: PYTHONPATH=. python3 agents/watcher_agent.py \\")
        print("       --run-pipeline --start-step 5 --end-step 6 \\")
        print("       --params '{\"trials\":1,\"enable_diagnostics\":true}'")
        print("  2. Verify freshness invalidation in logs: grep 'BUG-A' watcher.log")
        print("  3. Check skip registry: python3 training_health_check.py --status")
        print("  4. If all good, commit:")
        print("     git add agents/watcher_agent.py training_health_check.py diagnostics_llm_analyzer.py")
        print("     git commit -m 'fix(s93): Bug A/B/C — freshness invalidation, winner override, history guard'")
        print("     git push origin main && git push public main")
    else:
        print("\n⚠️  SOME PATCHES FAILED — review output above")
        print("  Backups available with suffix:", BACKUP_SUFFIX)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
