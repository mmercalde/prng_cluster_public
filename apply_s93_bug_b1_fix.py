#!/usr/bin/env python3
"""
S93 Bug B1 Fix — Health Check Winner Override (targeted retry)
Matches exact Zeus formatting: box-drawing chars ── and ───────────────
"""
import os, sys, shutil

HEALTH_CHECK_PATH = "training_health_check.py"

content = open(HEALTH_CHECK_PATH).read()

anchor = """    # \u2500\u2500 Determine if multi-model or single-model format \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if 'models' in diag:
        # Multi-model format (from --compare-models)
        return _evaluate_multi_model(diag, policies, metric_bounds)
    else:
        # Single-model format
        return _evaluate_single_model(diag, policies, metric_bounds)"""

replacement = """    # \u2500\u2500 S93 Bug B: Check for compare-models winner override \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # When compare-models was used, the winner model type should be
    # evaluated, not whatever model_type is in the diagnostics file.
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

    # \u2500\u2500 Determine if multi-model or single-model format \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if 'models' in diag:
        # Multi-model format (from --compare-models)
        return _evaluate_multi_model(diag, policies, metric_bounds)
    else:
        # Single-model format
        return _evaluate_single_model(diag, policies, metric_bounds)"""

if anchor not in content:
    print(f"\u274c Anchor NOT FOUND. Dumping bytes around line 157:")
    lines = content.split('\n')
    for i in range(155, min(165, len(lines))):
        print(f"  {i+1}: {repr(lines[i])}")
    sys.exit(1)

count = content.count(anchor)
if count != 1:
    print(f"\u274c Anchor found {count} times (expected 1)")
    sys.exit(1)

content = content.replace(anchor, replacement)
open(HEALTH_CHECK_PATH, 'w').write(content)

# Syntax check
import py_compile
try:
    py_compile.compile(HEALTH_CHECK_PATH, doraise=True)
    print("\u2705 PATCH B1: Applied + syntax OK")
except py_compile.PyCompileError as e:
    print(f"\u274c SYNTAX ERROR: {e}")
    shutil.copy2(HEALTH_CHECK_PATH + ".pre_s93_bug_fixes", HEALTH_CHECK_PATH)
    print("Restored from backup")
    sys.exit(1)
