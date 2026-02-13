#!/usr/bin/env python3
"""
apply_s81_phase7_watcher_patch.py
=================================

Chapter 14 Phase 7 — Wire LLM diagnostics into WATCHER RETRY path.

Python-based idempotent patcher using anchor markers.
No sed/awk heuristics. No AST guessing.

SCOPE: Step 5 RETRY path ONLY.

PREREQUISITES:
  - S76 patch applied (_build_retry_params, _handle_training_health exist)
  - diagnostics_llm_analyzer.py in project root
  - diagnostics_analysis_schema.py in project root
  - grammars/diagnostics_analysis.gbnf in grammars/

DESIGN CONSTRAINTS (from Team Beta review):
  1. Step gate: LLM only runs on step==5 AND health['action']=='RETRY'
  2. Advisory-only: heuristic params are primary, LLM refines
  3. Hard clamp: _is_within_policy_bounds() applied per proposal
  4. Timeout: 120s enforced in diagnostics_llm_analyzer.py
  5. Never block shutdown: no threads, opportunistic lifecycle
  6. Param shape: flat values only (retry_params[key] = scalar)

IDEMPOTENT: Checks for anchor markers before patching. Safe to re-run.

Session: 81
Date: 2026-02-11
"""

import os
import sys
import re
import shutil
from datetime import datetime

WATCHER_PATH = "agents/watcher_agent.py"
BACKUP_SUFFIX = f".bak.s81.{datetime.now():%Y%m%d_%H%M%S}"

# Anchor markers for idempotency
ANCHOR_IMPORT = "# --- S81_PHASE7_LLM_DIAGNOSTICS_IMPORT ---"
ANCHOR_IMPORT_END = "# --- S81_PHASE7_LLM_DIAGNOSTICS_IMPORT_END ---"
ANCHOR_POLICY_BOUNDS = "# --- S81_PHASE7_POLICY_BOUNDS ---"
ANCHOR_POLICY_BOUNDS_END = "# --- S81_PHASE7_POLICY_BOUNDS_END ---"
ANCHOR_BUILD_RETRY = "# --- S81_PHASE7_LLM_REFINEMENT ---"
ANCHOR_BUILD_RETRY_END = "# --- S81_PHASE7_LLM_REFINEMENT_END ---"


def check_prerequisites():
    """Verify S76 patch is applied and dependencies exist."""
    errors = []

    if not os.path.isfile(WATCHER_PATH):
        errors.append(f"{WATCHER_PATH} not found. Run from ~/distributed_prng_analysis/")
        return errors  # Fatal

    with open(WATCHER_PATH) as f:
        content = f.read()

    # S76 markers
    if "TRAINING_HEALTH_CHECK_AVAILABLE" not in content:
        errors.append("TRAINING_HEALTH_CHECK_AVAILABLE not found -- S76 patch required first")
    if "_build_retry_params" not in content:
        errors.append("_build_retry_params not found -- S76 patch required first")
    if "_handle_training_health" not in content:
        errors.append("_handle_training_health not found -- S76 patch required first")

    # Dependencies
    if not os.path.isfile("diagnostics_llm_analyzer.py"):
        errors.append("diagnostics_llm_analyzer.py not found in project root")
    if not os.path.isfile("diagnostics_analysis_schema.py"):
        errors.append("diagnostics_analysis_schema.py not found in project root")

    return errors


def is_already_patched(content: str) -> dict:
    """Check which patches are already applied."""
    return {
        'import': ANCHOR_IMPORT in content,
        'policy_bounds': ANCHOR_POLICY_BOUNDS in content,
        'build_retry': ANCHOR_BUILD_RETRY in content,
    }


def find_insertion_point(lines: list, pattern: str, after: bool = True) -> int:
    """Find line index matching pattern. Returns -1 if not found."""
    for i, line in enumerate(lines):
        if pattern in line:
            return i + (1 if after else 0)
    return -1


def patch_import_guard(lines: list) -> list:
    """
    PATCH 1: Add LLM_DIAGNOSTICS_AVAILABLE import guard.

    Inserts after the TRAINING_HEALTH_AVAILABLE = False line.
    Uses try/except with no logger calls (logger may not exist yet).
    Sets a module-level bool flag only.
    """
    # Find insertion point: after "TRAINING_HEALTH_AVAILABLE = False"
    insert_idx = find_insertion_point(lines, "TRAINING_HEALTH_CHECK_AVAILABLE = False")
    if insert_idx == -1:
        # Fallback: after last top-level import in first 150 lines
        for i in range(min(150, len(lines)) - 1, -1, -1):
            if lines[i].startswith(("import ", "from ")):
                insert_idx = i + 1
                break
        if insert_idx == -1:
            print("  ERROR: Cannot find import insertion point")
            return lines

    import_block = [
        "",
        ANCHOR_IMPORT,
        "try:",
        "    from diagnostics_llm_analyzer import (",
        "        request_llm_diagnostics_analysis,",
        "    )",
        "    LLM_DIAGNOSTICS_AVAILABLE = True",
        "except ImportError:",
        "    LLM_DIAGNOSTICS_AVAILABLE = False",
        ANCHOR_IMPORT_END,
        "",
    ]

    return lines[:insert_idx] + import_block + lines[insert_idx:]


def patch_policy_bounds(lines: list) -> list:
    """
    PATCH 2: Add _is_within_policy_bounds() method to WatcherAgent.

    Chapter 14 Section 7.4 specification. Whitelist-based parameter clamp.
    Inserted BEFORE _build_retry_params() method.

    Team Beta approved with required None guard on proposed_value.
    """
    # Find _build_retry_params method
    insert_idx = -1
    for i, line in enumerate(lines):
        if "def _build_retry_params(self" in line:
            insert_idx = i
            break

    if insert_idx == -1:
        print("  ERROR: _build_retry_params not found — cannot insert _is_within_policy_bounds")
        return lines

    indent = "    "  # method indent (4 spaces, class method)
    body = "        "  # body indent (8 spaces)

    bounds_block = [
        "",
        f"{indent}{ANCHOR_POLICY_BOUNDS}",
        f"{indent}def _is_within_policy_bounds(self, param_name, proposed_value):",
        f'{body}"""Validate LLM parameter proposal against policy bounds.',
        f"{body}",
        f"{body}Returns True if proposed value is within allowed range.",
        f"{body}Unknown parameters are rejected by default (whitelist approach).",
        f"{body}",
        f"{body}Chapter 14 Section 7.4 -- Session 81.",
        f'{body}"""',
        f"{body}# None guard (Team Beta requirement)",
        f"{body}if proposed_value is None:",
        f"{body}    return False",
        f"",
        f"{body}td_config = {{}}",
        f"{body}if hasattr(self, 'config') and hasattr(self.config, 'get'):",
        f"{body}    td_config = self.config.get('training_diagnostics', {{}})",
        f"{body}elif hasattr(self, 'policies'):",
        f"{body}    td_config = self.policies.get('training_diagnostics', {{}})",
        f"",
        f"{body}allowed_params = td_config.get('llm_adjustable_params', {{",
        f"{body}    'normalize_features': {{'type': 'bool'}},",
        f"{body}    'nn_activation': {{'type': 'enum', 'values': [0, 1, 2]}},",
        f"{body}    'learning_rate': {{'type': 'float', 'min': 1e-6, 'max': 0.1}},",
        f"{body}    'dropout': {{'type': 'float', 'min': 0.0, 'max': 0.5}},",
        f"{body}    'n_estimators': {{'type': 'int', 'min': 50, 'max': 2000}},",
        f"{body}    'max_depth': {{'type': 'int', 'min': 3, 'max': 15}},",
        f"{body}}})",
        f"",
        f"{body}if param_name not in allowed_params:",
        f"{body}    return False",
        f"",
        f"{body}bounds = allowed_params[param_name]",
        f"{body}if bounds.get('type') == 'bool':",
        f"{body}    return proposed_value in (0, 1, True, False)",
        f"{body}if bounds.get('type') == 'enum':",
        f"{body}    return proposed_value in bounds.get('values', [])",
        f"{body}if 'min' in bounds and proposed_value < bounds['min']:",
        f"{body}    return False",
        f"{body}if 'max' in bounds and proposed_value > bounds['max']:",
        f"{body}    return False",
        f"{body}return True",
        f"{indent}{ANCHOR_POLICY_BOUNDS_END}",
        "",
    ]

    return lines[:insert_idx] + bounds_block + lines[insert_idx:]


def patch_build_retry_params(lines: list) -> list:
    """
    PATCH 3: Add LLM refinement block inside _build_retry_params().

    Inserts BEFORE the final 'return retry_params' line of the method.

    Design:
      - Step gate: health.get('action') == 'RETRY' (calling context enforces step==5)
      - Lifecycle: opportunistic — uses llm_lifecycle.session() context manager
        if available, skips cleanly if not
      - Param shape: extracts .proposed_value (scalar) from DiagnosticsAnalysis
        proposals, applies flat to retry_params[key] = value
      - Clamp: every value passes through _is_within_policy_bounds()
      - Best-effort: entire block wrapped in try/except, returns heuristic
        params unmodified on any failure
    """
    # Find _build_retry_params method
    method_start = -1
    for i, line in enumerate(lines):
        if "def _build_retry_params(self" in line:
            method_start = i
            break

    if method_start == -1:
        print("  ERROR: _build_retry_params method not found")
        return lines

    # Find the 'return retry_params' within this method
    # Scan until next method def at same/lower indent (true method boundary)
    return_idx = -1
    method_indent = len(lines[method_start]) - len(lines[method_start].lstrip())
    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        # Detect end of method: next def/class at same or lower indentation
        if stripped and not stripped.startswith('#'):
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= method_indent and (
                stripped.startswith('def ') or stripped.startswith('class ')
            ):
                break
        if stripped == "return retry_params":
            return_idx = i
            # Don't break — take the LAST return in the method

    if return_idx == -1:
        print("  ERROR: 'return retry_params' not found in _build_retry_params")
        return lines

    # Determine indentation from the return line
    indent = "        "  # 8 spaces (method body)

    refinement_block = [
        "",
        f"{indent}{ANCHOR_BUILD_RETRY}",
        f"{indent}# Phase 7: LLM diagnostics refinement (Session 81)",
        f"{indent}# Step gate enforced by calling context (Step 5 retry path only).",
        f"{indent}# Defense-in-depth gate: health.get('action') == 'RETRY'.",
        f"{indent}# LLM proposals are advisory -- each value clamped via policy bounds.",
        f"{indent}# Lifecycle: opportunistic session() -- skip if LLM unavailable.",
        f"{indent}# Step gate: _build_retry_params() is only called from the Step 5",
        f"{indent}# RETRY path in run_pipeline() (line ~1849). The calling context",
        f"{indent}# enforces step==5 + health_action=='retry'. We gate on the health",
        f"{indent}# dict action value as defense-in-depth.",
        f"{indent}if (LLM_DIAGNOSTICS_AVAILABLE",
        f"{indent}        and health.get('action') == 'RETRY'):",
        f"{indent}    try:",
        f"{indent}        _llm_analysis = None",
        f"{indent}        _diag_path = 'diagnostics_outputs/training_diagnostics.json'",
        f"{indent}        _tier_path = 'diagnostics_outputs/tier_comparison.json'",
        f"",
        f"{indent}        if os.path.isfile(_diag_path):",
        f"{indent}            # Opportunistic lifecycle: use session() if available",
        f"{indent}            _lifecycle = getattr(self, 'llm_lifecycle', None)",
        f"{indent}            if _lifecycle and hasattr(_lifecycle, 'session'):",
        f"{indent}                try:",
        f"{indent}                    with _lifecycle.session():",
        f"{indent}                        _llm_analysis = request_llm_diagnostics_analysis(",
        f"{indent}                            diagnostics_path=_diag_path,",
        f"{indent}                            tier_comparison_path=(",
        f"{indent}                                _tier_path if os.path.isfile(_tier_path) else None",
        f"{indent}                            ),",
        f"{indent}                            timeout=120,",
        f"{indent}                        )",
        f"{indent}                except Exception as _sess_err:",
        f"{indent}                    logger.warning(",
        f"{indent}                        '[WATCHER][LLM_DIAG] Lifecycle session failed: %s',",
        f"{indent}                        _sess_err,",
        f"{indent}                    )",
        f"{indent}            else:",
        f"{indent}                # No lifecycle manager -- call directly (server may be up)",
        f"{indent}                _llm_analysis = request_llm_diagnostics_analysis(",
        f"{indent}                    diagnostics_path=_diag_path,",
        f"{indent}                    tier_comparison_path=(",
        f"{indent}                        _tier_path if os.path.isfile(_tier_path) else None",
        f"{indent}                    ),",
        f"{indent}                    timeout=120,",
        f"{indent}                )",
        f"",
        f"{indent}        # Apply proposals with clamp -- flat values only",
        f"{indent}        if _llm_analysis and hasattr(_llm_analysis, 'parameter_proposals'):",
        f"{indent}            if not hasattr(self, '_is_within_policy_bounds'):",
        f"{indent}                logger.warning(",
        f"{indent}                    '[WATCHER][LLM_DIAG] _is_within_policy_bounds missing -- skipping all proposals'",
        f"{indent}                )",
        f"{indent}            else:",
        f"{indent}                for _prop in _llm_analysis.parameter_proposals:",
        f"{indent}                    _pname = _prop.parameter",
        f"{indent}                    _pval = _prop.proposed_value",
        f"{indent}                    if self._is_within_policy_bounds(_pname, _pval):",
        f"{indent}                        retry_params[_pname] = _pval",
        f"{indent}                        logger.info(",
        f"{indent}                            '[WATCHER][LLM_DIAG] Applied: %s = %s (%s)',",
        f"{indent}                            _pname, _pval, _prop.rationale[:80],",
        f"{indent}                        )",
        f"{indent}                    else:",
        f"{indent}                        logger.warning(",
        f"{indent}                            '[WATCHER][LLM_DIAG] REJECTED (bounds): %s = %s',",
        f"{indent}                            _pname, _pval,",
        f"{indent}                        )",
        f"",
        f"{indent}                logger.info(",
        f"{indent}                    '[WATCHER][LLM_DIAG] focus=%s confidence=%.2f',",
        f"{indent}                    _llm_analysis.focus_area.value,",
        f"{indent}                    _llm_analysis.root_cause_confidence,",
        f"{indent}                )",
        f"",
        f"{indent}    except Exception as _llm_err:",
        f"{indent}        logger.warning(",
        f"{indent}            '[WATCHER][LLM_DIAG] Refinement failed (non-fatal): %s',",
        f"{indent}            _llm_err,",
        f"{indent}        )",
        f"{indent}{ANCHOR_BUILD_RETRY_END}",
        "",
    ]

    return lines[:return_idx] + refinement_block + lines[return_idx:]


def verify_syntax(path: str) -> bool:
    """Compile to check for syntax errors."""
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"  SYNTAX ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("Chapter 14 Phase 7 — WATCHER LLM Wiring Patch")
    print("Session 81 — Python idempotent patcher")
    print("=" * 60)
    print()

    # Step 0: Prerequisites
    print("STEP 0: Checking prerequisites...")
    errors = check_prerequisites()
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print("\nPatch ABORTED.")
        sys.exit(1)
    print("  All prerequisites met.")
    print()

    # Read current file
    with open(WATCHER_PATH) as f:
        content = f.read()
    lines = content.splitlines()

    # Check existing patches
    applied = is_already_patched(content)

    # Step 1: Import guard
    if applied['import']:
        print("STEP 1: SKIP -- import guard already present")
    else:
        print("STEP 1: Adding LLM_DIAGNOSTICS_AVAILABLE import guard...")
        lines = patch_import_guard(lines)
        print("  DONE")
    print()

    # Step 2: _is_within_policy_bounds method
    if applied['policy_bounds']:
        print("STEP 2: SKIP -- _is_within_policy_bounds already present")
    else:
        print("STEP 2: Adding _is_within_policy_bounds() method...")
        lines = patch_policy_bounds(lines)
        print("  DONE")
    print()

    # Step 3: _build_retry_params refinement
    if applied['build_retry']:
        print("STEP 3: SKIP -- _build_retry_params LLM refinement already present")
    else:
        print("STEP 3: Adding LLM refinement to _build_retry_params()...")
        lines = patch_build_retry_params(lines)
        print("  DONE")
    print()

    if applied['import'] and applied['policy_bounds'] and applied['build_retry']:
        print("All patches already applied. Nothing to do.")
        return

    # Step 4: Write patched file
    print("STEP 4: Writing patched file...")

    # Backup
    backup_path = WATCHER_PATH + BACKUP_SUFFIX
    shutil.copy2(WATCHER_PATH, backup_path)
    print(f"  Backup: {backup_path}")

    # Write
    patched_content = "\n".join(lines) + "\n"
    with open(WATCHER_PATH, "w") as f:
        f.write(patched_content)

    # Step 5: Verify
    print()
    print("STEP 5: Verification...")

    if not verify_syntax(WATCHER_PATH):
        print("  RESTORING from backup...")
        shutil.copy2(backup_path, WATCHER_PATH)
        print("  Restored. Patch FAILED.")
        sys.exit(1)
    print("  PASS: Syntax check OK")

    # Verify anchors
    with open(WATCHER_PATH) as f:
        final = f.read()

    markers = {
        'LLM_DIAGNOSTICS_AVAILABLE': 'Import guard',
        ANCHOR_IMPORT: 'Import anchor',
        ANCHOR_POLICY_BOUNDS: 'Policy bounds anchor',
        ANCHOR_BUILD_RETRY: 'Build retry anchor',
        "health.get('action') == 'RETRY'": 'Action gate',
        '_is_within_policy_bounds': 'Clamp enforcement',
        'if proposed_value is None': 'None guard (Team Beta)',
        'llm_lifecycle': 'Lifecycle awareness',
        'timeout=120': 'Timeout enforcement',
    }

    for marker, label in markers.items():
        if marker in final:
            print(f"  PASS: {label}")
        else:
            print(f"  WARN: Missing {label}")

    new_lines = final.count("\n")
    print()
    print(f"  watcher_agent.py: {new_lines} lines")
    print()
    print("=" * 60)
    print("PATCH COMPLETE")
    print("=" * 60)
    print()
    print("Test with:")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --status")
    print()
    print("End-to-end test:")
    print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\")
    print("    --start-step 5 --end-step 6 \\")
    print("    --params '{\"trials\":3,\"max_seeds\":5000,\"enable_diagnostics\":true}'")


if __name__ == "__main__":
    main()
