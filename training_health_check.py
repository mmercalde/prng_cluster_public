#!/usr/bin/env python3
"""
Training Health Check — Chapter 14 Phase 6
===========================================

WATCHER integration module for post-Step-5 training diagnostics evaluation.
Called between Step 5 and Step 6 in the pipeline to evaluate training health
and decide on pipeline action.

Design Invariants (non-negotiable):
1. ABSENT ≠ FAILURE — Missing diagnostics maps to PROCEED, not BLOCK
2. BEST-EFFORT — If this module fails, pipeline continues normally
3. NO TRAINING MODIFICATION — Read-only access to diagnostics

Usage:
    from training_health_check import check_training_health, reset_skip_registry
    
    # In WATCHER pipeline after Step 5
    health = check_training_health()
    if health['action'] == 'PROCEED':
        reset_skip_registry(health['model_type'])
        # Continue to Step 6
    elif health['action'] == 'RETRY':
        # Retry Step 5 with modified params
    elif health['action'] == 'SKIP_MODEL':
        # Skip this model type, continue with others

Author: Distributed PRNG Analysis System
Date: February 2026

Version History:
    1.0.0   2026-02-08  Session 72  Initial implementation (Phase 6)
                                    - check_training_health() main function
                                    - Skip registry tracking
                                    - Policy threshold evaluation
                                    - Diagnostics archival
                                    - Multi-model support
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

MODULE_VERSION = "1.0.0"

# Path constants — same convention as existing WATCHER paths
DIAGNOSTICS_PATH = "diagnostics_outputs/training_diagnostics.json"
DIAGNOSTICS_HISTORY_DIR = "diagnostics_outputs/history/"
WATCHER_POLICIES_PATH = "watcher_policies.json"
SKIP_REGISTRY_PATH = "diagnostics_outputs/model_skip_registry.json"

# Default policy values (used if watcher_policies.json doesn't have training_diagnostics section)
DEFAULT_METRIC_BOUNDS = {
    "nn_dead_neuron_pct": {"warning": 25.0, "critical": 50.0},
    "nn_gradient_spread_ratio": {"warning": 100.0, "critical": 1000.0},
    "overfit_ratio": {"warning": 1.3, "critical": 1.5},
    "early_stop_ratio": {"warning": 0.3, "critical": 0.15},
    "unused_feature_pct": {"warning": 40.0, "critical": 70.0},
}

DEFAULT_SKIP_RULES = {
    "max_consecutive_critical": 3,
    "skip_duration_hours": 24,
}

# [CH14-DIAG wave 1, 2026-11-23] Declared settings, read from
# watcher_policies.json -> training_diagnostics.<key>; absent key = default.
#   grading_rules         legacy (default, today's reads) | fixed (CD-01, CD-02 reader, CD-06 reader)
#   early_stop_round_base fixed only: added to the producer's 0-based best_val_round before
#                         dividing by rounds_captured (0 = best_val_round / rounds, the policy text)
#   early_stop_min_rounds fixed only: fewer captured rounds than this -> ratio not graded
#   grade_scope           fixed only: winner (default) | all_families (action from worst family)
GRADING_RULES = ("legacy", "fixed")
DEFAULT_GRADING_RULES = "legacy"
DEFAULT_EARLY_STOP_ROUND_BASE = 0
DEFAULT_EARLY_STOP_MIN_ROUNDS = 2
GRADE_SCOPES = ("winner", "all_families")
DEFAULT_GRADE_SCOPE = "winner"
# Written by meta_prediction_optimizer_anti_overfit.py at --diagnostics-level full (CD-06).
COMPARE_POINTER_FILENAME = "compare_models_summary_latest.json"
FAMILY_REPORT_TEMPLATE = "{model}_diagnostics.json"
_SEVERITY_ORDER = {'absent': -1, 'stale': -1, 'ok': 0, 'warning': 1, 'critical': 2}

# [CH14-DIAG wave 2, 2026-11-23] Declared settings, read from watcher_policies.json ->
# training_diagnostics.<key> (model_skip_rules.<key> for expiry_mode); absent key = default,
# and every default is today's behaviour.
#   unused_feature_grading  CD-08, fixed rules only: off (default) | all | exclude_constant
#   stale_report_action     CD-07: PROCEED_WITH_NOTE (default) | RETRY | grade (grade anyway, with a note)
#                           read on the fixed path, and on the legacy path only when the caller
#                           passes expected_run_id / run_started_at (WATCHER draft line, not applied)
#   model_skip_rules.expiry_mode  CD-09: until_reset (default, today's) | hours (skip_duration_hours
#                           after last_critical)
#   history_prune           CD-10: off (default, today's: the health archive is never pruned) | on
#   history_prune_mode      CD-10: move (default) | delete
#   history_max_files       CD-10: default training_diagnostics.MAX_HISTORY_FILES
#   history_archive_dir     CD-10: where 'move' puts pruned files
#   history_prune_include_root_cause  CD-10: false (default; Chapter 13 root_cause_* files kept)
#   training_health_log     CD-11: off (default) | on (append one line per check)
UNUSED_FEATURE_GRADINGS = ("off", "all", "exclude_constant")
DEFAULT_UNUSED_FEATURE_GRADING = "off"
STALE_REPORT_ACTIONS = ("PROCEED_WITH_NOTE", "RETRY", "grade")
DEFAULT_STALE_REPORT_ACTION = "PROCEED_WITH_NOTE"
SKIP_EXPIRY_MODES = ("until_reset", "hours")
DEFAULT_SKIP_EXPIRY_MODE = "until_reset"
# [BENCHING switch, 2026-11-26] model_skip_rules.benching. Owner ruling 2026-11-24 10:47-10:50: "a poor
# health grade never removes a model family from the comparison; it triggers a why-report. Benching stays
# in the code as a switch, off." DEFAULT CHANGED: before this switch a family was always benched (SKIP_MODEL,
# and left out of the compare) at max_consecutive_critical.
#   off (default): never skip. The count is kept; at >= max_consecutive_critical the action stays RETRY,
#                  the result carries why_report_needed, a warning is logged and one line is appended to
#                  WHY_REPORT_FLAGS_PATH. The automatic why-report itself is not built (TODO-WHY-1).
#   on:  the skip as before. until_reset: identical. hours: once a timed skip has expired the count
#        restarts, so the family gets its full retries again (READY_LOOP_DIAG_2_20261123.md:229).
BENCHING_SWITCH = ("off", "on")
DEFAULT_BENCHING = "off"
WHY_REPORT_FLAGS_PATH = "diagnostics_outputs/why_report_flags.jsonl"
WHY_REPORT_FLAG_SCHEMA = "ch14_why_report_flag/1"
HISTORY_PRUNE_SWITCH = ("off", "on")
DEFAULT_HISTORY_PRUNE = "off"
HISTORY_PRUNE_MODES = ("move", "delete")
DEFAULT_HISTORY_PRUNE_MODE = "move"
DEFAULT_HISTORY_ARCHIVE_DIR = "diagnostics_outputs/history_archive/"
DEFAULT_HISTORY_PRUNE_INCLUDE_ROOT_CAUSE = False
# Prefixes in history/ written by other writers: td's compare_models FIFO and Chapter 13.
HISTORY_OTHER_WRITER_PREFIXES = ("compare_models_",)
HISTORY_ROOT_CAUSE_PREFIX = "root_cause_"
TRAINING_HEALTH_LOG_SWITCH = ("off", "on")
DEFAULT_TRAINING_HEALTH_LOG = "off"
TRAINING_HEALTH_LOG_FILENAME = "training_health_log.jsonl"
TRAINING_HEALTH_LOG_SCHEMA = "ch14_training_health_log/1"


def _choice(policies: Dict[str, Any], key: str, allowed, default):
    v = policies.get(key, default)
    if v not in allowed:
        raise ValueError(f"{key} must be one of {tuple(allowed)}, got {v!r}")
    return v


def benching_mode(skip_rules: Optional[Dict[str, Any]]) -> str:
    """model_skip_rules.benching, validated (fail closed on an unknown value, like expiry_mode)."""
    v = (skip_rules or {}).get('benching', DEFAULT_BENCHING)
    if v not in BENCHING_SWITCH:
        raise ValueError(f"model_skip_rules.benching must be one of {BENCHING_SWITCH}, got {v!r}")
    return v


def _wave2_settings(policies: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve and validate the CH14-DIAG wave 2 settings (fail closed on an unknown value)."""
    skip_rules = policies.get('model_skip_rules', DEFAULT_SKIP_RULES) or {}
    expiry = skip_rules.get('expiry_mode', DEFAULT_SKIP_EXPIRY_MODE)
    if expiry not in SKIP_EXPIRY_MODES:
        raise ValueError(f"model_skip_rules.expiry_mode must be one of {SKIP_EXPIRY_MODES}, got {expiry!r}")
    max_files = policies.get('history_max_files')
    if max_files is not None and (not isinstance(max_files, int) or max_files < 1):
        raise ValueError(f"history_max_files must be an int >= 1, got {max_files!r}")
    inc_rc = policies.get('history_prune_include_root_cause', DEFAULT_HISTORY_PRUNE_INCLUDE_ROOT_CAUSE)
    if not isinstance(inc_rc, bool):
        raise ValueError(f"history_prune_include_root_cause must be true or false, got {inc_rc!r}")
    return {
        'stale_report_action': _choice(policies, 'stale_report_action', STALE_REPORT_ACTIONS,
                                       DEFAULT_STALE_REPORT_ACTION),
        'expiry_mode': expiry,
        'benching': benching_mode(skip_rules),
        'history_prune': _choice(policies, 'history_prune', HISTORY_PRUNE_SWITCH, DEFAULT_HISTORY_PRUNE),
        'history_prune_mode': _choice(policies, 'history_prune_mode', HISTORY_PRUNE_MODES,
                                      DEFAULT_HISTORY_PRUNE_MODE),
        'history_max_files': max_files,
        'history_archive_dir': policies.get('history_archive_dir', DEFAULT_HISTORY_ARCHIVE_DIR),
        'history_prune_include_root_cause': inc_rc,
        'training_health_log': _choice(policies, 'training_health_log', TRAINING_HEALTH_LOG_SWITCH,
                                       DEFAULT_TRAINING_HEALTH_LOG),
    }


def _grading_settings(policies: Dict[str, Any], grading_rules: Optional[str] = None) -> Dict[str, Any]:
    """Resolve and validate the CH14-DIAG settings (fail closed on an unknown value)."""
    rules = grading_rules if grading_rules is not None else policies.get('grading_rules', DEFAULT_GRADING_RULES)
    if rules not in GRADING_RULES:
        raise ValueError(f"grading_rules must be one of {GRADING_RULES}, got {rules!r}")
    base = policies.get('early_stop_round_base', DEFAULT_EARLY_STOP_ROUND_BASE)
    if base not in (0, 1):
        raise ValueError(f"early_stop_round_base must be 0 or 1, got {base!r}")
    min_rounds = policies.get('early_stop_min_rounds', DEFAULT_EARLY_STOP_MIN_ROUNDS)
    if not isinstance(min_rounds, int) or min_rounds < 1:
        raise ValueError(f"early_stop_min_rounds must be an int >= 1, got {min_rounds!r}")
    scope = policies.get('grade_scope', DEFAULT_GRADE_SCOPE)
    if scope not in GRADE_SCOPES:
        raise ValueError(f"grade_scope must be one of {GRADE_SCOPES}, got {scope!r}")
    unused = _choice(policies, 'unused_feature_grading', UNUSED_FEATURE_GRADINGS,
                     DEFAULT_UNUSED_FEATURE_GRADING)
    return {'grading_rules': rules, 'early_stop_round_base': base,
            'early_stop_min_rounds': min_rounds, 'grade_scope': scope,
            'unused_feature_grading': unused}


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def check_training_health(diagnostics_path: str = DIAGNOSTICS_PATH,
                          grading_rules: Optional[str] = None,
                          expected_run_id: Optional[str] = None,
                          run_started_at: Optional[str] = None) -> Dict[str, Any]:
    """
    Post-Step-5 training health check.

    Called by WATCHER between Step 5 and Step 6 dispatch.
    Reads training_diagnostics.json, evaluates against watcher_policies.json
    thresholds, returns an action decision.

    Args:
        diagnostics_path: Path to diagnostics JSON file

    Returns:
        dict: {
            'action': 'PROCEED' | 'PROCEED_WITH_NOTE' | 'RETRY' | 'SKIP_MODEL',
            'model_type': str,
            'severity': 'ok' | 'warning' | 'critical' | 'absent',
            'issues': list[str],
            'suggested_fixes': list[str],
            'confidence': float,
            'note': Optional[str],
            'consecutive_critical': int (only for SKIP_MODEL),
        }

    If diagnostics file doesn't exist (--enable-diagnostics not used),
    returns PROCEED with a note. Absence of diagnostics is NOT a failure.

    [CH14-DIAG CD-07] expected_run_id / run_started_at: the Step 5 run the caller just finished.
    A report from another run grades 'stale' (action from stale_report_action) and never counts
    toward a model skip. Both None (today's call) = no such check on the legacy path.
    [CH14-DIAG CD-11] with training_health_log 'on', every result is appended to
    diagnostics_outputs/training_health_log.jsonl (one writer).
    """
    run_ctx = {'expected_run_id': expected_run_id, 'run_started_at': run_started_at}
    try:
        result = _check_training_health_impl(diagnostics_path, grading_rules, run_ctx)
    except Exception as e:
        # Best-effort — if this module fails, pipeline continues
        logger.error(f"Training health check failed (non-fatal): {e}")
        result = {
            'action': 'PROCEED_WITH_NOTE',
            'model_type': 'unknown',
            'severity': 'absent',
            'issues': [f'Health check module error: {e}'],
            'suggested_fixes': ['Check training_health_check.py logs'],
            'confidence': 0.5,
            'note': 'Health check failed — proceeding anyway (best-effort)',
        }
    _write_training_health_log(result, diagnostics_path, run_ctx)
    return result


def _check_training_health_impl(diagnostics_path: str, grading_rules: Optional[str] = None,
                                run_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal implementation — separated for clean exception handling."""
    run_ctx = run_ctx or {}
    _wave2_settings(_load_policies())  # [CH14-DIAG wave 2] fail closed on an unknown setting value

    # [CH14-DIAG] grading_rules 'fixed' takes its own path; 'legacy' runs the original body below.
    if _grading_settings(_load_policies(), grading_rules)['grading_rules'] == 'fixed':
        return _check_training_health_fixed(diagnostics_path, run_ctx)
    
    # ── No diagnostics file → proceed normally ────────────────────────
    if not os.path.isfile(diagnostics_path):
        logger.info("No training diagnostics found — proceeding without health check")
        return {
            'action': 'PROCEED',
            'model_type': 'unknown',
            'severity': 'absent',
            'issues': [],
            'suggested_fixes': [],
            'confidence': 0.5,
            'note': 'Diagnostics not enabled for this run',
        }

    # ── Load diagnostics ──────────────────────────────────────────────
    try:
        with open(diagnostics_path) as f:
            diag = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read diagnostics: {e}")
        return {
            'action': 'PROCEED_WITH_NOTE',
            'model_type': 'unknown',
            'severity': 'warning',
            'issues': [f'Diagnostics file unreadable: {e}'],
            'suggested_fixes': ['Check diagnostics_outputs/ for corruption'],
            'confidence': 0.3,
        }

    # ── Load policies ─────────────────────────────────────────────────
    policies = _load_policies()
    metric_bounds = policies.get('metric_bounds', DEFAULT_METRIC_BOUNDS)

    # [CH14-DIAG CD-07] only when the caller names its run (today's call passes nothing)
    fr = None
    if run_ctx.get('expected_run_id') or run_ctx.get('run_started_at'):
        fr = report_freshness(diag, diagnostics_path, run_ctx)
        if fr['freshness'] == 'stale':
            stale = _stale_result(diag.get('model_type', 'unknown'), fr, policies,
                                  {'grading_rules': 'legacy', 'report_path': diagnostics_path})
            if stale is not None:
                return stale
    
    # ── S93 Bug B: Check for compare-models winner override ─────────
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

    # ── Determine if multi-model or single-model format ───────────────
    if 'models' in diag:
        # Multi-model format (from --compare-models)
        result = _evaluate_multi_model(diag, policies, metric_bounds)
    else:
        # Single-model format
        result = _evaluate_single_model(diag, policies, metric_bounds)
    if fr is not None:  # [CH14-DIAG CD-07] caller named its run: say what was found
        result['freshness'] = fr
        if fr['freshness'] != 'fresh':
            result['issues'] = list(result.get('issues', [])) + [f"WATCHER: report freshness {fr['freshness']}: {fr['reason']}"]
    return result


def _evaluate_single_model(
    diag: Dict[str, Any],
    policies: Dict[str, Any],
    metric_bounds: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate single-model diagnostics."""
    
    model_type, severity, all_issues, fixes = _grade_single(diag, metric_bounds)
    
    # ── Map severity to action ────────────────────────────────────────
    action, consecutive, why = _severity_to_action(severity, model_type, policies)
    
    # ── Archive diagnostics for Strategy Advisor history ──────────────
    _archive_diagnostics(diag, severity, all_issues, model_type)
    
    result = {
        'action': action,
        'model_type': model_type,
        'severity': severity,
        'issues': all_issues,
        'suggested_fixes': fixes,
        'confidence': _severity_to_confidence(severity),
    }
    
    _attach_skip_fields(result, action, consecutive, why)
    
    logger.info(
        "Training health check: model=%s severity=%s action=%s issues=%d",
        model_type, severity, action, len(all_issues)
    )
    
    return result


def _grade_single(diag: Dict[str, Any], metric_bounds: Dict[str, Any],
                  settings: Optional[Dict[str, Any]] = None) -> tuple:
    """Pure grading of a single-model report: (model_type, severity, issues, fixes). No side effects."""
    model_type = diag.get('model_type', 'unknown')
    diagnosis = diag.get('diagnosis', {})
    severity = diagnosis.get('severity', 'ok')
    issues = list(diagnosis.get('issues', []))
    fixes = list(diagnosis.get('suggested_fixes', []))
    
    # ── Evaluate against WATCHER's own metric bounds ──────────────────
    # This cross-checks diagnostics severity against WATCHER thresholds
    # to prevent a buggy diagnostics module from under-reporting
    watcher_issues, severity = _evaluate_metrics(
        diag, metric_bounds, severity, settings=settings
    )

    # Combine diagnostics issues with WATCHER's own findings
    all_issues = issues + watcher_issues
    return model_type, severity, all_issues, fixes


def _grade_multi(diag: Dict[str, Any], metric_bounds: Dict[str, Any],
                 settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pure grading of a multi-model ('models') report. No side effects."""
    models = diag.get('models', {})
    comparison = diag.get('comparison', {})
    winner = comparison.get('winner', 'unknown')
    model_results = {}
    worst_severity = 'ok'
    severity_order = {'ok': 0, 'warning': 1, 'critical': 2}
    for model_type, model_report in models.items():
        diagnosis = model_report.get('diagnosis', {})
        model_severity = diagnosis.get('severity', 'ok')
        model_issues = list(diagnosis.get('issues', []))
        watcher_issues, model_severity = _evaluate_metrics(
            model_report, metric_bounds, model_severity, model_type, settings=settings
        )
        model_results[model_type] = {
            'severity': model_severity,
            'issues': model_issues + watcher_issues,
        }
        if severity_order.get(model_severity, 0) > severity_order.get(worst_severity, 0):
            worst_severity = model_severity
    all_issues = []
    for model_type, result in model_results.items():
        for issue in result['issues']:
            all_issues.append(f"[{model_type}] {issue}")
    all_fixes = []
    if winner in models:
        all_fixes = list(models[winner].get('diagnosis', {}).get('suggested_fixes', []))
    winner_severity = model_results.get(winner, {}).get('severity', 'ok')
    return {'winner': winner, 'model_results': model_results, 'worst_severity': worst_severity,
            'winner_severity': winner_severity, 'issues': all_issues, 'fixes': all_fixes}


def _evaluate_multi_model(
    diag: Dict[str, Any],
    policies: Dict[str, Any],
    metric_bounds: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate multi-model diagnostics (from --compare-models).

    Returns worst-case severity across all models, but tracks
    issues per model for targeted retry/skip decisions.
    """
    g = _grade_multi(diag, metric_bounds)
    winner = g['winner']
    model_results = g['model_results']
    worst_severity = g['worst_severity']
    all_issues = g['issues']
    all_fixes = g['fixes']
    
    # ── For multi-model, we care most about the winner ────────────────
    # If winner is healthy, proceed even if other models had issues
    winner_severity = g['winner_severity']
    
    if winner_severity == 'ok' and worst_severity != 'ok':
        # Winner is fine, other models had issues — proceed with note
        logger.info(
            "Winner model %s is healthy; other models had issues (worst=%s)",
            winner, worst_severity
        )
        action = 'PROCEED_WITH_NOTE'
        severity = 'warning'
        consecutive, why = 0, None
    else:
        action, consecutive, why = _severity_to_action(winner_severity, winner, policies)
        severity = winner_severity
    
    # ── Archive ───────────────────────────────────────────────────────
    _archive_diagnostics(diag, severity, all_issues, f"multi_{winner}")
    
    result = {
        'action': action,
        'model_type': winner,
        'severity': severity,
        'issues': all_issues,
        'suggested_fixes': all_fixes,
        'confidence': _severity_to_confidence(severity),
        'multi_model': True,
        'per_model_severity': {m: r['severity'] for m, r in model_results.items()},
    }
    
    _attach_skip_fields(result, action, consecutive, why)
    
    logger.info(
        "Training health check (multi-model): winner=%s severity=%s action=%s",
        winner, severity, action
    )
    
    return result


def _evaluate_metrics(
    diag: Dict[str, Any],
    metric_bounds: Dict[str, Any],
    current_severity: str,
    model_type: str = None,
    settings: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Evaluate diagnostics against WATCHER metric bounds.

    settings None or grading_rules 'legacy': the reads below are exactly today's.
    grading_rules 'fixed' (CH14-DIAG): CD-01 early-stop key, CD-02 spread key.
    
    Returns:
        tuple: (watcher_issues: list, updated_severity: str)
    """
    if settings is not None and settings.get('grading_rules') == 'fixed':
        return _evaluate_metrics_fixed(diag, metric_bounds, current_severity, settings)
    watcher_issues = []
    severity = current_severity
    
    # ── Check NN-specific metrics ─────────────────────────────────────
    # Look for model_specific or nn_specific depending on schema version
    nn_data = diag.get('model_specific', {})
    if not nn_data:
        nn_data = diag.get('nn_specific', {})
    
    gradient_health = nn_data.get('gradient_health', {})
    layer_health = nn_data.get('layer_health', {})
    
    # Dead neuron check
    if layer_health:
        bounds = metric_bounds.get('nn_dead_neuron_pct', DEFAULT_METRIC_BOUNDS['nn_dead_neuron_pct'])
        for layer_name, layer_data in layer_health.items():
            dead_pct = layer_data.get('dead_pct', 0)
            if dead_pct >= bounds.get('critical', 50):
                watcher_issues.append(
                    f"WATCHER: {layer_name} dead neurons {dead_pct:.1f}% >= critical threshold"
                )
                severity = 'critical'
            elif dead_pct >= bounds.get('warning', 25):
                watcher_issues.append(
                    f"WATCHER: {layer_name} dead neurons {dead_pct:.1f}% >= warning threshold"
                )
                if severity == 'ok':
                    severity = 'warning'
    
    # Gradient spread check
    gradient_spread = nn_data.get('feature_gradient_spread', 1)
    if gradient_spread > 1:
        bounds = metric_bounds.get('nn_gradient_spread_ratio', DEFAULT_METRIC_BOUNDS['nn_gradient_spread_ratio'])
        if gradient_spread >= bounds.get('critical', 1000):
            watcher_issues.append(
                f"WATCHER: Feature gradient spread {gradient_spread:.0f}x >= critical"
            )
            severity = 'critical'
        elif gradient_spread >= bounds.get('warning', 100):
            watcher_issues.append(
                f"WATCHER: Feature gradient spread {gradient_spread:.0f}x >= warning"
            )
            if severity == 'ok':
                severity = 'warning'
    
    # ── Check universal metrics (all model types) ─────────────────────
    training_summary = diag.get('training_summary', {})
    
    # Early stop ratio check
    best_round = training_summary.get('best_round', 0)
    total_rounds = training_summary.get('rounds_captured', 1)
    if total_rounds > 0:
        early_stop_ratio = best_round / max(total_rounds, 1)
        
        bounds = metric_bounds.get('early_stop_ratio', DEFAULT_METRIC_BOUNDS['early_stop_ratio'])
        if early_stop_ratio <= bounds.get('critical', 0.15):
            watcher_issues.append(
                f"WATCHER: Early stop ratio {early_stop_ratio:.2f} "
                f"(peaked at round {best_round}/{total_rounds}) — severe overfitting"
            )
            severity = 'critical'
        elif early_stop_ratio <= bounds.get('warning', 0.3):
            watcher_issues.append(
                f"WATCHER: Early stop ratio {early_stop_ratio:.2f} — possible overfitting"
            )
            if severity == 'ok':
                severity = 'warning'
    
    # Overfit ratio check (val_loss / train_loss)
    final_train = training_summary.get('final_train_loss')
    final_val = training_summary.get('final_val_loss')
    if final_train and final_val and final_train > 0:
        overfit_ratio = final_val / final_train
        
        bounds = metric_bounds.get('overfit_ratio', DEFAULT_METRIC_BOUNDS['overfit_ratio'])
        if overfit_ratio >= bounds.get('critical', 1.5):
            watcher_issues.append(
                f"WATCHER: Overfit ratio {overfit_ratio:.2f} >= critical (val/train loss)"
            )
            severity = 'critical'
        elif overfit_ratio >= bounds.get('warning', 1.3):
            watcher_issues.append(
                f"WATCHER: Overfit ratio {overfit_ratio:.2f} >= warning (val/train loss)"
            )
            if severity == 'ok':
                severity = 'warning'
    
    return watcher_issues, severity


def _escalate(severity: str, to: str) -> str:
    return to if _SEVERITY_ORDER.get(to, 0) > _SEVERITY_ORDER.get(severity, 0) else severity


def compute_fixed_metrics(diag: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    CH14-DIAG: the values the fixed rules grade, with how each was read. Pure.
      early_stop_ratio  (best_val_round + early_stop_round_base) / rounds_captured   (CD-01, R-60)
                        legacy key best_round is the fallback (health check self-test fixture only)
      gradient_spread   model_specific.gradient_health.gradient_spread_ratio          (CD-02, R-61)
                        legacy key feature_gradient_spread is the fallback
      overfit_ratio     final_val_loss / final_train_loss (unchanged)
    """
    nn_data = diag.get('model_specific', {}) or diag.get('nn_specific', {}) or {}
    gh = nn_data.get('gradient_health', {}) or {}
    spread = gh.get('gradient_spread_ratio', nn_data.get('feature_gradient_spread'))
    ts = diag.get('training_summary', {}) or {}
    if 'best_val_round' in ts:
        best, best_key = ts.get('best_val_round'), 'best_val_round'
    elif 'best_round' in ts:
        best, best_key = ts.get('best_round'), 'best_round'
    else:
        best, best_key = None, None
    total = ts.get('rounds_captured')
    ratio = None
    ratio_note = None
    if best is None or total is None:
        ratio_note = 'not computable (no best round or rounds_captured in report)'
    elif total < settings['early_stop_min_rounds']:
        ratio_note = f'not graded ({total} rounds < early_stop_min_rounds {settings["early_stop_min_rounds"]})'
    else:
        ratio = (best + settings['early_stop_round_base']) / max(total, 1)
    ft, fv = ts.get('final_train_loss'), ts.get('final_val_loss')
    overfit = (fv / ft) if (ft and fv and ft > 0) else None
    # CD-08 (wave 2): unused_feature_pct from the level-full report (None when not reported)
    uf = diag.get('unused_features') or {}
    basis = 'pct_excluding_constant' if settings.get('unused_feature_grading') == 'exclude_constant' else 'pct'
    unused = uf.get(basis) if uf else None
    return {'unused_feature_pct': unused, 'unused_feature_basis': basis if uf else None,
            'early_stop_ratio': ratio, 'early_stop_note': ratio_note, 'best_round': best,
            'best_round_key': best_key, 'rounds_captured': total,
            'gradient_spread': spread, 'overfit_ratio': overfit,
            'layer_dead_pct': {k: v.get('dead_pct', 0) for k, v in (nn_data.get('layer_health', {}) or {}).items()}}


def _evaluate_metrics_fixed(diag, metric_bounds, current_severity, settings) -> tuple:
    """CH14-DIAG fixed rules. Same bounds and messages as legacy, correct keys."""
    issues = []
    severity = current_severity
    m = compute_fixed_metrics(diag, settings)
    b = metric_bounds.get('nn_dead_neuron_pct', DEFAULT_METRIC_BOUNDS['nn_dead_neuron_pct'])
    for layer_name, dead_pct in m['layer_dead_pct'].items():
        if dead_pct >= b.get('critical', 50):
            issues.append(f"WATCHER: {layer_name} dead neurons {dead_pct:.1f}% >= critical threshold")
            severity = 'critical'
        elif dead_pct >= b.get('warning', 25):
            issues.append(f"WATCHER: {layer_name} dead neurons {dead_pct:.1f}% >= warning threshold")
            severity = _escalate(severity, 'warning')
    spread = m['gradient_spread']
    if spread is not None and spread > 1:
        b = metric_bounds.get('nn_gradient_spread_ratio', DEFAULT_METRIC_BOUNDS['nn_gradient_spread_ratio'])
        if spread >= b.get('critical', 1000):
            issues.append(f"WATCHER: Feature gradient spread {spread:.0f}x >= critical")
            severity = 'critical'
        elif spread >= b.get('warning', 100):
            issues.append(f"WATCHER: Feature gradient spread {spread:.0f}x >= warning")
            severity = _escalate(severity, 'warning')
    ratio = m['early_stop_ratio']
    if ratio is None:
        issues.append(f"WATCHER: Early stop ratio {m['early_stop_note']}")
    else:
        b = metric_bounds.get('early_stop_ratio', DEFAULT_METRIC_BOUNDS['early_stop_ratio'])
        where = f"best {m['best_round_key']}={m['best_round']} of {m['rounds_captured']} rounds, base {settings['early_stop_round_base']}"
        if ratio <= b.get('critical', 0.15):
            issues.append(f"WATCHER: Early stop ratio {ratio:.2f} ({where}) — severe overfitting")
            severity = 'critical'
        elif ratio <= b.get('warning', 0.3):
            issues.append(f"WATCHER: Early stop ratio {ratio:.2f} ({where}) — possible overfitting")
            severity = _escalate(severity, 'warning')
    if settings.get('unused_feature_grading', DEFAULT_UNUSED_FEATURE_GRADING) != 'off':
        u = m['unused_feature_pct']
        if u is None:
            issues.append("WATCHER: unused_feature_pct not in report (needs --diagnostics-level full)")
        else:
            b = metric_bounds.get('unused_feature_pct', DEFAULT_METRIC_BOUNDS['unused_feature_pct'])
            if u >= b.get('critical', 70):
                issues.append(f"WATCHER: Unused features {u:.1f}% ({m['unused_feature_basis']}) >= critical")
                severity = 'critical'
            elif u >= b.get('warning', 40):
                issues.append(f"WATCHER: Unused features {u:.1f}% ({m['unused_feature_basis']}) >= warning")
                severity = _escalate(severity, 'warning')
    ov = m['overfit_ratio']
    if ov is not None:
        b = metric_bounds.get('overfit_ratio', DEFAULT_METRIC_BOUNDS['overfit_ratio'])
        if ov >= b.get('critical', 1.5):
            issues.append(f"WATCHER: Overfit ratio {ov:.2f} >= critical (val/train loss)")
            severity = 'critical'
        elif ov >= b.get('warning', 1.3):
            issues.append(f"WATCHER: Overfit ratio {ov:.2f} >= warning (val/train loss)")
            severity = _escalate(severity, 'warning')
    return issues, severity


def grade_report(diag: Dict[str, Any], policies: Dict[str, Any],
                 grading_rules: Optional[str] = None) -> Dict[str, Any]:
    """
    CH14-DIAG CD-00: pure grade of one report dict (no registry, no archive, no file I/O).
    Returns severity, model_type, issues and (fixed rules) the graded metric values.
    """
    settings = _grading_settings(policies, grading_rules)
    metric_bounds = policies.get('metric_bounds', DEFAULT_METRIC_BOUNDS)
    pass_settings = settings if settings['grading_rules'] == 'fixed' else None
    if 'models' in diag:
        g = _grade_multi(diag, metric_bounds, pass_settings)
        sev = g['winner_severity']
        if sev == 'ok' and g['worst_severity'] != 'ok':
            sev = 'warning'
        out = {'model_type': g['winner'], 'severity': sev, 'issues': g['issues'],
               'multi_model': True,
               'per_model_severity': {k: v['severity'] for k, v in g['model_results'].items()}}
    else:
        mt, sev, issues, _ = _grade_single(diag, metric_bounds, pass_settings)
        if pass_settings is not None and sev == 'ok' and not (diag.get('training_summary') or {}).get('rounds_captured'):
            # fixed: a report with no captured rounds carries no evidence; say 'absent', not 'ok'
            # (same action: ABSENT != FAILURE maps to PROCEED)
            sev = 'absent'
            issues = issues + [f"WATCHER: report has no captured rounds (status {diag.get('status')!r}) — nothing to grade"]
        out = {'model_type': mt, 'severity': sev, 'issues': issues, 'multi_model': False}
    out['grading_rules'] = settings['grading_rules']
    if settings['grading_rules'] == 'fixed' and 'models' not in diag:
        out['metrics'] = compute_fixed_metrics(diag, settings)
    return out


def _winner_from_summary(cs: Dict[str, Any]) -> Optional[str]:
    """CD-06: winner from a compare-models summary. mpo writes winner_best_effort as a dict (mpo:311)."""
    wbe = cs.get('winner_best_effort')
    if isinstance(wbe, dict) and isinstance(wbe.get('model_type'), str):
        return wbe['model_type']
    for k in ('winner_model_type', 'winner'):
        if isinstance(cs.get(k), str):
            return cs[k]
    return None


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _check_training_health_fixed(diagnostics_path: str, run_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CH14-DIAG grading_rules 'fixed'.
      CD-06 reader: when diagnostics_outputs/compare_models_summary_latest.json exists, grade the
      winner's own report ({winner}_diagnostics.json), not whichever family wrote the canonical
      file last; grade every family found and list them (per_family). grade_scope decides whether
      the action follows the winner (default) or the worst family.
      Without the pointer: the canonical file, as legacy.
    CD-07 (wave 2): each report's run_id is compared with the pointer's run_id, and, when the
    caller names its run (expected_run_id / run_started_at), with that. A stale winner report
    gives a 'stale' result (stale_report_action); a stale family report is listed, not graded.
    Limit: without the caller's run id, a pointer and family reports that all come from one
    older run agree with each other and are not detected.
    """
    run_ctx = run_ctx or {}
    policies = _load_policies()
    settings = _grading_settings(policies, 'fixed')
    diag_dir = os.path.dirname(diagnostics_path)
    ptr = os.path.join(diag_dir, COMPARE_POINTER_FILENAME)
    winner, report_path, report_source, family_paths, compare_run_id = None, diagnostics_path, 'canonical', {}, None
    pointer_diag_run_id = None
    if os.path.isfile(ptr):
        cs = _load_json(ptr)
        winner = _winner_from_summary(cs)
        compare_run_id = cs.get('run_id')
        pointer_diag_run_id = cs.get('diagnostics_run_id')  # CD-07 stamp (None for older pointers)
        family_paths = {m: os.path.join(diag_dir, FAMILY_REPORT_TEMPLATE.format(model=m))
                        for m in (cs.get('models') or {})}
        if winner:
            report_path = os.path.join(diag_dir, FAMILY_REPORT_TEMPLATE.format(model=winner))
            report_source = 'compare_pointer_winner_report'
    base = {'grading_rules': 'fixed', 'report_source': report_source, 'report_path': report_path,
            'compare_run_id': compare_run_id}
    if not os.path.isfile(report_path):
        note = ('Diagnostics not enabled for this run' if report_source == 'canonical'
                else f'Winner {winner} has no diagnostics report ({report_path})')
        return {'action': 'PROCEED', 'model_type': winner or 'unknown', 'severity': 'absent',
                'issues': [], 'suggested_fixes': [], 'confidence': 0.5, 'note': note, **base}
    try:
        diag = _load_json(report_path)
    except (json.JSONDecodeError, IOError) as e:
        return {'action': 'PROCEED_WITH_NOTE', 'model_type': winner or 'unknown', 'severity': 'warning',
                'issues': [f'Diagnostics file unreadable: {e}'],
                'suggested_fixes': ['Check diagnostics_outputs/ for corruption'], 'confidence': 0.3, **base}
    pointer_run_id = pointer_diag_run_id if report_source == 'compare_pointer_winner_report' else None
    fr = report_freshness(diag, report_path, run_ctx, pointer_run_id)
    base['freshness'] = fr
    if fr['freshness'] == 'stale':
        stale = _stale_result(winner or diag.get('model_type', 'unknown'), fr, policies, base)
        if stale is not None:
            return stale
    g = grade_report(diag, policies, 'fixed')
    model_type = winner or g['model_type']
    severity = g['severity']
    issues = list(g['issues'])
    per_family = {}
    family_reports = {}  # G-2: each graded family's own report and grade, kept for promotion
    for m, fp in family_paths.items():
        if not os.path.isfile(fp):
            per_family[m] = {'severity': 'absent', 'path': fp}
            continue
        try:
            frep = _load_json(fp)
            ffr = report_freshness(frep, fp, run_ctx, pointer_diag_run_id)
            if ffr['freshness'] == 'stale':
                per_family[m] = {'severity': 'stale', 'path': fp, 'freshness': ffr,
                                 'issues': [f"WATCHER: not graded, stale report: {ffr['reason']}"]}
                continue
            fg = grade_report(frep, policies, 'fixed')
            per_family[m] = {'severity': fg['severity'], 'issues': fg['issues'], 'path': fp,
                             'freshness': ffr['freshness']}
            family_reports[m] = (frep, fg, ffr)
        except Exception as e:  # one unreadable family never blocks the check
            per_family[m] = {'severity': 'absent', 'path': fp, 'error': str(e)}
    graded = {k: v for k, v in per_family.items() if v['severity'] not in ('absent', 'stale')}
    worst_family = max(graded, key=lambda k: _SEVERITY_ORDER.get(graded[k]['severity'], 0)) if graded else None
    winner_name, promoted = model_type, None
    if per_family and settings['grade_scope'] == 'all_families' and worst_family:
        if _SEVERITY_ORDER.get(graded[worst_family]['severity'], 0) > _SEVERITY_ORDER.get(severity, 0):
            model_type, severity = worst_family, graded[worst_family]['severity']
            promoted = worst_family
    # G-2: a promoted family's own report, grade and issues; the winner's issues follow, prefixed.
    # worst_family comes from `graded`, filled only after a successful load and grade, so its
    # report is always in family_reports; the guard below reports it if that ever changes.
    out_diag, out_g = diag, g
    if promoted is not None:
        if promoted in family_reports:
            out_diag, out_g, _ = family_reports[promoted]
            issues = list(out_g['issues']) + [f"[{winner_name}] {i}" for i in g['issues']]
        else:
            out_diag, out_g = {}, {}
            issues = [f"WATCHER: report of {promoted} could not be loaded; its issues, fixes and "
                      f"metrics are not shown"] + [f"[{winner_name}] {i}" for i in g['issues']]
    if severity == 'ok' and worst_family and graded[worst_family]['severity'] != 'ok' \
            and settings['grade_scope'] == 'winner':
        action, consecutive, severity, why = 'PROCEED_WITH_NOTE', 0, 'warning', None
    else:
        action, consecutive, why = _severity_to_action(severity, model_type, policies)
    for m, v in per_family.items():
        if m != model_type and (promoted is None or m != winner_name):
            issues += [f"[{m}] {i}" for i in v.get('issues', [])]
    if fr['freshness'] != 'fresh':
        issues.append((f"[{winner_name}] " if promoted is not None else '')
                      + f"WATCHER: report freshness {fr['freshness']}: {fr['reason']}")
    _archive_diagnostics(out_diag, severity, issues, model_type)
    result = {'action': action, 'model_type': model_type, 'severity': severity, 'issues': issues,
              'suggested_fixes': list(out_diag.get('diagnosis', {}).get('suggested_fixes', [])),
              'confidence': _severity_to_confidence(severity), 'per_family': per_family,
              'metrics': out_g.get('metrics'), 'grade_scope': settings['grade_scope'], **base}
    if promoted is not None:
        result.update({'winner_model_type': winner_name, 'grade_set_by': promoted,
                       'grade_note': f"grade set by {promoted}, not by the winner {winner_name} "
                                     f"(grade_scope all_families)"})
        # [R1 F4] report_path, report_source and freshness are the grade-setting family's own; the
        # winner's are kept under winner_*. (The retry LLM still opens the canonical report itself:
        # agents/watcher_agent.py, not changed here.)
        result.update({'winner_report_path': base['report_path'],
                       'winner_report_source': base['report_source'],
                       'winner_freshness': base.get('freshness'),
                       'report_path': per_family[promoted]['path'],
                       'report_source': 'all_families_grade_setter_report',
                       'freshness': family_reports[promoted][2] if promoted in family_reports else None})
    _attach_skip_fields(result, action, consecutive, why)
    logger.info("Training health check (fixed): model=%s severity=%s action=%s source=%s",
                model_type, severity, action, report_source)
    return result


# =============================================================================
# CH14-DIAG WAVE 2: freshness (CD-07), skip expiry (CD-09), history prune (CD-10), log (CD-11)
# =============================================================================

def report_freshness(report: Dict[str, Any], report_path: Optional[str],
                     run_ctx: Optional[Dict[str, Any]] = None,
                     pointer_run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    CD-07. Is this report from the run being checked? Pure apart from one stat() of report_path.
      1. caller's expected_run_id: equal -> fresh; another id -> stale
      2. compare-models pointer run_id: equal -> fresh; another id, or no id at all while the
         pointer has one (written by a legacy-level run) -> stale
      3. caller's run_started_at: report's own run_started_at equal -> fresh, different -> stale;
         else the file's mtime before the start -> stale, after -> fresh
      4. nothing to compare -> unverified (graded as before)
    """
    run_ctx = run_ctx or {}
    rid = report.get('diagnostics_run_id')
    exp = run_ctx.get('expected_run_id')
    out = {'report_run_id': rid, 'expected_run_id': exp, 'pointer_run_id': pointer_run_id}
    if exp:
        if rid == exp:
            return {**out, 'freshness': 'fresh', 'reason': 'run_id matches the run being checked'}
        if rid:
            return {**out, 'freshness': 'stale', 'reason': f'report run_id {rid} != expected {exp}'}
    if pointer_run_id:
        if rid == pointer_run_id:
            return {**out, 'freshness': 'fresh', 'reason': 'run_id matches the compare-models pointer'}
        return {**out, 'freshness': 'stale',
                'reason': f'report run_id {rid!r} != compare-models pointer run_id {pointer_run_id}'}
    started = _parse_utc(run_ctx.get('run_started_at'))
    if started is not None:
        rs = report.get('diagnostics_run_started_at')
        if rs:
            same = _parse_utc(rs) == started
            return {**out, 'freshness': 'fresh' if same else 'stale',
                    'reason': f"report run_started_at {rs} {'==' if same else '!='} {run_ctx.get('run_started_at')}"}
        try:
            mtime = datetime.fromtimestamp(os.stat(report_path).st_mtime, tz=timezone.utc)
        except (OSError, TypeError):
            mtime = None
        if mtime is not None:
            old = mtime < started
            return {**out, 'freshness': 'stale' if old else 'fresh',
                    'reason': f"report file written {mtime.isoformat()} {'before' if old else 'after'} "
                              f"run start {started.isoformat()}"}
    return {**out, 'freshness': 'unverified', 'reason': 'no run id or start time to compare'}


def _stale_result(model_type: str, fr: Dict[str, Any], policies: Dict[str, Any],
                  base: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """CD-07: result for a stale report, or None when stale_report_action is 'grade'.
    Never touches the skip registry and never archives: a stale report is no new evidence."""
    action = _wave2_settings(policies)['stale_report_action']
    if action == 'grade':
        return None
    logger.warning("Training health check: stale report (%s) -> %s", fr['reason'], action)
    return {'action': action, 'model_type': model_type, 'severity': 'stale',
            'issues': [f"WATCHER: stale diagnostics report: {fr['reason']}"],
            'suggested_fixes': ['Check that Step 5 ran with --enable-diagnostics and wrote this run\'s report'],
            'confidence': _severity_to_confidence('absent'),
            'note': 'Report is not from the run being checked; not graded', **base, 'freshness': fr}


def skip_active(entry: Dict[str, Any], skip_rules: Dict[str, Any],
                now: Optional[datetime] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    CD-09 (B8, R-67). Is this skip-registry entry an active skip?
      until_reset (default, today's): consecutive_critical >= max_consecutive_critical
      hours: the same AND last_critical less than skip_duration_hours ago. An entry with no
             readable last_critical stays skipped (today's behaviour) and says so.
    [BENCHING] model_skip_rules.benching 'off' (default): never an active skip; at or over the
    limit info carries why_report_needed=True instead. 'on': the rules above.
    """
    max_consecutive = skip_rules.get('max_consecutive_critical', DEFAULT_SKIP_RULES['max_consecutive_critical'])
    mode = skip_rules.get('expiry_mode', DEFAULT_SKIP_EXPIRY_MODE)
    if mode not in SKIP_EXPIRY_MODES:
        raise ValueError(f"model_skip_rules.expiry_mode must be one of {SKIP_EXPIRY_MODES}, got {mode!r}")
    bench = benching_mode(skip_rules)
    consecutive = (entry or {}).get('consecutive_critical', 0)
    info: Dict[str, Any] = {'expiry_mode': mode, 'consecutive': consecutive, 'threshold': max_consecutive,
                            'benching': bench}
    if consecutive < max_consecutive:
        return False, info
    if bench == 'off':
        info['why_report_needed'] = True
        return False, info
    if mode == 'until_reset':
        return True, info
    hours = float(skip_rules.get('skip_duration_hours', DEFAULT_SKIP_RULES['skip_duration_hours']))
    last = _parse_utc((entry or {}).get('last_critical'))
    if last is None:
        info['expiry_note'] = 'no readable last_critical; skip kept'
        return True, info
    expires = last + timedelta(hours=hours)
    now = now or datetime.now(timezone.utc)
    info['expires_at'] = expires.isoformat()
    return now < expires, info


def _prune_health_history(policies: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """CD-10 (B10, R-69): cap the health-check archive. Off by default (today: never pruned)."""
    st = _wave2_settings(policies)
    if st['history_prune'] != 'on':
        return None
    import training_diagnostics as _td
    exclude = HISTORY_OTHER_WRITER_PREFIXES + (() if st['history_prune_include_root_cause']
                                               else (HISTORY_ROOT_CAUSE_PREFIX,))
    return _td._prune_history_fifo(
        DIAGNOSTICS_HISTORY_DIR, st['history_max_files'] or _td.MAX_HISTORY_FILES,
        pattern="*.json", mode=st['history_prune_mode'], archive_dir=st['history_archive_dir'],
        exclude_prefixes=exclude)


def _registry_count(model_type: Optional[str]) -> Optional[int]:
    try:
        with open(SKIP_REGISTRY_PATH) as f:
            return (json.load(f).get(model_type) or {}).get('consecutive_critical')
    except (OSError, ValueError, AttributeError):
        return None


def _write_training_health_log(result: Dict[str, Any], diagnostics_path: str,
                               run_ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """CD-11 (B9, R-68): the one writer of training_health_log.jsonl. Off by default. Never raises."""
    try:
        policies = _load_policies()
        if _wave2_settings(policies)['training_health_log'] != 'on':
            return None
        run_ctx = run_ctx or {}
        fr = result.get('freshness') or {}
        rec = {
            'schema': TRAINING_HEALTH_LOG_SCHEMA,
            'logged_at': datetime.now(timezone.utc).isoformat(),
            'action': result.get('action'),
            'severity': result.get('severity'),
            'model_type': result.get('model_type'),
            'grading_rules': result.get('grading_rules', 'legacy'),
            'report_source': result.get('report_source', 'canonical'),
            'report_path': result.get('report_path', diagnostics_path),
            'report_run_id': fr.get('report_run_id'),
            'expected_run_id': run_ctx.get('expected_run_id'),
            'run_started_at': run_ctx.get('run_started_at'),
            'compare_run_id': result.get('compare_run_id'),
            'freshness': fr.get('freshness'),
            # [R1 F4] promoted case only: the winner's report, kept apart from the grade setter's
            **({k: result.get(k) for k in ('grade_set_by', 'winner_model_type', 'winner_report_path',
                                            'winner_report_source')} if result.get('grade_set_by') else {}),
            **({'winner_freshness': (result.get('winner_freshness') or {}).get('freshness')}
               if result.get('grade_set_by') else {}),
            'consecutive_critical': result.get('consecutive_critical'),
            'registry_consecutive_critical': _registry_count(result.get('model_type')),
            'per_family': {k: v.get('severity') for k, v in (result.get('per_family') or {}).items()},
            'n_issues': len(result.get('issues') or []),
            'issues': list(result.get('issues') or [])[:10],
            'health_check_version': MODULE_VERSION,
            # settings in force for this check (run record of every grading switch)
            'settings': {**_grading_settings(policies), **_wave2_settings(policies)},
        }
        path = os.path.join(os.path.dirname(diagnostics_path) or '.', TRAINING_HEALTH_LOG_FILENAME)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        return path
    except Exception as e:  # the log never changes the check's outcome
        logger.warning(f"training health log write failed (non-fatal): {e}")
        return None


def _severity_to_action(
    severity: str,
    model_type: str,
    policies: Dict[str, Any]
) -> tuple:
    """
    Map severity to action, checking skip registry for consecutive failures.
    
    Returns:
        tuple: (action: str, consecutive_critical: int, why_report_flag: dict or None)
    """
    severity_map = policies.get('severity_thresholds', {})
    policy = severity_map.get(severity, {})
    base_action = policy.get('action', 'PROCEED')
    
    # Handle absent/unknown severity
    if severity in ('absent', 'unknown'):
        return 'PROCEED', 0, None
    
    if severity == 'ok':
        return 'PROCEED', 0, None
    
    if severity == 'warning':
        return 'PROCEED_WITH_NOTE', 0, None
    
    if severity == 'critical':
        # Check skip registry for consecutive failures
        return _check_skip_registry(model_type, policies)
    
    # Fallback
    return base_action, 0, None


def _check_skip_registry(model_type: str, policies: Dict[str, Any]) -> tuple:
    """
    Track consecutive critical failures per model type.
    If threshold exceeded, return SKIP_MODEL instead of RETRY.

    Skip registry file format:
    {
        "neural_net": {"consecutive_critical": 2, "last_critical": "2026-02-10T..."},
        "catboost": {"consecutive_critical": 0, "last_critical": null}
    }
    
    [BENCHING] benching 'off' (default): the count is kept, the action stays RETRY at and over the
    limit, and a why-report flag is raised (_raise_why_flag). 'on': SKIP_MODEL at the limit; with
    expiry_mode 'hours', a critical after the timed skip expired restarts the count first.

    Returns:
        tuple: (action: str, consecutive_critical: int, why_report_flag: dict or None)
    """
    skip_rules = policies.get('model_skip_rules', DEFAULT_SKIP_RULES)
    max_consecutive = skip_rules.get('max_consecutive_critical', 3)
    bench = benching_mode(skip_rules)

    registry = {}
    if os.path.isfile(SKIP_REGISTRY_PATH):
        try:
            with open(SKIP_REGISTRY_PATH) as f:
                registry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read skip registry: {e}")

    entry = registry.get(model_type, {'consecutive_critical': 0, 'last_critical': None})
    if bench == 'on' and entry.get('consecutive_critical', 0) >= max_consecutive \
            and not skip_active(entry, skip_rules)[0]:
        # hours mode: the timed skip has expired, so the count restarts (full retries again)
        logger.info("Model %s: timed skip expired; consecutive critical count restarts", model_type)
        entry['consecutive_critical'] = 0
        for k in _WHY_FLAG_ENTRY_KEYS:  # a new run of criticals: no stale flag identity (absent unless off ran)
            entry.pop(k, None)
    now_iso = datetime.now(timezone.utc).isoformat()
    if bench == 'off' and entry['consecutive_critical'] == 0:
        entry['critical_since'] = now_iso       # start of this run of criticals (for the flag)
        entry.pop('why_flag_first_at', None)
    entry['consecutive_critical'] += 1
    entry['last_critical'] = now_iso
    if bench == 'off' and entry['consecutive_critical'] - entry.get('why_flag_base', 0) > max_consecutive \
            and _why_flag_attached(why_flag_id(model_type, _why_flag_anchor(entry))):
        # [BENCHING R1 F3] the flag of this run of criticals has a why-report attached and the family is
        # still critical: a new segment starts here, so another max_consecutive criticals with no reset
        # raise a NEW flag (linked by previous_flag_id). A family that keeps failing never goes quiet.
        entry['why_flag_prev'] = why_flag_id(model_type, _why_flag_anchor(entry))
        entry['why_flag_base'] = entry['consecutive_critical'] - 1
        entry['critical_since'] = now_iso
        entry.pop('why_flag_first_at', None)
    if bench == 'off' and entry['consecutive_critical'] >= max_consecutive \
            and not entry.get('critical_since') and not entry.get('why_flag_first_at'):
        # [BENCHING review fix c] the count started before critical_since was recorded: the first
        # flagged_at of this run of criticals is kept so every repeat flag carries one flag_id
        entry['why_flag_first_at'] = now_iso

    registry[model_type] = entry

    # Write updated registry
    try:
        _write_skip_registry(registry)
    except (IOError, OSError) as e:
        logger.warning(f"Failed to write skip registry: {e}")

    consecutive = entry['consecutive_critical']
    
    if consecutive >= max_consecutive:
        if bench == 'off':
            if consecutive - entry.get('why_flag_base', 0) < max_consecutive:
                return 'RETRY', consecutive, None   # [R1 F3] after an attached why-report, not yet again
            return 'RETRY', consecutive, _raise_why_flag(model_type, entry, max_consecutive)
        logger.warning(
            "Model %s hit %d consecutive critical failures — SKIP_MODEL",
            model_type, consecutive
        )
        return 'SKIP_MODEL', consecutive, None

    return 'RETRY', consecutive, None


def _attach_skip_fields(result: Dict[str, Any], action: str, consecutive: int,
                        why: Optional[Dict[str, Any]]) -> None:
    """consecutive_critical on SKIP_MODEL (as before); with a why-report flag, the flag as well."""
    if action == 'SKIP_MODEL' or why:
        result['consecutive_critical'] = consecutive
    if why:
        result['why_report_needed'] = why


_WHY_FLAG_ENTRY_KEYS = ('critical_since', 'why_flag_first_at', 'why_flag_base', 'why_flag_prev')
WHY_REPORT_ATTACH_PATH = "diagnostics_outputs/why_report_attachments.jsonl"  # == why_report_flags.ATTACH_PATH


def _write_skip_registry(registry: Dict[str, Any]) -> None:
    """[BENCHING R1 F1] The one writer of SKIP_REGISTRY_PATH: a temp file in the same directory, then
    os.replace, so an interrupted write leaves the previous registry intact (never truncated JSON)."""
    d = os.path.dirname(SKIP_REGISTRY_PATH) or '.'
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(SKIP_REGISTRY_PATH)}.{os.getpid()}.tmp")
    try:
        with open(tmp, 'w') as f:
            json.dump(registry, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SKIP_REGISTRY_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _why_flag_anchor(rec: Dict[str, Any]) -> Optional[str]:
    """The anchor of a flag_id: critical_since, else why_flag_first_at. The same chain as
    why_report_flags.flag_id_of (which falls back to flagged_at for lines older than both)."""
    return rec.get('critical_since') or rec.get('why_flag_first_at')


def _why_flag_attached(flag_id: str) -> bool:
    """[BENCHING R1 F3] Does flag_id have a why-report attached (why_report_flags attach record)?
    Unreadable lines are skipped; a missing or unreadable file means no attachment."""
    try:
        with open(WHY_REPORT_ATTACH_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get('flag_id') == flag_id:
                    return True
    except OSError:
        pass
    return False


def why_flag_id(model_type: str, anchor: str) -> str:
    """[BENCHING review fix c] Stable id of one run of critical grades of one family: the family plus
    critical_since, or the first flagged_at when critical_since is not known. Repeats share it."""
    return f"{model_type}@{anchor}"


def _raise_why_flag(model_type: str, entry: Dict[str, Any], threshold: int) -> Dict[str, Any]:
    """
    [BENCHING off] The family stays in the comparison; a why-report is needed. Logged at warning level
    and appended as one JSON line to WHY_REPORT_FLAGS_PATH (append-only, one writer, like the CD-11
    training health log). Raises the flag only: the automatic why-report is TODO-WHY-1. Never raises.
    """
    flagged_at = datetime.now(timezone.utc).isoformat()
    since = entry.get('critical_since')
    flag = {
        'schema': WHY_REPORT_FLAG_SCHEMA,
        'flag_id': why_flag_id(model_type, _why_flag_anchor(entry) or flagged_at),
        'flagged_at': flagged_at,
        # [R1 F1] the anchor is recorded, so why_report_flags.flag_id_of derives the same id from the line
        'why_flag_first_at': entry.get('why_flag_first_at'),
        # [R1 F3] the flag this one follows (its why-report attached, the family still critical), or None
        'previous_flag_id': entry.get('why_flag_prev'),
        'model_type': model_type,
        'consecutive_critical': entry.get('consecutive_critical'),
        'threshold': threshold,
        # [BENCHING review fix b] None when the count started before critical_since was recorded
        'critical_since': since,
        'critical_since_known': since is not None,
        'last_critical': entry.get('last_critical'),
        'benching': 'off',
        'flag_file': WHY_REPORT_FLAGS_PATH,
        'note': 'why-report needed: family kept in the comparison (owner ruling 2026-11-24); '
                'a human or AI checks it (automatic why-report not built, TODO-WHY-1)',
        'health_check_version': MODULE_VERSION,
    }
    logger.warning(
        "[WHY-REPORT] Model %s hit %d consecutive critical grades (limit %d) since %s; benching off, "
        "family kept in the comparison; why-report needed (flag appended to %s)",
        model_type, flag['consecutive_critical'], threshold, flag['critical_since'], WHY_REPORT_FLAGS_PATH)
    try:
        os.makedirs(os.path.dirname(WHY_REPORT_FLAGS_PATH) or '.', exist_ok=True)
        with open(WHY_REPORT_FLAGS_PATH, 'a') as f:
            f.write(json.dumps(flag, sort_keys=True, default=str) + "\n")
    except OSError as e:  # the flag file never changes the check's outcome
        logger.warning(f"why-report flag write failed (non-fatal): {e}")
        flag['flag_file_error'] = str(e)
    return flag


def reset_skip_registry(model_type: str) -> None:
    """
    Reset consecutive critical count for a model type.
    Called when a model type succeeds (severity != critical).
    """
    if not os.path.isfile(SKIP_REGISTRY_PATH):
        return

    try:
        with open(SKIP_REGISTRY_PATH) as f:
            registry = json.load(f)

        if model_type in registry:
            registry[model_type]['consecutive_critical'] = 0
            for k in _WHY_FLAG_ENTRY_KEYS:  # [BENCHING] the run of criticals ended; absent unless off ran
                registry[model_type].pop(k, None)
            _write_skip_registry(registry)   # [R1 F1] temp file + os.replace
            logger.info("Reset skip registry for %s", model_type)
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.warning(f"Failed to reset skip registry: {e}")


def _archive_diagnostics(
    diag: Dict[str, Any],
    severity: str,
    issues: List[str],
    model_type: str
) -> None:
    """
    Archive each diagnostics run to history/ for Strategy Advisor consumption.
    One JSON per run, timestamped filename.
    """
    try:
        os.makedirs(DIAGNOSTICS_HISTORY_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_{timestamp}.json"

        archive = {
            'diagnostics': diag,
            'watcher_severity': severity,
            'watcher_issues': issues,
            'archived_at': datetime.now(timezone.utc).isoformat(),
            'health_check_version': MODULE_VERSION,
        }

        filepath = os.path.join(DIAGNOSTICS_HISTORY_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(archive, f, indent=2)
        
        logger.debug(f"Archived diagnostics to {filepath}")
        
    except IOError as e:
        logger.warning(f"Failed to archive diagnostics (non-fatal): {e}")
    try:  # [CH14-DIAG CD-10] cap the archive (setting history_prune, default off)
        _prune_health_history(_load_policies())
    except Exception as e:
        logger.warning(f"History prune failed (non-fatal): {e}")


def _severity_to_confidence(severity: str) -> float:
    """Map severity to WATCHER confidence for proceed decision."""
    return {
        'ok': 0.90,
        'warning': 0.65,
        'critical': 0.30,
        'absent': 0.50,
    }.get(severity, 0.5)


def _load_policies() -> Dict[str, Any]:
    """Load training_diagnostics policies from watcher_policies.json."""
    policies = {}
    
    if os.path.isfile(WATCHER_POLICIES_PATH):
        try:
            with open(WATCHER_POLICIES_PATH) as f:
                all_policies = json.load(f)
            policies = all_policies.get('training_diagnostics', {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load watcher_policies.json: {e}")
    
    # Apply defaults for missing sections
    if 'metric_bounds' not in policies:
        policies['metric_bounds'] = DEFAULT_METRIC_BOUNDS
    if 'model_skip_rules' not in policies:
        policies['model_skip_rules'] = DEFAULT_SKIP_RULES
    if 'severity_thresholds' not in policies:
        policies['severity_thresholds'] = {
            'ok': {'action': 'PROCEED', 'log_level': 'info'},
            'warning': {'action': 'PROCEED_WITH_NOTE', 'log_level': 'warning'},
            'critical': {'action': 'RETRY_OR_SKIP', 'log_level': 'error', 'max_retries': 2},
        }
    
    return policies


# =============================================================================
# WATCHER INTEGRATION HELPERS
# =============================================================================

def get_retry_params_suggestions(health: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build modified params for Step 5 retry based on diagnostics.
    
    IMPORTANT: This function is for SINGLE-MODEL retry scenarios only.
    
    When using --compare-models mode:
    - All 4 models train regardless of individual failures
    - The winner (lowest MSE) determines pipeline progression
    - If winner is healthy, pipeline proceeds even if other models failed
    - Failed models are logged for Strategy Advisor but don't trigger RETRY
    
    This function is only invoked when:
    1. Running single-model mode (--model-type X) AND that model hits CRITICAL
    2. OR when --compare-models winner itself hits CRITICAL (rare)
    
    The suggestions here help WATCHER decide what to try on a retry attempt,
    NOT to override the --compare-models multi-model behavior.
    
    Args:
        health: Result from check_training_health()
        
    Returns:
        dict: Suggested parameter modifications for retry
    """
    suggestions = {}
    
    model_type = health.get('model_type', 'unknown')
    issues = health.get('issues', [])
    
    # If NN failed, suggest switching to tree model
    if model_type == 'neural_net':
        suggestions['model_type'] = 'catboost'
        suggestions['reason'] = 'Neural net had critical issues — switching to CatBoost'
    
    # If feature scaling issue, suggest normalization
    if any('scaling' in i.lower() or 'spread' in i.lower() or 'gradient' in i.lower()
           for i in issues):
        suggestions['normalize_features'] = True
        suggestions['feature_scaling_reason'] = 'Gradient/scaling issues detected'
    
    # If overfitting, suggest stronger regularization
    if any('overfit' in i.lower() for i in issues):
        suggestions['increase_regularization'] = True
        suggestions['regularization_reason'] = 'Overfitting detected'
    
    # If dead neurons, suggest LeakyReLU
    if any('dead neuron' in i.lower() for i in issues):
        suggestions['use_leaky_relu'] = True
        suggestions['activation_reason'] = 'Dead ReLU neurons detected'
    
    return suggestions


def is_model_skipped(model_type: str) -> bool:
    """
    Check if a model type is currently in skip state.
    
    Args:
        model_type: Model type to check
        
    Returns:
        bool: True if model should be skipped
    """
    if not os.path.isfile(SKIP_REGISTRY_PATH):
        return False
    
    try:
        with open(SKIP_REGISTRY_PATH) as f:
            registry = json.load(f)
        
        policies = _load_policies()
        skip_rules = policies.get('model_skip_rules', DEFAULT_SKIP_RULES)
        
        entry = registry.get(model_type, {})
        return skip_active(entry, skip_rules)[0]  # [CH14-DIAG CD-09] expiry_mode
        
    except (json.JSONDecodeError, IOError):
        return False


def get_skip_status() -> Dict[str, Any]:
    """
    Get current skip status for all model types.
    
    Returns:
        dict: {model_type: {'skipped': bool, 'consecutive': int, 'last': str}}
    """
    if not os.path.isfile(SKIP_REGISTRY_PATH):
        return {}
    
    try:
        with open(SKIP_REGISTRY_PATH) as f:
            registry = json.load(f)
        
        policies = _load_policies()
        skip_rules = policies.get('model_skip_rules', DEFAULT_SKIP_RULES)
        max_consecutive = skip_rules.get('max_consecutive_critical', 3)
        
        status = {}
        for model_type, entry in registry.items():
            consecutive = entry.get('consecutive_critical', 0)
            active, info = skip_active(entry, skip_rules)  # [CH14-DIAG CD-09]
            status[model_type] = {
                'skipped': active,
                'consecutive': consecutive,
                'last_critical': entry.get('last_critical'),
                'threshold': max_consecutive,
            }
            if info['expiry_mode'] != DEFAULT_SKIP_EXPIRY_MODE:
                status[model_type].update({k: v for k, v in info.items()
                                           if k in ('expiry_mode', 'expires_at', 'expiry_note')})
            if info['benching'] != 'on':   # [BENCHING] off: never skipped; say when a why-report is due
                status[model_type].update({'benching': info['benching'],
                                           'why_report_needed': bool(info.get('why_report_needed'))})
        
        return status
        
    except (json.JSONDecodeError, IOError):
        return {}


# =============================================================================
# CD-00 OFFLINE RE-GRADE (read-only)
# =============================================================================

REGRADE_SCHEMA = "ch14_regrade/1.0"
DEFAULT_REGRADE_MATCH_WINDOW_HOURS = 6.0


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _classify_json(obj) -> tuple:
    """(kind, report, recorded_grade, archived_at). kind: history_archive|report|compare_summary|other."""
    if isinstance(obj, dict):
        if 'diagnostics' in obj and 'watcher_severity' in obj:
            return 'history_archive', obj['diagnostics'], obj['watcher_severity'], obj.get('archived_at')
        if 'models' in obj and isinstance(obj.get('models'), dict) and 'comparison' in obj:
            return 'report', obj, None, None
        if 'training_summary' in obj and 'diagnosis' in obj:
            return 'report', obj, None, None
        if 'winner_best_effort' in obj or str(obj.get('run_id', '')).startswith('S88_'):
            return 'compare_summary', None, None, None
    return 'other', None, None, None


def _parse_utc(ts: Optional[str]):
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _critical_reasons(report: Dict[str, Any], graded: Dict[str, Any]) -> List[str]:
    if graded['severity'] != 'critical':
        return []
    reasons = []
    if (report.get('diagnosis') or {}).get('severity') == 'critical':
        reasons.append('producer_diagnosis_critical')
    for i in graded['issues']:
        if not i.startswith('WATCHER:'):
            continue
        if 'Early stop ratio' in i and 'severe' in i:
            reasons.append('early_stop_ratio')
        elif 'Overfit ratio' in i and 'critical' in i:
            reasons.append('overfit_ratio')
        elif 'gradient spread' in i and 'critical' in i:
            reasons.append('gradient_spread')
        elif 'dead neurons' in i and 'critical' in i:
            reasons.append('dead_neurons')
    return sorted(set(reasons))


def regrade(paths: List[str], policies: Dict[str, Any], out_path: Optional[str] = None,
            match_window_hours: float = DEFAULT_REGRADE_MATCH_WINDOW_HOURS) -> Dict[str, Any]:
    """
    CD-00: re-grade every diagnostics report found under `paths` (files or directories, recursive)
    with the legacy rules (today's) and the fixed rules, side by side. READ-ONLY: inputs are hashed
    before and after and the call fails if any changed; the skip registry and history are never
    touched (grade_report is pure). `out_path` (JSON) must not exist and must not be under `paths`.

    Columns: recorded (the grade WATCHER wrote into a history archive, when there is one),
    legacy (today's rules recomputed), fixed (policy settings, grading_rules forced 'fixed'),
    fixed_base1 (sensitivity: early_stop_round_base = 1).
    """
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, fns in os.walk(p):
                files += [os.path.join(root, fn) for fn in fns if fn.endswith('.json')]
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    files = sorted(set(os.path.abspath(f) for f in files))
    if out_path:
        out_abs = os.path.abspath(out_path)
        if os.path.exists(out_abs):
            raise FileExistsError(f"refusing to overwrite {out_abs}")
        for p in paths:
            if out_abs.startswith(os.path.abspath(p).rstrip(os.sep) + os.sep):
                raise ValueError(f"out_path {out_abs} is inside an input path {p}")
    before = {f: _sha256_file(f) for f in files}

    pol_fixed = dict(policies, grading_rules='fixed')
    pol_base1 = dict(pol_fixed, early_stop_round_base=1)
    settings = _grading_settings(pol_fixed)

    summaries = []
    loaded = {}
    for f in files:
        try:
            with open(f) as fh:
                loaded[f] = json.load(fh)
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            loaded[f] = e
    for f, obj in loaded.items():
        if isinstance(obj, Exception):
            continue
        kind, _, _, _ = _classify_json(obj)
        if kind == 'compare_summary':
            rid = str(obj.get('run_id', ''))
            try:
                t = datetime.strptime(rid.replace('S88_', ''), '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
            except ValueError:
                t = None
            summaries.append({'path': f, 'run_id': rid, 'time': t, 'winner': _winner_from_summary(obj),
                              'models': sorted((obj.get('models') or {}).keys())})
    summaries = [x for x in summaries if x['time'] is not None]
    summaries.sort(key=lambda x: x['time'])

    rows, kinds = [], {}
    for f, obj in loaded.items():
        if isinstance(obj, Exception):
            kinds['unreadable'] = kinds.get('unreadable', 0) + 1
            rows.append({'path': f, 'kind': 'unreadable', 'error': str(obj)})
            continue
        kind, report, recorded, archived_at = _classify_json(obj)
        kinds[kind] = kinds.get(kind, 0) + 1
        if report is None:
            continue
        g_legacy = grade_report(report, policies, 'legacy')
        g_fixed = grade_report(report, pol_fixed, 'fixed')
        g_base1 = grade_report(report, pol_base1, 'fixed')
        row = {
            'path': f, 'kind': kind, 'sha256': before[f],
            'model_type': g_fixed['model_type'],
            'schema_version': report.get('schema_version'),
            'generated_at': report.get('generated_at'),
            'archived_at': archived_at,
            'producer_severity': (report.get('diagnosis') or {}).get('severity'),
            'recorded': recorded,
            'legacy': g_legacy['severity'],
            'fixed': g_fixed['severity'],
            'fixed_base1': g_base1['severity'],
            'fixed_critical_reasons': _critical_reasons(report, g_fixed),
            'fixed_issues': [i for i in g_fixed['issues'] if i.startswith('WATCHER:')],
            'metrics': g_fixed.get('metrics'),
        }
        at = _parse_utc(archived_at)
        if at is not None and summaries:
            prior = [x for x in summaries if x['time'] <= at
                     and (at - x['time']).total_seconds() <= match_window_hours * 3600]
            if prior:
                w = prior[-1]
                row['compare_run_by_time'] = w['run_id']
                row['winner_by_time'] = w['winner']
                row['graded_model_is_winner'] = (w['winner'] == g_fixed['model_type'])
        rows.append(row)

    after = {f: _sha256_file(f) for f in files}
    changed = [f for f in files if after.get(f) != before[f]]
    if changed:
        raise RuntimeError(f"input files changed during re-grade: {changed[:5]}")

    graded = [r for r in rows if 'legacy' in r]
    def count(key):
        c = {}
        for r in graded:
            c[r[key]] = c.get(r[key], 0) + 1
        return c
    transitions = {}
    for r in graded:
        k = f"{r['legacy']}->{r['fixed']}"
        transitions[k] = transitions.get(k, 0) + 1
    reasons = {}
    for r in graded:
        for x in r['fixed_critical_reasons']:
            reasons[x] = reasons.get(x, 0) + 1
    with_rec = [r for r in graded if r.get('recorded') is not None]
    matched = [r for r in graded if 'winner_by_time' in r]
    out = {
        'schema': REGRADE_SCHEMA,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'inputs': [os.path.abspath(p) for p in paths],
        'read_only_check': {'files_hashed': len(files), 'changed': 0},
        'settings_fixed': settings,
        'metric_bounds': policies.get('metric_bounds', DEFAULT_METRIC_BOUNDS),
        'files_by_kind': kinds,
        'graded_reports': len(graded),
        'unique_report_sha256': len({r['sha256'] for r in graded}),
        'recorded_grades': len(with_rec),
        'recorded_equals_legacy_recomputed': sum(1 for r in with_rec if r['recorded'] == r['legacy']),
        'legacy': count('legacy'), 'fixed': count('fixed'), 'fixed_base1': count('fixed_base1'),
        'legacy_to_fixed': transitions,
        'fixed_critical_reasons': reasons,
        'compare_summaries_found': len(summaries),
        'time_matched_to_compare_run': len(matched),
        'time_matched_graded_model_not_winner': sum(1 for r in matched if not r['graded_model_is_winner']),
        'match_window_hours': match_window_hours,
        'rows': rows,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'x') as f:
            json.dump(out, f, indent=2, default=str)
    return out


def _print_regrade(out: Dict[str, Any]) -> None:
    print(f"\n=== CH14 offline re-grade ({out['graded_reports']} reports, "
          f"{out['unique_report_sha256']} unique; {out['read_only_check']['files_hashed']} files hashed, "
          f"{out['read_only_check']['changed']} changed) ===")
    print(f"files by kind: {out['files_by_kind']}")
    print(f"recorded grade present: {out['recorded_grades']}; recorded == legacy recomputed: "
          f"{out['recorded_equals_legacy_recomputed']}")
    for k in ('legacy', 'fixed', 'fixed_base1'):
        print(f"{k:12s} {out[k]}")
    print(f"legacy->fixed {out['legacy_to_fixed']}")
    print(f"fixed critical reasons (a report may have several): {out['fixed_critical_reasons']}")
    print(f"time-matched to a compare-models run: {out['time_matched_to_compare_run']}; "
          f"graded model was not that run's winner: {out['time_matched_graded_model_not_winner']}")


# =============================================================================
# CLI INTERFACE (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Health Check CLI")
    parser.add_argument("--check", action="store_true", help="Run health check")
    parser.add_argument("--diagnostics-path", type=str, default=DIAGNOSTICS_PATH,
                        help="Path to diagnostics JSON")
    parser.add_argument("--status", action="store_true", help="Show skip registry status")
    parser.add_argument("--reset", type=str, help="Reset skip registry for model type")
    parser.add_argument("--test", action="store_true", help="Run with test diagnostics")
    # [CH14-DIAG CD-00] offline, read-only re-grade
    parser.add_argument("--regrade", nargs="+", metavar="PATH",
                        help="Re-grade every report under these files/dirs (read-only), legacy vs fixed")
    parser.add_argument("--out", type=str, default=None,
                        help="--regrade: JSON output path (must not exist, not under an input)")
    parser.add_argument("--policies", type=str, default=WATCHER_POLICIES_PATH,
                        help="--regrade: watcher_policies.json to read bounds from (read-only)")
    parser.add_argument("--match-window-hours", type=float, default=DEFAULT_REGRADE_MATCH_WINDOW_HOURS,
                        help="--regrade: max hours between a compare-models run start and an archived check")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.regrade:
        with open(args.policies) as _pf:
            _pol = json.load(_pf).get('training_diagnostics', {})
        _pol.setdefault('metric_bounds', DEFAULT_METRIC_BOUNDS)
        _pol.setdefault('model_skip_rules', DEFAULT_SKIP_RULES)
        _res = regrade(args.regrade, _pol, args.out, args.match_window_hours)
        _print_regrade(_res)
        if args.out:
            print(f"written: {args.out}")

    elif args.check:
        health = check_training_health(args.diagnostics_path)
        print(f"\n=== Training Health Check ===")
        print(f"Action:     {health['action']}")
        print(f"Model:      {health['model_type']}")
        print(f"Severity:   {health['severity']}")
        print(f"Confidence: {health['confidence']:.2f}")
        if health.get('issues'):
            print(f"Issues ({len(health['issues'])}):")
            for issue in health['issues'][:5]:
                print(f"  - {issue}")
        if health.get('suggested_fixes'):
            print(f"Suggested fixes:")
            for fix in health['suggested_fixes'][:3]:
                print(f"  - {fix}")
        if health.get('note'):
            print(f"Note: {health['note']}")
    
    elif args.status:
        status = get_skip_status()
        print("\n=== Skip Registry Status ===")
        if not status:
            print("No models in skip registry")
        else:
            for model, info in status.items():
                skip_str = "SKIPPED" if info['skipped'] else "active"
                if info.get('why_report_needed'):   # [BENCHING] off
                    skip_str = "active, WHY-REPORT NEEDED (benching off)"
                print(f"{model}: {skip_str} ({info['consecutive']}/{info['threshold']} critical)")
    
    elif args.reset:
        reset_skip_registry(args.reset)
        print(f"Reset skip registry for {args.reset}")
    
    elif args.test:
        print("\n=== Testing with mock diagnostics ===")
        
        # Create test diagnostics
        test_diag = {
            'model_type': 'neural_net',
            'status': 'complete',
            'training_summary': {
                'rounds_captured': 100,
                'best_round': 15,
                'final_train_loss': 0.01,
                'final_val_loss': 0.02,
            },
            'diagnosis': {
                'severity': 'warning',
                'issues': ['Possible overfitting detected'],
                'suggested_fixes': ['Try early stopping'],
            },
            'model_specific': {
                'gradient_health': {'vanishing': False, 'exploding': False},
                'layer_health': {
                    'layer_0': {'dead_pct': 5.0, 'gradient_norm': 1.2},
                    'layer_2': {'dead_pct': 12.0, 'gradient_norm': 0.8},
                },
            },
        }
        
        # Write test file
        os.makedirs('diagnostics_outputs', exist_ok=True)
        test_path = 'diagnostics_outputs/test_diagnostics.json'
        with open(test_path, 'w') as f:
            json.dump(test_diag, f, indent=2)
        
        print(f"Created test diagnostics at {test_path}")
        
        # Run check
        health = check_training_health(test_path)
        print(f"\nResult:")
        print(f"  Action: {health['action']}")
        print(f"  Severity: {health['severity']}")
        print(f"  Issues: {len(health['issues'])}")
        
        # Cleanup
        os.remove(test_path)
        print("\nTest complete — cleaned up test file")
    
    else:
        parser.print_help()
