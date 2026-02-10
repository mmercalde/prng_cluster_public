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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def check_training_health(diagnostics_path: str = DIAGNOSTICS_PATH) -> Dict[str, Any]:
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
    """
    try:
        return _check_training_health_impl(diagnostics_path)
    except Exception as e:
        # Best-effort — if this module fails, pipeline continues
        logger.error(f"Training health check failed (non-fatal): {e}")
        return {
            'action': 'PROCEED_WITH_NOTE',
            'model_type': 'unknown',
            'severity': 'absent',
            'issues': [f'Health check module error: {e}'],
            'suggested_fixes': ['Check training_health_check.py logs'],
            'confidence': 0.5,
            'note': 'Health check failed — proceeding anyway (best-effort)',
        }


def _check_training_health_impl(diagnostics_path: str) -> Dict[str, Any]:
    """Internal implementation — separated for clean exception handling."""
    
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
    
    # ── Determine if multi-model or single-model format ───────────────
    if 'models' in diag:
        # Multi-model format (from --compare-models)
        return _evaluate_multi_model(diag, policies, metric_bounds)
    else:
        # Single-model format
        return _evaluate_single_model(diag, policies, metric_bounds)


def _evaluate_single_model(
    diag: Dict[str, Any],
    policies: Dict[str, Any],
    metric_bounds: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate single-model diagnostics."""
    
    model_type = diag.get('model_type', 'unknown')
    diagnosis = diag.get('diagnosis', {})
    severity = diagnosis.get('severity', 'ok')
    issues = list(diagnosis.get('issues', []))
    fixes = list(diagnosis.get('suggested_fixes', []))
    
    # ── Evaluate against WATCHER's own metric bounds ──────────────────
    # This cross-checks diagnostics severity against WATCHER thresholds
    # to prevent a buggy diagnostics module from under-reporting
    watcher_issues, severity = _evaluate_metrics(
        diag, metric_bounds, severity
    )
    
    # Combine diagnostics issues with WATCHER's own findings
    all_issues = issues + watcher_issues
    
    # ── Map severity to action ────────────────────────────────────────
    action, consecutive = _severity_to_action(severity, model_type, policies)
    
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
    
    if action == 'SKIP_MODEL':
        result['consecutive_critical'] = consecutive
    
    logger.info(
        "Training health check: model=%s severity=%s action=%s issues=%d",
        model_type, severity, action, len(all_issues)
    )
    
    return result


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
    models = diag.get('models', {})
    comparison = diag.get('comparison', {})
    winner = comparison.get('winner', 'unknown')
    
    # Track per-model evaluations
    model_results = {}
    worst_severity = 'ok'
    severity_order = {'ok': 0, 'warning': 1, 'critical': 2}
    
    for model_type, model_report in models.items():
        diagnosis = model_report.get('diagnosis', {})
        model_severity = diagnosis.get('severity', 'ok')
        model_issues = list(diagnosis.get('issues', []))
        
        # Evaluate against WATCHER bounds
        watcher_issues, model_severity = _evaluate_metrics(
            model_report, metric_bounds, model_severity, model_type
        )
        
        model_results[model_type] = {
            'severity': model_severity,
            'issues': model_issues + watcher_issues,
        }
        
        # Track worst severity
        if severity_order.get(model_severity, 0) > severity_order.get(worst_severity, 0):
            worst_severity = model_severity
    
    # Combine all issues
    all_issues = []
    all_fixes = []
    
    for model_type, result in model_results.items():
        for issue in result['issues']:
            all_issues.append(f"[{model_type}] {issue}")
    
    # Get fixes from winner model (most relevant)
    if winner in models:
        all_fixes = list(models[winner].get('diagnosis', {}).get('suggested_fixes', []))
    
    # ── For multi-model, we care most about the winner ────────────────
    # If winner is healthy, proceed even if other models had issues
    winner_result = model_results.get(winner, {})
    winner_severity = winner_result.get('severity', 'ok')
    
    if winner_severity == 'ok' and worst_severity != 'ok':
        # Winner is fine, other models had issues — proceed with note
        logger.info(
            "Winner model %s is healthy; other models had issues (worst=%s)",
            winner, worst_severity
        )
        action = 'PROCEED_WITH_NOTE'
        severity = 'warning'
    else:
        action, consecutive = _severity_to_action(winner_severity, winner, policies)
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
    
    if action == 'SKIP_MODEL':
        result['consecutive_critical'] = consecutive
    
    logger.info(
        "Training health check (multi-model): winner=%s severity=%s action=%s",
        winner, severity, action
    )
    
    return result


def _evaluate_metrics(
    diag: Dict[str, Any],
    metric_bounds: Dict[str, Any],
    current_severity: str,
    model_type: str = None
) -> tuple:
    """
    Evaluate diagnostics against WATCHER metric bounds.
    
    Returns:
        tuple: (watcher_issues: list, updated_severity: str)
    """
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


def _severity_to_action(
    severity: str,
    model_type: str,
    policies: Dict[str, Any]
) -> tuple:
    """
    Map severity to action, checking skip registry for consecutive failures.
    
    Returns:
        tuple: (action: str, consecutive_critical: int)
    """
    severity_map = policies.get('severity_thresholds', {})
    policy = severity_map.get(severity, {})
    base_action = policy.get('action', 'PROCEED')
    
    # Handle absent/unknown severity
    if severity in ('absent', 'unknown'):
        return 'PROCEED', 0
    
    if severity == 'ok':
        return 'PROCEED', 0
    
    if severity == 'warning':
        return 'PROCEED_WITH_NOTE', 0
    
    if severity == 'critical':
        # Check skip registry for consecutive failures
        action, consecutive = _check_skip_registry(model_type, policies)
        return action, consecutive
    
    # Fallback
    return base_action, 0


def _check_skip_registry(model_type: str, policies: Dict[str, Any]) -> tuple:
    """
    Track consecutive critical failures per model type.
    If threshold exceeded, return SKIP_MODEL instead of RETRY.

    Skip registry file format:
    {
        "neural_net": {"consecutive_critical": 2, "last_critical": "2026-02-10T..."},
        "catboost": {"consecutive_critical": 0, "last_critical": null}
    }
    
    Returns:
        tuple: (action: str, consecutive_critical: int)
    """
    skip_rules = policies.get('model_skip_rules', DEFAULT_SKIP_RULES)
    max_consecutive = skip_rules.get('max_consecutive_critical', 3)

    registry = {}
    if os.path.isfile(SKIP_REGISTRY_PATH):
        try:
            with open(SKIP_REGISTRY_PATH) as f:
                registry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read skip registry: {e}")

    entry = registry.get(model_type, {'consecutive_critical': 0, 'last_critical': None})
    entry['consecutive_critical'] += 1
    entry['last_critical'] = datetime.now(timezone.utc).isoformat()

    registry[model_type] = entry

    # Write updated registry
    try:
        os.makedirs(os.path.dirname(SKIP_REGISTRY_PATH), exist_ok=True)
        with open(SKIP_REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to write skip registry: {e}")

    consecutive = entry['consecutive_critical']
    
    if consecutive >= max_consecutive:
        logger.warning(
            "Model %s hit %d consecutive critical failures — SKIP_MODEL",
            model_type, consecutive
        )
        return 'SKIP_MODEL', consecutive

    return 'RETRY', consecutive


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
            with open(SKIP_REGISTRY_PATH, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info("Reset skip registry for %s", model_type)
    except (json.JSONDecodeError, IOError) as e:
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
        max_consecutive = skip_rules.get('max_consecutive_critical', 3)
        
        entry = registry.get(model_type, {})
        return entry.get('consecutive_critical', 0) >= max_consecutive
        
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
            status[model_type] = {
                'skipped': consecutive >= max_consecutive,
                'consecutive': consecutive,
                'last_critical': entry.get('last_critical'),
                'threshold': max_consecutive,
            }
        
        return status
        
    except (json.JSONDecodeError, IOError):
        return {}


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
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.check:
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
