"""
WATCHER Pipeline Integration — Chapter 14 Phase 6
==================================================

This file contains the code snippets to add to agents/watcher_agent.py
to integrate training health checks between Step 5 and Step 6.

Integration Points:
1. Import the health check module
2. Add to run_pipeline() after Step 5 evaluation
3. Add helper methods for retry params and incident recording

Version: 1.0.0
Session: 72 - February 8, 2026
"""

# =============================================================================
# 1. IMPORTS — Add to top of watcher_agent.py
# =============================================================================

# Add these imports near the top of the file:
"""
from training_health_check import (
    check_training_health,
    reset_skip_registry,
    get_retry_params_suggestions,
    is_model_skipped,
)
"""


# =============================================================================
# 2. PIPELINE INTEGRATION — Add to run_pipeline() after Step 5
# =============================================================================

def _step_5_with_health_check(self, step, params, evaluation):
    """
    Post-Step-5 training health check integration.
    
    Add this logic after the existing step 5 evaluation in run_pipeline().
    
    Returns:
        bool: True if should continue to Step 6, False if pipeline should stop
    """
    if evaluation['action'] != 'PROCEED':
        # Step 5 itself failed — don't run health check
        return False
    
    # ── NEW: Post-Step-5 diagnostics check ────────────────────────────
    health = check_training_health()
    
    if health['action'] == 'PROCEED':
        # Reset skip registry — model is healthy
        reset_skip_registry(health['model_type'])
        logger.info("Training health OK — proceeding to Step 6")
        return True
    
    elif health['action'] == 'PROCEED_WITH_NOTE':
        # Log warning for Strategy Advisor, continue to Step 6
        reset_skip_registry(health['model_type'])
        logger.warning(
            "Training health WARNING for %s: %s",
            health['model_type'],
            "; ".join(health['issues'][:3])
        )
        # Record incident for LLM bundle
        self._record_training_incident(health)
        return True
    
    elif health['action'] == 'RETRY':
        logger.warning(
            "Training health CRITICAL for %s — retrying Step 5",
            health['model_type']
        )
        
        # Get retry parameter suggestions
        retry_suggestions = get_retry_params_suggestions(health)
        retry_params = self._build_retry_params(params, retry_suggestions)
        
        # Retry Step 5
        result = self.dispatch_step(5, retry_params)
        evaluation = self.evaluate_step(5, result)
        
        if evaluation['action'] != 'PROCEED':
            self.escalate(f"Step 5 retry failed after health check CRITICAL")
            return False
        
        # Re-check health after retry
        health_after = check_training_health()
        if health_after['severity'] == 'critical':
            self.escalate(f"Step 5 still CRITICAL after retry: {health_after['issues']}")
            return False
        
        reset_skip_registry(health['model_type'])
        return True
    
    elif health['action'] == 'SKIP_MODEL':
        logger.error(
            "Model %s skipped — %d consecutive critical failures",
            health['model_type'],
            health.get('consecutive_critical', 0)
        )
        
        # Record the skip
        self._record_model_skip(health['model_type'], health)
        
        # Still proceed to Step 6 with remaining models
        # (if using --compare-models, other models may be fine)
        logger.info("Proceeding to Step 6 with remaining model types")
        return True
    
    # Unknown action — proceed with caution
    logger.warning(f"Unknown health action: {health['action']} — proceeding")
    return True


# =============================================================================
# 3. HELPER METHODS — Add to WatcherAgent class
# =============================================================================

def _record_training_incident(self, health: dict) -> None:
    """
    Record training health incident for Strategy Advisor consumption.
    Appends to open_incidents list for next LLM bundle.
    """
    incident = {
        'type': 'training_health',
        'model_type': health.get('model_type', 'unknown'),
        'severity': health.get('severity', 'unknown'),
        'issues': health.get('issues', [])[:5],  # Cap at 5 issues
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    # Append to incidents file (or in-memory list)
    incidents_path = "watcher_incidents.jsonl"
    try:
        with open(incidents_path, 'a') as f:
            f.write(json.dumps(incident) + '\n')
    except IOError as e:
        logger.warning(f"Failed to record incident: {e}")


def _build_retry_params(self, original_params: dict, suggestions: dict) -> dict:
    """
    Build modified params for Step 5 retry based on health check suggestions.
    """
    retry_params = dict(original_params or {})
    
    # Apply suggestions from health check
    for key, value in suggestions.items():
        if not key.endswith('_reason'):  # Skip reason fields
            retry_params[key] = value
            logger.info(f"Retry param: {key} = {value}")
    
    return retry_params


def _record_model_skip(self, model_type: str, health: dict) -> None:
    """
    Record that a model type has been skipped due to repeated failures.
    """
    skip_record = {
        'model_type': model_type,
        'reason': 'consecutive_critical_failures',
        'consecutive': health.get('consecutive_critical', 0),
        'issues': health.get('issues', [])[:5],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    skip_log_path = "watcher_model_skips.jsonl"
    try:
        with open(skip_log_path, 'a') as f:
            f.write(json.dumps(skip_record) + '\n')
    except IOError as e:
        logger.warning(f"Failed to record model skip: {e}")


# =============================================================================
# 4. EXAMPLE: Full run_pipeline() integration
# =============================================================================

"""
In the existing run_pipeline() method, find the Step 5 section and modify:

BEFORE:
    if step == 5:
        result = self.dispatch_step(5, params)
        evaluation = self.evaluate_step(5, result)
        
        if evaluation['action'] == 'PROCEED':
            continue  # Go to Step 6

AFTER:
    if step == 5:
        result = self.dispatch_step(5, params)
        evaluation = self.evaluate_step(5, result)
        
        # ── NEW: Training health check ────────────────────────
        if not self._step_5_with_health_check(step, params, evaluation):
            return  # Pipeline stopped
        
        continue  # Go to Step 6
"""


# =============================================================================
# 5. STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== WATCHER Integration Test ===")
    print()
    print("This module provides integration code for watcher_agent.py")
    print()
    print("To test the health check module directly:")
    print("  python3 training_health_check.py --test")
    print()
    print("To check current status:")
    print("  python3 training_health_check.py --status")
