#!/usr/bin/env python3
"""
Agent Doctrine - Shared reasoning rules for all AI agents.

The doctrine provides consistent decision-making guidelines that
apply across all pipeline steps, ensuring predictable behavior.

Version: 3.2.0
"""

from typing import Dict, Any


# Doctrine version
DOCTRINE_VERSION = "1.0.0"


def get_doctrine() -> Dict[str, Any]:
    """
    Get the agent doctrine as a structured dict.
    
    The doctrine defines:
    1. Decision framework (when to proceed/retry/escalate)
    2. Confidence calibration rules
    3. Safety priorities
    4. Communication guidelines
    """
    return {
        "version": DOCTRINE_VERSION,
        "purpose": "PRNG pattern analysis through systematic pipeline execution",
        
        "decision_framework": {
            "proceed": {
                "conditions": [
                    "success_condition_met = true",
                    "confidence >= 0.70",
                    "no critical anomalies detected"
                ],
                "description": "Advance to next pipeline step"
            },
            "retry": {
                "conditions": [
                    "success_condition_met = false",
                    "retries_remaining > 0",
                    "adjustable parameters available",
                    "no safety violations"
                ],
                "description": "Re-run current step with adjusted parameters"
            },
            "escalate": {
                "conditions": [
                    "confidence < 0.50",
                    "anomalies detected",
                    "retries exhausted",
                    "safety concerns"
                ],
                "description": "Halt for human review"
            }
        },
        
        "confidence_calibration": {
            "1.00": "Certain - all metrics excellent, no concerns",
            "0.85-0.99": "High confidence - metrics good/excellent",
            "0.70-0.84": "Acceptable - metrics meet minimum thresholds",
            "0.50-0.69": "Uncertain - mixed results, consider retry",
            "0.00-0.49": "Low confidence - escalate for review"
        },
        
        "safety_priorities": [
            "Never proceed if kill switch is triggered",
            "Respect retry limits to prevent infinite loops",
            "Escalate anomalies rather than ignore them",
            "Preserve all artifacts for debugging"
        ],
        
        "parameter_adjustment_rules": [
            "Only suggest parameters listed in ADJUSTABLE PARAMETERS",
            "Values must be within specified min/max bounds",
            "Provide clear reasoning for each adjustment",
            "Prefer conservative changes over aggressive ones"
        ],
        
        "output_requirements": {
            "format": "JSON only, no markdown or extra text",
            "required_fields": [
                "success_condition_met",
                "confidence",
                "reasoning",
                "recommended_action"
            ],
            "conditional_fields": {
                "suggested_param_adjustments": "Required if action is 'retry'",
                "warnings": "Include if any concerns detected"
            }
        }
    }


def get_doctrine_summary() -> str:
    """
    Get a minimal doctrine summary for prompt inclusion.
    
    This is intentionally brief - the full doctrine is in the dict.
    """
    return """DECISION RULES:
- PROCEED: success=true AND confidence>=0.70
- RETRY: success=false AND retries available AND parameters adjustable
- ESCALATE: confidence<0.50 OR anomalies OR safety concerns

OUTPUT: JSON only. Required: success_condition_met, confidence, reasoning, recommended_action."""


def validate_decision_against_doctrine(
    decision: Dict[str, Any],
    context: Dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Validate an agent decision against doctrine rules.
    
    Args:
        decision: The agent's decision dict
        context: The context dict used for the decision
        
    Returns:
        (is_valid, list of violations)
    """
    violations = []
    
    action = decision.get("recommended_action", "")
    confidence = decision.get("confidence", 0)
    success = decision.get("success_condition_met", False)
    adjustments = decision.get("suggested_param_adjustments", {})
    
    # Check action validity
    if action not in ["proceed", "retry", "escalate"]:
        violations.append(f"Invalid action: {action}")
    
    # Check confidence bounds
    if not (0.0 <= confidence <= 1.0):
        violations.append(f"Confidence out of bounds: {confidence}")
    
    # Check proceed conditions
    if action == "proceed":
        if not success:
            violations.append("Cannot proceed when success_condition_met=false")
        if confidence < 0.70:
            violations.append(f"Cannot proceed with confidence {confidence} < 0.70")
    
    # Check retry conditions
    if action == "retry":
        if not adjustments:
            violations.append("Retry requires suggested_param_adjustments")
        
        # Validate adjustments against parameter bounds if available
        params = context.get("parameters", {}).get("adjustable_parameters", [])
        param_bounds = {p.get("name"): p for p in params}
        
        for param, value in adjustments.items():
            if param in param_bounds:
                bounds = param_bounds[param]
                min_val = bounds.get("min")
                max_val = bounds.get("max")
                
                if min_val is not None and value < min_val:
                    violations.append(f"{param}={value} below minimum {min_val}")
                if max_val is not None and value > max_val:
                    violations.append(f"{param}={value} above maximum {max_val}")
    
    # Check escalate conditions
    if action == "escalate":
        if confidence >= 0.70 and success:
            violations.append("Escalation unexpected when metrics are good")
    
    return len(violations) == 0, violations
