"""
Threshold Guardrail - Deterministic override for LLM decisions.
Team Beta Approved: 2026-01-05

Rule: LLMs reason. Deterministic code decides.

If a decision depends on > or <, it must be enforced in code.
If a decision depends on "why", it belongs to the LLM.
"""
from typing import Dict, Any, Tuple


def enforce_thresholds(
    llm_decision: Dict[str, Any],
    derived_metrics: Dict[str, float],
    threshold_priors: Dict[str, Any],
    data_source_type: str = "synthetic"
) -> Tuple[Dict[str, Any], bool]:
    """
    Hard guardrail: numeric truth wins over LLM opinion.
    
    Args:
        llm_decision: The LLM's proposed decision
        derived_metrics: Computed rates from raw metrics
        threshold_priors: Config-driven thresholds
        data_source_type: "synthetic", "real", or "hybrid"
    
    Returns:
        Tuple of (final_decision, was_overridden)
    """
    was_overridden = False
    final = llm_decision.copy()
    
    # Get bidirectional rate thresholds
    bi_priors = threshold_priors.get("bidirectional_rate", {})
    fail_max = bi_priors.get("fail_max", 0.30)
    warn_max = bi_priors.get("warn_max", 0.10)
    
    bi_rate = derived_metrics.get("bidirectional_rate", 0)
    
    # Check fail condition
    if bi_rate > fail_max:
        if final.get("decision") != "escalate":
            final["decision"] = "escalate"
            final["retry_reason"] = None
            final["confidence"] = min(float(final.get("confidence", 0.5)), 0.8)
            final["_override_reason"] = f"threshold_violation: bidirectional_rate {bi_rate:.4f} > fail_max {fail_max}"
            was_overridden = True
    
    # Check warn condition
    elif bi_rate > warn_max:
        if final.get("decision") == "proceed":
            final["decision"] = "retry"
            final["retry_reason"] = "tighten"
            final["confidence"] = min(float(final.get("confidence", 0.5)), 0.7)
            final["_override_reason"] = f"threshold_violation: bidirectional_rate {bi_rate:.4f} > warn_max {warn_max}"
            was_overridden = True
    
    # Additional checks: overlap_ratio too low
    overlap_priors = threshold_priors.get("overlap_ratio", {})
    overlap_min = overlap_priors.get("good_min", 0.25)
    overlap_rate = derived_metrics.get("overlap_ratio", 1.0)
    
    if overlap_rate < overlap_min:
        if final.get("decision") == "proceed":
            final["decision"] = "retry"
            final["retry_reason"] = "investigate"
            final["confidence"] = min(float(final.get("confidence", 0.5)), 0.6)
            final["_override_reason"] = f"threshold_violation: overlap_ratio {overlap_rate:.4f} < good_min {overlap_min}"
            was_overridden = True
    
    return final, was_overridden


def validate_decision(
    llm_decision: Dict[str, Any],
    derived_metrics: Dict[str, float],
    threshold_priors: Dict[str, Any],
    data_source_type: str = "synthetic"
) -> Dict[str, Any]:
    """
    Convenience wrapper that applies guardrail and logs override.
    """
    final, was_overridden = enforce_thresholds(
        llm_decision, derived_metrics, threshold_priors, data_source_type
    )
    
    if was_overridden:
        print(f"[GUARDRAIL] LLM decision overridden: {llm_decision.get('decision')} → {final.get('decision')}")
        print(f"[GUARDRAIL] Reason: {final.get('_override_reason')}")
    
    return final
