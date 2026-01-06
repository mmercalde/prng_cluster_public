"""
Step 1 Prompt Builder - Window Optimizer evaluation.
Team Beta Approved: 2026-01-04

This replaces the legacy to_context_dict() approach with:
- Global mission context
- Step-specific mission
- Raw + derived metrics (no interpretation)
- Relevant threshold priors only
"""
import json
from typing import Dict, Any

from agents.prompts.global_mission import GLOBAL_MISSION


STEP1_MISSION = """Your role is to evaluate whether the current search window is sufficiently
constrained before downstream computation. Be conservative: retry > escalate > proceed."""


def build_step1_prompt(
    raw_metrics: Dict[str, Any],
    derived_metrics: Dict[str, float],
    threshold_priors: Dict[str, Any],
    data_source_type: str = "synthetic"
) -> str:
    """
    Build the LLM prompt for Step 1 evaluation.
    
    Args:
        raw_metrics: Unprocessed counts from window optimizer
        derived_metrics: Computed rates/ratios
        threshold_priors: Relevant thresholds for this data_source_type
        data_source_type: "synthetic", "real", or "hybrid"
    
    Returns:
        Complete prompt string for LLM
    """
    # Extract only relevant priors (bidirectional_rate, overlap_ratio)
    relevant_priors = {}
    for key in ["bidirectional_rate", "overlap_ratio", "forward_rate", "reverse_rate"]:
        if key in threshold_priors:
            relevant_priors[key] = threshold_priors[key]
    
    prompt = f"""SYSTEM:
You are the Watcher agent for a distributed PRNG analysis system.

GLOBAL MISSION:
{GLOBAL_MISSION}

STEP MISSION (STEP 1 – WINDOW OPTIMIZER):
{STEP1_MISSION}

DATA SOURCE TYPE:
{data_source_type}

THRESHOLD PRIORS (contextual, not absolute rules):
{json.dumps(relevant_priors, indent=2)}

RAW METRICS:
{json.dumps(raw_metrics, indent=2)}

DERIVED METRICS:
{json.dumps(derived_metrics, indent=2)}

TASK:
Using DERIVED RATES (not absolute counts), decide:
- proceed: rates within good range
- retry: rates in warning range, suggest parameter tightening
- escalate: rates exceed fail threshold or anomaly detected

REQUIREMENTS:
- Reference bidirectional_rate explicitly in reasoning
- Mention data_source_type in reasoning
- Set checks.used_rates = true, checks.mentioned_data_source = true

OUTPUT:
JSON only. Must match this schema:
{{
  "decision": "proceed" | "retry" | "escalate",
  "retry_reason": "tighten" | "widen" | "rerun" | "investigate" | null,
  "confidence": 0.0 to 1.0,
  "reasoning": "your analysis here",
  "primary_signal": "bidirectional_rate",
  "suggested_params": {{"param": "value"}} or null,
  "warnings": [],
  "checks": {{
    "used_rates": true,
    "mentioned_data_source": true,
    "avoided_absolute_only": true
  }}
}}"""
    
    return prompt


def build_step1_corrective_prompt(
    raw_metrics: Dict[str, Any],
    derived_metrics: Dict[str, float],
    threshold_priors: Dict[str, Any],
    data_source_type: str = "synthetic",
    violation: str = "checks failed"
) -> str:
    """
    Build corrective prompt after first LLM failure.
    
    Team Beta rule: One retry with corrective prefix, then fallback.
    """
    base_prompt = build_step1_prompt(
        raw_metrics, derived_metrics, threshold_priors, data_source_type
    )
    
    corrective_prefix = f"""CORRECTION REQUIRED:
Your previous response violated the rules ({violation}).
You MUST base your decision on RATES and mention data_source_type.
Do NOT use absolute counts alone.

"""
    
    return corrective_prefix + base_prompt
