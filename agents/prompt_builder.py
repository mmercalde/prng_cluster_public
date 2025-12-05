#!/usr/bin/env python3
"""
Prompt Builder - Assembles hybrid JSON prompts for LLM evaluation.

Combines context dicts from various sources into a clean prompt
with minimal framing and maximum data clarity.

Version: 3.2.0
"""

from typing import Dict, Any, Optional, List
import json

from agents.agent_decision import AgentDecision


def build_evaluation_prompt(
    manifest_context: Dict[str, Any],
    parameter_context: Dict[str, Any],
    results: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
    history: Optional[Dict[str, Any]] = None,
    runtime: Optional[Dict[str, Any]] = None,
    task_instruction: Optional[str] = None
) -> str:
    """
    Build a complete evaluation prompt from context dicts.
    
    This is the hybrid approach:
    - Section labels (short, uppercase) for anchoring
    - All data as clean JSON
    - Minimal instruction text
    - Schema-enforced output format
    
    Args:
        manifest_context: From AgentManifest.to_context_dict()
        parameter_context: From ParameterContext.to_context_dict()
        results: The results JSON to evaluate
        thresholds: Optional evaluation thresholds
        history: Optional historical run data
        runtime: Optional runtime environment info
        task_instruction: Optional custom task instruction
        
    Returns:
        Complete prompt string ready for LLM
    """
    sections = []
    
    # ══════════════════════════════════════════════════════════════════════
    # MANIFEST
    # ══════════════════════════════════════════════════════════════════════
    sections.append("MANIFEST:")
    sections.append(json.dumps(manifest_context, indent=2))
    
    # ══════════════════════════════════════════════════════════════════════
    # PARAMETERS
    # ══════════════════════════════════════════════════════════════════════
    if parameter_context.get("adjustable_parameters"):
        sections.append("\nPARAMETERS:")
        sections.append(json.dumps(parameter_context, indent=2))
    
    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════
    sections.append("\nRESULTS:")
    # Truncate large results
    results_str = json.dumps(results, indent=2, default=str)
    if len(results_str) > 4000:
        results_str = results_str[:4000] + "\n... (truncated)"
    sections.append(results_str)
    
    # ══════════════════════════════════════════════════════════════════════
    # THRESHOLDS (if provided)
    # ══════════════════════════════════════════════════════════════════════
    if thresholds:
        sections.append("\nTHRESHOLDS:")
        sections.append(json.dumps(thresholds, indent=2))
    
    # ══════════════════════════════════════════════════════════════════════
    # HISTORY (if provided)
    # ══════════════════════════════════════════════════════════════════════
    if history:
        sections.append("\nHISTORY:")
        sections.append(json.dumps(history, indent=2))
    
    # ══════════════════════════════════════════════════════════════════════
    # RUNTIME (if provided)
    # ══════════════════════════════════════════════════════════════════════
    if runtime:
        sections.append("\nRUNTIME:")
        sections.append(json.dumps(runtime, indent=2))
    
    # ══════════════════════════════════════════════════════════════════════
    # TASK
    # ══════════════════════════════════════════════════════════════════════
    default_task = (
        "Evaluate the RESULTS against the success_condition in MANIFEST. "
        "Determine if criteria are met and recommend next action."
    )
    sections.append(f"\nTASK: {task_instruction or default_task}")
    
    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT FORMAT
    # ══════════════════════════════════════════════════════════════════════
    sections.append("\nOUTPUT FORMAT:")
    sections.append(json.dumps(AgentDecision.output_schema(), indent=2))
    
    sections.append("\nEXAMPLE OUTPUT:")
    sections.append(json.dumps(AgentDecision.output_example(), indent=2))
    
    sections.append("\nRespond with valid JSON only. No markdown, no explanation outside JSON.")
    
    return "\n".join(sections)


def build_simple_prompt(
    results: Dict[str, Any],
    success_condition: str,
    parameters: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build a minimal evaluation prompt.
    
    For quick evaluations without full context.
    
    Args:
        results: The results to evaluate
        success_condition: The condition to check
        parameters: Optional list of adjustable parameters
        
    Returns:
        Simple prompt string
    """
    sections = []
    
    sections.append("RESULTS:")
    sections.append(json.dumps(results, indent=2, default=str))
    
    sections.append(f"\nSUCCESS_CONDITION: {success_condition}")
    
    if parameters:
        sections.append("\nADJUSTABLE_PARAMETERS:")
        sections.append(json.dumps(parameters, indent=2))
    
    sections.append("\nTASK: Evaluate if success_condition is met.")
    
    sections.append("\nOUTPUT FORMAT:")
    sections.append(json.dumps(AgentDecision.output_schema(), indent=2))
    
    sections.append("\nRespond with valid JSON only.")
    
    return "\n".join(sections)


def build_doctrine_header(
    mission: str = "Distributed PRNG Analysis System",
    version: str = "3.2.0"
) -> str:
    """
    Build a minimal doctrine header.
    
    Unlike the old prose approach, this is just metadata.
    """
    return json.dumps({
        "system": mission,
        "version": version,
        "decision_options": ["proceed", "retry", "escalate"],
        "proceed_threshold": 0.7,
        "escalate_threshold": 0.5
    }, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# STEP-SPECIFIC THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════════

STEP_THRESHOLDS = {
    1: {  # Window Optimizer
        "bidirectional_count": {
            "excellent": {"min": 1, "max": 10},
            "good": {"min": 11, "max": 100},
            "acceptable": {"min": 101, "max": 1000},
            "poor": {"min": 1001, "max": 10000},
            "fail": {"min": 10001, "max": None}
        },
        "zero_survivors": "fail"
    },
    2: {  # Scorer Meta Optimizer
        "validation_score": {
            "excellent": {"min": 0.95},
            "good": {"min": 0.85},
            "acceptable": {"min": 0.70},
            "poor": {"min": 0.50},
            "fail": {"max": 0.50}
        },
        "convergence_required": True
    },
    3: {  # Full Scoring
        "completion_rate": {
            "required": 1.0,
            "acceptable": 0.99
        },
        "features_required": 64
    },
    4: {  # ML Meta Optimizer
        "architecture_score": {
            "excellent": {"min": 0.85},
            "good": {"min": 0.70},
            "acceptable": {"min": 0.55},
            "fail": {"max": 0.55}
        },
        "optimal_layers": {"min": 2, "max": 4}
    },
    5: {  # Anti-Overfit
        "overfit_ratio": {
            "excellent": {"min": 0.95, "max": 1.05},
            "good": {"min": 1.05, "max": 1.15},
            "warning": {"min": 1.15, "max": 1.30},
            "fail": {"min": 1.30}
        },
        "kfold_std_max": 0.05
    },
    6: {  # Prediction
        "pool_size": {
            "optimal": {"min": 100, "max": 300},
            "acceptable": {"min": 50, "max": 500}
        },
        "mean_confidence": {
            "high": {"min": 0.7},
            "moderate": {"min": 0.5},
            "low": {"max": 0.5}
        }
    }
}


def get_thresholds_for_step(step: int) -> Dict[str, Any]:
    """Get evaluation thresholds for a pipeline step."""
    return STEP_THRESHOLDS.get(step, {})
