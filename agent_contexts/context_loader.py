#!/usr/bin/env python3
"""
Mission Context Loader for WATCHER Agent
Version: 1.0.0

Loads step-specific mission context templates and injects runtime data.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


# Step name mappings
STEP_TEMPLATES = {
    1: "step1_window_optimizer.md",
    2: "step2_bidirectional_sieve.md",
    2.5: "step2_5_scorer_meta.md",
    3: "step3_full_scoring.md",
    4: "step4_adaptive_meta.md",
    5: "step5_anti_overfit.md",
    6: "step6_prediction.md"
}

STEP_NAMES = {
    1: "Window Optimizer",
    2: "Bidirectional Sieve",
    2.5: "Scorer Meta-Optimizer",
    3: "Full Scoring",
    4: "Adaptive Meta-Optimizer",
    5: "Anti-Overfit Training",
    6: "Prediction Generator"
}


def load_mission_template(
    step: float,
    context_dir: str = "agent_contexts"
) -> str:
    """
    Load the mission context template for a specific step.
    
    Args:
        step: Pipeline step number (1, 2, 2.5, 3, 4, 5, 6)
        context_dir: Directory containing template files
    
    Returns:
        Template content as string
    
    Raises:
        ValueError: If step not recognized
        FileNotFoundError: If template file missing
    """
    if step not in STEP_TEMPLATES:
        valid_steps = list(STEP_TEMPLATES.keys())
        raise ValueError(f"Unknown step {step}. Valid steps: {valid_steps}")
    
    template_file = Path(context_dir) / STEP_TEMPLATES[step]
    
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    
    return template_file.read_text()


def build_prompt(
    step: float,
    current_data: Dict[str, Any],
    context_dir: str = "agent_contexts"
) -> str:
    """
    Build complete prompt with mission context and current data.
    
    Args:
        step: Pipeline step number
        current_data: Runtime data to inject
        context_dir: Directory containing template files
    
    Returns:
        Complete prompt ready for LLM
    """
    template = load_mission_template(step, context_dir)
    data_json = json.dumps(current_data, indent=2)
    return template.replace("{current_data_json}", data_json)


def get_step_name(step: float) -> str:
    """Get human-readable step name."""
    return STEP_NAMES.get(step, f"Unknown Step {step}")


def get_follow_up_step(step: float) -> Optional[float]:
    """Get the next step in the pipeline."""
    sequence = [1, 2, 2.5, 3, 4, 5, 6]
    try:
        idx = sequence.index(step)
        if idx + 1 < len(sequence):
            return sequence[idx + 1]
        return None  # Step 6 is terminal
    except ValueError:
        return None


# Example usage
if __name__ == "__main__":
    # Example: Build prompt for Step 1
    step1_data = {
        "trials_completed": 50,
        "bidirectional_count": 847,
        "seeds_tested": 50000000,
        "bidirectional_rate": 0.000017,
        "forward_count": 12543,
        "reverse_count": 9876,
        "best_config": {
            "window_size": 256,
            "offset": 50,
            "skip_min": 0,
            "skip_max": 30
        }
    }
    
    prompt = build_prompt(1, step1_data)
    print(f"Step 1 prompt length: {len(prompt)} chars")
    print(f"Follow-up step: {get_follow_up_step(1)}")
    
    # List all templates
    print("\nAvailable templates:")
    for step, name in STEP_NAMES.items():
        print(f"  Step {step}: {name}")
