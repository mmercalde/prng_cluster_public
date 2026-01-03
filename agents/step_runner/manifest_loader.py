#!/usr/bin/env python3
"""
Manifest Loader - Load and parse agent manifest files
======================================================
Version: 1.0.0
Date: 2026-01-02

Loads JSON manifest files from agent_manifests/ directory
and converts them to typed StepManifest objects.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .models import (
    StepManifest,
    ActionConfig,
    EvaluationConfig,
    MetricsExtractionConfig,
    STEP_MANIFEST_MAP,
    get_manifest_filename
)

logger = logging.getLogger(__name__)


# =============================================================================
# MANIFEST LOADING
# =============================================================================

def load_manifest(
    step: int,
    manifest_dir: str = "agent_manifests",
    work_dir: Optional[Path] = None
) -> StepManifest:
    """
    Load and parse manifest for a pipeline step.
    
    Args:
        step: Pipeline step number (1-6)
        manifest_dir: Directory containing manifest files
        work_dir: Working directory (defaults to current directory)
    
    Returns:
        Parsed StepManifest object
    
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest is invalid
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    
    # Get manifest filename for step
    filename = get_manifest_filename(step)
    manifest_path = work_dir / manifest_dir / filename
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    logger.info(f"Loading manifest: {manifest_path}")
    
    # Load JSON
    with open(manifest_path, 'r') as f:
        raw_data = json.load(f)
    
    # Transform raw manifest to StepManifest format
    manifest_data = _transform_manifest(raw_data, step)
    
    # Parse into typed model
    manifest = StepManifest(**manifest_data)
    
    logger.info(f"Loaded manifest for step {step}: {manifest.agent_name}")
    return manifest


def _transform_manifest(raw: Dict[str, Any], step: int) -> Dict[str, Any]:
    """
    Transform raw manifest JSON to StepManifest-compatible format.
    
    Handles variations in manifest structure across different steps.
    Now supports multi-action manifests.
    """
    data = {}
    
    # Direct mappings
    data["agent_name"] = raw.get("agent_name", f"step_{step}_agent")
    data["description"] = raw.get("description", "")
    data["pipeline_step"] = raw.get("pipeline_step", step)
    data["version"] = raw.get("version", "1.0.0")
    
    # Parse actions into ActionConfig objects
    actions = []
    if "actions" in raw and isinstance(raw["actions"], list):
        for action_data in raw["actions"]:
            action = ActionConfig(
                type=action_data.get("type", "run_script"),
                script=action_data.get("script", ""),
                args_map=action_data.get("args_map", {}),
                distributed=action_data.get("distributed", False),
                timeout_minutes=action_data.get("timeout_minutes", 240)
            )
            actions.append(action)
    data["actions"] = actions
    
    # Extract script/args_map from first action for backward compatibility
    if "script" in raw:
        data["script"] = raw["script"]
    elif actions:
        data["script"] = actions[0].script
        data["distributed"] = actions[0].distributed
        data["timeout_minutes"] = actions[0].timeout_minutes
    
    # Args map - might be at top level or in first action
    if "args_map" in raw:
        data["args_map"] = raw["args_map"]
    elif actions:
        data["args_map"] = actions[0].args_map
    else:
        data["args_map"] = {}
    
    # Outputs
    data["outputs"] = raw.get("outputs", [])
    
    # Success criteria
    data["success_condition"] = raw.get("success_condition", "")
    data["success_metrics"] = raw.get("success_metrics", {})
    
    # Defaults and bounds
    data["default_params"] = raw.get("default_params", {})
    data["parameter_bounds"] = raw.get("parameter_bounds", {})
    
    # Metrics extraction config
    if "metrics_extraction" in raw:
        data["metrics_extraction"] = MetricsExtractionConfig(**raw["metrics_extraction"])
    else:
        # Build default metrics extraction from outputs
        data["metrics_extraction"] = _build_default_metrics_config(raw, step)
    
    # Evaluation config
    if "evaluation" in raw:
        data["evaluation"] = EvaluationConfig(**raw["evaluation"])
    else:
        data["evaluation"] = EvaluationConfig(
            prompt_file=f"agent_prompts/step{step}_eval.txt",
            grammar_file="agent_grammars/step_decision.gbnf",
            llm_model="math"
        )
    
    # Execution control
    data["retry"] = raw.get("retry", 2)
    data["timeout_minutes"] = raw.get("timeout_minutes", data.get("timeout_minutes", 240))
    data["distributed"] = raw.get("distributed", data.get("distributed", False))
    
    # Pipeline flow
    data["follow_up_agents"] = raw.get("follow_up_agents", [])
    
    return data


def _build_default_metrics_config(raw: Dict[str, Any], step: int) -> MetricsExtractionConfig:
    """
    Build default metrics extraction config based on step.
    
    Each step has known output files and expected metrics.
    """
    configs = {
        1: {  # Window Optimizer
            "from_output_files": {
                "bidirectional_survivors.json": {
                    "survivor_count": "len(data) if isinstance(data, list) else len(data.get('survivors', []))"
                },
                "optimal_window_config.json": {
                    "window_size": "data.get('window_size')",
                    "forward_threshold": "data.get('forward_threshold')",
                    "reverse_threshold": "data.get('reverse_threshold')",
                    "skip_mode": "data.get('skip_mode', 'constant')"
                }
            },
            "computed": {
                "survival_rate": "metrics.get('survivor_count', 0) / params.get('seed_count', 1)"
            }
        },
        2: {  # Scorer Meta
            "from_output_files": {
                "optimal_scorer_config.json": {
                    "best_score": "data.get('best_score')",
                    "best_params": "data.get('best_params')"
                }
            },
            "computed": {}
        },
        3: {  # Full Scoring
            "from_output_files": {
                "survivors_with_scores.json": {
                    "scored_count": "len(data) if isinstance(data, list) else len(data.get('survivors', []))",
                    "feature_count": "len(data[0].keys()) if isinstance(data, list) and len(data) > 0 else 0"
                }
            },
            "computed": {}
        },
        4: {  # ML Meta
            "from_output_files": {
                "reinforcement_engine_config.json": {
                    "recommended_epochs": "data.get('epochs')",
                    "recommended_batch_size": "data.get('batch_size')"
                }
            },
            "computed": {}
        },
        5: {  # Anti-Overfit
            "from_output_files": {
                "models/reinforcement/best_model.meta.json": {
                    "signal_quality": "data.get('signal_quality', {}).get('quality')",
                    "signal_confidence": "data.get('signal_quality', {}).get('confidence')",
                    "model_type": "data.get('model_type')",
                    "fingerprint": "data.get('data_context', {}).get('fingerprint_hash')"
                }
            },
            "computed": {}
        },
        6: {  # Prediction
            "from_output_files": {},  # Dynamic output path
            "computed": {}
        }
    }
    
    config = configs.get(step, {"from_output_files": {}, "computed": {}})
    return MetricsExtractionConfig(**config)


def list_available_manifests(manifest_dir: str = "agent_manifests") -> Dict[int, str]:
    """
    List all available manifest files.
    
    Returns:
        Dict mapping step numbers to manifest filenames
    """
    manifest_path = Path(manifest_dir)
    available = {}
    
    for step, filename in STEP_MANIFEST_MAP.items():
        if (manifest_path / filename).exists():
            available[step] = filename
    
    return available


def validate_manifest(manifest: StepManifest) -> tuple[bool, list[str]]:
    """
    Validate a manifest has required fields.
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    if not manifest.script:
        errors.append("Missing required field: script")
    
    if not manifest.outputs:
        errors.append("Missing required field: outputs")
    
    if manifest.pipeline_step < 1 or manifest.pipeline_step > 6:
        errors.append(f"Invalid pipeline_step: {manifest.pipeline_step}")
    
    if manifest.timeout_minutes < 1:
        errors.append(f"Invalid timeout_minutes: {manifest.timeout_minutes}")
    
    return len(errors) == 0, errors
