#!/usr/bin/env python3
"""
Metadata Writer Utility - Phase 2 Agent Infrastructure

Injects agent_metadata into result dictionaries for autonomous pipeline operation.
Designed to be lightweight, stateless, and compatible with all existing scripts.

Location: integration/metadata_writer.py
Schema: results_schema_v1.json v1.0.3

Usage:
    from integration.metadata_writer import inject_agent_metadata
    
    result = {"run_metadata": {...}, "results_summary": {...}}
    result = inject_agent_metadata(
        result,
        inputs=[{"file": "survivors.json", "required": True}],
        outputs=["optimal_config.json"],
        pipeline_step=2,
        follow_up_agent="full_scoring_agent",
        confidence=0.92
    )
"""

import os
import json
import hashlib
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = "1.0.3"

VALID_PIPELINE_STEPS = {1, 2, 3, 4, 5, 6}

PIPELINE_STEP_NAMES = {
    1: "window_optimizer",
    2: "scorer_meta_optimizer",
    3: "full_scoring",
    4: "ml_meta_optimizer",
    5: "anti_overfit_training",
    6: "prediction"
}

VALID_FOLLOW_UP_AGENTS = {
    "window_optimizer_agent",
    "scorer_meta_agent",
    "full_scoring_agent",
    "ml_meta_agent",
    "reinforcement_agent",
    "prediction_agent",
    None  # Valid for final step
}


# =============================================================================
# Core Injection Function
# =============================================================================

def inject_agent_metadata(
        result_dict: Dict[str, Any],
        inputs: Optional[List[Union[str, Dict]]] = None,
        outputs: Optional[List[str]] = None,
        parent_run_id: Optional[str] = None,
        pipeline_step: Optional[int] = None,
        follow_up_agent: Optional[str] = None,
        confidence: Optional[float] = None,
        suggested_params: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        success_criteria_met: Optional[bool] = None,
        retry_count: Optional[int] = None,
        cluster_resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
    """
    Inject agent_metadata section into a result dictionary.
    
    Args:
        result_dict: The result dictionary to modify (modified in place and returned)
        inputs: Input files consumed by this step. Can be:
                - List of strings: ["file1.json", "file2.json"]
                - List of dicts: [{"file": "file1.json", "hash": "sha256:...", "required": True}]
        outputs: Output files produced by this step (list of relative paths)
        parent_run_id: Run ID of parent/previous step (None for step 1)
        pipeline_step: Position in pipeline (1-6)
        follow_up_agent: Next agent to trigger (None for final step)
        confidence: AI confidence in results (0.0 to 1.0)
        suggested_params: Parameters to pass to next step
        reasoning: Human-readable explanation
        success_criteria_met: Whether step met success criteria
        retry_count: Number of retry attempts
        cluster_resources: Cluster resources used (nodes, GPUs, etc.)
    
    Returns:
        The modified result_dict with agent_metadata injected
    
    Example:
        result = inject_agent_metadata(
            result,
            inputs=[{"file": "bidirectional_survivors.json", "required": True}],
            outputs=["optimal_scorer_config.json"],
            parent_run_id="step1_20251201_143022_abc123",
            pipeline_step=2,
            follow_up_agent="full_scoring_agent",
            confidence=0.92,
            suggested_params={"threshold": 0.012},
            reasoning="High overlap suggests strong signal",
            success_criteria_met=True
        )
    """
    
    # Validate inputs (warn but don't fail)
    _validate_inputs(
        pipeline_step=pipeline_step,
        follow_up_agent=follow_up_agent,
        confidence=confidence,
        parent_run_id=parent_run_id
    )
    
    # Normalize inputs list
    normalized_inputs = _normalize_inputs(inputs) if inputs else None
    
    # Auto-derive pipeline_step_name if pipeline_step provided
    pipeline_step_name = None
    if pipeline_step is not None and pipeline_step in PIPELINE_STEP_NAMES:
        pipeline_step_name = PIPELINE_STEP_NAMES[pipeline_step]
    
    # Build agent_metadata section
    agent_metadata = {}
    
    # Only include non-None values (keep output clean)
    if normalized_inputs is not None:
        agent_metadata["inputs"] = normalized_inputs
    
    if outputs is not None:
        agent_metadata["outputs"] = outputs
    
    if parent_run_id is not None:
        agent_metadata["parent_run_id"] = parent_run_id
    elif pipeline_step == 1:
        # Explicitly set null for step 1 (no parent)
        agent_metadata["parent_run_id"] = None
    
    if pipeline_step is not None:
        agent_metadata["pipeline_step"] = pipeline_step
    
    if pipeline_step_name is not None:
        agent_metadata["pipeline_step_name"] = pipeline_step_name
    
    # Handle follow_up_agent (can be string or None)
    if follow_up_agent is not None:
        agent_metadata["follow_up_agent"] = follow_up_agent
    elif pipeline_step == 6:
        # Explicitly set null for final step
        agent_metadata["follow_up_agent"] = None
    
    if confidence is not None:
        agent_metadata["confidence"] = confidence
    
    if suggested_params is not None:
        agent_metadata["suggested_params"] = suggested_params
    
    if reasoning is not None:
        agent_metadata["reasoning"] = reasoning
    
    if success_criteria_met is not None:
        agent_metadata["success_criteria_met"] = success_criteria_met
    
    if retry_count is not None:
        agent_metadata["retry_count"] = retry_count
    
    if cluster_resources is not None:
        agent_metadata["cluster_resources"] = cluster_resources
    
    # Auto-add timestamp (recommended refinement)
    agent_metadata["timestamp_injected"] = datetime.now(timezone.utc).isoformat()
    
    # Inject into result_dict
    result_dict["agent_metadata"] = agent_metadata
    
    # Ensure schema_version is set in run_metadata
    if "run_metadata" in result_dict:
        result_dict["run_metadata"]["schema_version"] = SCHEMA_VERSION
    
    return result_dict


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_inputs(inputs: List[Union[str, Dict]]) -> List[Dict]:
    """
    Normalize inputs to consistent dict format.
    
    Accepts:
        - ["file1.json", "file2.json"]  -> converts to dict format
        - [{"file": "file1.json", ...}] -> passes through
    
    Returns:
        List of dicts with at least "file" key
    """
    normalized = []
    
    for inp in inputs:
        if isinstance(inp, str):
            # Simple string -> convert to dict
            normalized.append({"file": inp, "required": True})
        elif isinstance(inp, dict):
            # Already dict -> ensure "file" key exists
            if "file" not in inp:
                warnings.warn(f"Input dict missing 'file' key: {inp}")
                continue
            # Set default for required if not specified
            if "required" not in inp:
                inp["required"] = True
            normalized.append(inp)
        else:
            warnings.warn(f"Invalid input type: {type(inp)}, skipping")
    
    return normalized


def _validate_inputs(
        pipeline_step: Optional[int],
        follow_up_agent: Optional[str],
        confidence: Optional[float],
        parent_run_id: Optional[str]
    ) -> None:
    """
    Validate inputs and warn on issues (never fails).
    """
    
    # Validate pipeline_step
    if pipeline_step is not None and pipeline_step not in VALID_PIPELINE_STEPS:
        warnings.warn(
            f"pipeline_step {pipeline_step} not in valid range 1-6. "
            "Proceeding anyway."
        )
    
    # Validate follow_up_agent
    if follow_up_agent is not None and follow_up_agent not in VALID_FOLLOW_UP_AGENTS:
        warnings.warn(
            f"follow_up_agent '{follow_up_agent}' not in known agents. "
            "Proceeding anyway."
        )
    
    # Validate confidence range
    if confidence is not None:
        if not (0.0 <= confidence <= 1.0):
            warnings.warn(
                f"confidence {confidence} outside valid range 0.0-1.0. "
                "Proceeding anyway."
            )
    
    # Warn if step > 1 but no parent_run_id
    if pipeline_step is not None and pipeline_step > 1 and parent_run_id is None:
        warnings.warn(
            f"pipeline_step {pipeline_step} > 1 but no parent_run_id provided. "
            "Lineage tracking will be incomplete."
        )


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> Optional[str]:
    """
    Compute hash of a file for input verification.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ("sha256" or "md5")
    
    Returns:
        Hash string prefixed with algorithm, e.g., "sha256:a1b2c3..."
        Returns None if file not found
    
    Example:
        hash = compute_file_hash("bidirectional_survivors.json")
        # Returns: "sha256:a1b2c3d4e5f6..."
    """
    if not os.path.exists(filepath):
        warnings.warn(f"File not found for hashing: {filepath}")
        return None
    
    try:
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        else:
            warnings.warn(f"Unknown hash algorithm: {algorithm}, using sha256")
            hasher = hashlib.sha256()
            algorithm = "sha256"
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return f"{algorithm}:{hasher.hexdigest()}"
    
    except Exception as e:
        warnings.warn(f"Error computing hash for {filepath}: {e}")
        return None


def create_input_entry(
        filepath: str,
        required: bool = True,
        compute_hash: bool = True
    ) -> Dict[str, Any]:
    """
    Create a properly formatted input entry with optional hash.
    
    Args:
        filepath: Path to input file (will be converted to relative)
        required: Whether this input is required for the step
        compute_hash: Whether to compute and include file hash
    
    Returns:
        Dict ready for inputs list
    
    Example:
        entry = create_input_entry("results/bidirectional_survivors.json")
        # Returns: {"file": "bidirectional_survivors.json", "hash": "sha256:...", "required": True}
    """
    # Use basename for relative path
    filename = os.path.basename(filepath)
    
    entry = {
        "file": filename,
        "required": required
    }
    
    if compute_hash and os.path.exists(filepath):
        file_hash = compute_file_hash(filepath)
        if file_hash:
            entry["hash"] = file_hash
    
    return entry


def get_default_cluster_resources() -> Dict[str, Any]:
    """
    Return default cluster resources for the 26-GPU cluster.
    
    Can be overridden by passing custom cluster_resources to inject_agent_metadata.
    """
    return {
        "nodes": ["zeus", "rig-6600", "rig-6600b", "rig-6600c"],
        "total_gpus": 26,
        "total_tflops": 285.69,
        "platform": "Mixed CUDA/ROCm"
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_step1_metadata(
        outputs: List[str],
        confidence: float,
        reasoning: str = None,
        suggested_params: Dict = None,
        success_criteria_met: bool = True
    ) -> Dict[str, Any]:
    """
    Convenience function for Window Optimizer (Step 1) metadata.
    
    Step 1 has no parent_run_id and follows up with scorer_meta_agent.
    """
    return {
        "outputs": outputs,
        "parent_run_id": None,
        "pipeline_step": 1,
        "pipeline_step_name": "window_optimizer",
        "follow_up_agent": "scorer_meta_agent",
        "confidence": confidence,
        "suggested_params": suggested_params,
        "reasoning": reasoning,
        "success_criteria_met": success_criteria_met,
        "timestamp_injected": datetime.now(timezone.utc).isoformat()
    }


def create_step6_metadata(
        inputs: List[Union[str, Dict]],
        outputs: List[str],
        parent_run_id: str,
        confidence: float,
        reasoning: str = None,
        success_criteria_met: bool = True
    ) -> Dict[str, Any]:
    """
    Convenience function for Prediction (Step 6) metadata.
    
    Step 6 has no follow_up_agent (pipeline ends).
    """
    return {
        "inputs": _normalize_inputs(inputs) if inputs else None,
        "outputs": outputs,
        "parent_run_id": parent_run_id,
        "pipeline_step": 6,
        "pipeline_step_name": "prediction",
        "follow_up_agent": None,
        "confidence": confidence,
        "reasoning": reasoning,
        "success_criteria_met": success_criteria_met,
        "timestamp_injected": datetime.now(timezone.utc).isoformat()
    }


# =============================================================================
# File I/O Helpers
# =============================================================================

def save_result_with_metadata(
        result_dict: Dict[str, Any],
        output_path: str,
        create_dirs: bool = True
    ) -> str:
    """
    Save result dictionary to JSON file, creating directories if needed.
    
    Args:
        result_dict: The result dict (should already have agent_metadata injected)
        output_path: Path to save the JSON file
        create_dirs: Create parent directories if they don't exist
    
    Returns:
        The output_path on success
    
    Example:
        save_result_with_metadata(result, "results/step2_output.json")
    """
    if create_dirs:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    return output_path


def load_result_with_metadata(input_path: str) -> Dict[str, Any]:
    """
    Load a result JSON file and return the dict.
    
    Args:
        input_path: Path to the JSON file
    
    Returns:
        The result dictionary
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def extract_agent_metadata(result_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract agent_metadata from a result dictionary.
    
    Returns None if agent_metadata section doesn't exist.
    """
    return result_dict.get("agent_metadata")


def get_follow_up_info(result_dict: Dict[str, Any]) -> tuple:
    """
    Extract follow-up information for pipeline chaining.
    
    Returns:
        (follow_up_agent, suggested_params, confidence)
        
    Example:
        agent, params, conf = get_follow_up_info(result)
        if conf >= 0.8 and agent:
            trigger_agent(agent, params)
    """
    metadata = extract_agent_metadata(result_dict)
    if not metadata:
        return (None, None, None)
    
    return (
        metadata.get("follow_up_agent"),
        metadata.get("suggested_params"),
        metadata.get("confidence")
    )


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    # Quick self-test
    print("Testing metadata_writer.py...")
    
    # Create sample result
    sample_result = {
        "run_metadata": {
            "run_id": "test_20251201_160000_abc123",
            "analysis_type": "scorer_meta_optimization",
            "timestamp_start": "2025-12-01T16:00:00-08:00"
        },
        "results_summary": {
            "total_survivors": 278,
            "survival_rate": 0.0000278,
            "analysis_complete": True
        }
    }
    
    # Inject metadata
    result = inject_agent_metadata(
        sample_result,
        inputs=[
            {"file": "bidirectional_survivors.json", "required": True},
            "optimal_window_config.json"  # Simple string format
        ],
        outputs=["optimal_scorer_config.json"],
        parent_run_id="step1_20251201_143022_xyz789",
        pipeline_step=2,
        follow_up_agent="full_scoring_agent",
        confidence=0.92,
        suggested_params={"threshold": 0.012, "k_folds": 5},
        reasoning="High survivor overlap suggests strong PRNG signal.",
        success_criteria_met=True,
        retry_count=0
    )
    
    # Verify
    assert "agent_metadata" in result
    assert result["agent_metadata"]["pipeline_step"] == 2
    assert result["agent_metadata"]["pipeline_step_name"] == "scorer_meta_optimizer"
    assert result["agent_metadata"]["confidence"] == 0.92
    assert result["agent_metadata"]["follow_up_agent"] == "full_scoring_agent"
    assert "timestamp_injected" in result["agent_metadata"]
    assert result["run_metadata"]["schema_version"] == "1.0.3"
    
    print("✓ All tests passed!")
    print("\nSample output:")
    print(json.dumps(result, indent=2, default=str))
