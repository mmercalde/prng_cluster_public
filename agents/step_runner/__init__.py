#!/usr/bin/env python3
"""
Step Runner - Modular step automation framework
===============================================
Version: 1.0.0
Date: 2026-01-02

Provides modular, configuration-driven execution of pipeline steps.

Usage (Human-Directed Mode):
    from step_runner import run_step
    
    result = run_step(
        step=1,
        params={"lottery_file": "synthetic_lottery.json"}
    )
    
    print(f"Status: {result.status}")
    print(f"Survivors: {result.metrics.get('survivor_count')}")

Components:
    - models: Pydantic models (StepManifest, StepResult, etc.)
    - manifest_loader: Load and parse agent manifests
    - command_builder: Build subprocess commands
    - step_executor: Execute steps with timeout handling
    - output_validator: Validate output files exist
    - metrics_extractor: Extract metrics from outputs
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Models
from .models import (
    # Enums
    AssessmentLevel,
    RecommendationAction,
    RunMode,
    StepStatus,
    # Models
    StepManifest,
    ActionConfig,
    ActionResult,
    StepResult,
    StepDecision,
    MetricsResult,
    PreRunCheck,
    EvaluationConfig,
    MetricsExtractionConfig,
    # Mappings
    STEP_MANIFEST_MAP,
    STEP_DISPLAY_NAMES,
    # Utilities
    get_manifest_filename,
    get_step_display_name,
)

# Components
from .manifest_loader import (
    load_manifest,
    list_available_manifests,
    validate_manifest,
)

from .command_builder import (
    build_command,
    build_command_for_action,
    merge_params,
    validate_params,
    get_required_params,
    format_command_display,
)

from .step_executor import (
    execute_step,
    execute_multi_action_step,
    print_result_summary,
    check_script_exists,
    check_python_available,
)

from .output_validator import (
    validate_outputs,
    all_outputs_exist,
    get_missing_outputs,
    get_output_sizes,
    validate_output_not_empty,
    format_validation_report,
)

from .metrics_extractor import (
    extract_metrics,
    extract_step1_metrics,
    extract_step5_metrics,
    format_metrics_report,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def run_step(
    step: int,
    params: Optional[Dict[str, Any]] = None,
    work_dir: Optional[str] = None,
    manifest_dir: str = "agent_manifests",
    stream_output: bool = True
) -> StepResult:
    """
    Run a pipeline step (human-directed mode).
    
    This is the main entry point for running steps. It:
    1. Loads the manifest for the step
    2. Merges params with defaults
    3. Validates parameters
    4. Builds and executes the command (handles multi-action steps)
    5. Validates outputs exist
    6. Extracts metrics from outputs
    
    Args:
        step: Pipeline step number (1-6)
        params: Runtime parameters (merged with manifest defaults)
        work_dir: Working directory (defaults to current directory)
        manifest_dir: Directory containing manifest files
        stream_output: If True, stream stdout in real-time
    
    Returns:
        StepResult with status, metrics, and execution details
    
    Example:
        result = run_step(
            step=1,
            params={
                "lottery_file": "synthetic_lottery.json",
                "prng_type": "java_lcg",
                "window_trials": 50
            }
        )
        
        if result.success:
            print(f"Survivors: {result.metrics['survivor_count']}")
        else:
            print(f"Failed: {result.error_message}")
    """
    params = params or {}
    work_path = Path(work_dir) if work_dir else Path.cwd()
    
    logger.info(f"Starting Step {step}: {get_step_display_name(step)}")
    
    # 1. Load manifest
    try:
        manifest = load_manifest(step, manifest_dir, work_path)
    except FileNotFoundError as e:
        return _error_result(step, str(e))
    
    # 2. Merge params with defaults
    full_params = merge_params(manifest, params)
    logger.debug(f"Merged params: {full_params}")
    
    # 3. Validate parameters
    valid, errors = validate_params(manifest, full_params)
    if not valid:
        return _error_result(step, f"Invalid parameters: {errors}")
    
    # 4. Check if multi-action or single-action step
    if manifest.is_multi_action:
        # Multi-action step: use execute_multi_action_step
        logger.info(f"Multi-action step detected: {manifest.action_count} actions")
        result = execute_multi_action_step(
            manifest=manifest,
            params=full_params,
            work_dir=work_path,
            stream_output=stream_output
        )
    else:
        # Single-action step: build command and execute
        if not check_script_exists(manifest.script, work_path):
            return _error_result(step, f"Script not found: {manifest.script}")
        
        command = build_command(manifest, full_params)
        logger.info(f"Command: {format_command_display(command)}")
        
        result = execute_step(
            command=command,
            step=step,
            timeout_minutes=manifest.timeout_minutes,
            work_dir=work_path,
            params=full_params,
            stream_output=stream_output
        )
    
    # 5. Validate outputs
    result.outputs_found = validate_outputs(manifest, work_path)
    logger.info(format_validation_report(result.outputs_found))
    
    # 6. Extract metrics
    if all_outputs_exist(result.outputs_found):
        metrics_result = extract_metrics(manifest, work_path, full_params)
        result.metrics = metrics_result.metrics
        result.metrics_errors = metrics_result.errors
        
        if not metrics_result.success:
            result.status = StepStatus.FAILED_METRICS
            logger.warning("Metrics extraction had errors")
        
        logger.info(format_metrics_report(metrics_result))
    else:
        missing = get_missing_outputs(result.outputs_found)
        logger.warning(f"Skipping metrics extraction - missing outputs: {missing}")
        result.metrics_errors = [f"Missing outputs: {missing}"]
    
    # 7. Final status check
    if result.exit_code == 0 and all_outputs_exist(result.outputs_found):
        result.status = StepStatus.SUCCESS
    elif result.status not in (StepStatus.TIMEOUT, StepStatus.FAILED_METRICS):
        result.status = StepStatus.FAILED
    
    return result


def _error_result(step: int, error: str) -> StepResult:
    """Create an error result before execution."""
    from datetime import datetime
    return StepResult(
        step=step,
        step_name=get_step_display_name(step),
        status=StepStatus.FAILED,
        exit_code=-1,
        duration_seconds=0,
        error_message=error,
        started_at=datetime.now(),
        completed_at=datetime.now()
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # High-level API
    "run_step",
    
    # Enums
    "AssessmentLevel",
    "RecommendationAction", 
    "RunMode",
    "StepStatus",
    
    # Models
    "StepManifest",
    "ActionConfig",
    "ActionResult",
    "StepResult",
    "StepDecision",
    "MetricsResult",
    "PreRunCheck",
    "EvaluationConfig",
    "MetricsExtractionConfig",
    
    # Mappings
    "STEP_MANIFEST_MAP",
    "STEP_DISPLAY_NAMES",
    
    # Manifest loading
    "load_manifest",
    "list_available_manifests",
    "validate_manifest",
    "get_manifest_filename",
    "get_step_display_name",
    
    # Command building
    "build_command",
    "build_command_for_action",
    "merge_params",
    "validate_params",
    "get_required_params",
    "format_command_display",
    
    # Execution
    "execute_step",
    "execute_multi_action_step",
    "print_result_summary",
    "check_script_exists",
    "check_python_available",
    
    # Output validation
    "validate_outputs",
    "all_outputs_exist",
    "get_missing_outputs",
    "get_output_sizes",
    "validate_output_not_empty",
    "format_validation_report",
    
    # Metrics extraction
    "extract_metrics",
    "extract_step1_metrics",
    "extract_step5_metrics",
    "format_metrics_report",
]


# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.2.0"  # Added multi-action + distributed handler
