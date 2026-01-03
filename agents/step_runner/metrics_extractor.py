#!/usr/bin/env python3
"""
Metrics Extractor - Extract metrics from step output files
==========================================================
Version: 1.0.0
Date: 2026-01-02

Extracts metrics from JSON output files based on manifest configuration.
Fails loudly with structured errors - no silent failures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .models import StepManifest, MetricsResult

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE EXPRESSION EVALUATION
# =============================================================================

# Allowed functions/names for safe_eval
SAFE_BUILTINS = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


def safe_eval(expression: str, context: Dict[str, Any]) -> Any:
    """
    Safely evaluate a simple Python expression.
    
    Limited to basic operations and allowed functions.
    
    Args:
        expression: Python expression string
        context: Variable context (e.g., {"data": {...}})
    
    Returns:
        Evaluated result
    
    Raises:
        Exception on invalid/unsafe expressions
    """
    # Build safe namespace
    namespace = {**SAFE_BUILTINS, **context}
    
    try:
        # Use eval with restricted namespace
        result = eval(expression, {"__builtins__": {}}, namespace)
        return result
    except Exception as e:
        raise ValueError(f"Expression evaluation failed: {expression} -> {e}")


# =============================================================================
# METRICS EXTRACTION
# =============================================================================

def extract_metrics(
    manifest: StepManifest,
    work_dir: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None
) -> MetricsResult:
    """
    Extract metrics from output files based on manifest configuration.
    
    Args:
        manifest: StepManifest with metrics_extraction config
        work_dir: Working directory containing output files
        params: Runtime parameters (available for computed metrics)
    
    Returns:
        MetricsResult with extracted metrics and any errors
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    params = params or {}
    
    metrics = {}
    errors = []
    warnings = []
    
    extraction_config = manifest.metrics_extraction
    
    # Extract from output files
    for output_file, extractors in extraction_config.from_output_files.items():
        file_path = work_dir / output_file
        
        if not file_path.exists():
            errors.append(f"Metrics file not found: {output_file}")
            continue
        
        # Load JSON data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {output_file}: {e}")
            continue
        except Exception as e:
            errors.append(f"Failed to read {output_file}: {e}")
            continue
        
        # Extract each metric
        for metric_name, expression in extractors.items():
            try:
                value = safe_eval(expression, {"data": data})
                metrics[metric_name] = value
                logger.debug(f"Extracted {metric_name}: {value}")
            except Exception as e:
                errors.append(f"Failed to extract {metric_name} from {output_file}: {e}")
                metrics[metric_name] = None  # Explicit None, not silent skip
    
    # Compute derived metrics
    for metric_name, expression in extraction_config.computed.items():
        try:
            # Computed metrics have access to both extracted metrics and params
            context = {"metrics": metrics, "params": params}
            value = safe_eval(expression, context)
            metrics[metric_name] = value
            logger.debug(f"Computed {metric_name}: {value}")
        except Exception as e:
            # Computed metrics failing is a warning, not error
            warnings.append(f"Failed to compute {metric_name}: {e}")
            metrics[metric_name] = None
    
    # Determine success
    success = len(errors) == 0
    
    if not success:
        logger.warning(f"Metrics extraction had {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return MetricsResult(
        success=success,
        metrics=metrics,
        errors=errors,
        warnings=warnings
    )


def extract_step1_metrics(work_dir: Optional[Path] = None) -> MetricsResult:
    """
    Specialized metrics extraction for Step 1 (Window Optimizer).
    
    Uses hardcoded extraction when manifest config isn't available.
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    
    metrics = {}
    errors = []
    warnings = []
    
    # Extract from bidirectional_survivors.json
    survivors_file = work_dir / "bidirectional_survivors.json"
    if survivors_file.exists():
        try:
            with open(survivors_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                metrics["survivor_count"] = len(data)
                
                # Count by direction
                forward = sum(1 for s in data if s.get("direction") == "forward")
                reverse = sum(1 for s in data if s.get("direction") == "reverse")
                metrics["forward_survivors"] = forward
                metrics["reverse_survivors"] = reverse
                
                # Count by skip mode
                skip_modes = {}
                for s in data:
                    mode = s.get("skip_mode", "unknown")
                    skip_modes[mode] = skip_modes.get(mode, 0) + 1
                metrics["skip_mode_distribution"] = skip_modes
                
            elif isinstance(data, dict):
                survivors = data.get("survivors", [])
                metrics["survivor_count"] = len(survivors)
            else:
                errors.append("Unexpected format in bidirectional_survivors.json")
                
        except Exception as e:
            errors.append(f"Failed to extract from bidirectional_survivors.json: {e}")
    else:
        errors.append("bidirectional_survivors.json not found")
    
    # Extract from optimal_window_config.json
    config_file = work_dir / "optimal_window_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            metrics["window_size"] = config.get("window_size")
            metrics["forward_threshold"] = config.get("forward_threshold")
            metrics["reverse_threshold"] = config.get("reverse_threshold")
            metrics["skip_mode"] = config.get("skip_mode", "constant")
            metrics["best_trial_score"] = config.get("best_score")
            
        except Exception as e:
            errors.append(f"Failed to extract from optimal_window_config.json: {e}")
    else:
        warnings.append("optimal_window_config.json not found")
    
    return MetricsResult(
        success=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        warnings=warnings
    )


def extract_step5_metrics(work_dir: Optional[Path] = None) -> MetricsResult:
    """
    Specialized metrics extraction for Step 5 (Anti-Overfit Training).
    
    Extracts from sidecar file with signal quality information.
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    
    metrics = {}
    errors = []
    warnings = []
    
    # Extract from sidecar
    sidecar_file = work_dir / "models" / "reinforcement" / "best_model.meta.json"
    if sidecar_file.exists():
        try:
            with open(sidecar_file, 'r') as f:
                sidecar = json.load(f)
            
            # Signal quality
            signal_quality = sidecar.get("signal_quality", {})
            metrics["signal_quality"] = signal_quality.get("quality")
            metrics["signal_confidence"] = signal_quality.get("confidence")
            metrics["target_variance"] = signal_quality.get("target_variance")
            
            # Model info
            metrics["model_type"] = sidecar.get("model_type")
            metrics["checkpoint_path"] = sidecar.get("checkpoint_path")
            
            # Data context
            data_context = sidecar.get("data_context", {})
            metrics["fingerprint"] = data_context.get("fingerprint_hash")
            metrics["fingerprint_version"] = data_context.get("fingerprint_version")
            
            training_window = data_context.get("training_window", {})
            metrics["training_draws"] = training_window.get("draw_count")
            
            holdout_window = data_context.get("holdout_window", {})
            metrics["holdout_draws"] = holdout_window.get("draw_count")
            
            # PRNG hypothesis
            prng = data_context.get("prng_hypothesis", {})
            metrics["prng_type"] = prng.get("prng_type")
            
        except Exception as e:
            errors.append(f"Failed to extract from sidecar: {e}")
    else:
        errors.append("Sidecar not found: models/reinforcement/best_model.meta.json")
    
    return MetricsResult(
        success=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        warnings=warnings
    )


def format_metrics_report(metrics_result: MetricsResult) -> str:
    """
    Format metrics result as human-readable report.
    
    Returns:
        Formatted string
    """
    lines = ["Extracted Metrics:"]
    
    for name, value in metrics_result.metrics.items():
        if value is None:
            lines.append(f"  ❌ {name}: (extraction failed)")
        elif isinstance(value, float):
            lines.append(f"  ✅ {name}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"  ✅ {name}: {json.dumps(value)}")
        else:
            lines.append(f"  ✅ {name}: {value}")
    
    if metrics_result.errors:
        lines.append("")
        lines.append("Errors:")
        for error in metrics_result.errors:
            lines.append(f"  ❌ {error}")
    
    if metrics_result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in metrics_result.warnings:
            lines.append(f"  ⚠️  {warning}")
    
    lines.append("")
    status = "✅ Success" if metrics_result.success else "❌ Failed"
    lines.append(f"Status: {status}")
    
    return "\n".join(lines)
