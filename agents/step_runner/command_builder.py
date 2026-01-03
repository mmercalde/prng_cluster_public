#!/usr/bin/env python3
"""
Command Builder - Build subprocess commands from manifests
==========================================================
Version: 1.0.0
Date: 2026-01-02

Converts manifest args_map + runtime params into
subprocess-ready command lists.
"""

import logging
from typing import Dict, List, Any, Optional

from .models import StepManifest, ActionConfig

logger = logging.getLogger(__name__)


# =============================================================================
# COMMAND BUILDING
# =============================================================================

def build_command_for_action(
    action: ActionConfig,
    params: Dict[str, Any],
    python_executable: str = "python3"
) -> List[str]:
    """
    Build subprocess command for a specific action.
    
    Used for multi-action steps where each action has its own args_map.
    
    Args:
        action: ActionConfig from manifest
        params: Runtime parameters
        python_executable: Python interpreter to use
    
    Returns:
        Command list suitable for subprocess.run()
    """
    cmd = [python_executable, action.script]
    
    # Process args_map: CLI arg name → param key
    for cli_arg, param_key in action.args_map.items():
        value = params.get(param_key)
        
        if value is None:
            continue
        
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{cli_arg}")
        else:
            cmd.extend([f"--{cli_arg}", str(value)])
    
    logger.debug(f"Built action command: {' '.join(cmd)}")
    return cmd


def build_command(
    manifest: StepManifest,
    params: Dict[str, Any],
    python_executable: str = "python3"
) -> List[str]:
    """
    Build subprocess command from manifest and runtime parameters.
    
    Args:
        manifest: Parsed StepManifest
        params: Runtime parameters (merged with defaults)
        python_executable: Python interpreter to use
    
    Returns:
        Command list suitable for subprocess.run()
    
    Example:
        manifest.script = "window_optimizer.py"
        manifest.args_map = {"lottery-file": "lottery_file", "trials": "window_trials"}
        params = {"lottery_file": "synthetic.json", "window_trials": 50}
        
        → ["python3", "window_optimizer.py", "--lottery-file", "synthetic.json", "--trials", "50"]
    """
    cmd = [python_executable, manifest.script]
    
    # Process args_map: CLI arg name → param key
    for cli_arg, param_key in manifest.args_map.items():
        value = params.get(param_key)
        
        if value is None:
            # Skip None values
            continue
        
        if isinstance(value, bool):
            # Boolean flags: only add if True
            if value:
                cmd.append(f"--{cli_arg}")
        else:
            # Key-value pairs
            cmd.extend([f"--{cli_arg}", str(value)])
    
    logger.debug(f"Built command: {' '.join(cmd)}")
    return cmd


def merge_params(
    manifest: StepManifest,
    runtime_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge runtime params with manifest defaults.
    
    Priority: runtime_params > default_params
    
    Args:
        manifest: StepManifest with default_params
        runtime_params: User-provided parameters
    
    Returns:
        Merged parameter dictionary
    """
    merged = {}
    
    # Start with defaults
    merged.update(manifest.default_params)
    
    # Override with runtime params (non-None only)
    for key, value in runtime_params.items():
        if value is not None:
            merged[key] = value
    
    return merged


def validate_params(
    manifest: StepManifest,
    params: Dict[str, Any]
) -> tuple[bool, List[str]]:
    """
    Validate parameters against manifest bounds.
    
    Args:
        manifest: StepManifest with parameter_bounds
        params: Parameters to validate
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    for param_name, bounds in manifest.parameter_bounds.items():
        if param_name not in params:
            continue
        
        value = params[param_name]
        param_type = bounds.get("type", "any")
        
        # Type validation
        if param_type == "int":
            if not isinstance(value, int):
                errors.append(f"{param_name}: expected int, got {type(value).__name__}")
                continue
            
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            
            if min_val is not None and value < min_val:
                errors.append(f"{param_name}: {value} < minimum {min_val}")
            if max_val is not None and value > max_val:
                errors.append(f"{param_name}: {value} > maximum {max_val}")
        
        elif param_type == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"{param_name}: expected float, got {type(value).__name__}")
                continue
            
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            
            if min_val is not None and value < min_val:
                errors.append(f"{param_name}: {value} < minimum {min_val}")
            if max_val is not None and value > max_val:
                errors.append(f"{param_name}: {value} > maximum {max_val}")
        
        elif param_type == "choice":
            choices = bounds.get("choices", [])
            if value not in choices:
                errors.append(f"{param_name}: {value} not in choices {choices}")
        
        elif param_type == "bool":
            if not isinstance(value, bool):
                errors.append(f"{param_name}: expected bool, got {type(value).__name__}")
    
    return len(errors) == 0, errors


def get_required_params(manifest: StepManifest) -> List[str]:
    """
    Get list of parameters that have no defaults.
    
    These must be provided at runtime.
    """
    required = []
    
    # Params referenced in args_map but not in default_params
    for cli_arg, param_key in manifest.args_map.items():
        if param_key not in manifest.default_params:
            required.append(param_key)
    
    return required


def format_command_display(command: List[str], max_width: int = 80) -> str:
    """
    Format command for display (with line wrapping).
    
    Args:
        command: Command list
        max_width: Maximum line width
    
    Returns:
        Formatted string suitable for logging
    """
    cmd_str = " ".join(command)
    
    if len(cmd_str) <= max_width:
        return cmd_str
    
    # Wrap at argument boundaries
    lines = []
    current_line = command[0] + " " + command[1]  # python3 script.py
    
    i = 2
    while i < len(command):
        arg = command[i]
        
        if arg.startswith("--"):
            # Start of new argument
            if i + 1 < len(command) and not command[i + 1].startswith("--"):
                # Has value
                arg_str = f"{arg} {command[i + 1]}"
                i += 2
            else:
                # Flag only
                arg_str = arg
                i += 1
        else:
            arg_str = arg
            i += 1
        
        if len(current_line) + len(arg_str) + 1 > max_width:
            lines.append(current_line + " \\")
            current_line = "    " + arg_str
        else:
            current_line += " " + arg_str
    
    lines.append(current_line)
    return "\n".join(lines)
