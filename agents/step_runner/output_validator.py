#!/usr/bin/env python3
"""
Output Validator - Validate step outputs exist
==============================================
Version: 1.0.0
Date: 2026-01-02

Checks that expected output files were created by step execution.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from .models import StepManifest

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_outputs(
    manifest: StepManifest,
    work_dir: Optional[Path] = None
) -> Dict[str, bool]:
    """
    Check which output files exist.
    
    Args:
        manifest: StepManifest with outputs list
        work_dir: Working directory to check
    
    Returns:
        Dict mapping output filename to exists status
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    results = {}
    
    for output_file in manifest.outputs:
        output_path = work_dir / output_file
        exists = output_path.exists()
        results[output_file] = exists
        
        if exists:
            logger.debug(f"✅ Found: {output_file}")
        else:
            logger.warning(f"❌ Missing: {output_file}")
    
    return results


def all_outputs_exist(outputs_found: Dict[str, bool]) -> bool:
    """Check if all outputs were found."""
    return all(outputs_found.values())


def get_missing_outputs(outputs_found: Dict[str, bool]) -> list[str]:
    """Get list of missing output files."""
    return [name for name, exists in outputs_found.items() if not exists]


def get_output_sizes(
    manifest: StepManifest,
    work_dir: Optional[Path] = None
) -> Dict[str, int]:
    """
    Get file sizes of output files (in bytes).
    
    Returns:
        Dict mapping filename to size (-1 if not found)
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    sizes = {}
    
    for output_file in manifest.outputs:
        output_path = work_dir / output_file
        if output_path.exists():
            sizes[output_file] = output_path.stat().st_size
        else:
            sizes[output_file] = -1
    
    return sizes


def validate_output_not_empty(
    manifest: StepManifest,
    work_dir: Optional[Path] = None,
    min_size_bytes: int = 10
) -> Dict[str, bool]:
    """
    Check that output files are not empty/minimal.
    
    Args:
        manifest: StepManifest with outputs list
        work_dir: Working directory
        min_size_bytes: Minimum acceptable file size
    
    Returns:
        Dict mapping filename to "is valid" status
    """
    work_dir = Path(work_dir) if work_dir else Path.cwd()
    results = {}
    
    for output_file in manifest.outputs:
        output_path = work_dir / output_file
        
        if not output_path.exists():
            results[output_file] = False
            continue
        
        size = output_path.stat().st_size
        results[output_file] = size >= min_size_bytes
        
        if size < min_size_bytes:
            logger.warning(f"⚠️  {output_file}: File too small ({size} bytes)")
    
    return results


def format_validation_report(
    outputs_found: Dict[str, bool],
    output_sizes: Optional[Dict[str, int]] = None
) -> str:
    """
    Format validation results as human-readable report.
    
    Returns:
        Formatted string
    """
    lines = ["Output Validation:"]
    
    for name, exists in outputs_found.items():
        status = "✅" if exists else "❌"
        size_str = ""
        
        if output_sizes and name in output_sizes:
            size = output_sizes[name]
            if size >= 0:
                size_str = f" ({_format_size(size)})"
        
        lines.append(f"  {status} {name}{size_str}")
    
    all_ok = all(outputs_found.values())
    lines.append("")
    lines.append(f"Status: {'All outputs found ✅' if all_ok else 'Missing outputs ❌'}")
    
    return "\n".join(lines)


def _format_size(bytes: int) -> str:
    """Format file size in human-readable form."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.1f} GB"
