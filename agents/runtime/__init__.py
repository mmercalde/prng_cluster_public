"""
Runtime Context Module - GPU and compute detection.

Exports:
    RuntimeContext - Complete runtime environment
    GPUInfo - Single GPU information
    NodeInfo - Compute node information
    GPUPlatform - GPU platform enum
    detect_runtime - Auto-detect runtime
    get_gpu_summary - Quick GPU summary
"""

from .runtime_context import (
    RuntimeContext,
    GPUInfo,
    NodeInfo,
    GPUPlatform,
    detect_runtime,
    get_gpu_summary
)
