"""
GPU Memory Reporting Mixin (v3.1.2)

Provides honest GPU memory reporting (NOT compute utilization).
True utilization requires nvidia-smi polling or NVML integration.

Usage:
    class MyWrapper(ModelInterface, GPUMemoryMixin):
        def fit(self, ...):
            # Training code
            self._gpu_info = self.log_gpu_memory()
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class GPUMemoryMixin:
    """Mixin for GPU memory reporting (NOT utilization)."""
    
    device: str  # Expected to be set by the class using this mixin
    
    def log_gpu_memory(self) -> Dict[str, Any]:
        """
        Log GPU memory usage during/after training.
        
        NOTE: This reports MEMORY allocation, not compute utilization.
        True utilization requires nvidia-smi polling or NVML integration.
        
        Returns:
            Dict with cuda_available, device_requested, memory_report, etc.
        """
        try:
            import torch
        except ImportError:
            return {
                "cuda_available": False,
                "device_requested": getattr(self, 'device', 'unknown'),
                "memory_report": [],
                "fallback_to_cpu": True,
                "warnings": ["PyTorch not available"]
            }
        
        gpu_info: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
            "device_requested": getattr(self, 'device', 'unknown'),
            "memory_report": [],
            "fallback_to_cpu": False,
            "warnings": []
        }
        
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                    memory_total = props.total_memory / 1e9
                    
                    gpu_info["memory_report"].append({
                        "device_id": i,
                        "name": props.name,
                        "memory_allocated_gb": round(memory_allocated, 2),
                        "memory_reserved_gb": round(memory_reserved, 2),
                        "memory_total_gb": round(memory_total, 2),
                        "memory_pct": round(memory_allocated / memory_total * 100, 1) if memory_total > 0 else 0.0
                    })
            except Exception as e:
                gpu_info["warnings"].append(f"Error reading GPU memory: {e}")
        else:
            gpu_info["fallback_to_cpu"] = True
            gpu_info["warnings"].append("CUDA not available - using CPU")
        
        return gpu_info
    
    def get_device_name(self) -> str:
        """Get the name of the current device."""
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = 0
                if hasattr(self, 'device') and ':' in str(self.device):
                    device_idx = int(str(self.device).split(':')[1])
                return torch.cuda.get_device_name(device_idx)
            return "CPU"
        except Exception:
            return "Unknown"
