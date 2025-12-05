#!/usr/bin/env python3
"""
Runtime Context - Real GPU and compute environment detection.

Detects actual hardware using nvidia-smi and rocm-smi, providing
accurate runtime context for AI agent decision-making.

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import subprocess
import os
import re
import json


class GPUPlatform(str, Enum):
    """GPU platform type."""
    CUDA = "cuda"
    ROCM = "rocm"
    UNKNOWN = "unknown"


class GPUInfo(BaseModel):
    """Information about a single GPU."""
    
    index: int
    name: str
    platform: GPUPlatform
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    utilization_percent: float = 0.0
    temperature_c: int = 0
    
    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for context."""
        return {
            "index": self.index,
            "name": self.name,
            "platform": self.platform.value,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_utilization_pct": round(self.memory_utilization, 1),
            "gpu_utilization_pct": self.utilization_percent,
            "temperature_c": self.temperature_c
        }


class NodeInfo(BaseModel):
    """Information about a compute node."""
    
    hostname: str
    gpus: List[GPUInfo] = Field(default_factory=list)
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    
    @property
    def total_gpu_memory_mb(self) -> int:
        """Total GPU memory across all GPUs."""
        return sum(g.memory_total_mb for g in self.gpus)
    
    @property
    def available_gpu_memory_mb(self) -> int:
        """Available GPU memory across all GPUs."""
        return sum(g.memory_free_mb for g in self.gpus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for context."""
        return {
            "hostname": self.hostname,
            "gpu_count": len(self.gpus),
            "gpus": [g.to_dict() for g in self.gpus],
            "cpu_count": self.cpu_count,
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb
        }


class RuntimeContext(BaseModel):
    """
    Complete runtime environment context.
    
    Detects actual hardware configuration for AI agent awareness.
    """
    
    nodes: List[NodeInfo] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    detection_method: str = "auto"
    
    # Cluster totals
    total_gpus: int = 0
    total_cuda_gpus: int = 0
    total_rocm_gpus: int = 0
    estimated_tflops: float = 0.0
    
    @classmethod
    def detect(cls) -> "RuntimeContext":
        """
        Auto-detect runtime environment.
        
        Tries nvidia-smi first, then rocm-smi.
        """
        context = cls()
        
        # Get hostname
        hostname = os.uname().nodename
        
        # Try CUDA first
        cuda_gpus = cls._detect_nvidia_gpus()
        
        # Try ROCm
        rocm_gpus = cls._detect_rocm_gpus()
        
        # Build node info
        gpus = cuda_gpus + rocm_gpus
        
        if gpus:
            node = NodeInfo(
                hostname=hostname,
                gpus=gpus,
                cpu_count=os.cpu_count() or 0,
                memory_total_gb=cls._get_system_memory()[0],
                memory_available_gb=cls._get_system_memory()[1]
            )
            context.nodes.append(node)
        
        # Calculate totals
        context.total_gpus = len(gpus)
        context.total_cuda_gpus = len(cuda_gpus)
        context.total_rocm_gpus = len(rocm_gpus)
        context.estimated_tflops = context._estimate_tflops()
        context.detection_method = "nvidia-smi/rocm-smi"
        
        return context
    
    @staticmethod
    def _detect_nvidia_gpus() -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        gpus = []
        
        try:
            # Query GPU info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpus.append(GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            platform=GPUPlatform.CUDA,
                            memory_total_mb=int(parts[2]),
                            memory_used_mb=int(parts[3]),
                            memory_free_mb=int(parts[4]),
                            utilization_percent=float(parts[5]),
                            temperature_c=int(parts[6])
                        ))
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return gpus
    
    @staticmethod
    def _detect_rocm_gpus() -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi."""
        gpus = []
        
        try:
            # Get GPU list
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return gpus
            
            # Parse GPU names
            gpu_names = {}
            for line in result.stdout.split('\n'):
                match = re.search(r'GPU\[(\d+)\].*Card series:\s*(.+)', line)
                if match:
                    gpu_names[int(match.group(1))] = match.group(2).strip()
            
            # Get memory info
            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            gpu_memory = {}
            if mem_result.returncode == 0:
                current_gpu = None
                for line in mem_result.stdout.split('\n'):
                    gpu_match = re.search(r'GPU\[(\d+)\]', line)
                    if gpu_match:
                        current_gpu = int(gpu_match.group(1))
                    
                    if current_gpu is not None:
                        total_match = re.search(r'Total Memory.*:\s*(\d+)', line)
                        used_match = re.search(r'Used Memory.*:\s*(\d+)', line)
                        
                        if total_match:
                            gpu_memory.setdefault(current_gpu, {})['total'] = int(total_match.group(1)) // (1024 * 1024)
                        if used_match:
                            gpu_memory.setdefault(current_gpu, {})['used'] = int(used_match.group(1)) // (1024 * 1024)
            
            # Get temperature
            temp_result = subprocess.run(
                ["rocm-smi", "--showtemp"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            gpu_temps = {}
            if temp_result.returncode == 0:
                for line in temp_result.stdout.split('\n'):
                    match = re.search(r'GPU\[(\d+)\].*Temperature.*:\s*(\d+)', line)
                    if match:
                        gpu_temps[int(match.group(1))] = int(match.group(2))
            
            # Build GPU info
            for idx in gpu_names:
                mem = gpu_memory.get(idx, {})
                total = mem.get('total', 8192)  # Default 8GB
                used = mem.get('used', 0)
                
                gpus.append(GPUInfo(
                    index=idx,
                    name=gpu_names[idx],
                    platform=GPUPlatform.ROCM,
                    memory_total_mb=total,
                    memory_used_mb=used,
                    memory_free_mb=total - used,
                    utilization_percent=0.0,  # ROCm doesn't always report this
                    temperature_c=gpu_temps.get(idx, 0)
                ))
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return gpus
    
    @staticmethod
    def _get_system_memory() -> Tuple[float, float]:
        """Get system memory in GB (total, available)."""
        try:
            with open('/proc/meminfo') as f:
                meminfo = f.read()
            
            total = 0
            available = 0
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) / (1024 * 1024)
            
            return round(total, 1), round(available, 1)
        except Exception:
            return 0.0, 0.0
    
    def _estimate_tflops(self) -> float:
        """Estimate total TFLOPS based on known GPU specs."""
        # Approximate FP32 TFLOPS for common GPUs
        tflops_map = {
            "3080": 29.8,
            "3080 Ti": 34.1,
            "3090": 35.6,
            "4080": 48.7,
            "4090": 82.6,
            "6600": 10.6,
            "6600 XT": 10.6,
            "6700": 13.2,
            "6800": 16.2,
            "6900": 23.0,
        }
        
        total = 0.0
        for node in self.nodes:
            for gpu in node.gpus:
                for pattern, tflops in tflops_map.items():
                    if pattern in gpu.name:
                        total += tflops
                        break
        
        return round(total, 1)
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate runtime context as clean dict for LLM.
        
        Hybrid JSON approach - data only.
        """
        return {
            "detected_at": self.detected_at.isoformat(),
            "cluster_summary": {
                "total_gpus": self.total_gpus,
                "cuda_gpus": self.total_cuda_gpus,
                "rocm_gpus": self.total_rocm_gpus,
                "estimated_tflops": self.estimated_tflops
            },
            "nodes": [n.to_dict() for n in self.nodes]
        }
    
    def get_available_gpus(self, platform: Optional[GPUPlatform] = None) -> List[GPUInfo]:
        """Get list of available GPUs, optionally filtered by platform."""
        gpus = []
        for node in self.nodes:
            for gpu in node.gpus:
                if platform is None or gpu.platform == platform:
                    gpus.append(gpu)
        return gpus
    
    def has_sufficient_memory(self, required_mb: int) -> bool:
        """Check if cluster has sufficient GPU memory."""
        total_free = sum(
            g.memory_free_mb 
            for n in self.nodes 
            for g in n.gpus
        )
        return total_free >= required_mb


# Convenience functions
def detect_runtime() -> RuntimeContext:
    """Detect current runtime environment."""
    return RuntimeContext.detect()


def get_gpu_summary() -> Dict[str, Any]:
    """Get quick GPU summary dict."""
    ctx = RuntimeContext.detect()
    return ctx.to_context_dict()["cluster_summary"]
