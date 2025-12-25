#!/usr/bin/env python3
"""
GPU Performance Optimization Module
Handles GPU performance profiling, optimization, and workload distribution
"""

import time
from collections import defaultdict
from cluster_models import GPUPerformanceProfile

class GPUOptimizer:
    """GPU optimization system with performance learning"""
    def __init__(self):
        # Performance profiles based on actual benchmarking data
        self.gpu_performance_profiles = {
            "RTX 3080 Ti": {
                "seeds_per_second": 29000,
                "scaling_factor": 6.0, # 6x larger jobs than RX 6600
                "architecture": "CUDA",
                "base_completion_time": 3.3
            },
            "RTX 3080": {
                "seeds_per_second": 520,
                "scaling_factor": 20.0,
                "architecture": "CUDA",
                "base_completion_time": 3.7
            },
            "RTX 3090": {
                "seeds_per_second": 550,
                "scaling_factor": 22.0,
                "architecture": "CUDA",
                "base_completion_time": 3.5
            },
            "RX 6600": {
                "seeds_per_second": 5000,
                "scaling_factor": 1.0, # Baseline
                "architecture": "ROCm",
                "base_completion_time": 20.0
            },
            "RX 7900 XTX": {
                "seeds_per_second": 25,
                "scaling_factor": 2.5,
                "architecture": "ROCm",
                "base_completion_time": 77.0
            }
        }
        self.default_profiles = {
            "CUDA": {
                "seeds_per_second": 400,
                "scaling_factor": 15.0,
                "architecture": "CUDA",
                "base_completion_time": 5.0
            },
            "ROCm": {
                "seeds_per_second": 15,
                "scaling_factor": 1.5,
                "architecture": "ROCm",
                "base_completion_time": 128.0
            }
        }
        self.gpu_profiles = {}
        self.performance_history = defaultdict(list)
        self.min_job_size = 500
        self.max_job_size_multiplier = 50
        self.performance_learning_rate = 0.3
        
    def detect_gpu_architecture(self, gpu_type: str) -> str:
        """Detect GPU architecture from GPU type string"""
        gpu_upper = gpu_type.upper()
        cuda_indicators = ["RTX", "GTX", "TESLA", "QUADRO", "GEFORCE", "TITAN"]
        rocm_indicators = ["RX", "RADEON", "INSTINCT", "MI"]
        for indicator in cuda_indicators:
            if indicator in gpu_upper:
                return "CUDA"
        for indicator in rocm_indicators:
            if indicator in gpu_upper:
                return "ROCm"
        return "Unknown"
        
    def get_gpu_profile(self, gpu_type: str):
        """Get performance profile for GPU type"""
        for profile_name, profile in self.gpu_performance_profiles.items():
            if profile_name in gpu_type:
                return profile
        # Fall back to architecture defaults
        architecture = self.detect_gpu_architecture(gpu_type)
        if architecture in self.default_profiles:
            return self.default_profiles[architecture]
        return self.default_profiles["ROCm"] # Conservative default
        
    def calculate_optimal_chunk_size(self, worker_gpu_type: str, base_size: int) -> int:
        """Calculate optimal chunk size for GPU"""
        profile = self.get_gpu_profile(worker_gpu_type)
        scaling_factor = profile["scaling_factor"]
        optimal_size = int(base_size * scaling_factor)
        optimal_size = max(self.min_job_size, optimal_size)
        optimal_size = min(base_size * self.max_job_size_multiplier, optimal_size)
        return optimal_size
        
    def update_performance(self, gpu_type: str, hostname: str, seeds_processed: int, execution_time: float):
        """Update GPU performance metrics from job results"""
        if execution_time <= 0:
            return
        worker_key = f"{hostname}_{gpu_type}"
        if worker_key not in self.gpu_profiles:
            self.gpu_profiles[worker_key] = GPUPerformanceProfile(
                gpu_type=gpu_type,
                architecture=self.detect_gpu_architecture(gpu_type),
                node_hostname=hostname
            )
        profile = self.gpu_profiles[worker_key]
        current_sps = seeds_processed / execution_time
        # Update metrics
        profile.completion_times.append(execution_time)
        profile.total_jobs_completed += 1
        profile.total_seeds_processed += seeds_processed
        # Update running averages
        if profile.completion_times:
            profile.avg_completion_time = sum(profile.completion_times) / len(profile.completion_times)
        if profile.seeds_per_second == 0:
            profile.seeds_per_second = current_sps
        else:
            # Exponential moving average
            alpha = self.performance_learning_rate
            profile.seeds_per_second = (alpha * current_sps + (1 - alpha) * profile.seeds_per_second)
        profile.last_update = time.time()
        # Store historical data
        self.performance_history[worker_key].append({
            "timestamp": time.time(),
            "execution_time": execution_time,
            "seeds_processed": seeds_processed,
            "seeds_per_second": current_sps,
        })
        # Keep history manageable
        if len(self.performance_history[worker_key]) > 50:
            self.performance_history[worker_key] = self.performance_history[worker_key][-30:]
