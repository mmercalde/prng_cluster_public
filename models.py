#!/usr/bin/env python3
"""
Data Models for Distributed PRNG Analysis System
Extracted from coordinator.py - contains all dataclass definitions exactly as they exist
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
import time

@dataclass
class WorkerNode:
    hostname: str
    username: str
    gpu_count: int
    gpu_type: str
    python_env: str
    script_path: str
    password: Optional[str] = None

@dataclass
class JobSpec:
    job_id: str
    prng_type: str
    mapping_type: str
    seeds: List[int]
    samples: int
    lmax: int
    grid_size: int
    mining_mode: bool
    search_type: str
    target_draw: Optional[List[int]] = None
    payload: Optional[Dict[str, Any]] = None
    analysis_type: str = 'statistical'
    # retry/timeout tuning info
    attempt: int = 0

@dataclass
class GPUWorker:
    node: WorkerNode
    gpu_id: int
    worker_id: str
    
    def __post_init__(self):
        self.worker_id = f"{self.node.hostname}_gpu{self.gpu_id}"

@dataclass
class JobResult:
    job_id: str
    node: str
    success: bool
    results: Optional[Dict]
    error: Optional[str]
    runtime: float

@dataclass
class ProgressState:
    """Internal progress tracking for automatic fault tolerance"""
    analysis_id: str
    total_jobs: int
    completed_jobs: Dict[str, Dict] # job_id -> result_data
    failed_jobs: Dict[str, str] # job_id -> error_message
    pending_jobs: List[Dict] # Serialized job assignments
    retry_count: Dict[str, int] # job_id -> retry_attempts
    start_time: float
    last_update: float

@dataclass
class GPUPerformanceProfile:
    """Track GPU performance characteristics for optimization"""
    gpu_type: str
    architecture: str
    node_hostname: str
    # Performance metrics
    avg_completion_time: float = 0.0
    seeds_per_second: float = 0.0
    # Historical data (rolling window)
    completion_times: deque = field(default_factory=lambda: deque(maxlen=10))
    total_jobs_completed: int = 0
    total_seeds_processed: int = 0
    # Performance tracking
    last_update: float = field(default_factory=time.time)
