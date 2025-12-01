#!/usr/bin/env python3
"""
Centralized Progress Display Module for PRNG Cluster Operations
================================================================
Version: 2.0 - Dual Terminal Support
Date: 2025-11-30

Two modes:
1. WRITER MODE: Called by coordinator, writes progress to JSON file
2. MONITOR MODE: Run separately, reads JSON and displays rich panel

Usage:
    # In coordinator (writes progress):
    from progress_display import ProgressWriter
    writer = ProgressWriter()
    writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds, runtime)
    
    # In separate terminal (displays progress):
    python3 progress_monitor.py
"""

import json
import time
import os
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Progress file location
PROGRESS_FILE = "/tmp/cluster_progress.json"


@dataclass
class NodeStats:
    hostname: str
    gpu_type: str = "Unknown"
    total_gpus: int = 0
    active_gpus: int = 0
    jobs_completed: int = 0
    seeds_processed: int = 0
    current_seeds_per_sec: float = 0.0


class ProgressWriter:
    """
    Writes progress updates to a JSON file.
    Used by coordinator.py during execution.
    """
    
    def __init__(self, step_name: str = "Distributed GPU Processing", total_jobs: int = 100, total_seeds: int = 0):
        self.step_name = step_name
        self.total_jobs = total_jobs
        self.total_seeds = total_seeds
        self.jobs_completed = 0
        self.seeds_completed = 0
        self.start_time = time.time()
        self.nodes: Dict[str, dict] = {}
        self._write_progress()
    
    def register_node(self, hostname: str, gpu_type: str, gpu_count: int):
        """Register a cluster node"""
        self.nodes[hostname] = {
            "hostname": hostname,
            "gpu_type": gpu_type,
            "total_gpus": gpu_count,
            "active_gpus": 0,
            "jobs_completed": 0,
            "seeds_processed": 0,
            "current_seeds_per_sec": 0.0
        }
        self._write_progress()
    
    def log_gpu_result(self, hostname: str, gpu_id: int, gpu_type: str,
                       seeds: int, runtime: float, success: bool = True):
        """Log a GPU job result"""
        if hostname not in self.nodes:
            self.nodes[hostname] = {
                "hostname": hostname,
                "gpu_type": gpu_type,
                "total_gpus": 1,
                "active_gpus": 0,
                "jobs_completed": 0,
                "seeds_processed": 0,
                "current_seeds_per_sec": 0.0
            }
        
        seeds_per_sec = seeds / runtime if runtime > 0 else 0
        
        if success:
            self.nodes[hostname]["jobs_completed"] += 1
            self.nodes[hostname]["seeds_processed"] += seeds
            self.nodes[hostname]["current_seeds_per_sec"] = seeds_per_sec
            self.jobs_completed += 1
            self.seeds_completed += seeds
        
        self._write_progress()
    
    def update_progress(self, jobs_done: int = None, chunks_total: int = None):
        """Update overall progress"""
        if jobs_done is not None:
            self.jobs_completed = jobs_done
        if chunks_total is not None:
            self.total_jobs = chunks_total
        self._write_progress()
    
    def finish(self):
        """Mark as complete"""
        self.jobs_completed = self.total_jobs
        self._write_progress(finished=True)
    
    def _write_progress(self, finished: bool = False):
        """Write current state to JSON file"""
        elapsed = time.time() - self.start_time
        
        state = {
            "step_name": self.step_name,
            "total_jobs": self.total_jobs,
            "total_seeds": self.total_seeds,
            "jobs_completed": self.jobs_completed,
            "seeds_completed": self.seeds_completed,
            "elapsed_seconds": elapsed,
            "start_time": self.start_time,
            "nodes": self.nodes,
            "finished": finished,
            "updated_at": time.time()
        }
        
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            pass  # Silent fail - don't break coordinator


# For backward compatibility
ClusterProgress = ProgressWriter


if __name__ == "__main__":
    print("This module provides ProgressWriter for coordinator.py")
    print(f"Progress file: {PROGRESS_FILE}")
    print("\nTo monitor progress, run: python3 progress_monitor.py")
